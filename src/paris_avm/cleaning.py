"""Cleaning: from the raw extract to a modelling-ready table.

The order of the steps matters and is deliberate:

1. repair geography *before* dropping the identifiers used to repair it;
2. drop columns that carry no information;
3. de-duplicate re-listings, keeping the most recent one;
4. keep residential only;
5. turn impossible values into NaN (they are errors, not observations);
6. drop rows that are outside the residential market altogether;
7. impute — structural zeros first, then medians.

Every step returns the frame and appends a line to `log`, so the row count
is traceable from 100% down to whatever is left.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import config


@dataclass
class CleaningLog:
    """Row-level audit trail of the cleaning run."""

    n_raw: int
    steps: list[tuple[str, int, str]] = field(default_factory=list)

    def record(self, name: str, n_after: int, note: str = "") -> None:
        self.steps.append((name, n_after, note))

    def to_frame(self) -> pd.DataFrame:
        rows = [("raw", self.n_raw, "")] + self.steps
        out = pd.DataFrame(rows, columns=["step", "rows", "note"])
        out["retained_pct"] = (out["rows"] / self.n_raw * 100).round(2)
        return out


def standardize_zipcodes(df: pd.DataFrame) -> pd.DataFrame:
    """Repair Paris zipcodes.

    Three passes, from the most certain fix to the least:
      1. known mis-encodings mapped directly (75116 -> 75016, ...);
      2. ambiguous codes inferred from other listings at the same address;
      3. whatever is still not a Paris code becomes NaN — the row is kept,
         because everything else about the property is still usable.
    """
    df = df.copy()
    df["zipcode"] = df["zipcode"].replace(config.ZIPCODE_REPLACEMENTS)

    has_address = "address_id" in df.columns and "street_id" in df.columns
    if has_address:
        amb_mask = (
            df["zipcode"].isin(config.AMBIGUOUS_ZIPCODES)
            & df["address_id"].notna()
            & df["street_id"].notna()
        )
        if amb_mask.any():
            # Ambiguous codes are rare (a handful of rows out of 100k+), but a
            # plain groupby(...).transform(python_fn) still evaluates its
            # function once per (address_id, street_id) group across the
            # *whole* frame — tens of thousands of Python-level calls to fix
            # a dozen rows. Restrict the expensive per-group work to only the
            # groups that actually contain an ambiguous zipcode.
            keys = df.loc[amb_mask, ["address_id", "street_id"]].drop_duplicates()
            same_group = df.merge(keys, on=["address_id", "street_id"], how="inner")
            candidates = same_group[
                ~same_group["zipcode"].isin(config.AMBIGUOUS_ZIPCODES)
                & same_group["zipcode"].notna()
            ]
            if len(candidates):
                group_fill = candidates.groupby(["address_id", "street_id"])["zipcode"].agg(
                    lambda s: s.mode().iloc[0]
                )
                fill = df.loc[amb_mask, ["address_id", "street_id"]].merge(
                    group_fill.rename("fill_zip"), on=["address_id", "street_id"], how="left"
                )["fill_zip"]
                df.loc[amb_mask, "zipcode"] = fill.values

    invalid = ~df["zipcode"].between(config.PARIS_ZIP_MIN, config.PARIS_ZIP_MAX)
    df.loc[invalid, "zipcode"] = np.nan
    return df


def drop_uninformative_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Drop empty / free-text / identifier columns, then near-constant ones."""
    df = df.drop(columns=[c for c in config.DROP_COLS if c in df.columns])

    dominance = {}
    for col in df.columns:
        vc = df[col].value_counts(normalize=True, dropna=True)
        if len(vc):
            dominance[col] = vc.iloc[0]
    near_constant = [
        c for c, share in dominance.items() if share >= config.NEAR_CONSTANT_THRESHOLD
    ]
    return df.drop(columns=near_constant), near_constant


def deduplicate_relistings(df: pd.DataFrame) -> pd.DataFrame:
    """One row per property: the same flat relisted twice is one flat.

    The most recent listing wins, so the price reflects the latest state
    of the market rather than an abandoned earlier attempt.
    """
    if "external_id" not in df.columns:
        return df
    return df.sort_values("end_date").drop_duplicates(subset="external_id", keep="last")


def normalize_property_types(df: pd.DataFrame) -> pd.DataFrame:
    """Strip the `ITEM_TYPE.` prefix from the type enums."""
    df = df.copy()
    for col in ("item_type", "item_subtype"):
        if col in df.columns:
            df[col] = df[col].str.replace("ITEM_TYPE.", "", regex=False)
    return df


def keep_residential(df: pd.DataFrame) -> pd.DataFrame:
    """A parking space and a flat are not the same market, and one model
    cannot price both. Offices, storage and parking go."""
    return df[df["item_type"].isin(config.RESIDENTIAL_TYPES)]


def nullify_impossible_values(df: pd.DataFrame) -> pd.DataFrame:
    """A 300-room flat is a typo, not a mansion: replace with NaN.

    Nulling rather than dropping keeps the rest of the row — area, price
    and location are usually fine on those listings.
    """
    df = df.copy()
    for col, (lo, hi) in config.PLAUSIBLE_RANGES.items():
        if col in df.columns:
            df.loc[(df[col] < lo) | (df[col] > hi), col] = np.nan
    return df


def drop_market_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows outside the residential market on price, area or EUR/sqm.

    These are dropped rather than imputed: a listing at EUR 2,000/sqm in
    Paris is not a mispriced flat, it is something else entirely
    (a share of undivided property, a leasehold, a data error).
    """
    price_ok = df["price"].between(*config.PRICE_BOUNDS)
    area_ok = df["area"].between(*config.AREA_BOUNDS)
    ppsqm_ok = (df["price"] / df["area"]).between(*config.PRICE_PER_SQM_BOUNDS)
    return df[price_ok & area_ok & ppsqm_ok]


def impute(df: pd.DataFrame) -> pd.DataFrame:
    """Two kinds of missing, two treatments.

    `balcony_count` missing means there is no balcony -> 0.
    `room_count` missing means nobody filled the field -> median.
    Treating the first as unknown would invent balconies; treating the
    second as zero would invent studios.
    """
    df = df.copy()
    for col in config.STRUCTURAL_ZERO_COLS:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    med_cols = [c for c in config.MEDIAN_FILL_COLS if c in df.columns]
    df[med_cols] = df[med_cols].fillna(df[med_cols].median())
    return df


def encode_energy_certificate(df: pd.DataFrame) -> pd.DataFrame:
    """Map the DPE letter to an ordinal where higher = more efficient."""
    df = df.copy()
    col = "energy_certificate_efficiency_score"
    if col in df.columns:
        df["energy_score_ordinal"] = df[col].map(config.ENERGY_ORDER)
    return df


def restrict_to_listing_year(df: pd.DataFrame, year: int = config.LISTING_YEAR) -> pd.DataFrame:
    """Keep a single year of listings so price levels are comparable."""
    if "start_date" not in df.columns:
        return df
    return df[df["start_date"].dt.year == year]


def clean(df: pd.DataFrame, verbose: bool = True) -> tuple[pd.DataFrame, CleaningLog]:
    """Run the full cleaning sequence and return the frame plus its audit log."""
    log = CleaningLog(n_raw=len(df))

    df = standardize_zipcodes(df)
    log.record("zipcodes repaired", len(df), f"{df['zipcode'].isna().sum()} set to NaN")

    df, near_constant = drop_uninformative_columns(df)
    log.record("columns dropped", len(df), f"near-constant: {near_constant}")

    n = len(df)
    df = deduplicate_relistings(df)
    log.record("re-listings removed", len(df), f"-{n - len(df)} rows")

    df = normalize_property_types(df)
    n = len(df)
    df = keep_residential(df)
    log.record("residential only", len(df), f"-{n - len(df)} rows")

    df = nullify_impossible_values(df)
    log.record("impossible values nulled", len(df))

    n = len(df)
    df = drop_market_outliers(df)
    log.record("market outliers removed", len(df), f"-{n - len(df)} rows")

    df = impute(df)
    df = encode_energy_certificate(df)
    left = df.isna().sum()
    left = left[left > 0]
    log.record(
        "imputed",
        len(df),
        "NaN left: " + (", ".join(f"{c} ({n})" for c, n in left.items()) if len(left) else "none"),
    )

    n = len(df)
    df = restrict_to_listing_year(df)
    log.record(f"{config.LISTING_YEAR} listings only", len(df), f"-{n - len(df)} rows")

    if verbose:
        print(log.to_frame().to_string(index=False))
    return df.reset_index(drop=True), log
