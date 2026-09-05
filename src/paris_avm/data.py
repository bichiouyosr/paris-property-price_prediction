"""Loading the raw listing extract."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from . import config


def load_raw(path: str | Path) -> pd.DataFrame:
    """Read the raw listing extract from .xlsx, .csv or .parquet.

    Date columns are parsed here so that every downstream step can assume
    real datetimes.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. This repository ships no listing data; "
            "run `python scripts/make_synthetic_data.py` to generate a "
            "synthetic extract with the same schema."
        )

    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        df = pd.read_excel(path)
    elif suffix == ".parquet":
        df = pd.read_parquet(path)
    elif suffix in {".csv", ".gz"}:
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported extension: {suffix}")

    for col in config.DATE_COLS:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    return df


def quality_report(df: pd.DataFrame) -> dict[str, pd.DataFrame | pd.Series]:
    """Audit the raw extract before anything is changed.

    Returns the four tables worth looking at: missingness, duplicate
    listings, zipcodes that are not a Paris arrondissement, and the
    min/max of every numeric column (where impossible values show up).
    """
    missing = (df.isna().mean() * 100).sort_values(ascending=False)

    dupes = pd.Series(dtype=int)
    if "external_id" in df.columns:
        counts = df["external_id"].value_counts()
        dupes = counts[counts > 1]

    bad_zip = pd.Series(dtype=int)
    if "zipcode" in df.columns:
        invalid = ~df["zipcode"].between(config.PARIS_ZIP_MIN, config.PARIS_ZIP_MAX)
        bad_zip = df.loc[invalid, "zipcode"].value_counts()

    numeric = df.select_dtypes("number")
    ranges = pd.DataFrame(
        {
            "min": numeric.min(),
            "max": numeric.max(),
            "n_negative": (numeric < 0).sum(),
            "pct_missing": numeric.isna().mean().mul(100).round(2),
        }
    )

    return {
        "missing_pct": missing[missing > 0],
        "duplicate_external_ids": dupes,
        "invalid_zipcodes": bad_zip,
        "numeric_ranges": ranges,
    }
