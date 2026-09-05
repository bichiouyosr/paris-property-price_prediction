"""Generate a synthetic listing extract with the same schema as the real one.

No listing data ships with this repository. This script fabricates an extract
that has the same 35 columns, the same dtypes, and — importantly — the same
defects as a real marketplace export, so the cleaning code has something to
actually clean:

  * the same property relisted under one external_id,
  * zipcodes that are not a Paris arrondissement (75116, 75107, 75550, ...),
  * negative and absurd counts (-1 bedroom, 300 rooms),
  * build years of 0 and 3021,
  * columns that are 100% missing, and one that is constant,
  * non-residential rows (parking, offices, storage),
  * missingness concentrated in the fields sellers skip.

Prices are generated from a per-arrondissement EUR/sqm base with adjustments
for floor, lift, outdoor space, energy rating and age, plus noise — so the
relationships the model recovers are real relationships in this data, not an
accident. They are invented; they are not a market estimate.

Usage:
    python scripts/make_synthetic_data.py --rows 100000 --out data/listings_synthetic.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# Rough 2025 EUR/sqm level by arrondissement. Invented, but ordered the way
# the Paris market is ordered, so the geography features carry signal.
BASE_PPSQM = {
    1: 13_600, 2: 12_400, 3: 13_100, 4: 14_000, 5: 13_500, 6: 15_100, 7: 14_600,
    8: 12_600, 9: 11_700, 10: 10_700, 11: 11_100, 12: 10_600, 13: 10_200,
    14: 10_900, 15: 11_200, 16: 12_100, 17: 11_600, 18: 10_400, 19: 9_500,
    20: 9_900,
}

SUBTYPES = {
    "APARTMENT": [
        ("ITEM_TYPE.APARTMENT.FLAT", 0.72),
        ("ITEM_TYPE.APARTMENT.STUDIO", 0.14),
        ("ITEM_TYPE.APARTMENT.DUPLEX", 0.07),
        ("ITEM_TYPE.APARTMENT.LOFT", 0.04),
        ("ITEM_TYPE.APARTMENT.PENTHOUSE", 0.03),
    ],
    "HOUSE": [("ITEM_TYPE.HOUSE.VILLA", 0.5), ("ITEM_TYPE.HOUSE.TOWNHOUSE", 0.5)],
}

ENERGY_LETTERS = list("ABCDEFG")
ENERGY_WEIGHTS = [0.02, 0.05, 0.14, 0.30, 0.28, 0.13, 0.08]
ENERGY_MULT = {"A": 1.09, "B": 1.06, "C": 1.03, "D": 1.00, "E": 0.97, "F": 0.93, "G": 0.89}


def generate(n_rows: int = 100_000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # ---- geography -------------------------------------------------------
    arr_choices = np.array(list(BASE_PPSQM))
    # Volume is not uniform: the outer, denser arrondissements list more.
    weights = np.array([0.02, 0.02, 0.03, 0.02, 0.04, 0.03, 0.03, 0.03, 0.05,
                        0.05, 0.07, 0.06, 0.07, 0.06, 0.09, 0.07, 0.07, 0.07,
                        0.06, 0.06])
    weights = weights / weights.sum()
    arrondissement = rng.choice(arr_choices, size=n_rows, p=weights)
    zipcode = 75_000 + arrondissement

    # ---- property attributes --------------------------------------------
    area = np.clip(rng.lognormal(mean=3.95, sigma=0.52, size=n_rows), 9, 420).round(0)
    room_count = np.clip(np.round(area / 24 + rng.normal(0, 0.7, n_rows)), 1, 11)
    bedroom_count = np.clip(room_count - rng.integers(1, 3, n_rows), 0, 9)
    bathroom_count = np.clip(np.round(bedroom_count / 2 + rng.normal(0, 0.4, n_rows)), 1, 5)
    toilet_count = np.clip(np.round(bathroom_count + rng.normal(0, 0.4, n_rows)), 1, 5)

    floor_count = np.clip(np.round(rng.gamma(6, 1.0, n_rows)), 1, 22)
    floor = np.array([rng.integers(0, fc + 1) for fc in floor_count])

    build_year = np.clip(np.round(rng.normal(1935, 45, n_rows)), 1700, 2026)
    has_passenger_lift = (rng.random(n_rows) < np.clip(0.25 + floor_count / 20, 0, 0.95)).astype(int)
    has_cellar = (rng.random(n_rows) < 0.42).astype(int)

    balcony_count = np.where(rng.random(n_rows) < 0.30, rng.integers(1, 3, n_rows), 0)
    terrace_count = np.where(rng.random(n_rows) < 0.12, 1, 0)
    balcony_area = np.where(balcony_count > 0, np.round(rng.gamma(2, 3, n_rows), 1), 0.0)
    terrace_area = np.where(terrace_count > 0, np.round(rng.gamma(3, 6, n_rows), 1), 0.0)
    parking_boxe_count = np.where(rng.random(n_rows) < 0.13, 1, 0)
    parking_inside_count = np.where(rng.random(n_rows) < 0.08, 1, 0)

    energy = rng.choice(ENERGY_LETTERS, size=n_rows, p=ENERGY_WEIGHTS)

    item_type = rng.choice(["APARTMENT", "HOUSE"], size=n_rows, p=[0.985, 0.015])
    item_subtype = np.array(
        [
            rng.choice([s for s, _ in SUBTYPES[t]], p=[p for _, p in SUBTYPES[t]])
            for t in item_type
        ]
    )

    # ---- price -----------------------------------------------------------
    ppsqm = np.array([BASE_PPSQM[a] for a in arrondissement], dtype=float)
    ppsqm *= np.array([ENERGY_MULT[e] for e in energy])
    ppsqm *= 1 + 0.010 * np.where(floor_count > 1, floor / floor_count, 0) * 3  # top floors
    ppsqm *= 1 + 0.035 * has_passenger_lift
    ppsqm *= 1 + 0.030 * (terrace_area > 0) + 0.012 * (balcony_area > 0)
    ppsqm *= 1 + 0.020 * (parking_boxe_count + parking_inside_count)
    ppsqm *= 1 - 0.00035 * np.clip(2025 - build_year, 0, 200)  # mild age discount
    ppsqm *= np.exp(rng.normal(0, 0.11, n_rows))  # everything the data cannot see
    # Small flats trade at a premium per sqm, large ones at a discount.
    ppsqm *= (area / 55) ** -0.06

    price = np.round(ppsqm * area, -3)

    # ---- dates -----------------------------------------------------------
    start = pd.Timestamp("2025-01-01") + pd.to_timedelta(rng.integers(0, 365, n_rows), unit="D")
    # August is quiet; September rebounds.
    duration = rng.integers(7, 260, n_rows)
    end = start + pd.to_timedelta(duration, unit="D")
    publish = start + pd.to_timedelta(rng.integers(0, 3, n_rows), unit="D")

    df = pd.DataFrame(
        {
            "Unnamed: 0": np.arange(n_rows),
            "external_id": rng.integers(10_000_000, 99_999_999, n_rows),
            "client_id": rng.integers(1_000, 9_999, n_rows),
            "address_id": rng.integers(100_000, 999_999, n_rows),
            "street_id": rng.integers(10_000, 99_999, n_rows),
            "iris_id": rng.integers(751_000, 751_999, n_rows),
            "parcel_id": np.nan,
            "city": "Paris",
            "city_id": 75_056,
            "zipcode": zipcode,
            "price": price,
            "area": area,
            "floor": floor,
            "floor_count": floor_count,
            "room_count": room_count,
            "bedroom_count": bedroom_count,
            "bathroom_count": bathroom_count,
            "toilet_count": toilet_count,
            "balcony_count": balcony_count,
            "terrace_count": terrace_count,
            "balcony_area": balcony_area,
            "terrace_area": terrace_area,
            "parking_boxe_count": parking_boxe_count,
            "parking_inside_count": parking_inside_count,
            "parking_garage_count": np.nan,
            "site_area": np.nan,
            "has_cellar": has_cellar,
            "has_passenger_lift": has_passenger_lift,
            "build_year": build_year,
            "item_type": ["ITEM_TYPE." + t for t in item_type],
            "item_subtype": item_subtype,
            "energy_certificate_efficiency_score": [
                f"ENERGY.CERTIFICATE_EFFICIENCY_SCORE.{e}" for e in energy
            ],
            "start_date": start,
            "end_date": end,
            "publish_start_date": publish,
        }
    )

    return _inject_defects(df, rng)


def _inject_defects(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Make the extract dirty in the ways a real marketplace export is dirty."""
    n = len(df)

    # Free-text city, inconsistently spelled.
    spellings = ["Paris", "PARIS", "paris", "Paris 16", "Paris Cedex", "  Paris"]
    idx = rng.choice(n, size=int(n * 0.05), replace=False)
    df.loc[idx, "city"] = rng.choice(spellings, size=len(idx))

    # Zipcodes that are not a Paris arrondissement.
    for bad, share in ((75_116, 0.006), (75_107, 0.001), (75_108, 0.001),
                       (75_000, 0.001), (75_550, 0.001), (75_023, 0.0005)):
        idx = rng.choice(n, size=max(1, int(n * share)), replace=False)
        df.loc[idx, "zipcode"] = bad

    # The same property listed twice under one external_id.
    n_dupes = max(1, int(n * 0.001))
    src = rng.choice(n, size=n_dupes, replace=False)
    dst = rng.choice(n, size=n_dupes, replace=False)
    df.loc[dst, "external_id"] = df.loc[src, "external_id"].values

    # Negative counts.
    for col in ["room_count", "bedroom_count", "bathroom_count", "balcony_count"]:
        idx = rng.choice(n, size=int(n * 0.002), replace=False)
        df.loc[idx, col] = -1

    # Absurd maxima.
    df.loc[rng.choice(n, 40, replace=False), "room_count"] = rng.integers(50, 300, 40)
    df.loc[rng.choice(n, 25, replace=False), "floor"] = rng.integers(60, 120, 25)
    df.loc[rng.choice(n, 30, replace=False), "build_year"] = rng.choice([0, 1, 3021], 30)
    df.loc[rng.choice(n, 20, replace=False), "terrace_area"] = rng.integers(500, 3000, 20)

    # Prices and areas outside the residential market.
    df.loc[rng.choice(n, int(n * 0.02), replace=False), "price"] *= rng.uniform(0.05, 0.2)
    df.loc[rng.choice(n, int(n * 0.02), replace=False), "price"] *= rng.uniform(3, 8)
    df.loc[rng.choice(n, int(n * 0.005), replace=False), "area"] = rng.integers(500, 4000, int(n * 0.005))
    df["price"] = df["price"].round(-3)

    # Non-residential rows.
    idx = rng.choice(n, size=int(n * 0.004), replace=False)
    other = rng.choice(["PARKING", "OFFICE", "STORAGE_PRODUCTION", "TRADING", "MISCELLANEOUS"], size=len(idx))
    df.loc[idx, "item_type"] = ["ITEM_TYPE." + t for t in other]
    df.loc[idx, "item_subtype"] = ["ITEM_TYPE." + t for t in other]

    # Missingness where sellers skip fields.
    missing_rates = {
        "build_year": 0.53, "energy_certificate_efficiency_score": 0.48,
        "floor": 0.22, "floor_count": 0.31, "toilet_count": 0.36,
        "bathroom_count": 0.19, "bedroom_count": 0.08, "balcony_area": 0.66,
        "terrace_area": 0.71, "parking_boxe_count": 0.62, "parking_inside_count": 0.64,
        "has_cellar": 0.41, "has_passenger_lift": 0.29, "address_id": 0.04,
        "street_id": 0.04, "room_count": 0.03,
    }
    for col, rate in missing_rates.items():
        idx = rng.choice(n, size=int(n * rate), replace=False)
        df.loc[idx, col] = np.nan

    # A few stale relists from earlier years.
    idx = rng.choice(n, size=int(n * 0.0005), replace=False)
    df.loc[idx, "start_date"] = df.loc[idx, "start_date"] - pd.DateOffset(years=2)

    return df.sample(frac=1, random_state=0).reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rows", type=int, default=100_000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=Path("data/listings_synthetic.csv"))
    args = ap.parse_args()

    df = generate(args.rows, args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.suffix == ".parquet":
        df.to_parquet(args.out, index=False)
    elif args.out.suffix in {".xlsx", ".xls"}:
        df.to_excel(args.out, index=False)
    else:
        df.to_csv(args.out, index=False)

    print(f"Wrote {len(df):,} synthetic listings x {df.shape[1]} columns -> {args.out}")
    print(f"Median price: EUR {df['price'].median():,.0f} | median EUR/sqm: "
          f"{(df['price'] / df['area']).median():,.0f}")


if __name__ == "__main__":
    main()
