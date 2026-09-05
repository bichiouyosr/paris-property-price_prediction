"""Central configuration: schema, business rules and thresholds.

Every magic number used by the pipeline lives here so the rules can be
reviewed, tuned and unit-tested without touching the transformation code.
"""

from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

RANDOM_STATE = 42
TEST_SIZE = 0.2

# --------------------------------------------------------------------------
# Raw schema
# --------------------------------------------------------------------------
TARGET = "price"

#: Identifiers and administrative columns. Useful for de-duplication and for
#: repairing geography, never fed to the model.
ID_COLS = [
    "external_id",
    "client_id",
    "address_id",
    "street_id",
    "iris_id",
    "parcel_id",
    "city_id",
]

DATE_COLS = ["start_date", "end_date", "publish_start_date"]

#: Dropped during cleaning, with the reason kept next to the name.
DROP_COLS = {
    "Unnamed: 0": "spare row index, no information",
    "parcel_id": "100% missing",
    "site_area": ">99% missing",
    "parking_garage_count": ">99% missing",
    "city": "free-text and inconsistent, superseded by zipcode",
    "iris_id": "high-cardinality identifier, not a property attribute",
    "street_id": "high-cardinality identifier, not a property attribute",
    "address_id": "high-cardinality identifier, not a property attribute",
    "client_id": "agency identifier, not a property attribute",
}

#: A column where a single value covers this share of rows carries no signal.
NEAR_CONSTANT_THRESHOLD = 0.99

# --------------------------------------------------------------------------
# Geography
# --------------------------------------------------------------------------
PARIS_ZIP_MIN, PARIS_ZIP_MAX = 75001, 75020

#: Zipcodes that are unambiguously a mis-encoding of a real arrondissement.
#: 75116 is a legacy postal code for the 16th; 75107 / 75108 are typos;
#: 75000 is a generic "Paris" placeholder mapped to the historic centre.
ZIPCODE_REPLACEMENTS = {75116: 75016, 75107: 75007, 75108: 75008, 75000: 75003}

#: Zipcodes that are wrong but not self-explanatory. They are repaired from
#: other listings at the same address when possible, otherwise set to NaN.
AMBIGUOUS_ZIPCODES = [75550, 75023]

LEFT_BANK = {5, 6, 7, 13, 14, 15}
PRIME_WEST = {6, 7, 8, 16}

# --------------------------------------------------------------------------
# Business rules — plausibility bounds
# --------------------------------------------------------------------------
RESIDENTIAL_TYPES = ["APARTMENT", "HOUSE"]

#: (min, max) accepted range. Values outside become NaN and are imputed later.
PLAUSIBLE_RANGES = {
    "room_count": (0, 12),
    "bedroom_count": (0, 10),
    "bathroom_count": (0, 6),
    "toilet_count": (0, 6),
    "balcony_count": (0, 8),
    "terrace_count": (0, 8),
    "parking_boxe_count": (0, 6),
    "parking_inside_count": (0, 6),
    "floor": (-2, 25),
    "floor_count": (0, 40),
    "build_year": (1700, 2026),
    "terrace_area": (0, 200),
    "balcony_area": (0, 100),
}

#: Rows outside these bounds are dropped, not imputed: they are not noisy
#: measurements of a real flat, they are a different object altogether
#: (a parking space priced at EUR 15k, a whole building, a data-entry ghost).
PRICE_BOUNDS = (20_000, 15_000_000)
#: 9 sqm is the legal minimum habitable surface in France (decree 2002-120).
AREA_BOUNDS = (9, 400)
#: Realistic Paris band; the observed per-arrondissement medians sit inside it.
PRICE_PER_SQM_BOUNDS = (6_000, 18_000)

#: Missing here means "the property does not have one", not "unknown".
STRUCTURAL_ZERO_COLS = [
    "balcony_count",
    "terrace_count",
    "terrace_area",
    "balcony_area",
    "parking_boxe_count",
    "parking_inside_count",
    "has_cellar",
    "has_passenger_lift",
]

#: Missing here means the field was not filled in; the median is a safer guess
#: than dropping the row.
MEDIAN_FILL_COLS = [
    "room_count",
    "bedroom_count",
    "bathroom_count",
    "toilet_count",
    "floor",
    "floor_count",
    "build_year",
]

#: A (best) to G (worst), encoded so that a higher number is a better rating.
ENERGY_ORDER = {
    "ENERGY.CERTIFICATE_EFFICIENCY_SCORE.A": 7,
    "ENERGY.CERTIFICATE_EFFICIENCY_SCORE.B": 6,
    "ENERGY.CERTIFICATE_EFFICIENCY_SCORE.C": 5,
    "ENERGY.CERTIFICATE_EFFICIENCY_SCORE.D": 4,
    "ENERGY.CERTIFICATE_EFFICIENCY_SCORE.E": 3,
    "ENERGY.CERTIFICATE_EFFICIENCY_SCORE.F": 2,
    "ENERGY.CERTIFICATE_EFFICIENCY_SCORE.G": 1,
}
ORDINAL_TO_LETTER = {7: "A", 6: "B", 5: "C", 4: "D", 3: "E", 2: "F", 1: "G"}

#: The snapshot is a single year of listings; anything else is a stale relist.
LISTING_YEAR = 2025

# --------------------------------------------------------------------------
# Modelling
# --------------------------------------------------------------------------
#: Excluded on purpose:
#:   price_per_sqm, log_price  -> built from the target (leakage)
#:   days_on_market            -> only known after the listing closes (leakage)
#:   log_area                  -> collinear with area, EDA only
BASE_FEATURES = [
    "area",
    "room_count",
    "bedroom_count",
    "bathroom_count",
    "toilet_count",
    "floor",
    "floor_count",
    "floor_ratio",
    "build_year",
    "property_age",
    "arrondissement",
    "rive_gauche",
    "prime_arrondissement",
    "has_passenger_lift",
    "has_cellar",
    "balcony_count",
    "terrace_count",
    "outdoor_area",
    "total_parking",
    "energy_score_ordinal",
    "area_per_room",
    "area_per_bedroom",
    "listing_month",
    "listing_quarter",
    "listing_dow",
]
