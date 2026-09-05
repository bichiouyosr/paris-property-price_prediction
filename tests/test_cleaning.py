"""Tests for the cleaning rules.

Each test pins one decision from `config.py`, so changing a threshold has to
be a deliberate act rather than a silent one.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from paris_avm import cleaning, config


@pytest.fixture
def frame() -> pd.DataFrame:
    """Six rows, each carrying one specific defect."""
    return pd.DataFrame(
        {
            "external_id": [1, 1, 2, 3, 4, 5],
            "address_id": [10, 10, 11, 12, 13, 14],
            "street_id": [100, 100, 101, 102, 103, 104],
            "zipcode": [75116, 75116, 75107, 75550, 75011, 99999],
            "price": [900_000, 950_000, 400_000, 600_000, 12_000, 500_000],
            "area": [60, 60, 30, 45, 40, 50],
            "room_count": [3, 3, -1, 300, 2, 2],
            "build_year": [1900, 1900, 0, 3021, 1980, 1970],
            "item_type": ["ITEM_TYPE.APARTMENT"] * 5 + ["ITEM_TYPE.PARKING"],
            "item_subtype": ["ITEM_TYPE.APARTMENT.FLAT"] * 6,
            "end_date": pd.to_datetime(
                ["2025-03-01", "2025-09-01", "2025-04-01", "2025-05-01", "2025-06-01", "2025-07-01"]
            ),
        }
    )


def test_known_zipcodes_are_remapped(frame):
    out = cleaning.standardize_zipcodes(frame)
    assert out.loc[0, "zipcode"] == 75016  # 75116 is the legacy 16th
    assert out.loc[2, "zipcode"] == 75007  # 75107 is a typo


def test_unrecoverable_zipcode_becomes_nan_but_keeps_the_row(frame):
    out = cleaning.standardize_zipcodes(frame)
    assert np.isnan(out.loc[5, "zipcode"])
    assert len(out) == len(frame), "a bad zipcode must not cost the whole row"


def test_relisting_keeps_the_most_recent(frame):
    out = cleaning.deduplicate_relistings(frame)
    assert len(out) == 5
    kept = out[out["external_id"] == 1]
    assert kept["price"].item() == 950_000, "the later listing should win"


def test_non_residential_rows_are_dropped(frame):
    out = cleaning.keep_residential(cleaning.normalize_property_types(frame))
    assert set(out["item_type"]) == {"APARTMENT"}
    assert len(out) == 5


def test_impossible_values_are_nulled_not_dropped(frame):
    out = cleaning.nullify_impossible_values(frame)
    assert len(out) == len(frame)
    assert np.isnan(out.loc[2, "room_count"])  # -1 rooms
    assert np.isnan(out.loc[3, "room_count"])  # 300 rooms
    assert np.isnan(out.loc[2, "build_year"])  # year 0
    assert np.isnan(out.loc[3, "build_year"])  # year 3021
    assert out.loc[0, "room_count"] == 3, "valid values must survive untouched"


def test_out_of_market_rows_are_dropped(frame):
    out = cleaning.drop_market_outliers(frame)
    assert 12_000 not in out["price"].values, "below the residential price floor"
    assert len(out) < len(frame)


def test_structural_zero_and_median_imputation_differ():
    df = pd.DataFrame(
        {
            "balcony_count": [np.nan, 1.0, 2.0],
            "room_count": [np.nan, 2.0, 4.0],
        }
    )
    out = cleaning.impute(df)
    assert out.loc[0, "balcony_count"] == 0, "missing balcony means no balcony"
    assert out.loc[0, "room_count"] == 3, "missing room count means unknown -> median"


def test_energy_certificate_is_ordered_a_best():
    df = pd.DataFrame(
        {"energy_certificate_efficiency_score": ["ENERGY.CERTIFICATE_EFFICIENCY_SCORE.A",
                                                 "ENERGY.CERTIFICATE_EFFICIENCY_SCORE.G"]}
    )
    out = cleaning.encode_energy_certificate(df)
    assert out.loc[0, "energy_score_ordinal"] > out.loc[1, "energy_score_ordinal"]


def test_cleaning_log_tracks_every_step(frame):
    frame = frame.assign(
        publish_start_date=frame["end_date"],
        start_date=frame["end_date"],
        **{c: 0 for c in config.STRUCTURAL_ZERO_COLS},
    )
    _, log = cleaning.clean(frame, verbose=False)
    steps = log.to_frame()
    assert steps.loc[0, "step"] == "raw"
    assert steps["rows"].is_monotonic_decreasing, "no step may add rows"
