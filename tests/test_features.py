"""Tests for feature engineering — mostly guarding against target leakage."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from paris_avm import config, features


@pytest.fixture
def frame() -> pd.DataFrame:
    dates = pd.to_datetime(["2025-03-15", "2025-08-02"])
    return pd.DataFrame(
        {
            "price": [800_000.0, 400_000.0],
            "area": [80.0, 40.0],
            "room_count": [4.0, 2.0],
            "bedroom_count": [2.0, 1.0],
            "zipcode": [75006.0, 75019.0],
            "floor": [3.0, 1.0],
            "floor_count": [6.0, 4.0],
            "build_year": [1900.0, 1975.0],
            "terrace_area": [10.0, 0.0],
            "balcony_area": [0.0, 4.0],
            "parking_boxe_count": [1.0, 0.0],
            "parking_inside_count": [0.0, 0.0],
            "item_type": ["APARTMENT", "APARTMENT"],
            "item_subtype": ["APARTMENT.FLAT", "APARTMENT.STUDIO"],
            "publish_start_date": dates,
            "start_date": dates,
            "end_date": dates + pd.Timedelta(days=60),
        }
    )


def test_geography_flags(frame):
    out = features.add_geography(frame)
    assert out.loc[0, "arrondissement"] == 6
    assert out.loc[0, "rive_gauche"] == 1
    assert out.loc[0, "prime_arrondissement"] == 1
    assert out.loc[1, "prime_arrondissement"] == 0


def test_size_ratios(frame):
    out = features.add_size_ratios(frame)
    assert out.loc[0, "area_per_room"] == 20
    assert out.loc[0, "outdoor_area"] == 10
    assert out.loc[0, "total_parking"] == 1


def test_floor_ratio_is_relative(frame):
    out = features.add_building_position(frame)
    assert out.loc[0, "floor_ratio"] == pytest.approx(0.5)
    assert out.loc[1, "floor_ratio"] == pytest.approx(0.25)


def test_property_age_is_never_negative(frame):
    df = features.add_temporal_features(frame)
    df.loc[0, "build_year"] = 2030  # a listing for something not yet built
    out = features.add_property_age(df)
    assert out["property_age"].min() >= 0


def test_target_derived_columns_never_reach_the_feature_list(frame):
    _, feature_cols = features.engineer(frame)
    leaks = {"price", "log_price", "price_per_sqm", "days_on_market", "log_area"}
    assert not leaks & set(feature_cols), f"leaking: {leaks & set(feature_cols)}"


def test_engineer_returns_only_existing_columns(frame):
    df, feature_cols = features.engineer(frame)
    assert set(feature_cols) <= set(df.columns)
    assert "area" in feature_cols and "arrondissement" in feature_cols


def test_one_hot_columns_are_included(frame):
    _, feature_cols = features.engineer(frame)
    assert any(c.startswith("subtype_") for c in feature_cols)


def test_base_features_are_all_known_names():
    assert len(config.BASE_FEATURES) == len(set(config.BASE_FEATURES)), "duplicate feature name"
