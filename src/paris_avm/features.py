"""Feature engineering.

Fourteen features are derived from the cleaned columns. They fall into four
groups: time, size/layout ratios, position in the building, and Paris
geography. The geography ones encode knowledge the model cannot learn from
a zipcode integer — that the 6th and the 19th are not neighbours on any
axis that matters to price.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    pub = df["publish_start_date"]
    df["listing_year"] = pub.dt.year
    df["listing_month"] = pub.dt.month
    df["listing_quarter"] = pub.dt.quarter
    df["listing_dow"] = pub.dt.dayofweek
    # Diagnostic only: unknown at prediction time, never a model feature.
    df["days_on_market"] = (df["end_date"] - df["start_date"]).dt.days
    return df


def add_property_age(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["property_age"] = (df["listing_year"] - df["build_year"]).clip(lower=0)
    return df


def add_size_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """Area alone says how big; area per room says how it is laid out.

    A 90 sqm two-room flat and a 90 sqm four-room flat are different
    products at the same surface.
    """
    df = df.copy()
    df["area_per_room"] = df["area"] / df["room_count"].replace(0, np.nan)
    df["area_per_bedroom"] = df["area"] / df["bedroom_count"].replace(0, np.nan)
    df["outdoor_area"] = df["terrace_area"] + df["balcony_area"]
    df["total_parking"] = df[["parking_boxe_count", "parking_inside_count"]].sum(axis=1)
    return df


def add_building_position(df: pd.DataFrame) -> pd.DataFrame:
    """Third floor out of four is a top floor; out of twenty it is not."""
    df = df.copy()
    df["floor_ratio"] = df["floor"] / df["floor_count"].replace(0, np.nan)
    return df


def add_geography(df: pd.DataFrame) -> pd.DataFrame:
    """Turn the zipcode into arrondissement plus two market flags."""
    df = df.copy()
    df["arrondissement"] = df["zipcode"] - 75000
    df["rive_gauche"] = df["arrondissement"].isin(config.LEFT_BANK).astype(int)
    df["prime_arrondissement"] = df["arrondissement"].isin(config.PRIME_WEST).astype(int)
    return df


def add_target_transforms(df: pd.DataFrame) -> pd.DataFrame:
    """EDA helpers. `price_per_sqm` and `log_price` are derived from the
    target and must never reach the feature matrix."""
    df = df.copy()
    df["log_price"] = np.log(df["price"])
    df["log_area"] = np.log(df["area"])
    df["price_per_sqm"] = df["price"] / df["area"]
    return df


def encode_property_types(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """One-hot encode type and subtype, returning the generated columns."""
    df = df.copy()
    dummies = []
    for col, prefix in (("item_type", "type"), ("item_subtype", "subtype")):
        if col in df.columns:
            d = pd.get_dummies(df[col], prefix=prefix, drop_first=True).astype(int)
            df = pd.concat([df, d], axis=1)
            dummies += d.columns.tolist()
    return df, dummies


def engineer(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Run the whole feature pipeline; return the frame and the feature list."""
    df = add_temporal_features(df)
    df = add_property_age(df)
    df = add_size_ratios(df)
    df = add_building_position(df)
    df = add_geography(df)
    df = add_target_transforms(df)
    df, dummy_cols = encode_property_types(df)

    feature_cols = [c for c in config.BASE_FEATURES + dummy_cols if c in df.columns]
    return df, feature_cols
