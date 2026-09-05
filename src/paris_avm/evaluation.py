"""Diagnostics: where the model is wrong, and by how much.

A single headline MAE hides the thing a valuation product actually needs to
know — that the error is not spread evenly across price segments, and that
the estimate should be shown as an interval whose width depends on the
segment.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def error_frame(y_true, y_pred) -> pd.DataFrame:
    out = pd.DataFrame({"actual": np.asarray(y_true), "pred": np.asarray(y_pred)})
    out["error"] = out["actual"] - out["pred"]
    out["abs_error"] = out["error"].abs()
    out["pct_error"] = out["abs_error"] / out["actual"] * 100
    return out


def error_by_segment(y_true, y_pred, q: int = 5) -> pd.DataFrame:
    """Break the error down by price quintile.

    Absolute error grows with price while percentage error usually shrinks:
    the model is relatively better on expensive flats and absolutely worse.
    """
    df = error_frame(y_true, y_pred)
    labels = ["Q1 (cheapest)", "Q2", "Q3", "Q4", "Q5 (priciest)"][:q]
    df["bucket"] = pd.qcut(df["actual"], q, labels=labels)
    return (
        df.groupby("bucket", observed=True)
        .agg(
            n=("actual", "size"),
            mean_price=("actual", "mean"),
            MAE=("abs_error", "mean"),
            MAPE=("pct_error", "mean"),
        )
        .round(1)
    )


def error_by_group(y_true, y_pred, groups: pd.Series, name: str = "group") -> pd.DataFrame:
    """Same idea on any grouping — arrondissement, property type, size band."""
    df = error_frame(y_true, y_pred)
    df[name] = np.asarray(groups)
    return (
        df.groupby(name, observed=True)
        .agg(
            n=("actual", "size"),
            median_actual=("actual", "median"),
            median_pred=("pred", "median"),
            MAE=("abs_error", "mean"),
            MAPE=("pct_error", "mean"),
        )
        .round(1)
    )


def feature_importance(model, feature_cols: list[str]) -> pd.DataFrame:
    """Tree importances when available, absolute coefficients otherwise."""
    if hasattr(model, "feature_importances_"):
        values = model.feature_importances_
        kind = "impurity importance"
    else:
        values = np.abs(model.coef_)
        kind = "|coefficient|"
    return (
        pd.DataFrame({"feature": feature_cols, "importance": values, "kind": kind})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def prediction_interval(y_true, y_pred, coverage: float = 0.8) -> tuple[float, float]:
    """Empirical residual quantiles, to publish an interval instead of a point.

    A valuation shown as a single number implies a precision the model does
    not have; this returns the (low, high) offsets covering `coverage` of
    the observed residuals on the test set.
    """
    resid = np.asarray(y_true) - np.asarray(y_pred)
    tail = (1 - coverage) / 2
    return float(np.quantile(resid, tail)), float(np.quantile(resid, 1 - tail))
