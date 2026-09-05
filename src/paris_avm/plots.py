"""Charts for the EDA and the model diagnostics.

Matplotlib only, on purpose: every figure is written to `reports/figures/`
as a PNG, so a notebook committed to git carries small static images rather
than an interactive payload with thousands of raw rows embedded in it.

The backend is left alone so the figures render inline in a notebook;
`run_pipeline.py` switches to Agg before importing this module.

Colour: one hue for magnitude, and a fixed three-slot categorical order
(blue / orange / aqua) for identity. Hues are assigned in that fixed order
and never cycled, so a series keeps its colour from one chart to the next.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from . import config

SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]  # categorical, fixed order
SEQ = "#2a78d6"  # single hue for magnitude
GRID = "#d9d9d6"
INK = "#0b0b0b"
INK_MUTED = "#52514e"


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 110,
            "savefig.dpi": 150,
            "savefig.bbox": "tight",
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "axes.edgecolor": GRID,
            "axes.labelcolor": INK_MUTED,
            "axes.titlecolor": INK,
            "axes.titleweight": "semibold",
            "text.color": INK,
            "xtick.color": INK_MUTED,
            "ytick.color": INK_MUTED,
            "legend.frameon": False,
            "font.size": 10,
        }
    )


def _save(fig, name: str, outdir: Path | None) -> Path | None:
    if outdir is None:
        return None
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / f"{name}.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def _euros(ax, axis: str = "y") -> None:
    fmt = mticker.StrMethodFormatter("{x:,.0f}")
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(fmt)


def missingness(df: pd.DataFrame, outdir: Path | None = config.FIGURES_DIR):
    """Share of missing values per column, worst first."""
    _style()
    miss = (df.isna().mean() * 100).sort_values(ascending=False)
    miss = miss[miss > 0]
    fig, ax = plt.subplots(figsize=(8, max(3, 0.32 * len(miss))))
    ax.barh(miss.index, miss.values, color=SEQ, height=0.62)
    ax.invert_yaxis()
    ax.set_xlabel("% missing")
    ax.set_title("Missing values by column (raw extract)")
    for y, v in enumerate(miss.values):
        ax.text(v + 0.6, y, f"{v:.0f}%", va="center", fontsize=8, color=INK_MUTED)
    ax.grid(axis="y", visible=False)
    return _save(fig, "01_missingness", outdir) or fig


def price_distribution(df: pd.DataFrame, outdir: Path | None = config.FIGURES_DIR):
    """Raw price versus log price — why the log target is even a question."""
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    axes[0].hist(df["price"] / 1000, bins=80, color=SERIES[0])
    axes[0].set_xlabel("Price (k EUR)")
    axes[0].set_title("Price")
    _euros(axes[0], "x")
    axes[1].hist(df["log_price"], bins=80, color=SERIES[1])
    axes[1].set_xlabel("log(price)")
    axes[1].set_title("log(Price)")
    for ax in axes:
        ax.set_ylabel("Listings")
        ax.grid(axis="x", visible=False)
    fig.suptitle(
        f"Median EUR {df['price'].median():,.0f} · mean EUR {df['price'].mean():,.0f}"
        "  — the right tail pulls the mean up",
        y=1.03,
        fontsize=10,
        color=INK_MUTED,
    )
    return _save(fig, "02_price_distribution", outdir) or fig


def price_vs_area(df: pd.DataFrame, outdir: Path | None = config.FIGURES_DIR, sample: int = 12_000):
    """Area against price, prime arrondissements separated from the rest.

    Two series rather than twenty: one colour per arrondissement would be a
    cycled rainbow nobody can read. The split that carries the message is
    prime versus the rest, at equal surface.
    """
    _style()
    d = df.dropna(subset=["arrondissement"])
    if len(d) > sample:
        d = d.sample(sample, random_state=config.RANDOM_STATE)
    fig, ax = plt.subplots(figsize=(9, 5.2))
    for flag, colour, label in ((0, SERIES[0], "Other arrondissements"), (1, SERIES[1], "Prime (6, 7, 8, 16)")):
        s = d[d["prime_arrondissement"] == flag]
        ax.scatter(s["area"], s["price"] / 1000, s=7, alpha=0.35, color=colour, label=label, linewidths=0)
    corr = df["area"].corr(df["price"], method="spearman")
    ax.set_xlabel("Area (sqm)")
    ax.set_ylabel("Price (k EUR)")
    ax.set_title(f"Price vs area — Spearman rho = {corr:.2f}")
    _euros(ax)
    ax.legend(loc="upper left", markerscale=2.2)
    return _save(fig, "03_price_vs_area", outdir) or fig


def price_per_sqm_by_arrondissement(df: pd.DataFrame, outdir: Path | None = config.FIGURES_DIR):
    """Median EUR/sqm per arrondissement, ranked."""
    _style()
    med = df.groupby("arrondissement")["price_per_sqm"].median().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(10, 4.6))
    ax.bar([str(int(a)) for a in med.index], med.values, color=SEQ, width=0.68)
    ax.set_xlabel("Arrondissement")
    ax.set_ylabel("Median EUR/sqm")
    ax.set_title(
        f"Price per sqm by arrondissement — {med.iloc[0]/med.iloc[-1]:.1f}x between "
        f"the {int(med.index[0])}th and the {int(med.index[-1])}th"
    )
    _euros(ax)
    ax.grid(axis="x", visible=False)
    ax.text(0, med.iloc[0] * 1.02, f"EUR {med.iloc[0]:,.0f}", ha="center", fontsize=8, color=INK_MUTED)
    ax.text(len(med) - 1, med.iloc[-1] * 1.02, f"EUR {med.iloc[-1]:,.0f}", ha="center", fontsize=8, color=INK_MUTED)
    return _save(fig, "04_price_per_sqm_by_arrondissement", outdir) or fig


def correlation_heatmap(df: pd.DataFrame, cols: list[str] | None = None, outdir: Path | None = config.FIGURES_DIR):
    """Spearman correlation on the numeric drivers.

    Diverging scale with a neutral midpoint: the sign of a correlation is
    the thing being read, so zero has to look like nothing.
    """
    _style()
    cols = cols or [
        "price", "area", "room_count", "bedroom_count", "bathroom_count", "floor",
        "floor_count", "property_age", "energy_score_ordinal", "total_parking",
        "outdoor_area", "arrondissement", "days_on_market",
    ]
    cols = [c for c in cols if c in df.columns]
    corr = df[cols].corr(method="spearman")

    fig, ax = plt.subplots(figsize=(8.5, 7))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols)), cols, rotation=90)
    ax.set_yticks(range(len(cols)), cols)
    for i in range(len(cols)):
        for j in range(len(cols)):
            v = corr.iloc[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if abs(v) > 0.55 else INK)
    ax.grid(visible=False)
    fig.colorbar(im, ax=ax, shrink=0.8, label="Spearman rho")
    ax.set_title("Correlation with price and between drivers")
    return _save(fig, "05_correlation", outdir) or fig


def seasonality(df: pd.DataFrame, outdir: Path | None = config.FIGURES_DIR):
    """Volume and price level by month of publication."""
    _style()
    m = df.groupby("listing_month").agg(n=("price", "size"), ppsqm=("price_per_sqm", "median"))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].bar(m.index, m["n"], color=SEQ, width=0.68)
    axes[0].set_title("Listings published per month")
    axes[0].set_ylabel("Listings")
    axes[0].grid(axis="x", visible=False)
    axes[1].plot(m.index, m["ppsqm"], marker="o", color=SERIES[1], linewidth=2, markersize=6)
    axes[1].set_title("Median EUR/sqm by month")
    axes[1].set_ylabel("EUR/sqm")
    _euros(axes[1])
    for ax in axes:
        ax.set_xlabel("Month")
        ax.set_xticks(range(1, 13))
    return _save(fig, "06_seasonality", outdir) or fig


def model_comparison(board: pd.DataFrame, outdir: Path | None = config.FIGURES_DIR):
    """MAE by model — one panel, one metric, sorted."""
    _style()
    b = board.sort_values("Test MAE (EUR)")
    best = b.iloc[0]["Model"]
    colours = [SERIES[1] if m == best else SEQ for m in b["Model"]]
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    ax.barh(b["Model"], b["Test MAE (EUR)"], color=colours, height=0.6)
    ax.invert_yaxis()
    ax.set_xlabel("Test MAE (EUR) — lower is better")
    ax.set_title(f"{best} has the lowest error")
    _euros(ax, "x")
    ax.grid(axis="y", visible=False)
    for y, (v, r2) in enumerate(zip(b["Test MAE (EUR)"], b["Test R2"])):
        ax.text(v * 1.01, y, f"EUR {v:,.0f} · R2 {r2:.3f}", va="center", fontsize=8, color=INK_MUTED)
    ax.set_xlim(0, b["Test MAE (EUR)"].max() * 1.28)
    return _save(fig, "07_model_comparison", outdir) or fig


def predicted_vs_actual(y_true, y_pred, model_name: str, outdir: Path | None = config.FIGURES_DIR):
    """The two diagnostics that matter: fit along y=x, and residual shape."""
    _style()
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    resid = y_true - y_pred

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    axes[0].scatter(y_true / 1000, y_pred / 1000, s=7, alpha=0.3, color=SERIES[0], linewidths=0)
    lims = [0, max(y_true.max(), y_pred.max()) / 1000]
    axes[0].plot(lims, lims, "--", color=INK_MUTED, linewidth=1.5)
    axes[0].set_xlabel("Actual (k EUR)")
    axes[0].set_ylabel("Predicted (k EUR)")
    axes[0].set_title(f"{model_name} — actual vs predicted")
    _euros(axes[0]); _euros(axes[0], "x")

    axes[1].hist(resid / 1000, bins=70, color=SERIES[1])
    axes[1].axvline(0, color=INK_MUTED, linestyle="--", linewidth=1.5)
    axes[1].set_xlabel("Residual (k EUR)")
    axes[1].set_ylabel("Listings")
    axes[1].set_title(f"Residuals — median EUR {np.median(resid):,.0f}")
    axes[1].grid(axis="x", visible=False)
    return _save(fig, "08_predicted_vs_actual", outdir) or fig


def error_by_segment(seg: pd.DataFrame, outdir: Path | None = config.FIGURES_DIR):
    """Absolute error and percentage error by price quintile.

    Two measures on different scales, so two panels rather than two y-axes.
    """
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    idx = [str(i) for i in seg.index]
    axes[0].bar(idx, seg["MAE"], color=SEQ, width=0.6)
    axes[0].set_title("MAE by price quintile")
    axes[0].set_ylabel("EUR")
    _euros(axes[0])
    axes[1].bar(idx, seg["MAPE"], color=SERIES[1], width=0.6)
    axes[1].set_title("MAPE by price quintile")
    axes[1].set_ylabel("%")
    for ax in axes:
        ax.tick_params(axis="x", rotation=20)
        ax.grid(axis="x", visible=False)
    return _save(fig, "09_error_by_segment", outdir) or fig


def feature_importance(importance: pd.DataFrame, top_n: int = 15, outdir: Path | None = config.FIGURES_DIR):
    """Top drivers of the winning model."""
    _style()
    top = importance.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 5.6))
    ax.barh(top["feature"], top["importance"], color=SEQ, height=0.62)
    ax.set_xlabel(top["kind"].iloc[0] if "kind" in top else "importance")
    ax.set_title(f"Top {top_n} features")
    ax.grid(axis="y", visible=False)
    return _save(fig, "10_feature_importance", outdir) or fig


def actual_vs_predicted_by_arrondissement(err_by_arr: pd.DataFrame, outdir: Path | None = config.FIGURES_DIR):
    """Does the model reproduce the geography it was shown?"""
    _style()
    fig, ax = plt.subplots(figsize=(10, 4.6))
    x = err_by_arr.index.astype(int)
    ax.plot(x, err_by_arr["median_actual"] / 1000, marker="o", color=SERIES[0], linewidth=2, label="Actual median")
    ax.plot(x, err_by_arr["median_pred"] / 1000, marker="s", color=SERIES[1], linewidth=2, label="Predicted median")
    ax.set_xlabel("Arrondissement")
    ax.set_ylabel("Median price (k EUR)")
    ax.set_title("Median price by arrondissement — test set")
    ax.set_xticks(sorted(x))
    _euros(ax)
    ax.legend()
    return _save(fig, "11_geography_check", outdir) or fig
