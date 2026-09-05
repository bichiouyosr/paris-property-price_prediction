"""Run the whole pipeline end to end: raw extract -> metrics, figures, model.

    python scripts/run_pipeline.py --data data/listings_synthetic.csv

Writes to reports/: metrics.json, model_comparison.csv, feature_importance.csv,
cleaning_log.csv, the figures under reports/figures/, and the fitted best model
under models/.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")  # headless: figures are written, never shown

from paris_avm import cleaning, data, evaluation, features, modeling, plots

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=ROOT / "data" / "listings_synthetic.csv")
    ap.add_argument("--reports", type=Path, default=ROOT / "reports")
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args()

    figures_dir = args.reports / "figures"
    args.reports.mkdir(parents=True, exist_ok=True)

    print("\n=== 1. Load ===")
    df_raw = data.load_raw(args.data)
    print(f"{df_raw.shape[0]:,} rows x {df_raw.shape[1]} columns")

    audit = data.quality_report(df_raw)
    print(f"{len(audit['duplicate_external_ids']):,} external_ids appear more than once")
    print(f"{int(audit['invalid_zipcodes'].sum()):,} rows carry a non-Paris zipcode")

    print("\n=== 2. Clean ===")
    df, log = cleaning.clean(df_raw)
    log.to_frame().to_csv(args.reports / "cleaning_log.csv", index=False)

    print("\n=== 3. Feature engineering ===")
    df, feature_cols = features.engineer(df)
    print(f"{len(feature_cols)} model features, {df.shape[1]} columns in total")

    print("\n=== 4. Target transform check ===")
    split = modeling.make_split(df, feature_cols)
    transforms = modeling.compare_target_transforms(split)
    print(transforms.to_string(index=False))

    print("\n=== 5. Models ===")
    results = modeling.train_all(split)
    board = modeling.leaderboard(results)
    board.to_csv(args.reports / "model_comparison.csv", index=False)

    best_name = board.loc[0, "Model"]
    best = results[best_name]
    print(f"\nBest: {best_name} (MAE EUR {best['MAE']:,.0f}, R2 {best['R2']:.4f})")

    cv_mean, cv_std = modeling.cross_validate_best(split, modeling.default_models()[best_name][0])
    print(f"5-fold CV R2 on train: {cv_mean:.4f} (+/- {cv_std:.4f})")

    print("\n=== 6. Diagnostics ===")
    seg = evaluation.error_by_segment(split.y_test, best["pred"])
    print(seg.to_string())
    importance = evaluation.feature_importance(best["model"], feature_cols)
    importance.to_csv(args.reports / "feature_importance.csv", index=False)
    print("\nTop 8 features:")
    print(importance.head(8).to_string(index=False))

    lo, hi = evaluation.prediction_interval(split.y_test, best["pred"], coverage=0.8)
    print(f"\n80% of test residuals fall in [EUR {lo:,.0f}, EUR {hi:,.0f}] "
          "-> publish an interval, not a point estimate")

    by_arr = evaluation.error_by_group(
        split.y_test, best["pred"], split.X_test["arrondissement"], "arrondissement"
    )

    metrics = {
        "rows_raw": int(len(df_raw)),
        "rows_modelled": int(len(df)),
        "n_features": len(feature_cols),
        "best_model": best_name,
        "leaderboard": board.to_dict(orient="records"),
        "cv_r2_mean": cv_mean,
        "cv_r2_std": cv_std,
        "interval_80pct": [lo, hi],
        "error_by_segment": json.loads(seg.reset_index().to_json(orient="records")),
    }
    (args.reports / "metrics.json").write_text(json.dumps(metrics, indent=2))

    if not args.no_figures:
        print("\n=== 7. Figures ===")
        plots.missingness(df_raw, figures_dir)
        plots.price_distribution(df, figures_dir)
        plots.price_vs_area(df, figures_dir)
        plots.price_per_sqm_by_arrondissement(df, figures_dir)
        plots.correlation_heatmap(df, outdir=figures_dir)
        plots.seasonality(df, figures_dir)
        plots.model_comparison(board, figures_dir)
        plots.predicted_vs_actual(split.y_test, best["pred"], best_name, figures_dir)
        plots.error_by_segment(seg, figures_dir)
        plots.feature_importance(importance, outdir=figures_dir)
        plots.actual_vs_predicted_by_arrondissement(by_arr, figures_dir)
        print(f"11 figures -> {figures_dir}")

    models_dir = ROOT / "models"
    models_dir.mkdir(exist_ok=True)
    joblib.dump(
        {"model": best["model"], "features": feature_cols, "imputer": split.imputer,
         "scaler": split.scaler, "name": best_name},
        models_dir / "best_model.joblib",
    )
    print(f"\nModel saved -> {models_dir / 'best_model.joblib'}")
    print(f"Reports     -> {args.reports}")


if __name__ == "__main__":
    main()
