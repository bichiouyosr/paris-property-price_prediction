"""Randomized hyperparameter search for the tree ensembles.

The headline numbers in the README come from sensible defaults, not a tuned
model — this script is the missing step. It searches one model's space
(default: Random Forest, the model `run_pipeline.py` picks by default),
scores the tuned model once on the held-out test set, and reports the gain
against the default hyperparameters side by side.

Each search draw is `--cv` full model fits, so the default (20 draws x
3-fold = 60 Random Forest fits on ~75k training rows) takes on the order of
15-20 minutes on a modern laptop. Lower --n-iter/--cv for a quicker look,
raise them for a more thorough search.

    python scripts/tune_hyperparameters.py --data data/listings_synthetic.csv
    python scripts/tune_hyperparameters.py --model "XGBoost" --n-iter 60

Writes reports/hyperparameter_search.json with the best params, the CV
score that selected them, and the before/after test metrics. Pass --save to
also overwrite models/best_model.joblib with the tuned model — off by
default, since a search run should be reviewed before it replaces the
deployed model.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib

from paris_avm import cleaning, data, features, modeling

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=ROOT / "data" / "listings_synthetic.csv")
    ap.add_argument("--reports", type=Path, default=ROOT / "reports")
    ap.add_argument(
        "--model",
        default="Random Forest",
        choices=sorted(modeling.SEARCH_SPACES),
        help="Which model's search space to tune (default: Random Forest).",
    )
    ap.add_argument(
        "--n-iter", type=int, default=20,
        help="Randomized search draws (default 20; each draw is `--cv` model fits).",
    )
    ap.add_argument(
        "--cv", type=int, default=3,
        help="CV folds during the search (default 3). 20x3=60 fits on the full "
        "100k-row extract takes on the order of 15-20 minutes on a modern "
        "laptop; raise --n-iter/--cv for a more thorough search if you have "
        "the time, or point --data at a smaller extract to iterate faster.",
    )
    ap.add_argument("--save", action="store_true", help="Overwrite models/best_model.joblib.")
    args = ap.parse_args()

    args.reports.mkdir(parents=True, exist_ok=True)

    print("=== Load, clean, engineer ===")
    df_raw = data.load_raw(args.data)
    df, _ = cleaning.clean(df_raw)
    df, feature_cols = features.engineer(df)
    split = modeling.make_split(df, feature_cols)

    print(f"\n=== Default {args.model} ===")
    default_model, flavour = modeling.default_models()[args.model]
    X_tr = split.X_train_scaled if flavour == "scaled" else split.X_train_imp
    X_te = split.X_test_scaled if flavour == "scaled" else split.X_test_imp
    default_model.fit(X_tr, split.y_train)
    default_metrics = modeling.metrics(split.y_test, default_model.predict(X_te))
    print(
        f"MAE: EUR {default_metrics['MAE']:,.0f} | "
        f"RMSE: EUR {default_metrics['RMSE']:,.0f} | R2: {default_metrics['R2']:.4f}"
    )

    print(
        f"\n=== Randomized search: {args.model} "
        f"({args.n_iter} draws x {args.cv}-fold CV, scoring neg MAE) ==="
    )
    start = time.time()
    search = modeling.hyperparameter_search(split, args.model, n_iter=args.n_iter, cv=args.cv)
    elapsed = time.time() - start
    print(f"Done in {elapsed:.0f}s. Best CV MAE: EUR {-search.best_score_:,.0f}")
    print("Best params:")
    for k, v in search.best_params_.items():
        print(f"  {k}: {v}")

    tuned_metrics = modeling.metrics(split.y_test, search.predict(X_te))
    print(f"\n=== Tuned {args.model} on held-out test set ===")
    print(
        f"MAE: EUR {tuned_metrics['MAE']:,.0f} | "
        f"RMSE: EUR {tuned_metrics['RMSE']:,.0f} | R2: {tuned_metrics['R2']:.4f}"
    )
    gain = default_metrics["MAE"] - tuned_metrics["MAE"]
    print(f"\nMAE change vs default: EUR {gain:+,.0f} ({'better' if gain > 0 else 'worse or flat'})")

    report = {
        "model": args.model,
        "n_iter": args.n_iter,
        "cv": args.cv,
        "search_time_sec": round(elapsed, 1),
        "best_cv_mae": float(-search.best_score_),
        "best_params": search.best_params_,
        "default_test_metrics": default_metrics,
        "tuned_test_metrics": tuned_metrics,
        "mae_improvement": float(gain),
    }
    out_path = args.reports / "hyperparameter_search.json"
    out_path.write_text(json.dumps(report, indent=2, default=str))
    print(f"\nReport -> {out_path}")

    if args.save:
        models_dir = ROOT / "models"
        models_dir.mkdir(exist_ok=True)
        joblib.dump(
            {
                "model": search.best_estimator_,
                "features": feature_cols,
                "imputer": split.imputer,
                "scaler": split.scaler,
                "name": f"{args.model} (tuned)",
            },
            models_dir / "best_model.joblib",
        )
        print(f"Tuned model saved -> {models_dir / 'best_model.joblib'}")
    else:
        print("Not saved. Re-run with --save to replace models/best_model.joblib.")


if __name__ == "__main__":
    main()
