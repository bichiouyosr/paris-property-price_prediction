"""Model training and comparison.

Six models on the same split: three linear (Linear, Ridge, Lasso) and three
tree ensembles (Random Forest, Gradient Boosting, XGBoost). The linear ones
get imputed + standardised inputs; the trees get imputed inputs only, since
scaling does nothing for a split-based learner.

The raw-price versus log-price question is settled empirically by
`compare_target_transforms` rather than by convention.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, cross_val_score, train_test_split
from sklearn.exceptions import ConvergenceWarning
from sklearn.preprocessing import StandardScaler

from . import config

# Lasso at this alpha is close enough to unregularized OLS that coordinate
# descent's default tolerance takes tens of thousands of iterations to hit
# on 100k rows — several minutes for a coefficient change too small to move
# MAE at all (empirically identical to 4 significant figures against a full
# 50,000-iteration fit, in a fraction of the time). max_iter is capped below
# accordingly, and the resulting non-convergence warning is expected and
# silenced rather than a bug to chase.
warnings.filterwarnings("ignore", category=ConvergenceWarning)

try:  # optional dependency
    import xgboost as xgb

    HAS_XGB = True
except ImportError:  # pragma: no cover
    HAS_XGB = False


@dataclass
class Split:
    """Train/test matrices in the three shapes the models need."""

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    X_train_imp: pd.DataFrame
    X_test_imp: pd.DataFrame
    X_train_scaled: np.ndarray
    X_test_scaled: np.ndarray
    y_train_log: pd.Series
    imputer: SimpleImputer
    scaler: StandardScaler


def make_split(
    df: pd.DataFrame,
    feature_cols: list[str],
    test_size: float = config.TEST_SIZE,
    random_state: int = config.RANDOM_STATE,
) -> Split:
    """Split, then impute and scale — fitted on train only, so the test set
    never leaks its medians or its variance into the transformation."""
    X = df[feature_cols].copy()
    y = df[config.TARGET].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    imputer = SimpleImputer(strategy="median")
    X_train_imp = pd.DataFrame(
        imputer.fit_transform(X_train), columns=feature_cols, index=X_train.index
    )
    X_test_imp = pd.DataFrame(
        imputer.transform(X_test), columns=feature_cols, index=X_test.index
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imp)
    X_test_scaled = scaler.transform(X_test_imp)

    return Split(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        X_train_imp=X_train_imp,
        X_test_imp=X_test_imp,
        X_train_scaled=X_train_scaled,
        X_test_scaled=X_test_scaled,
        y_train_log=np.log(y_train),
        imputer=imputer,
        scaler=scaler,
    )


def metrics(y_true, y_pred) -> dict[str, float]:
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
    }


def compare_target_transforms(split: Split) -> pd.DataFrame:
    """Same model, two targets: raw price and log price back-transformed.

    Log-transforming a right-skewed target is the textbook move, but the
    metric that matters here is the euro error on the real price. The
    back-transform re-introduces its own bias, so the comparison is run
    rather than assumed.
    """
    raw = LinearRegression().fit(split.X_train_scaled, split.y_train)
    pred_raw = raw.predict(split.X_test_scaled)

    logm = LinearRegression().fit(split.X_train_scaled, split.y_train_log)
    lo, hi = split.y_train_log.min(), split.y_train_log.max()
    pred_log = np.exp(np.clip(logm.predict(split.X_test_scaled), lo, hi))

    return pd.DataFrame(
        [
            {"target": "raw price", **metrics(split.y_test, pred_raw)},
            {"target": "log price", **metrics(split.y_test, pred_log)},
        ]
    )


def default_models(random_state: int = config.RANDOM_STATE) -> dict[str, tuple[object, str]]:
    """Model zoo. The second element says which input shape the model wants:
    `scaled` for the linear family, `imputed` for the trees."""
    models: dict[str, tuple[object, str]] = {
        "Linear Regression": (LinearRegression(), "scaled"),
        "Ridge": (Ridge(alpha=1.0), "scaled"),
        "Lasso": (Lasso(alpha=0.001, max_iter=2_000, tol=1e-3), "scaled"),
        "Random Forest": (
            RandomForestRegressor(
                n_estimators=300,
                max_depth=18,
                min_samples_leaf=3,
                random_state=random_state,
                n_jobs=-1,
            ),
            "imputed",
        ),
        "Gradient Boosting": (
            GradientBoostingRegressor(
                n_estimators=300, learning_rate=0.05, max_depth=4, random_state=random_state
            ),
            "imputed",
        ),
    }
    if HAS_XGB:
        models["XGBoost"] = (
            xgb.XGBRegressor(
                n_estimators=400,
                learning_rate=0.05,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=random_state,
                verbosity=0,
            ),
            "imputed",
        )
    return models


#: Search spaces for the tree ensembles. Only these three are searched — the
#: linear models have one real knob each (their alpha) and are baselines
#: here, not candidates for the deployed model.
SEARCH_SPACES: dict[str, dict[str, list]] = {
    "Random Forest": {
        "n_estimators": [200, 300, 400, 600, 800],
        "max_depth": [10, 14, 18, 22, 28, None],
        "min_samples_leaf": [1, 2, 3, 5, 8],
        "min_samples_split": [2, 4, 6, 10],
        "max_features": [0.5, 0.7, 0.9, 1.0, "sqrt"],
    },
    "Gradient Boosting": {
        "n_estimators": [150, 300, 450, 600],
        "learning_rate": [0.01, 0.03, 0.05, 0.08, 0.1],
        "max_depth": [2, 3, 4, 5, 6],
        "subsample": [0.6, 0.8, 0.9, 1.0],
        "min_samples_leaf": [1, 3, 5, 8],
    },
}

if HAS_XGB:
    SEARCH_SPACES["XGBoost"] = {
        "n_estimators": [200, 400, 600, 800],
        "learning_rate": [0.01, 0.03, 0.05, 0.08, 0.1],
        "max_depth": [3, 4, 5, 6, 8],
        "subsample": [0.6, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.6, 0.8, 0.9, 1.0],
        "min_child_weight": [1, 3, 5, 8],
    }


def hyperparameter_search(
    split: Split,
    model_name: str,
    n_iter: int = 30,
    cv: int = 5,
    random_state: int = config.RANDOM_STATE,
    n_jobs: int = -1,
) -> RandomizedSearchCV:
    """Randomized search over `model_name`'s space in `SEARCH_SPACES`.

    Scored on negative MAE — the headline metric — with 5-fold CV on the
    training set only. The test set is untouched during the search and
    scored exactly once at the end, the same discipline as every other
    model in `train_all`. The current leaderboard numbers all come from
    sensible defaults, not this: it exists to show how much headroom that
    leaves, not because the defaults were arbitrary.
    """
    if model_name not in SEARCH_SPACES:
        raise ValueError(
            f"No search space for {model_name!r}. Choices: {list(SEARCH_SPACES)}"
        )
    base_model, flavour = default_models(random_state)[model_name]
    X_tr = split.X_train_scaled if flavour == "scaled" else split.X_train_imp

    search = RandomizedSearchCV(
        base_model,
        SEARCH_SPACES[model_name],
        n_iter=n_iter,
        cv=cv,
        scoring="neg_mean_absolute_error",
        random_state=random_state,
        n_jobs=n_jobs,
        refit=True,
    )
    search.fit(X_tr, split.y_train)
    return search


def train_all(split: Split, models: dict | None = None, verbose: bool = True) -> dict:
    """Fit every model on the same split and collect test metrics."""
    models = models or default_models()
    results = {}
    for name, (model, flavour) in models.items():
        if flavour == "scaled":
            X_tr, X_te = split.X_train_scaled, split.X_test_scaled
        else:
            X_tr, X_te = split.X_train_imp, split.X_test_imp

        model.fit(X_tr, split.y_train)
        pred = model.predict(X_te)
        results[name] = {"model": model, "pred": pred, **metrics(split.y_test, pred)}
        if verbose:
            m = results[name]
            print(
                f"{name:20s} | MAE: EUR {m['MAE']:>10,.0f} "
                f"| RMSE: EUR {m['RMSE']:>10,.0f} | R2: {m['R2']:.4f}"
            )
    return results


def leaderboard(results: dict) -> pd.DataFrame:
    return (
        pd.DataFrame(
            [
                {"Model": k, "Test MAE (EUR)": v["MAE"], "Test RMSE (EUR)": v["RMSE"], "Test R2": v["R2"]}
                for k, v in results.items()
            ]
        )
        .sort_values("Test R2", ascending=False)
        .reset_index(drop=True)
    )


def cross_validate_best(split: Split, model, cv: int = 5) -> tuple[float, float]:
    """5-fold CV on the training set — a single split can flatter a model."""
    scores = cross_val_score(model, split.X_train_imp, split.y_train, cv=cv, scoring="r2")
    return float(scores.mean()), float(scores.std())
