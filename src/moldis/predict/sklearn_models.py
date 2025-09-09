from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class SKLearnRegressor:
    name: str
    feature_names: list[str]
    estimator: Any  # sklearn estimator/pipeline
    random_state: int


def _require_sklearn() -> None:
    try:
        import sklearn  # noqa: F401
    except Exception as e:  # pragma: no cover
        raise RuntimeError("scikit-learn not installed. Install extras: '.[ml]'") from e


def fit_ridge(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    random_state: int = 0,
    alpha: float = 1.0,
) -> SKLearnRegressor:
    _require_sklearn()
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=alpha, random_state=random_state)),
        ]
    )
    model.fit(X, y)
    return SKLearnRegressor(
        name="ridge", feature_names=feature_names, estimator=model, random_state=random_state
    )


def fit_random_forest(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    random_state: int = 0,
    n_estimators: int = 200,
    max_depth: int | None = None,
) -> SKLearnRegressor:
    _require_sklearn()
    from sklearn.ensemble import RandomForestRegressor

    rf = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1,
    )
    rf.fit(X, y)
    return SKLearnRegressor(
        name="rf", feature_names=feature_names, estimator=rf, random_state=random_state
    )


def evaluate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    return {"mae": mae, "rmse": rmse, "r2": r2}


def evaluate_holdout(
    estimator: SKLearnRegressor,
    X: np.ndarray,
    y: np.ndarray,
    idx_train: list[int],
    idx_val: list[int],
) -> dict[str, float]:
    mdl = estimator.estimator
    mdl.fit(X[idx_train], y[idx_train])
    y_hat = mdl.predict(X[idx_val])
    return evaluate_metrics(y[idx_val], y_hat)


def kfold_cv(
    estimator_factory, X: np.ndarray, y: np.ndarray, n_splits: int = 5, random_state: int = 0
) -> tuple[dict[str, float], np.ndarray]:
    _require_sklearn()
    from sklearn.model_selection import KFold

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    preds = np.zeros_like(y, dtype=float)
    metrics_list = []
    for train_idx, val_idx in kf.split(X):
        est = estimator_factory()
        est.estimator.fit(X[train_idx], y[train_idx])
        y_hat = est.estimator.predict(X[val_idx])
        preds[val_idx] = y_hat
        metrics_list.append(evaluate_metrics(y[val_idx], y_hat))
    # Average metrics
    agg = {k: float(np.mean([m[k] for m in metrics_list])) for k in metrics_list[0].keys()}
    return agg, preds


def residual_interval_coverage(
    y_true: np.ndarray, y_pred: np.ndarray, nominal_levels: list[float]
) -> dict[str, float]:
    resid = y_true - y_pred
    abs_resid = np.abs(resid)
    coverages: dict[str, float] = {}
    for level in nominal_levels:
        q = np.quantile(abs_resid, level)
        covered = float(np.mean(abs_resid <= q))
        coverages[str(level)] = covered
    return coverages


def cv_residual_quantiles(
    X: np.ndarray, y: np.ndarray, levels: list[float], n_splits: int = 5, random_state: int = 0
) -> dict[str, float]:
    """Estimate residual quantiles via KFold CV using a ridge baseline scaler+ridge."""
    _require_sklearn()
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import KFold
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    preds = np.zeros_like(y, dtype=float)
    for train_idx, val_idx in kf.split(X):
        model = Pipeline(
            steps=[("scaler", StandardScaler()), ("ridge", Ridge(random_state=random_state))]
        )
        model.fit(X[train_idx], y[train_idx])
        preds[val_idx] = model.predict(X[val_idx])
    resid = y - preds
    abs_resid = np.abs(resid)
    return {str(lv): float(np.quantile(abs_resid, lv)) for lv in levels}


def intervals_from_resid_quantiles(
    y_pred: np.ndarray, q_by_level: dict[str, float], level: float
) -> tuple[np.ndarray, np.ndarray]:
    q = float(q_by_level[str(level)])
    return y_pred - q, y_pred + q


def coverage_from_intervals(y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    return float(np.mean((y_true >= lower) & (y_true <= upper)))


def fit_gbr_quantile_models(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    random_state: int = 0,
    lower_alpha: float = 0.1,
    upper_alpha: float = 0.9,
) -> tuple[SKLearnRegressor, SKLearnRegressor]:
    """Fit gradient boosting quantile regressors for lower/upper bounds."""
    _require_sklearn()
    from sklearn.ensemble import GradientBoostingRegressor

    lower = GradientBoostingRegressor(loss="quantile", alpha=lower_alpha, random_state=random_state)
    upper = GradientBoostingRegressor(loss="quantile", alpha=upper_alpha, random_state=random_state)
    lower.fit(X, y)
    upper.fit(X, y)
    lower_est = SKLearnRegressor("gbr_q_lower", feature_names, lower, random_state)
    upper_est = SKLearnRegressor("gbr_q_upper", feature_names, upper, random_state)
    return lower_est, upper_est


def predict_intervals_from_quantiles(
    lower_est: SKLearnRegressor, upper_est: SKLearnRegressor, X: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    lower = lower_est.estimator.predict(X)
    upper = upper_est.estimator.predict(X)
    return lower, upper


def save_model(estimator: SKLearnRegressor, out_dir: str | Path, meta: dict[str, Any]) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    # Save estimator
    try:
        import joblib
    except Exception as e:  # pragma: no cover
        raise RuntimeError("joblib not installed. Install extras: '.[ml]'") from e
    joblib.dump(estimator.estimator, out / "model.joblib")
    # Save metadata
    meta_full = {
        "name": estimator.name,
        "feature_names": estimator.feature_names,
        "random_state": estimator.random_state,
        **meta,
    }
    (out / "model.json").write_text(
        __import__("json").dumps(meta_full, ensure_ascii=False, sort_keys=True, indent=2),
        encoding="utf-8",
    )
    return out
