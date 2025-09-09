from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from moldis.predict.datasets import load_esol_tiny
from moldis.predict.features import FEATURE_NAMES, featurize_smiles
from moldis.predict.sklearn_models import (
    SKLearnRegressor,
    evaluate_holdout,
    fit_random_forest,
    residual_interval_coverage,
    save_model,
)
from moldis.predict.splits import scaffold_split
from moldis.report.calibration import plot_interval_calibration

sklearn = pytest.importorskip("sklearn")
rdkit = pytest.importorskip("rdkit")
mpl = pytest.importorskip("matplotlib")


def test_sklearn_rf_scaffold_split_and_calibration(tmp_path: Path) -> None:
    smiles, y_list = load_esol_tiny()
    X = featurize_smiles(smiles).X
    y = np.asarray(y_list, dtype=float)
    idx_train, idx_val = scaffold_split(smiles, val_fraction=0.3, random_state=0)
    est: SKLearnRegressor = fit_random_forest(
        X[idx_train], y[idx_train], FEATURE_NAMES, random_state=0, n_estimators=64
    )
    metrics = evaluate_holdout(est, X, y, idx_train, idx_val)
    assert set(["mae", "rmse", "r2"]).issubset(metrics.keys())
    # Compute residual coverage on validation
    y_hat = est.estimator.predict(X[idx_val])
    cover = residual_interval_coverage(y[idx_val], y_hat, nominal_levels=[0.8, 0.9])
    assert all(0.0 <= v <= 1.0 for v in cover.values())
    # Save model and calibration plot
    out_dir = tmp_path / "model_rf"
    save_model(est, out_dir, meta={"metrics": metrics})
    assert (out_dir / "model.joblib").exists()
    plot_path = out_dir / "calibration.png"
    plot_interval_calibration([0.8, 0.9], [cover["0.8"], cover["0.9"]], plot_path)
    assert plot_path.exists()
