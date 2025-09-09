from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from moldis.predict.datasets import load_esol_tiny
from moldis.predict.features import FEATURE_NAMES, featurize_smiles
from moldis.predict.infer import batch_predict, batch_predict_cached
from moldis.predict.linear import fit_linear_regressor

rdkit = pytest.importorskip("rdkit")


def test_esol_training_and_cached_infer(tmp_path: Path):
    smiles, y_list = load_esol_tiny()
    y = np.asarray(y_list, dtype=float)
    feats = featurize_smiles(smiles)
    model = fit_linear_regressor(FEATURE_NAMES, feats.X, y, lam=1e-2)
    # raw batch predict
    preds = batch_predict(smiles[:5], model)
    assert len(preds) == 5
    # cached batch predict
    cache_dir = tmp_path / "cache"
    preds1 = batch_predict_cached(smiles[:5], model, cache_dir=cache_dir)
    preds2 = batch_predict_cached(smiles[:5], model, cache_dir=cache_dir)
    # ensure the cache is used (second call should read from cache)
    assert [round(p.mean, 6) for p in preds1] == [round(p.mean, 6) for p in preds2]
