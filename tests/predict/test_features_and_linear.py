from __future__ import annotations

import math

import numpy as np
import pytest

from moldis.predict.features import FEATURE_NAMES, featurize_smiles
from moldis.predict.linear import fit_linear_regressor

rdkit = pytest.importorskip("rdkit")


def test_featurize_shapes():
    smiles = ["CCO", "invalid", "CCN"]
    feats = featurize_smiles(smiles)
    assert feats.X.shape == (3, len(FEATURE_NAMES))
    assert feats.ok == [True, False, True]


def test_linear_fit_and_predict():
    smiles = ["CCO", "CCN", "OCCO", "CCOC"]
    y = np.array([-0.3, -0.2, -0.7, -0.5], dtype=float)
    feats = featurize_smiles(smiles)
    model = fit_linear_regressor(FEATURE_NAMES, feats.X, y, lam=1e-2)
    preds = model.predict(feats.X)
    assert preds.shape == (4,)
    assert math.isfinite(float(model.sigma))
