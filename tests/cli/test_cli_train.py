from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

sklearn = pytest.importorskip("sklearn")
rdkit = pytest.importorskip("rdkit")
mpl = pytest.importorskip("matplotlib")


def test_cli_train_esol(tmp_path: Path) -> None:
    out = tmp_path / "esol_rf"
    cmd = [
        sys.executable,
        "-m",
        "moldis",
        "train",
        "--task",
        "esol",
        "--out",
        str(out),
        "--model",
        "rf",
        "--val-fraction",
        "0.3",
        "--random-state",
        "0",
        "--n-estimators",
        "32",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, res.stderr
    assert (out / "model.joblib").exists()
    assert (out / "model.json").exists()
    assert (out / "calibration.png").exists()
