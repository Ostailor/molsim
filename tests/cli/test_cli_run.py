from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_cli_run_example_spec(tmp_path: Path) -> None:
    # Use the provided example spec
    spec_path = Path("configs/spec.example.yaml")
    assert spec_path.exists()
    out = subprocess.run(
        [sys.executable, "-m", "moldis", "run", "--spec", str(spec_path)],
        capture_output=True,
        text=True,
    )
    assert out.returncode == 0, out.stderr
    assert "run_id" in out.stdout


def test_cli_run_blocked_spec() -> None:
    spec_path = Path("configs/spec.blocked.yaml")
    assert spec_path.exists()
    out = subprocess.run(
        [sys.executable, "-m", "moldis", "run", "--spec", str(spec_path)],
        capture_output=True,
        text=True,
    )
    assert out.returncode != 0
    assert "blocked" in (out.stderr or "").lower()
