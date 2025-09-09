from __future__ import annotations

import subprocess
import sys

import moldis


def test_version_string() -> None:
    assert isinstance(moldis.__version__, str) and len(moldis.__version__) >= 5


def test_cli_help_exits_zero() -> None:
    out = subprocess.run([sys.executable, "-m", "moldis", "--help"], capture_output=True)
    assert out.returncode == 0
    assert b"Molecule discovery pipeline" in out.stdout + out.stderr
