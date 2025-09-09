from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path


def test_cli_score_pareto(tmp_path: Path) -> None:
    # Create a small candidate CSV
    data = [
        {"id": "a", "p1": 0.8, "p2": 0.2},
        {"id": "b", "p1": 0.6, "p2": 0.6},
        {"id": "c", "p1": 0.2, "p2": 0.9},
        {"id": "d", "p1": 0.5, "p2": 0.4},
    ]
    csv_path = tmp_path / "cands.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "p1", "p2"])
        w.writeheader()
        for r in data:
            w.writerow(r)

    out = tmp_path / "sel"
    cmd = [
        sys.executable,
        "-m",
        "moldis",
        "score",
        "--input",
        str(csv_path),
        "--out",
        str(out),
        "--objectives",
        "p1:max,p2:max",
        "--k",
        "2",
        "--method",
        "pareto",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, res.stderr
    assert (out / "selected.csv").exists()
    assert (out / "score.json").exists()
    # Score JSON should include selected_ids and possibly hypervolume
    js = json.loads((out / "score.json").read_text(encoding="utf-8"))
    assert js.get("selected_ids") and isinstance(js.get("selected_ids"), list)
