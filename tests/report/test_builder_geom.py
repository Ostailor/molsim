from __future__ import annotations

from pathlib import Path

import pytest

from moldis.geometry.conformers import summarize_conformers
from moldis.report.builder import build_geom_report

rdkit = pytest.importorskip("rdkit")


def test_build_geom_report(tmp_path: Path) -> None:
    geom_dir = tmp_path / "geom"
    geom_dir.mkdir()
    # Write two summaries
    s1 = summarize_conformers("CCO", n_confs=4, seed=0)
    s2 = summarize_conformers("CCN", n_confs=4, seed=0)
    (geom_dir / "summaries.jsonl").write_text(
        f"{s1.__dict__}\n".replace("'", '"') + f"{s2.__dict__}\n".replace("'", '"'),
        encoding="utf-8",
    )
    out_md = tmp_path / "report.md"
    out_json = tmp_path / "report.json"
    build_geom_report(geom_dir, out_md, out_json)
    assert out_md.exists()
    assert out_json.exists()
