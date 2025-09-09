from __future__ import annotations

import re

import pytest

from moldis.synth.high_level_routes import propose_routes

rdkit = pytest.importorskip("rdkit")


def test_routes_are_non_actionable_and_cover_common_groups():
    ideas = propose_routes("CC(=O)N")  # simple amide
    text = " ".join(i.transform + " " + i.rationale for i in ideas)
    # Should suggest amide coupling
    assert any("amide" in i.transform for i in ideas)
    # Non-actionable check: no temperatures, times, or solvents
    banned = [r"\d+\s?(°C|K)", r"hours?", r"minutes?", r"solvent", r"stir", r"dropwise"]
    for pat in banned:
        assert re.search(pat, text, flags=re.IGNORECASE) is None
