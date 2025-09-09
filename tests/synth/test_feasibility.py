from __future__ import annotations

import pytest

from moldis.synth.feasibility import assess_feasibility

rdkit = pytest.importorskip("rdkit")


def test_feasibility_basic():
    s = "CC(=O)Oc1ccccc1"  # phenyl acetate
    res = assess_feasibility(s, blocks=["CC(=O)O", "c1ccccc1O"])  # acetic acid, phenol
    assert res.sa_proxy is None or (0.0 <= res.sa_proxy <= 1.0)
    if res.block_match_rate is not None:
        assert 0.0 <= res.block_match_rate <= 1.0
    assert isinstance(res.hazard_flags, list)
