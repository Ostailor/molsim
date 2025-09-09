from __future__ import annotations

import pytest

from moldis.generate.brics_gen import generate_brics
from moldis.generate.common import GenerationConstraints

rdkit = pytest.importorskip("rdkit")


def test_brics_generation_deterministic():
    seeds = [
        "CCO",  # ethanol
        "CCN",  # ethylamine
    ]
    gc = GenerationConstraints.default()
    out1 = generate_brics(seeds, budget=10, constraints=gc, seed=42)
    out2 = generate_brics(seeds, budget=10, constraints=gc, seed=42)
    assert out1 == out2
    # Basic sanity: outputs meet constraints
    assert 1 <= len(out1) <= 10
