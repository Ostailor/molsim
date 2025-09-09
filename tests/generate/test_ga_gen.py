from __future__ import annotations

import pytest

from moldis.generate.common import GenerationConstraints
from moldis.generate.ga_gen import GAConfig, run_ga

rdkit = pytest.importorskip("rdkit")


def test_ga_runs_and_is_deterministic():
    seeds = [
        "CC(=O)OC1=CC=CC=C1C(=O)O",  # aspirin
        "CN1C=NC2=C1C(=O)N(C(=O)N2)C",  # caffeine
    ]
    cfg = GAConfig(
        population_size=16,
        generations=3,
        seed=123,
        mutation_rate=0.5,
        crossover_rate=0.5,
    )
    gc = GenerationConstraints.default()
    out1 = run_ga(seeds, constraints=gc, config=cfg)
    out2 = run_ga(seeds, constraints=gc, config=cfg)
    assert out1 == out2
    assert len(out1) >= 1
