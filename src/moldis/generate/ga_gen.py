from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass

from ..chem import rdkit_utils as RU
from .common import GenerationConstraints, mol_ok

ScoreFn = Callable[[str], float]


def _require_rdkit() -> None:
    if not RU.RDKIT_AVAILABLE:
        raise RuntimeError("RDKit not available. Install with extras: pip install -e '.[chem]' ")


def default_score(smiles: str) -> float:
    """A simple, fast score combining QED and Lipinski satisfaction."""
    try:
        d = RU.basic_descriptors(smiles)
        lip = RU.lipinski_violations(smiles)
        lip_ok = sum(lip.values()) / len(lip)
        return 0.8 * d.qed + 0.2 * lip_ok
    except Exception:
        return 0.0


@dataclass
class GAConfig:
    population_size: int = 32
    generations: int = 10
    mutation_rate: float = 0.3
    crossover_rate: float = 0.7
    elite_fraction: float = 0.1
    seed: int | None = 0


def _brics_recombine(parent_a: str, parent_b: str, rng: random.Random) -> str | None:
    from rdkit import Chem
    from rdkit.Chem import BRICS

    try:
        frags_a = list(BRICS.BRICSDecompose(Chem.MolFromSmiles(parent_a)))
        frags_b = list(BRICS.BRICSDecompose(Chem.MolFromSmiles(parent_b)))
    except Exception:
        return None
    pool = frags_a + frags_b
    if not pool:
        return None
    k = rng.randint(2, min(5, max(2, len(pool))))
    parts = rng.sample(pool, k)
    try:
        gen = BRICS.BRICSBuild(frags=parts)
        for mol in gen:
            try:
                Chem.SanitizeMol(mol)
            except Exception:
                continue
            return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return None
    return None


def _mutate_brics(smiles: str, rng: random.Random) -> str | None:
    return _brics_recombine(smiles, smiles, rng)


def run_ga(
    seeds: list[str],
    score_fn: ScoreFn | None = None,
    constraints: GenerationConstraints | None = None,
    config: GAConfig | None = None,
) -> list[str]:
    """Run a minimal GA over SMILES, using BRICS-based mutation/crossover.

    Returns a list of unique, high-scoring SMILES.
    Determinism is controlled by config.seed.
    """
    _require_rdkit()
    cfg = config or GAConfig()
    rng = random.Random(cfg.seed)
    gc = constraints or GenerationConstraints.default()
    score = score_fn or default_score

    # Initialize population from seeds; if fewer seeds than population, sample with replacement
    pop: list[str] = []
    if not seeds:
        return []
    while len(pop) < cfg.population_size:
        s = seeds[len(pop) % len(seeds)]
        if mol_ok(s, gc):
            pop.append(s)
        else:
            pop.append(seeds[len(pop) % len(seeds)])

    def evaluate(population: list[str]) -> list[tuple[float, str]]:
        scored: list[tuple[float, str]] = []
        seen: set[str] = set()
        for s in population:
            try:
                c = RU.canonical_smiles(s)
            except Exception:
                continue
            if c in seen or not mol_ok(c, gc):
                continue
            seen.add(c)
            scored.append((score(c), c))
        scored.sort(reverse=True, key=lambda t: t[0])
        return scored

    # Evolution loop
    for _ in range(cfg.generations):
        scored = evaluate(pop)
        if not scored:
            break
        elites_n = max(1, int(cfg.elite_fraction * cfg.population_size))
        elites = [s for _, s in scored[:elites_n]]
        # Generate offspring
        offspring: list[str] = []
        while len(offspring) + len(elites) < cfg.population_size:
            if rng.random() < cfg.crossover_rate and len(pop) >= 2:
                pa = rng.choice(pop)
                pb = rng.choice(pop)
                child = _brics_recombine(pa, pb, rng)
                if child is None:
                    continue
                offspring.append(child)
            elif rng.random() < cfg.mutation_rate:
                p = rng.choice(pop)
                child = _mutate_brics(p, rng)
                if child is None:
                    continue
                offspring.append(child)
            else:
                offspring.append(rng.choice(pop))
        pop = elites + offspring

    final = [s for _, s in evaluate(pop)]
    return final
