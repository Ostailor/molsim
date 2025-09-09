from __future__ import annotations

import random
from collections.abc import Iterable

from ..chem import rdkit_utils as RU


def novelty_filter_inchikey(candidates: Iterable[str], known_inchikeys: set[str]) -> list[str]:
    out: list[str] = []
    for s in candidates:
        try:
            ik = RU.to_inchikey(s)
        except Exception:
            continue
        if ik not in known_inchikeys:
            out.append(RU.canonical_smiles(s))
    # dedup on canonical
    return RU.deduplicate_smiles(out)


def _fp(smiles: str, radius: int, n_bits: int):
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("invalid SMILES")
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)


def tanimoto_novelty_filter(
    candidates: Iterable[str],
    reference: Iterable[str],
    threshold: float = 0.7,
    radius: int = 2,
    n_bits: int = 2048,
) -> list[str]:
    """Keep candidates whose max similarity to any reference is < threshold."""
    if not RU.RDKIT_AVAILABLE:
        return list(candidates)
    from rdkit import DataStructs

    ref_fps = []
    for s in reference:
        try:
            ref_fps.append(_fp(s, radius, n_bits))
        except Exception:
            continue
    kept: list[str] = []
    for s in candidates:
        try:
            f = _fp(s, radius, n_bits)
        except Exception:
            continue
        max_sim = 0.0
        for r in ref_fps:
            sim = DataStructs.TanimotoSimilarity(f, r)
            if sim > max_sim:
                max_sim = sim
                if max_sim >= threshold:
                    break
        if max_sim < threshold:
            kept.append(RU.canonical_smiles(s))
    return RU.deduplicate_smiles(kept)


def select_diverse_k_center(
    candidates: list[str],
    k: int,
    radius: int = 2,
    n_bits: int = 2048,
    seed: int | None = 0,
) -> list[str]:
    """Greedy k-center selection under Tanimoto distance (1 - similarity)."""
    if not RU.RDKIT_AVAILABLE or not candidates:
        return candidates[:k]
    from rdkit import DataStructs

    rng = random.Random(seed)
    fps = []
    valid_smiles = []
    for s in candidates:
        try:
            fps.append(_fp(s, radius, n_bits))
            valid_smiles.append(RU.canonical_smiles(s))
        except Exception:
            continue
    if not fps:
        return []
    # Start with a random candidate for deterministic seed
    start_idx = rng.randrange(len(valid_smiles))
    selected = [start_idx]

    # Precompute similarity function to avoid recomputing
    def tanimoto(i: int, j: int) -> float:
        return DataStructs.TanimotoSimilarity(fps[i], fps[j])

    while len(selected) < min(k, len(valid_smiles)):
        # Choose the point that maximizes its min distance to selected set
        best_idx = None
        best_min_dist = -1.0
        for i in range(len(valid_smiles)):
            if i in selected:
                continue
            min_dist = 1.0
            for j in selected:
                min_dist = min(min_dist, 1.0 - tanimoto(i, j))
                if min_dist == 0.0:
                    break
            if min_dist > best_min_dist:
                best_min_dist = min_dist
                best_idx = i
        if best_idx is None:
            break
        selected.append(best_idx)
    return [valid_smiles[i] for i in selected]
