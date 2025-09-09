from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Iterable

from ..chem import rdkit_utils as RU


def murcko_scaffold(smiles: str) -> str:
    """Return Bemis–Murcko scaffold SMILES using RDKit; falls back to canonical SMILES on error."""
    if not RU.RDKIT_AVAILABLE:
        return smiles
    try:
        from rdkit import Chem
        from rdkit.Chem.Scaffolds import MurckoScaffold

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return smiles
        scaf = MurckoScaffold.GetScaffoldForMol(mol)
        return Chem.MolToSmiles(scaf, canonical=True)
    except Exception:
        return smiles


def scaffold_split(
    smiles_list: Iterable[str],
    val_fraction: float = 0.2,
    random_state: int | None = 0,
) -> tuple[list[int], list[int]]:
    """Greedy scaffold split: assign entire scaffolds to validation until reaching val_fraction.

    Returns (train_indices, val_indices).
    """
    smiles = list(smiles_list)
    n = len(smiles)
    _ = random.Random(random_state)

    by_scaffold: dict[str, list[int]] = defaultdict(list)
    for idx, s in enumerate(smiles):
        by_scaffold[murcko_scaffold(s)].append(idx)

    # Sort scaffolds by size (desc) then by hash for determinism
    scaf_groups = sorted(by_scaffold.values(), key=lambda g: (-len(g), hash(tuple(g))))

    val_set: set[int] = set()
    target_val = int(round(val_fraction * n))
    for group in scaf_groups:
        if len(val_set) >= target_val:
            break
        # Simple greedy selection
        val_set.update(group)

    # If overshot heavily, randomly move some groups back (rare with small sets)
    if len(val_set) > target_val and scaf_groups:
        extra = len(val_set) - target_val
        for group in scaf_groups[::-1]:
            if extra <= 0:
                break
            moved = 0
            for idx in group:
                if idx in val_set:
                    val_set.remove(idx)
                    moved += 1
                    extra -= 1
                    if extra <= 0:
                        break

    val_idx = sorted(val_set)
    train_idx = sorted(set(range(n)) - val_set)
    return train_idx, val_idx
