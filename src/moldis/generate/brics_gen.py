from __future__ import annotations

import random
from collections.abc import Iterable

from ..chem import rdkit_utils as RU
from .common import GenerationConstraints, mol_ok


def _require_rdkit() -> None:
    if not RU.RDKIT_AVAILABLE:
        raise RuntimeError("RDKit not available. Install with extras: pip install -e '.[chem]' ")


def brics_fragments(smiles_list: Iterable[str]) -> set[str]:
    """Return a set of BRICS fragments (SMILES with attachment points)."""
    _require_rdkit()
    from rdkit import Chem
    from rdkit.Chem import BRICS

    frags: set[str] = set()
    for s in smiles_list:
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            continue
        for f in BRICS.BRICSDecompose(mol):
            frags.add(f)
    return frags


def generate_brics(
    seeds: list[str],
    budget: int = 100,
    constraints: GenerationConstraints | None = None,
    seed: int | None = 0,
) -> list[str]:
    """Generate candidate molecules by BRICS recombination of seed fragments.

    - seeds: input SMILES used to build a fragment pool
    - budget: max number of valid, unique molecules to return
    - constraints: structural constraints applied to outputs
    - seed: RNG seed for determinism (None for system randomness)
    """
    _require_rdkit()
    from rdkit import Chem
    from rdkit.Chem import BRICS

    _ = random.Random(seed)  # reserved for future randomization
    gc = constraints or GenerationConstraints.default()
    pool = sorted(list(brics_fragments(seeds)))

    def _fallback_from_seeds() -> list[str]:
        uniq_local: list[str] = []
        seen_local: set[str] = set()
        for s in seeds:
            try:
                c = RU.canonical_smiles(s)
            except Exception:
                c = s
            if c not in seen_local:
                seen_local.add(c)
                uniq_local.append(c)
        return uniq_local[:budget]

    if not pool:
        return _fallback_from_seeds()

    out: list[str] = []
    seen: set[str] = set()

    # Deterministic enumeration using full fragment pool; stop at budget.
    try:
        gen = BRICS.BRICSBuild(frags=pool)
    except Exception:
        return _fallback_from_seeds()
    for mol in gen:
        try:
            Chem.SanitizeMol(mol)
        except Exception:
            continue
        s = Chem.MolToSmiles(mol, canonical=True)
        if s in seen:
            continue
        if not mol_ok(s, gc):
            continue
        seen.add(s)
        out.append(s)
        if len(out) >= budget:
            break
    # Fallback: if no recombinations were produced, return canonicalized seeds (dedup)
    if not out:
        return _fallback_from_seeds()
    return out
