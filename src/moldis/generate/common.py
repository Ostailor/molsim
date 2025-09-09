from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from ..chem import rdkit_utils as RU


@dataclass
class GenerationConstraints:
    allowed_elements: set[str]
    max_heavy_atoms: int
    max_rings: int
    min_synth_score: float | None = None  # optional heuristic from RU.synthesizability_score

    @staticmethod
    def default() -> GenerationConstraints:
        return GenerationConstraints(
            allowed_elements={"C", "H", "N", "O", "F", "S", "P", "Cl"},
            max_heavy_atoms=40,
            max_rings=4,
        )


def mol_ok(smiles: str, gc: GenerationConstraints) -> bool:
    """Check structural constraints using RDKit-derived info."""
    if not RU.RDKIT_AVAILABLE:
        return False
    try:
        from rdkit import Chem
        from rdkit.Chem import rdMolDescriptors
    except Exception:
        return False

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False
    if mol.GetNumHeavyAtoms() > gc.max_heavy_atoms:
        return False
    rings = rdMolDescriptors.CalcNumRings(mol)
    if rings > gc.max_rings:
        return False
    # allowed elements
    for atom in mol.GetAtoms():
        if atom.GetSymbol() not in gc.allowed_elements:
            return False
    if gc.min_synth_score is not None:
        s = RU.synthesizability_score(smiles)
        if s is None or s < gc.min_synth_score:
            return False
    return True


def canonical_dedup(smiles_iter: Iterable[str]) -> list[str]:
    uniq: list[str] = []
    seen: set[str] = set()
    for s in smiles_iter:
        try:
            c = RU.canonical_smiles(s)
        except Exception:
            continue
        if c not in seen:
            seen.add(c)
            uniq.append(c)
    return uniq
