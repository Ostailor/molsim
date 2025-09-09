from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

try:  # optional dependency
    from rdkit import Chem
    from rdkit.Chem import QED, AllChem, Descriptors, FilterCatalog, rdMolDescriptors
    from rdkit.Chem import inchi as rd_inchi
    from rdkit.Chem.MolStandardize import rdMolStandardize

    RDKIT_AVAILABLE = True
except Exception:  # pragma: no cover - import guard
    Chem = None  # type: ignore[assignment]
    rd_inchi = None  # type: ignore[assignment]
    rdMolStandardize = None  # type: ignore[assignment]
    Descriptors = rdMolDescriptors = QED = AllChem = FilterCatalog = None  # type: ignore[assignment]
    RDKIT_AVAILABLE = False


class RDKitNotAvailable(RuntimeError):
    pass


def _require_rdkit() -> None:
    if not RDKIT_AVAILABLE:
        raise RDKitNotAvailable(
            "rdkit not installed. Install optional 'rdkit-pypi' to enable chem features."
        )


def mol_from_smiles(smiles: str, sanitize: bool = True):
    _require_rdkit()
    if sanitize:
        mol = Chem.MolFromSmiles(smiles)
    else:
        mol = Chem.MolFromSmiles(smiles, sanitize=False)
    return mol


def canonical_smiles(smiles: str) -> str:
    _require_rdkit()
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES")
    return Chem.MolToSmiles(mol, canonical=True)


def sanitize_mol(mol):
    _require_rdkit()
    Chem.SanitizeMol(mol)
    return mol


def standardize_tautomer(smiles: str) -> str:
    _require_rdkit()
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES")
    enumerator = rdMolStandardize.TautomerEnumerator()
    can = enumerator.Canonicalize(mol)
    return Chem.MolToSmiles(can, canonical=True)


def to_inchi(smiles: str) -> str:
    _require_rdkit()
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES")
    return rd_inchi.MolToInchi(mol)


def to_inchikey(smiles: str) -> str:
    _require_rdkit()
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES")
    return rd_inchi.MolToInchiKey(mol)


def ecfp_bits(smiles: str, radius: int = 2, n_bits: int = 2048) -> list[int]:
    _require_rdkit()
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES")
    bv = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    # Convert to list[int] of 0/1 rather than list[str]
    return [int(ch) for ch in bv.ToBitString()]


@dataclass
class BasicDescriptors:
    mw: float
    logp: float
    tpsa: float
    qed: float
    hbd: int
    hba: int
    rot_bonds: int
    rings: int


def basic_descriptors(smiles: str) -> BasicDescriptors:
    _require_rdkit()
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES")
    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    tpsa = rdMolDescriptors.CalcTPSA(mol)
    qed = float(QED.qed(mol))
    hbd = rdMolDescriptors.CalcNumHBD(mol)
    hba = rdMolDescriptors.CalcNumHBA(mol)
    rot = rdMolDescriptors.CalcNumRotatableBonds(mol)
    rings = rdMolDescriptors.CalcNumRings(mol)
    return BasicDescriptors(
        mw=mw,
        logp=logp,
        tpsa=tpsa,
        qed=qed,
        hbd=hbd,
        hba=hba,
        rot_bonds=rot,
        rings=rings,
    )


def lipinski_violations(smiles: str) -> dict[str, bool]:
    _require_rdkit()
    d = basic_descriptors(smiles)
    return {
        "HBD<=5": d.hbd <= 5,
        "HBA<=10": d.hba <= 10,
        "MW<=500": d.mw <= 500,
        "logP<=5": d.logp <= 5,
    }


def veber(smiles: str) -> dict[str, bool]:
    _require_rdkit()
    d = basic_descriptors(smiles)
    return {
        "RB<=10": d.rot_bonds <= 10,
        "TPSA<=140": d.tpsa <= 140,
    }


def pains_alerts(smiles: str) -> list[str]:
    _require_rdkit()
    params = FilterCatalog.FilterCatalogParams()
    params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS)
    cat = FilterCatalog.FilterCatalog(params)
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES")
    entries: list[str] = []
    for match in cat.GetMatches(mol):
        try:
            desc = match.GetEntry().GetDescription()
        except Exception:  # pragma: no cover - API fallback
            desc = "PAINS_match"
        entries.append(desc)
    return entries


def deduplicate_smiles(smiles_list: Iterable[str]) -> list[str]:
    _require_rdkit()
    seen: set[str] = set()
    out: list[str] = []
    for s in smiles_list:
        try:
            c = canonical_smiles(s)
        except Exception:
            continue
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def synthesizability_score(smiles: str) -> float | None:
    """Placeholder synthesizability proxy.

    Returns a normalized heuristic in [0,1] combining rings and rotatable bonds.
    If rdkit is unavailable, returns None.
    Note: For research use, integrate SA Score or SCScore in later phases.
    """
    if not RDKIT_AVAILABLE:
        return None
    d = basic_descriptors(smiles)
    # Heuristic: more rings and very high RB reduce score; QED contributes positively
    ring_penalty = min(d.rings / 6.0, 1.0)
    rb_penalty = min(d.rot_bonds / 15.0, 1.0)
    score = max(0.0, min(1.0, 0.6 * d.qed + 0.4 * (1.0 - 0.5 * (ring_penalty + rb_penalty))))
    return score
