from __future__ import annotations

import csv
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

from ..chem import rdkit_utils as RU


@dataclass
class FeasibilityResult:
    smiles: str
    inchikey: str | None
    sa_proxy: float | None
    block_match_rate: float | None
    hazard_flags: list[str]
    notes: list[str]


# Conservative high-risk SMARTS (non-exhaustive)
HAZARD_SMARTS: dict[str, str] = {
    "azide": "[N-]=[N+]=N",  # simple azide motif
    "peroxide": "[OX2][OX2]",  # -O-O-
    "nitro_aromatic": "[a][N+](=O)[O-]",
    "azo": "N=N",
}


def _require_rdkit() -> None:
    if not RU.RDKIT_AVAILABLE:
        raise RuntimeError("RDKit not available. Install with extras: '.[chem]'")


def load_blocks_csv(path: str | Path) -> list[str]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Building blocks file not found: {p}")
    blocks: list[str] = []
    with p.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if "smiles" not in (reader.fieldnames or []):
            raise ValueError("Blocks CSV must contain a 'smiles' column")
        for row in reader:
            blocks.append(row["smiles"])  # raw; canonicalize later
    return blocks


def sa_score(smiles: str, prefer: str = "auto") -> float | None:
    """Normalized synthesizability score in [0,1] (higher=more feasible).

    Order of attempts:
    - If prefer in {"auto", "sa"}: try sascorer (Ertl SA Score) and map 1–10 -> 0–1
    - Fallback to RDKit proxy (qed and simple penalties) from rdkit_utils
    """
    # Try sascorer if requested
    if prefer in {"auto", "sa"} and RU.RDKIT_AVAILABLE:
        try:
            from rdkit import Chem  # type: ignore

            try:
                from ..third_party import sascorer as _sasc  # type: ignore
            except Exception:
                import sascorer as _sasc  # type: ignore

            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                raw = float(_sasc.calculateScore(mol))  # 1 (easy) .. 10 (hard)
                # Map to 0..1 with higher = easier: (10 - raw)/9 clamped
                val = max(0.0, min(1.0, (10.0 - raw) / 9.0))
                return val
        except Exception:
            pass
    # Fallback proxy
    return RU.synthesizability_score(smiles)


def building_block_match_rate(
    smiles: str, blocks: Iterable[str], max_blocks: int | None = 200
) -> float | None:
    """Fraction of provided blocks that match as substructures (capped by max_blocks).

    Coarse availability signal; meaningful for small libraries only.
    """
    _require_rdkit()
    from rdkit import Chem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    count = 0
    total = 0
    for idx, b in enumerate(blocks):
        if max_blocks is not None and idx >= max_blocks:
            break
        bm = Chem.MolFromSmiles(b)
        if bm is None:
            continue
        total += 1
        try:
            if mol.HasSubstructMatch(bm):
                count += 1
        except Exception:
            continue
    if total == 0:
        return None
    return count / total


def hazard_motifs(smiles: str) -> list[str]:
    _require_rdkit()
    from rdkit import Chem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ["invalid_structure"]
    hits: list[str] = []
    for name, smarts in HAZARD_SMARTS.items():
        try:
            q = Chem.MolFromSmarts(smarts)
            if q is not None and mol.HasSubstructMatch(q):
                hits.append(name)
        except Exception:
            continue
    return hits


def assess_feasibility(
    smiles: str, blocks: Iterable[str] | None = None, sa_prefer: str = "auto"
) -> FeasibilityResult:
    ik = None
    try:
        ik = RU.to_inchikey(smiles)
    except Exception:
        pass
    sa = sa_score(smiles, prefer=sa_prefer)
    rate = building_block_match_rate(smiles, blocks or []) if blocks is not None else None
    flags = hazard_motifs(smiles)
    notes = [
        "Non-actionable synthesis guidance only; metrics are approximate.",
        "Hazard flags are conservative and not exhaustive.",
    ]
    return FeasibilityResult(
        smiles=smiles,
        inchikey=ik,
        sa_proxy=sa,
        block_match_rate=rate,
        hazard_flags=flags,
        notes=notes,
    )


def write_feasibility_jsonl(
    smiles_list: Iterable[str], out_path: str | Path, blocks: Iterable[str] | None = None
) -> Path:
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for s in smiles_list:
            rec = asdict(assess_feasibility(s, blocks=blocks))
            import json

            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return p
