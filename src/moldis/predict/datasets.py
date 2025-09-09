from __future__ import annotations

import csv
from pathlib import Path


def load_esol_tiny() -> tuple[list[str], list[float]]:
    """Load a tiny ESOL-like dataset bundled with the repo for tests.

    Returns (smiles_list, solubility_logS)
    """
    here = Path(__file__).resolve()
    # project root is three levels up from this file's directory (src/moldis/predict)
    root = here.parents[3]
    p = root / "data" / "processed" / "esol_tiny.csv"
    if not p.exists():
        # Fallback to running from src/ installation (if layout differs)
        p = root / "src" / "data" / "processed" / "esol_tiny.csv"
    if not p.exists():
        raise FileNotFoundError(f"Dataset not found: {p}")
    smiles: list[str] = []
    y: list[float] = []
    with p.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            smiles.append(row["smiles"])
            y.append(float(row["logS"]))
    return smiles, y
