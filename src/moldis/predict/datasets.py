from __future__ import annotations

import csv
from io import StringIO
from pathlib import Path

_ESOL_TINY_CSV = """smiles,logS
CCO,-0.3
CCN,-0.2
OCCO,-0.7
CCOC,-0.5
CC(=O)O,-0.6
OC=O,-0.8
OCCN,-0.4
CCCO,-0.2
CC(C)O,-0.1
CCCCO,0.0
CCOCC,-0.2
CCNCC,-0.1
CCOC(C)C,0.1
CC(C)CO,0.2
CCCOC,-0.1
COC,0.0
COCC,0.1
CCCOC(C)C,0.3
CC(C)OCC,0.2
CCCCCO,0.1
"""


def _read_csv_rows(path: Path) -> tuple[list[str], list[float]]:
    smiles: list[str] = []
    y: list[float] = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            smiles.append(row["smiles"])
            y.append(float(row["logS"]))
    return smiles, y


def _read_csv_rows_from_string(data: str) -> tuple[list[str], list[float]]:
    smiles: list[str] = []
    y: list[float] = []
    reader = csv.DictReader(StringIO(data))
    for row in reader:
        smiles.append(row["smiles"])
        y.append(float(row["logS"]))
    return smiles, y


def load_esol_tiny() -> tuple[list[str], list[float]]:
    """Load a tiny ESOL-like dataset.

    Tries repository paths first; falls back to an embedded CSV when installed.
    """
    here = Path(__file__).resolve()
    candidates: list[Path] = []
    try:
        root = here.parents[3]
        candidates.append(root / "data" / "processed" / "esol_tiny.csv")
        candidates.append(root / "src" / "data" / "processed" / "esol_tiny.csv")
    except Exception:
        pass
    for p in candidates:
        if p.exists():
            return _read_csv_rows(p)
    return _read_csv_rows_from_string(_ESOL_TINY_CSV)


def load_from_csv(
    path: str | Path,
    smiles_col: str = "smiles",
    y_col: str = "y",
) -> tuple[list[str], list[float]]:
    """Load a generic dataset from CSV with columns for SMILES and target.

    - path: CSV file path
    - smiles_col: column name containing SMILES strings
    - y_col: column name containing the target numeric value
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Dataset not found: {p}")
    smiles: list[str] = []
    y: list[float] = []
    with p.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if smiles_col not in (reader.fieldnames or []) or y_col not in (reader.fieldnames or []):
            raise ValueError(
                f"CSV must contain columns '{smiles_col}' and '{y_col}'; found {reader.fieldnames}"
            )
        for row in reader:
            try:
                s = row[smiles_col]
                t = float(row[y_col])
            except Exception as e:
                raise ValueError(f"Error parsing row: {row}") from e
            smiles.append(s)
            y.append(t)
    return smiles, y
