from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

from ..chem import rdkit_utils as RU

FEATURE_NAMES: list[str] = [
    "mw",
    "logp",
    "tpsa",
    "qed",
    "hbd",
    "hba",
    "rot_bonds",
    "rings",
]


@dataclass
class Features:
    names: list[str]
    X: np.ndarray  # shape (n, d)
    ok: list[bool]  # validity flags aligned to input order


def featurize_smiles(smiles_list: Iterable[str]) -> Features:
    if not RU.RDKIT_AVAILABLE:
        raise RuntimeError("RDKit required for feature computation; install extras: '.[chem]'")

    X_rows: list[list[float]] = []
    ok_flags: list[bool] = []
    for s in smiles_list:
        try:
            d = RU.basic_descriptors(s)
            X_rows.append(
                [
                    d.mw,
                    d.logp,
                    d.tpsa,
                    d.qed,
                    float(d.hbd),
                    float(d.hba),
                    float(d.rot_bonds),
                    float(d.rings),
                ]
            )
            ok_flags.append(True)
        except Exception:
            X_rows.append([0.0] * len(FEATURE_NAMES))
            ok_flags.append(False)
    X = np.asarray(X_rows, dtype=np.float64)
    return Features(names=list(FEATURE_NAMES), X=X, ok=ok_flags)
