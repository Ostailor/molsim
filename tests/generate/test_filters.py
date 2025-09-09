from __future__ import annotations

import pytest

from moldis.generate.filters import (
    novelty_filter_inchikey,
    select_diverse_k_center,
    tanimoto_novelty_filter,
)

rdkit = pytest.importorskip("rdkit")


def test_novelty_and_diversity():
    refs = [
        "CC(=O)OC1=CC=CC=C1C(=O)O",  # aspirin
        "CN1C=NC2=C1C(=O)N(C(=O)N2)C",  # caffeine
    ]
    cands = refs + ["c1ccccc1O", "CCO", "CCCCN"]
    # Novelty by inchikey
    from moldis.chem.rdkit_utils import to_inchikey

    known = {to_inchikey(s) for s in refs}
    novel = novelty_filter_inchikey(cands, known)
    assert all(s not in refs for s in novel)
    # Tanimoto novelty filter should drop references at threshold 1.0 and keep others
    nov2 = tanimoto_novelty_filter(cands, refs, threshold=0.9)
    assert all(s not in refs for s in nov2)
    # Diversity selection
    sel = select_diverse_k_center(nov2, k=2, seed=0)
    assert len(sel) == min(2, len(nov2))
