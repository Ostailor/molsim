from __future__ import annotations

import pytest

from moldis.chem import rdkit_utils as RU

rdkit = pytest.importorskip("rdkit")


def test_canonicalization_and_identifiers():
    smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"  # aspirin
    can = RU.canonical_smiles(smiles)
    assert isinstance(can, str) and can
    ik = RU.to_inchikey(smiles)
    assert len(ik) == 27 and "-" in ik


def test_basic_descriptors_and_rules():
    smiles = "CN1C=NC2=C1C(=O)N(C(=O)N2)C"  # caffeine
    d = RU.basic_descriptors(smiles)
    assert d.mw > 100 and d.tpsa > 0 and 0 <= d.qed <= 1
    lip = RU.lipinski_violations(smiles)
    assert all(isinstance(v, bool) for v in lip.values())
    v = RU.veber(smiles)
    assert all(isinstance(vv, bool) for vv in v.values())


def test_ecfp_bits_and_pains_and_synth():
    smiles = "c1ccccc1O"  # phenol
    bits = RU.ecfp_bits(smiles, radius=2, n_bits=128)
    assert len(bits) == 128 and set(bits) <= {0, 1}
    alerts = RU.pains_alerts(smiles)
    assert isinstance(alerts, list)
    sa = RU.synthesizability_score(smiles)
    assert sa is None or (0.0 <= sa <= 1.0)


def test_deduplication():
    a = "O=C(O)c1ccccc1OC(=O)C"  # aspirin alt form
    b = "CC(=O)OC1=CC=CC=C1C(=O)O"
    uniq = RU.deduplicate_smiles([a, b])
    assert len(uniq) == 1
