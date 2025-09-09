from __future__ import annotations

import pytest

from moldis.physics.semi_empirical import run_xtb_singlepoint, xtb_available

rdkit = pytest.importorskip("rdkit")


@pytest.mark.skipif(not xtb_available(), reason="xtb not installed")
def test_xtb_singlepoint_smoke() -> None:
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.AddHs(Chem.MolFromSmiles("CH4"))
    AllChem.EmbedMolecule(mol, AllChem.ETKDG())
    res = run_xtb_singlepoint(mol, conf_id=0)
    assert res.success
    # Energy may be None if parsing failed, but should not crash
