from __future__ import annotations

from pathlib import Path

import pytest

from moldis.geometry.conformers import generate_conformers, summarize_conformers, write_sdf

rdkit = pytest.importorskip("rdkit")


def test_summarize_and_export(tmp_path: Path) -> None:
    smiles = "CCO"  # ethanol
    summ = summarize_conformers(smiles, n_confs=8, seed=0)
    assert summ.success
    assert summ.n_confs >= 1
    assert isinstance(summ.min_energy, float)
    assert (summ.dipole_magnitude is None) or (summ.dipole_magnitude >= 0)

    mol, energies, _ = generate_conformers(smiles, n_confs=4, seed=0)
    out = write_sdf(mol, tmp_path / "mol.sdf", energies)
    assert out.exists()
