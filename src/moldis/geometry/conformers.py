from __future__ import annotations

import hashlib
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from ..chem import rdkit_utils as RU


@dataclass
class ConformerSummary:
    smiles: str
    inchikey: str | None
    forcefield: str
    n_confs: int
    energies: list[float]
    min_energy: float | None
    dipole_magnitude: float | None
    success: bool
    message: str | None = None
    xtb_energy: float | None = None
    xtb_dipole: float | None = None
    cache_key: str | None = None


def _require_rdkit() -> None:
    if not RU.RDKIT_AVAILABLE:
        raise RuntimeError("RDKit not available. Install with extras: '.[chem]'")


def _mmff_available(mol) -> bool:
    from rdkit.Chem import AllChem

    try:
        props = AllChem.MMFFGetMoleculeProperties(mol, mmffVariant="MMFF94")
        return props is not None
    except Exception:
        return False


def _optimize_and_energy(mol, conf_id: int, use_mmff: bool) -> float | None:
    from rdkit.Chem import AllChem

    try:
        if use_mmff:
            props = AllChem.MMFFGetMoleculeProperties(mol, mmffVariant="MMFF94")
            ff = AllChem.MMFFGetMoleculeForceField(mol, props, confId=conf_id)
        else:
            ff = AllChem.UFFGetMoleculeForceField(mol, confId=conf_id)
        ff.Initialize()
        ff.Minimize(maxIts=200)
        return float(ff.CalcEnergy())
    except Exception:
        return None


def _compute_gasteiger_dipole(mol, conf_id: int) -> float | None:
    """Approximate dipole magnitude (e·Å) using Gasteiger charges and 3D coordinates."""
    from rdkit.Chem import AllChem

    try:
        AllChem.ComputeGasteigerCharges(mol)
        conf = mol.GetConformer(conf_id)
        mu = np.zeros(3, dtype=float)
        for atom in mol.GetAtoms():
            q = atom.GetDoubleProp("_GasteigerCharge") if atom.HasProp("_GasteigerCharge") else 0.0
            pos = conf.GetAtomPosition(atom.GetIdx())
            r = np.array([pos.x, pos.y, pos.z], dtype=float)
            mu += q * r
        return float(np.linalg.norm(mu))
    except Exception:
        return None


def generate_conformers(
    smiles: str,
    n_confs: int = 20,
    seed: int | None = 0,
    prune_rms_thresh: float = 0.5,
    forcefield_preference: str = "MMFF",
) -> tuple[object, list[float], bool]:
    """Generate and optimize conformers; return (mol, energies, use_mmff).

    Energies in kcal/mol units as returned by RDKit force fields (arbitrary scale; relative use).
    """
    _require_rdkit()
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES")
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    if seed is not None:
        params.randomSeed = int(seed)
    params.pruneRmsThresh = float(prune_rms_thresh)
    ids = AllChem.EmbedMultipleConfs(mol, numConfs=int(n_confs), params=params)
    if not ids:
        # Try with a more permissive setting
        params.useRandomCoords = True
        ids = AllChem.EmbedMultipleConfs(mol, numConfs=max(1, int(n_confs // 2)), params=params)
    if not ids:
        return mol, [], False

    use_mmff = False
    if forcefield_preference.upper() == "MMFF" and _mmff_available(mol):
        use_mmff = True

    energies: list[float] = []
    for cid in ids:
        e = _optimize_and_energy(mol, cid, use_mmff=use_mmff)
        if e is not None:
            energies.append(e)
        else:
            # Drop conformer if optimization failed
            try:
                mol.RemoveConformer(cid)
            except Exception:
                pass
    return mol, energies, use_mmff


def _key_for(
    smiles: str, n_confs: int, seed: int | None, prune_rms_thresh: float, ff: str, do_xtb: bool
) -> str:
    try:
        ik = RU.to_inchikey(smiles)
    except Exception:
        ik = smiles
    payload = {
        "ik": ik,
        "n": int(n_confs),
        "seed": int(seed) if seed is not None else None,
        "prune": float(prune_rms_thresh),
        "ff": ff.upper(),
        "xtb": bool(do_xtb),
    }
    blob = (str(payload)).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def summarize_conformers(
    smiles: str,
    n_confs: int = 20,
    seed: int | None = 0,
    prune_rms_thresh: float = 0.5,
    forcefield_preference: str = "MMFF",
    cache_dir: str | Path | None = None,
    do_xtb: bool = False,
) -> ConformerSummary:
    """Generate conformers and return a summary with basic physics-like properties."""
    try:
        cache_key = _key_for(smiles, n_confs, seed, prune_rms_thresh, forcefield_preference, do_xtb)
        if cache_dir is not None:
            cpath = Path(cache_dir) / f"{cache_key}.json"
            if cpath.exists():
                import json

                data = json.loads(cpath.read_text(encoding="utf-8"))
                return ConformerSummary(**data)

        mol, energies, use_mmff = generate_conformers(
            smiles,
            n_confs=n_confs,
            seed=seed,
            prune_rms_thresh=prune_rms_thresh,
            forcefield_preference=forcefield_preference,
        )
    except Exception as e:
        return ConformerSummary(
            smiles=smiles,
            inchikey=None,
            forcefield=forcefield_preference.upper(),
            n_confs=0,
            energies=[],
            min_energy=None,
            dipole_magnitude=None,
            success=False,
            message=str(e),
            cache_key=None,
        )

    inchikey = None
    try:
        inchikey = RU.to_inchikey(smiles)
    except Exception:
        pass

    min_e = float(min(energies)) if energies else None
    dip = _compute_gasteiger_dipole(mol, 0) if energies else None
    xtb_e = None
    xtb_dip = None
    if do_xtb and energies:
        try:
            from ..physics.semi_empirical import run_xtb_singlepoint, xtb_available

            if xtb_available():
                res = run_xtb_singlepoint(mol, conf_id=0)
                if res.success:
                    xtb_e = res.energy
                    xtb_dip = res.dipole
        except Exception:
            pass

    summary = ConformerSummary(
        smiles=smiles,
        inchikey=inchikey,
        forcefield="MMFF" if use_mmff else "UFF",
        n_confs=len(energies) if energies else 0,
        energies=[float(x) for x in energies],
        min_energy=min_e,
        dipole_magnitude=dip,
        success=bool(energies),
        message=None,
        xtb_energy=xtb_e,
        xtb_dipole=xtb_dip,
        cache_key=cache_key,
    )
    if cache_dir is not None:
        import json

        cpath = Path(cache_dir)
        cpath.mkdir(parents=True, exist_ok=True)
        (cpath / f"{summary.cache_key}.json").write_text(
            json.dumps(asdict(summary), ensure_ascii=False, sort_keys=True), encoding="utf-8"
        )
    return summary


def write_sdf(mol, path: str | Path, energies: list[float] | None = None) -> Path:
    """Write all conformers to an SDF file; annotate energies if provided."""
    from rdkit import Chem

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    w = Chem.SDWriter(str(p))
    try:
        for idx, conf in enumerate(mol.GetConformers()):
            m = Chem.Mol(mol)
            m.RemoveAllConformers()
            m.AddConformer(conf, assignId=True)
            if energies is not None and idx < len(energies):
                m.SetProp("Energy", f"{energies[idx]:.6f}")
            w.write(m)
    finally:
        w.close()
    return p


def batch_conformers_to_artifacts(
    smiles_list: Iterable[str],
    out_dir: str | Path,
    n_confs: int = 20,
    seed: int | None = 0,
    prune_rms_thresh: float = 0.5,
    forcefield_preference: str = "MMFF",
    cache_dir: str | Path | None = "artifacts/cache/geom",
    do_xtb: bool = False,
) -> Path:
    """Process a list of SMILES, write SDF and JSON summaries in out_dir.

    - Writes `structures.sdf` with first conformer per molecule.
    - Writes `summaries.jsonl` containing one JSON per molecule summary.
    """
    import json

    from rdkit import Chem

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    sdf_path = out / "structures.sdf"
    writer = Chem.SDWriter(str(sdf_path))
    try:
        with (out / "summaries.jsonl").open("w", encoding="utf-8") as jf:
            for s in smiles_list:
                summ = summarize_conformers(
                    s,
                    n_confs=n_confs,
                    seed=seed,
                    prune_rms_thresh=prune_rms_thresh,
                    forcefield_preference=forcefield_preference,
                    cache_dir=cache_dir,
                    do_xtb=do_xtb,
                )
                jf.write(json.dumps(asdict(summ), ensure_ascii=False) + "\n")
                if summ.success:
                    # regenerate molecule to export first conformer cleanly
                    mol, energies, _ = generate_conformers(
                        s,
                        n_confs=1,
                        seed=seed,
                        prune_rms_thresh=prune_rms_thresh,
                        forcefield_preference=forcefield_preference,
                    )
                    writer.write(mol)
    finally:
        writer.close()
    return sdf_path
