from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from ..chem import rdkit_utils as RU
from ..generate.brics_gen import generate_brics
from ..generate.common import GenerationConstraints
from ..generate.filters import tanimoto_novelty_filter
from ..orchestrator.seeds import SEED_SMILES
from ..report.builder import build_geom_report
from ..scoring.selection import select_pareto
from ..spec.models import Spec
from ..synth.feasibility import assess_feasibility
from ..synth.high_level_routes import propose_routes
from ..utils.run import log_event, new_run_id, run_dir, snapshot_config


def _gc_from_spec(spec: Spec) -> GenerationConstraints:
    return GenerationConstraints(
        allowed_elements=set(spec.constraints.allowed_elements),
        max_heavy_atoms=spec.constraints.max_heavy_atoms,
        max_rings=spec.constraints.max_rings,
        min_synth_score=spec.constraints.synthesizability_min,
    )


def _property_map(smiles: str) -> dict[str, float]:
    d = RU.basic_descriptors(smiles)
    return {
        "MW": float(d.mw),
        "logP": float(d.logp),
        "TPSA": float(d.tpsa),
        "QED": float(d.qed),
        "HBD": float(d.hbd),
        "HBA": float(d.hba),
        "RB": float(d.rot_bonds),
        "RINGS": float(d.rings),
    }


def _score_from_objectives(
    spec: Spec, props_list: list[dict[str, float]]
) -> tuple[np.ndarray, list[str]]:
    # For each objective:
    # - If target+tolerance, score in [0,1] = 1 - min(1, |val-target|/tol)
    # - Else maximize raw property (normalized later); fallback to QED if empty
    n = len(props_list)
    if n == 0:
        return np.zeros((0, 0)), []

    dims: list[str] = []
    scores: list[list[float]] = []
    for obj in spec.objectives:
        pname = obj.name
        vals = [props.get(pname) for props in props_list]
        if any(v is None for v in vals):
            continue
        if obj.tolerance is not None:
            target = float(obj.target) if isinstance(obj.target, (int | float)) else 0.0
            tol = float(obj.tolerance)
            # 0..1 score
            dim = [float(max(0.0, 1.0 - min(1.0, abs(v - target) / max(tol, 1e-9)))) for v in vals]
        else:
            # Raw values; will be normalized by compute_bounds inside selector; here keep as is
            dim = [float(v) for v in vals]
        dims.append(pname)
        scores.append(dim)

    if not scores:
        # Fallback to QED
        vals = [props.get("QED", 0.0) for props in props_list]
        dims = ["QED"]
        scores = [list(map(float, vals))]
    arr = np.asarray(scores, dtype=float).T  # shape (n, d)
    return arr, dims


def _write_csv(path: Path, header: list[str], rows: list[list[Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow(r)


def _read_candidates_csv(path: Path) -> tuple[list[str], list[str], list[dict[str, float]]]:
    ids: list[str] = []
    smiles: list[str] = []
    props_list: list[dict[str, float]] = []
    import csv as _csv

    with path.open(encoding="utf-8") as f:
        reader = _csv.DictReader(f)
        for row in reader:
            rid = row.get("id") or row.get("ID") or ""
            smi = row.get("smiles") or row.get("SMILES") or ""
            if not smi:
                # skip malformed
                continue
            ids.append(rid if rid else f"row{len(ids)}")
            smiles.append(smi)
            props: dict[str, float] = {}
            for k, v in row.items():
                if k in {"id", "ID", "smiles", "SMILES"}:
                    continue
                try:
                    props[k] = float(v) if v is not None and v != "" else 0.0
                except Exception:
                    # Non-numeric field, ignore
                    continue
            props_list.append(props)
    return ids, smiles, props_list


def _read_selected_csv(path: Path) -> tuple[list[str], list[str]]:
    import csv as _csv

    ids: list[str] = []
    smiles: list[str] = []
    with path.open(encoding="utf-8") as f:
        reader = _csv.DictReader(f)
        for row in reader:
            rid = row.get("id") or row.get("ID") or ""
            smi = row.get("smiles") or row.get("SMILES") or ""
            if not smi:
                continue
            ids.append(rid if rid else f"row{len(ids)}")
            smiles.append(smi)
    return ids, smiles


def run_pipeline(spec: Spec, resume: str | None = None) -> str:
    # Initialize run directory
    run_id = resume or new_run_id()
    out = run_dir(run_id)
    if resume:
        log_event(run_id, "resume_start", request_id=spec.request_id, use_case=spec.use_case)
        if not (out / "config.spec.json").exists():
            snapshot_config(run_id, json.loads(spec.model_dump_json()))
    else:
        snapshot_config(run_id, json.loads(spec.model_dump_json()))
        log_event(run_id, "run_start", request_id=spec.request_id, use_case=spec.use_case)

    # Candidates (generate + properties) — allow resume from candidates.csv
    candidates_csv = out / "candidates.csv"
    if resume and candidates_csv.exists():
        ids, smiles_list, props_list = _read_candidates_csv(candidates_csv)
        # Infer dimension names from spec, else fallback to QED-only later
        _, dim_names = _score_from_objectives(spec, props_list)
        log_event(run_id, "candidates_loaded", count=len(ids))
    else:
        gc = _gc_from_spec(spec)
        budget = int(spec.compute.max_candidates)
        budget = max(100, min(budget, 2000))
        gen = generate_brics(SEED_SMILES, budget=budget, constraints=gc, seed=0)
        # Novelty: drop seeds and near-seed
        novel = tanimoto_novelty_filter(gen, SEED_SMILES, threshold=0.7)
        if len(novel) < 50:
            novel = gen  # fallback
        log_event(run_id, "generated", count=len(novel))

        props_list: list[dict[str, float]] = []
        ids: list[str] = []
        smiles_list: list[str] = []
        for i, s in enumerate(novel):
            try:
                props = _property_map(s)
                props_list.append(props)
                ids.append(f"cand_{i:05d}")
                smiles_list.append(s)
            except Exception:
                continue
        log_event(run_id, "properties_computed", count=len(ids))
        # Determine dim names for writing out
        _, dim_names = _score_from_objectives(spec, props_list)

        # Write candidates.csv
        cand_rows: list[list[Any]] = []
        header = ["id", "smiles", *dim_names, "QED", "MW", "logP", "TPSA"]
        for idx, sid in enumerate(ids):
            s = smiles_list[idx]
            props = props_list[idx]
            dims = _lookup_dims(dim_names, props)
            cand_rows.append(
                [
                    sid,
                    s,
                    *dims,
                    props.get("QED", 0.0),
                    props.get("MW", 0.0),
                    props.get("logP", 0.0),
                    props.get("TPSA", 0.0),
                ]
            )
        _write_csv(candidates_csv, header, cand_rows)

    # Selection — allow resume from selected.csv
    selected_csv = out / "selected.csv"
    if resume and selected_csv.exists():
        selected_ids, selected_smiles = _read_selected_csv(selected_csv)
        log_event(run_id, "selection_loaded", selected=len(selected_ids))
    else:
        means, _ = _score_from_objectives(spec, props_list)
        goals = ["max"] * means.shape[1]
        k = min(100, max(20, means.shape[0] // 10))
        selected_ids = select_pareto(
            ids, means, sigmas=None, goals=goals, k=k, penalty_k=0.0, seed=0
        )
        selected_idx = {sid: i for i, sid in enumerate(ids) if sid in selected_ids}
        selected_smiles = [smiles_list[selected_idx[sid]] for sid in selected_ids]

        # Build selected.csv from candidates file
        import csv as _csv

        with candidates_csv.open(encoding="utf-8") as f:
            reader = _csv.DictReader(f)
            header = list(reader.fieldnames or [])
            rows = [row for row in reader]
        # Ensure canonical header ordering
        if header and header[0] != "id":
            base = ["id", "smiles"]
            rest = [h for h in header if h not in base]
            header = base + rest
        sel_rows = []
        for row in rows:
            if row.get("id") in selected_ids:
                sel_rows.append([row.get(h, "") for h in header])
        _write_csv(selected_csv, header, sel_rows)
        log_event(run_id, "selection", selected=len(selected_ids))

    # Geometry
    from ..geometry.conformers import batch_conformers_to_artifacts

    geom_dir = out / "geom"
    summaries_path = geom_dir / "summaries.jsonl"
    if not (resume and summaries_path.exists()):
        batch_conformers_to_artifacts(
            selected_smiles, out_dir=geom_dir, n_confs=10, seed=0, forcefield_preference="MMFF"
        )
        log_event(run_id, "geometry", dir=str(geom_dir))
    else:
        log_event(run_id, "geometry_skipped", reason="resume")

    # Synthesis feasibility + routes (non-actionable)
    synth_dir = out / "synth"
    synth_dir.mkdir(parents=True, exist_ok=True)
    feas_path = synth_dir / "feasibility.json"
    routes_path = synth_dir / "routes.json"
    routes_map: dict[str, Any] = {}
    if not (resume and feas_path.exists() and routes_path.exists()):
        feas_list: list[dict[str, Any]] = []
        for sid, s in zip(selected_ids, selected_smiles, strict=False):
            feas = assess_feasibility(s)
            feas_list.append({"id": sid, **asdict(feas)})
            routes_map[sid] = [asdict(r) for r in propose_routes(s)]
        feas_path.write_text(
            json.dumps(feas_list, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
        )
        routes_path.write_text(
            json.dumps(routes_map, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
        )
        log_event(run_id, "synthesis", dir=str(synth_dir))
    else:
        log_event(run_id, "synthesis_skipped", reason="resume")
        try:
            routes_map = json.loads(routes_path.read_text(encoding="utf-8"))
        except Exception:
            routes_map = {}

    # Report (geometry summary for now)
    report_md = out / "report.md"
    report_json = out / "report.json"
    if not (resume and report_md.exists() and report_json.exists()):
        build_geom_report(geom_dir, report_md, report_json)
        log_event(run_id, "report", md=str(report_md))
    else:
        log_event(run_id, "report_skipped", reason="resume")

    # Align with expected layout: duplicate routes under routes/ for convenience
    routes_dir = out / "routes"
    routes_dir.mkdir(parents=True, exist_ok=True)
    (routes_dir / "routes.json").write_text(
        json.dumps(routes_map, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    return run_id


def _lookup_dims(dims: list[str], props: dict[str, float]) -> list[float]:
    out: list[float] = []
    for d in dims:
        out.append(float(props.get(d, 0.0)))
    return out
