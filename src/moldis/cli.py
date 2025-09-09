from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__
from .geometry.conformers import (
    batch_conformers_to_artifacts,
    summarize_conformers,
)
from .predict.datasets import load_esol_tiny
from .predict.features import FEATURE_NAMES, featurize_smiles
from .predict.sklearn_models import (
    SKLearnRegressor,
    coverage_from_intervals,
    cv_residual_quantiles,
    evaluate_holdout,
    fit_gbr_quantile_models,
    fit_random_forest,
    intervals_from_resid_quantiles,
    predict_intervals_from_quantiles,
    save_model,
)
from .predict.sklearn_models import (
    fit_ridge as sk_fit_ridge,
)
from .predict.splits import scaffold_split
from .report.calibration import plot_interval_calibration
from .safety.request_gate import screen_user_request
from .scoring.objectives import Goal, apply_uncertainty_penalty, compute_bounds, normalize
from .scoring.pareto import hypervolume_2d
from .scoring.selection import select_pareto, select_weighted_sum
from .spec.models import Spec, validate_spec_payload
from .utils.io import dump_json, load_yaml_or_json
from .utils.run import log_event, new_run_id, run_dir, snapshot_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="moldis",
        description=(
            "Molecule discovery pipeline — spec-driven generation, prediction, and reporting."
        ),
    )

    parser.add_argument(
        "--version",
        action="version",
        version=f"moldis {__version__}",
        help="Show version and exit",
    )

    subparsers = parser.add_subparsers(dest="command")

    # Placeholder subcommands for future phases
    run_parser = subparsers.add_parser(
        "run",
        help="Validate spec, run safety gate, create run stub",
    )
    run_parser.add_argument(
        "--spec",
        type=str,
        required=True,
        help="Path to spec YAML/JSON",
    )

    report_parser = subparsers.add_parser("report", help="Build a simple report from artifacts")
    report_parser.add_argument("--run", type=str, required=False, help="Run ID (reserved)")
    report_parser.add_argument(
        "--geom-dir",
        type=str,
        required=False,
        help="Directory containing summaries.jsonl for conformers",
    )
    report_parser.add_argument(
        "--out-md",
        type=str,
        required=False,
        help="Output Markdown path",
        default="artifacts/report.md",
    )
    report_parser.add_argument(
        "--out-json",
        type=str,
        required=False,
        help="Output JSON summary path",
        default="artifacts/report.json",
    )

    schema_parser = subparsers.add_parser("schema", help="Print the Spec JSON Schema")
    schema_parser.add_argument(
        "--out",
        type=str,
        required=False,
        help="Optional path to write the JSON schema",
    )

    train_parser = subparsers.add_parser(
        "train", help="Train a predictor on a known task and save artifacts"
    )
    train_parser.add_argument(
        "--task",
        type=str,
        default="esol",
        choices=["esol"],
        help="Training task",
    )
    train_parser.add_argument(
        "--out",
        type=str,
        default="artifacts/models/esol_rf",
        help="Output directory for model artifacts",
    )
    train_parser.add_argument(
        "--model",
        type=str,
        default="rf",
        choices=["rf", "ridge", "gbr-quantile"],
        help="Which model to train",
    )
    train_parser.add_argument(
        "--val-fraction", type=float, default=0.3, help="Validation fraction (scaffold split)"
    )
    train_parser.add_argument(
        "--random-state", type=int, default=0, help="Random seed for splits and models"
    )
    train_parser.add_argument(
        "--n-estimators", type=int, default=200, help="Number of trees (for rf)"
    )

    score_parser = subparsers.add_parser(
        "score", help="Score and select candidates from a CSV via Pareto or weighted methods"
    )
    score_parser.add_argument("--input", type=str, required=True, help="Path to candidates CSV")
    score_parser.add_argument(
        "--out",
        type=str,
        default="artifacts/score",
        help="Output directory for selection artifacts",
    )
    score_parser.add_argument(
        "--objectives",
        type=str,
        required=True,
        help="Comma list like 'p1:max,p2:min' mapping columns to goals",
    )
    score_parser.add_argument(
        "--sigmas",
        type=str,
        default="",
        help=(
            "Optional comma list of sigma columns aligned to objectives; "
            "defaults to <name>_sigma if present"
        ),
    )
    score_parser.add_argument("--k", type=int, default=20, help="Number of selections to keep")
    score_parser.add_argument(
        "--method",
        type=str,
        default="pareto",
        choices=["pareto", "weighted"],
        help="Selection method",
    )
    score_parser.add_argument(
        "--weights",
        type=str,
        default="",
        help="Comma list of weights for weighted method (length must match objectives)",
    )
    score_parser.add_argument(
        "--penalty-k", type=float, default=1.0, help="Uncertainty penalty multiplier"
    )
    score_parser.add_argument("--seed", type=int, default=0, help="Random seed for selection")

    conf_parser = subparsers.add_parser(
        "conformers", help="Generate 3D conformers and energies; write SDF and summaries"
    )
    conf_parser.add_argument("--smiles", type=str, help="Single SMILES input", required=False)
    conf_parser.add_argument(
        "--input",
        type=str,
        help="Optional CSV with 'smiles' column (and optional 'id')",
        required=False,
    )
    conf_parser.add_argument(
        "--out",
        type=str,
        default="artifacts/geom",
        help="Output directory (SDF and summaries.jsonl)",
    )
    conf_parser.add_argument("--n-confs", type=int, default=20)
    conf_parser.add_argument("--seed", type=int, default=0)
    conf_parser.add_argument("--ff", type=str, default="MMFF", choices=["MMFF", "UFF"])
    conf_parser.add_argument(
        "--xtb",
        action="store_true",
        help="Compute xTB single-point properties if xtb is installed",
    )

    return parser


def _cmd_run(spec_path: str) -> int:
    payload = load_yaml_or_json(spec_path)
    spec: Spec = validate_spec_payload(payload)
    allowed, reasons = screen_user_request(spec)
    if not allowed:
        sys.stderr.write("Request blocked by safety gate.\n")
        for r in reasons:
            sys.stderr.write(f"- {r}\n")
        return 2

    run_id = new_run_id()
    run_dir(run_id)  # ensure dirs
    snapshot_config(run_id, payload)
    log_event(run_id, "run_start", request_id=spec.request_id, use_case=spec.use_case)
    log_event(run_id, "safety_check", status="pass")
    log_event(run_id, "spec_validated", objectives=len(spec.objectives))
    # Later phases will do actual work; for now, print run_id for the user.
    sys.stdout.write(json.dumps({"run_id": run_id}) + "\n")
    return 0


def _cmd_schema(out: str | None) -> int:
    schema = Spec.json_schema()
    data = dump_json(schema)
    if out:
        Path(out).write_text(data, encoding="utf-8")
        sys.stdout.write(f"Wrote schema to {out}\n")
    else:
        sys.stdout.write(data + "\n")
    return 0


def _cmd_train(
    task: str,
    out: str,
    model: str,
    val_fraction: float,
    random_state: int,
    n_estimators: int,
) -> int:
    # Currently supports only ESOL
    if task != "esol":
        sys.stderr.write("Only 'esol' task is supported right now.\n")
        return 2
    # Load data and features
    smiles, y_list = load_esol_tiny()
    import numpy as np

    X = featurize_smiles(smiles).X
    y = np.asarray(y_list, dtype=float)
    idx_tr, idx_va = scaffold_split(smiles, val_fraction=val_fraction, random_state=random_state)

    # Fit base estimator
    if model == "rf":
        est: SKLearnRegressor = fit_random_forest(
            X[idx_tr],
            y[idx_tr],
            FEATURE_NAMES,
            random_state=random_state,
            n_estimators=n_estimators,
        )
        y_hat_val = est.estimator.predict(X[idx_va])
        # Conformal-style residual quantiles from CV on full data
        levels = [0.8, 0.9]
        q_by_level = cv_residual_quantiles(
            X[idx_tr], y[idx_tr], levels=levels, n_splits=5, random_state=random_state
        )
        coverages: dict[str, float] = {}
        for lv in levels:
            lo, hi = intervals_from_resid_quantiles(y_hat_val, q_by_level, lv)
            coverages[str(lv)] = coverage_from_intervals(y[idx_va], lo, hi)
    elif model == "ridge":
        est = sk_fit_ridge(X[idx_tr], y[idx_tr], FEATURE_NAMES, random_state=random_state)
        y_hat_val = est.estimator.predict(X[idx_va])
        levels = [0.8, 0.9]
        q_by_level = cv_residual_quantiles(
            X[idx_tr], y[idx_tr], levels=levels, n_splits=5, random_state=random_state
        )
        coverages = {}
        for lv in levels:
            lo, hi = intervals_from_resid_quantiles(y_hat_val, q_by_level, lv)
            coverages[str(lv)] = coverage_from_intervals(y[idx_va], lo, hi)
    elif model == "gbr-quantile":
        lower_est, upper_est = fit_gbr_quantile_models(
            X[idx_tr],
            y[idx_tr],
            FEATURE_NAMES,
            random_state=random_state,
            lower_alpha=0.1,
            upper_alpha=0.9,
        )
        # For point prediction we can use median ~ 0.5 quantile; approximate via midpoint of bounds
        lo, hi = predict_intervals_from_quantiles(lower_est, upper_est, X[idx_va])
        y_hat_val = 0.5 * (lo + hi)
        levels = [0.8, 0.9]
        # Estimate coverage at 0.8 and 0.9 using the 10/90 quantile bounds as a coarse proxy
        coverages = {
            "0.8": coverage_from_intervals(y[idx_va], lo, hi),
            "0.9": coverage_from_intervals(y[idx_va], lo, hi),
        }
        # Save lower/upper models; use lower_est metadata for save, but write extra fields
        est = lower_est
    else:
        sys.stderr.write(f"Unknown model: {model}\n")
        return 2

    # Metrics
    metrics = evaluate_holdout(est, X, y, idx_tr, idx_va)
    # Save artifacts
    out_dir = Path(out)
    meta = {
        "task": task,
        "model": model,
        "metrics": metrics,
        "val_fraction": val_fraction,
        "random_state": random_state,
        "coverage": coverages,
        "levels": levels,
    }
    save_model(est, out_dir, meta=meta)
    # Plot calibration
    plot_interval_calibration(
        levels, [coverages[str(lv)] for lv in levels], out_dir / "calibration.png"
    )
    sys.stdout.write(f"Saved artifacts to {out_dir}\n")
    return 0


def _parse_objectives(spec: str) -> tuple[list[str], list[Goal]]:
    parts = [p.strip() for p in spec.split(",") if p.strip()]
    names: list[str] = []
    goals: list[Goal] = []
    for p in parts:
        if ":" not in p:
            raise ValueError("Objective must be 'name:goal'")
        n, g = p.split(":", 1)
        n = n.strip()
        g = g.strip().lower()
        if g not in {"max", "min"}:
            raise ValueError("Goal must be 'max' or 'min'")
        names.append(n)
        goals.append(g)  # type: ignore[arg-type]
    return names, goals


def _read_csv(path: str) -> tuple[list[str], list[dict[str, str]]]:
    import csv

    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        header = list(reader.fieldnames or [])
        rows = [row for row in reader]
    return header, rows


def _cmd_score(
    input_path: str,
    out: str,
    objectives: str,
    sigmas: str,
    k: int,
    method: str,
    weights: str,
    penalty_k: float,
    seed: int,
) -> int:
    import json

    import numpy as np

    obj_names, goals = _parse_objectives(objectives)
    header, rows = _read_csv(input_path)
    id_key = "id" if "id" in header else ("smiles" if "smiles" in header else None)
    ids: list[str] = []
    means_list: list[list[float]] = []
    sig_list: list[list[float]] | None = None

    # Sigma columns mapping
    sigma_cols: list[str] | None = None
    if sigmas.strip():
        sigma_cols = [s.strip() for s in sigmas.split(",") if s.strip()]
        if len(sigma_cols) != len(obj_names):
            raise ValueError("Length of --sigmas must match number of objectives")

    for idx, row in enumerate(rows):
        rid = row[id_key] if id_key else f"row{idx}"
        ids.append(rid)
        vals: list[float] = []
        svals: list[float] = []
        for j, name in enumerate(obj_names):
            try:
                vals.append(float(row[name]))
            except Exception:
                vals.append(float("nan"))
            if sigma_cols is not None:
                sc = sigma_cols[j]
                svals.append(float(row.get(sc, "nan")))
            else:
                sc_auto = f"{name}_sigma"
                if sc_auto in row:
                    svals.append(float(row.get(sc_auto, "nan")))
        means_list.append(vals)
        if svals:
            if sig_list is None:
                sig_list = []
            sig_list.append(svals)

    means = np.asarray(means_list, dtype=float)
    sigmas_arr = None
    if (
        isinstance(sig_list, list)
        and sig_list
        and len(sig_list) == len(means_list)
        and len(sig_list[0]) == len(obj_names)
    ):
        sigmas_arr = np.asarray(sig_list, dtype=float)

    # Apply uncertainty penalty and normalize
    penalized = apply_uncertainty_penalty(means, sigmas_arr, goals, k=penalty_k)
    lo, hi = compute_bounds(penalized)
    norm = normalize(penalized, goals, lo, hi)

    # Selection
    if method == "pareto":
        selected_ids = select_pareto(
            ids, means, sigmas_arr, goals, k=k, penalty_k=penalty_k, seed=seed
        )
    else:
        if weights.strip():
            ws = [float(x) for x in weights.split(",")]
            if len(ws) != len(obj_names):
                raise ValueError("Length of --weights must match number of objectives")
        else:
            ws = [1.0 / len(obj_names)] * len(obj_names)
        ordered = select_weighted_sum(
            ids, means, sigmas_arr, goals, weights=ws, penalty_k=penalty_k
        )
        selected_ids = ordered[:k]

    # Hypervolume (2D only) on normalized values (maximize), transform to min for HV
    hv = None
    if norm.shape[1] == 2:
        pts_min = 1.0 - norm  # all in [0,1]
        hv = float(hypervolume_2d(pts_min, ref=(1.0, 1.0)))

    # Write artifacts
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    # selected.csv
    import csv

    with (out_dir / "selected.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", *obj_names])
        id_set = set(selected_ids)
        for i, rid in enumerate(ids):
            if rid in id_set:
                row_vals = [rid] + [means[i, j] for j in range(len(obj_names))]
                writer.writerow(row_vals)
    # score.json
    summary = {
        "objectives": [{"name": n, "goal": g} for n, g in zip(obj_names, goals, strict=False)],
        "method": method,
        "k": k,
        "penalty_k": penalty_k,
        "seed": seed,
        "hypervolume_2d": hv,
        "bounds_lo": lo.tolist(),
        "bounds_hi": hi.tolist(),
        "selected_ids": selected_ids,
        "input": input_path,
    }
    (out_dir / "score.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    sys.stdout.write(f"Saved selection to {out_dir}\n")
    return 0


def _cmd_conformers(
    smiles: str | None,
    input_path: str | None,
    out: str,
    n_confs: int,
    seed: int,
    ff: str,
    xtb: bool,
) -> int:
    import csv
    import json
    from dataclasses import asdict
    from pathlib import Path as _P

    if not smiles and not input_path:
        sys.stderr.write("Provide either --smiles or --input CSV with 'smiles' column.\n")
        return 2
    out_dir = _P(out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if smiles:
        summ = summarize_conformers(
            smiles,
            n_confs=n_confs,
            seed=seed,
            forcefield_preference=ff,
            do_xtb=xtb,
        )
        (out_dir / "summary.json").write_text(
            json.dumps(asdict(summ), ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
        )
        # Also write SDF with all conformers
        from .geometry.conformers import generate_conformers, write_sdf

        mol, energies, _ = generate_conformers(
            smiles, n_confs=n_confs, seed=seed, forcefield_preference=ff
        )
        write_sdf(mol, out_dir / "molecule.sdf", energies)
        sys.stdout.write(f"Wrote conformers to {out_dir}\n")
        return 0

    # CSV path
    if input_path is None:
        sys.stderr.write("--input required when --smiles is not provided.\n")
        return 2
    with open(input_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = [r for r in reader]
    if not rows or "smiles" not in rows[0]:
        sys.stderr.write("Input CSV must contain a 'smiles' column.\n")
        return 2
    sdf_path = batch_conformers_to_artifacts(
        [r["smiles"] for r in rows],
        out_dir=out_dir,
        n_confs=n_confs,
        seed=seed,
        forcefield_preference=ff,
        do_xtb=xtb,
    )
    sys.stdout.write(f"Wrote SDF: {sdf_path}\n")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "run":
        return _cmd_run(args.spec)
    if args.command == "schema":
        return _cmd_schema(args.out)
    if args.command == "report":
        # Currently only supports geom-dir; run-based reporting to be added in P9
        from .report.builder import build_geom_report

        if args.geom_dir:
            build_geom_report(args.geom_dir, args.out_md, args.out_json)
            sys.stdout.write(f"Wrote report: {args.out_md}\n")
            return 0
        sys.stderr.write("Nothing to report; provide --geom-dir.\n")
        return 2
    if args.command == "train":
        return _cmd_train(
            task=args.task,
            out=args.out,
            model=args.model,
            val_fraction=args.val_fraction,
            random_state=args.random_state,
            n_estimators=args.n_estimators,
        )
    if args.command == "score":
        return _cmd_score(
            input_path=args.input,
            out=args.out,
            objectives=args.objectives,
            sigmas=args.sigmas,
            k=args.k,
            method=args.method,
            weights=args.weights,
            penalty_k=args.penalty_k,
            seed=args.seed,
        )
    if args.command == "conformers":
        return _cmd_conformers(
            smiles=args.smiles,
            input_path=args.input,
            out=args.out,
            n_confs=args.n_confs,
            seed=args.seed,
            ff=args.ff,
            xtb=args.xtb,
        )
    # Default: print help
    parser.print_help()
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
