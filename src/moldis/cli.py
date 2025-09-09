from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__
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

    report_parser = subparsers.add_parser(
        "report", help="Rebuild a report from artifacts (P9). Placeholder."
    )
    report_parser.add_argument("--run", type=str, required=False, help="Run ID")

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


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "run":
        return _cmd_run(args.spec)
    if args.command == "schema":
        return _cmd_schema(args.out)
    if args.command == "train":
        return _cmd_train(
            task=args.task,
            out=args.out,
            model=args.model,
            val_fraction=args.val_fraction,
            random_state=args.random_state,
            n_estimators=args.n_estimators,
        )
    # Default: print help
    parser.print_help()
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
