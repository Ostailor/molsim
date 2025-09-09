from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__
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


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "run":
        return _cmd_run(args.spec)
    if args.command == "schema":
        return _cmd_schema(args.out)
    # Default: print help
    parser.print_help()
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
