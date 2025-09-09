from __future__ import annotations

import argparse
import sys

from . import __version__


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
        "run", help="Run the pipeline (P1+). Placeholder for future implementation."
    )
    run_parser.add_argument(
        "--spec",
        type=str,
        required=False,
        help="Path to spec YAML/JSON (implemented in later phases)",
    )

    report_parser = subparsers.add_parser(
        "report", help="Rebuild a report from artifacts (P9). Placeholder."
    )
    report_parser.add_argument("--run", type=str, required=False, help="Run ID")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    _ = parser.parse_args(argv)
    # No-op for P0; subcommands are placeholders.
    parser.print_help()
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
