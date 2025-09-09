from __future__ import annotations

import datetime as _dt
import uuid as _uuid
from pathlib import Path
from typing import Any

from .io import dump_json, write_text


def new_run_id(prefix: str = "run") -> str:
    ts = _dt.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    short = str(_uuid.uuid4())[:8]
    return f"{prefix}-{ts}-{short}"


def run_dir(run_id: str) -> Path:
    p = Path("artifacts") / run_id
    p.mkdir(parents=True, exist_ok=True)
    (p / "logs").mkdir(parents=True, exist_ok=True)
    return p


def log_event(run_id: str, event: str, **fields: Any) -> None:
    payload: dict[str, Any] = {
        "ts": _dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "event": event,
        **fields,
    }
    log_path = run_dir(run_id) / "logs" / "run.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(dump_json(payload) + "\n")


def snapshot_config(run_id: str, spec_payload: dict[str, Any]) -> None:
    # Save both YAML and JSON for convenience
    json_path = run_dir(run_id) / "config.spec.json"
    write_text(json_path, dump_json(spec_payload))
