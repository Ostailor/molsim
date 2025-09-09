from __future__ import annotations

import json
import re
from pathlib import Path

BANNED_PATTERNS = [
    r"\b\d+\s?(°C|K)\b",
    r"\b(hours?|minutes?|seconds?)\b",
    r"\b(stir|reflux|heat|cool)\b",
    r"\b(solvent|add dropwise|vacuum|inert atmosphere)\b",
]


def find_banned_in_text(text: str) -> list[str]:
    hits: list[str] = []
    for pat in BANNED_PATTERNS:
        if re.search(pat, text, flags=re.IGNORECASE):
            hits.append(pat)
    return hits


def lint_routes_file(path: str | Path) -> list[dict[str, str]]:
    p = Path(path)
    data = json.loads(p.read_text(encoding="utf-8"))
    issues: list[dict[str, str]] = []
    for idx, item in enumerate(data):
        parts = [
            str(item.get("transform", "")),
            str(item.get("rationale", "")),
            str(item.get("disclaimer", "")),
        ]
        text = " \n".join(parts)
        hits = find_banned_in_text(text)
        if hits:
            issues.append({"index": str(idx), "hits": ",".join(hits)})
    return issues
