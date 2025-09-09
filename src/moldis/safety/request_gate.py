from __future__ import annotations

from collections.abc import Iterable

from ..spec.models import Spec

# Minimal v0 keyword list. Conservative and intentionally non-exhaustive.
PROHIBITED_KEYWORDS: list[str] = [
    # general harm/weaponization
    "weapon",
    "weaponize",
    "chemical weapon",
    "toxic agent",
    "nerve agent",
    "sarin",
    "vx",
    "mustard gas",
    # explosives (common terms)
    "explosive",
    "tatp",
    "rdx",
    # illicit drugs (examples)
    "fentanyl",
    "heroin",
    "methamphetamine",
]


def _text_fields(spec: Spec) -> Iterable[str]:
    yield spec.use_case
    if spec.notes:
        yield spec.notes
    for obj in spec.objectives:
        yield obj.name
        if isinstance(obj.target, str):
            yield obj.target


def screen_user_request(spec: Spec) -> tuple[bool, list[str]]:
    """Return (allowed, reasons). Conservative: blocks on match when safety_mode is 'strict'."""
    text = " ".join(s.lower() for s in _text_fields(spec))
    hits = [kw for kw in PROHIBITED_KEYWORDS if kw in text]

    reasons: list[str] = []
    allowed = True

    if hits and spec.ethics.safety_mode in {"strict", "standard"}:
        allowed = False
        reasons.append(
            "request matches prohibited keywords; cannot assist with harmful/illegal designs"
        )
        reasons.append(f"matched: {sorted(set(hits))}")

    return allowed, reasons
