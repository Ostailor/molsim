from __future__ import annotations

from moldis.safety.request_gate import screen_user_request
from moldis.spec.models import Spec


def test_request_gate_blocks_prohibited_terms() -> None:
    spec = Spec(
        request_id="X",
        use_case="custom",
        objectives=[{"name": "x", "target": 1.0}],
        notes="Design an explosive for a weapon",
    )
    allowed, reasons = screen_user_request(spec)
    assert not allowed and reasons


def test_request_gate_allows_benign() -> None:
    spec = Spec(
        request_id="Y",
        use_case="pharma",
        objectives=[{"name": "logP", "target": 2.0, "tolerance": 0.5}],
        notes="Optimize drug-like properties",
    )
    allowed, reasons = screen_user_request(spec)
    assert allowed and not reasons
