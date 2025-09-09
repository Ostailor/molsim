from __future__ import annotations

from moldis.spec.models import Spec


def test_spec_json_schema_includes_core_fields() -> None:
    schema = Spec.json_schema()
    props = schema.get("properties", {})
    assert "use_case" in props
    assert "objectives" in props
    assert "constraints" in props
