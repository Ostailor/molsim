from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, ValidationError, field_validator

UseCase = Literal["pharma", "materials", "agro", "custom"]
Budget = Literal["fast", "standard", "thorough"]
PhysicsTier = Literal["none", "semi_empirical", "dft_screen"]
Novelty = Literal["strict", "medium", "relaxed"]
ToxProfile = Literal["low_risk_only", "open"]
SafetyMode = Literal["strict", "standard", "open"]


class Objective(BaseModel):
    name: str = Field(..., description="Property name (e.g., logP, band_gap, IC50_targetX)")
    target: float | str = Field(..., description="Target value (numeric or categorical)")
    tolerance: float | None = Field(
        default=None, description="Acceptable deviation for target (numeric targets)"
    )
    weight: float = Field(1.0, ge=0.0, description="Relative importance in multi-objective scoring")

    @field_validator("tolerance")
    @classmethod
    def non_negative_tolerance(cls, v: float | None) -> float | None:
        if v is not None and v < 0:
            raise ValueError("tolerance must be non-negative")
        return v


class Constraints(BaseModel):
    allowed_elements: list[str] = Field(
        default_factory=lambda: ["C", "H", "N", "O", "F", "S", "P", "Cl"],
        description="Permitted chemical elements",
    )
    max_heavy_atoms: int = Field(40, ge=1, description="Maximum heavy atom count")
    max_rings: int = Field(4, ge=0, description="Maximum ring count")
    novelty_threshold: Novelty = Field("medium", description="Novelty strictness vs. corpora")
    synthesizability_min: float = Field(
        0.6, ge=0.0, le=1.0, description="Minimum normalized synthesizability score"
    )
    toxicology_profile: ToxProfile = Field(
        "low_risk_only", description="Toxicology posture for candidate selection"
    )


class Compute(BaseModel):
    budget: Budget = Field("standard", description="Compute budget tier")
    max_candidates: int = Field(5000, ge=1, description="Generation budget")
    physics_tier: PhysicsTier = Field("none", description="Optional physics-based evaluation tier")


class Ethics(BaseModel):
    safety_mode: SafetyMode = Field(
        "strict",
        description="Safety strictness. 'strict' blocks ambiguous or risky requests.",
    )
    prohibited_motifs: list[str] = Field(
        default_factory=list, description="Domain-specific motifs to explicitly avoid"
    )


class Spec(BaseModel):
    request_id: str = Field(..., description="Unique request identifier")
    use_case: UseCase = Field(..., description="Domain context for the design task")
    objectives: list[Objective] = Field(..., min_length=1)
    constraints: Constraints = Field(default=Constraints())
    compute: Compute = Field(default=Compute())
    ethics: Ethics = Field(default=Ethics())
    notes: str | None = Field(
        default=None, description="Free-text context. Must be non-harmful and non-actionable."
    )

    @classmethod
    def json_schema(cls) -> dict:
        # pydantic v2: BaseModel.model_json_schema()
        return cls.model_json_schema()


def validate_spec_payload(payload: dict) -> Spec:
    try:
        return Spec.model_validate(payload)
    except ValidationError as e:
        # Raise a ValueError with concise message suitable for CLI output
        raise ValueError(e) from e
