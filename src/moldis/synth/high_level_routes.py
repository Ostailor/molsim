from __future__ import annotations

from dataclasses import dataclass

from ..chem import rdkit_utils as RU


@dataclass
class RouteIdea:
    transform: str
    rationale: str
    disclaimer: str = "High-level concept only. No conditions or reagents specified."


NON_ACTIONABLE_DISCLAIMER = (
    "High-level retrosynthesis idea only; no operational parameters provided."
)


def _require_rdkit() -> None:
    if not RU.RDKIT_AVAILABLE:
        raise RuntimeError("RDKit not available. Install with extras: '.[chem]'")


def propose_routes(smiles: str) -> list[RouteIdea]:
    """Suggest non-actionable, coarse retrosynthetic ideas based on functional groups."""
    _require_rdkit()
    from rdkit import Chem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return [
            RouteIdea(
                "structure_invalid",
                "Input structure could not be parsed",
                NON_ACTIONABLE_DISCLAIMER,
            )
        ]

    ideas: list[RouteIdea] = []

    def add(transform: str, rationale: str) -> None:
        ideas.append(RouteIdea(transform, rationale, NON_ACTIONABLE_DISCLAIMER))

    # Amide
    if mol.HasSubstructMatch(Chem.MolFromSmarts("C(=O)N")):
        add(
            "amide_coupling",
            "Assemble from a carboxylic acid (or activated equivalent) and an amine precursor.",
        )
    # Ester
    if mol.HasSubstructMatch(Chem.MolFromSmarts("C(=O)O")):
        add(
            "ester_formation",
            "Assembled from carboxylic acid and alcohol precursors via generic ester formation.",
        )
    # Urea
    if mol.HasSubstructMatch(Chem.MolFromSmarts("NC(=O)N")):
        add(
            "urea_assembly",
            "Combine an isocyanate-equivalent with an amine precursor to form a urea linkage.",
        )
    # Ether
    if mol.HasSubstructMatch(Chem.MolFromSmarts("[CX4]-O-[CX4]")):
        add(
            "ether_formation",
            "Assemble from an alcohol and an alkyl electrophile (e.g., halide equivalent).",
        )
    # Biaryl (cross-coupling idea)
    if mol.HasSubstructMatch(Chem.MolFromSmarts("a-a")):
        add(
            "aryl_aryl_cross_coupling",
            "Join aryl fragments via generic aryl–aryl cross-coupling.",
        )

    if not ideas:
        ideas.append(
            RouteIdea(
                "generic_divergent_synthesis",
                "Consider disconnections at key functional groups (carbonyl, heteroatom links).",
                NON_ACTIONABLE_DISCLAIMER,
            )
        )

    # Enforce non-actionable content (no parameters or reagent lists beyond precursor classes)
    banned_terms = ["°C", "K", "solvent", "heat", "stir", "hours", "minutes", "dropwise"]
    for idea in ideas:
        for term in banned_terms:
            if term.lower() in idea.rationale.lower():
                idea.rationale = idea.rationale.replace(term, "[redacted]")
    return ideas
