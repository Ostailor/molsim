from __future__ import annotations

# A small, diverse set of aromatic/heterocyclic seeds to avoid trivial molecules
# Source: common scaffolds and benign examples

SEED_SMILES: list[str] = [
    # Aromatics and simple biaryls
    "c1ccccc1",  # benzene
    "c1ccc(cc1)C",  # toluene
    "c1ccc(cc1)O",  # phenol
    "c1ccc(cc1)N",  # aniline
    "c1cc(ccc1)Cl",
    "c1ccc2ccccc2c1",  # naphthalene
    # Heterocycles
    "c1ccncc1",  # pyridine
    "c1ccncc1O",
    "c1ncccc1N",
    "c1ncc[nH]1",  # imidazole-like
    "c1ncccc1Cl",
    "c1ccoc1",  # furan-like
    "c1ccnc2ccccc12",  # quinoline
    "c1ccc2ncccc2c1",  # isoquinoline
    # Common pharmacophores
    "CC(=O)Oc1ccccc1C(=O)O",  # aspirin
    "CN1C=NC2=C1C(=O)N(C(=O)N2)C",  # caffeine
    "CC(C)CC1=CC(=O)Oc2ccccc21",  # ibuprofen-like core
    "CC(=O)NC1=CC=CC=C1",  # acetanilide
    "CC(=O)NCCC1=CC=CC=C1",  # acetylated aniline linker
    # Simple linkers
    "O=C(N)N",  # urea
    "O=C(N)C",  # acetamide
    "CCOC(=O)C",  # ethyl acetate
    "CCN(CC)CC",  # triethylamine-like skeleton (for fragments)
    # Benign heteroaromatics
    "c1ncc[nH]1C",
    "c1nc[nH]c1O",
    "c1ncccc1S",
]
