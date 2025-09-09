# Molecule Discovery

Specification‑driven generation of novel molecules with property prediction (including uncertainty), high‑level non‑actionable synthesis ideas, and research‑grade reporting.

## Quickstart (Dev)

```bash
# Requires Python 3.11+
make setup  # optionally: make setup PY=python3.11
make test
```

CLI skeleton:

```bash
moldis --help
```

Safety: This project blocks harmful/illegal designs and never emits actionable synthesis instructions. All synthesis suggestions are high‑level only.

## Optional: RDKit for P2

RDKit wheels on pip (rdkit-pypi==2022.09.5) are compiled against NumPy 1.x. Use the helper target to install ABI-compatible wheels:

```bash
make chem-setup          # installs numpy<2 and rdkit-pypi
make chem-test           # runs only RDKit-dependent tests
```

If you manage the environment manually:

```bash
.venv/bin/pip install --no-cache-dir "numpy<2" "rdkit-pypi==2022.9.5"
```
