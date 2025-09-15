# Molecule Discovery (moldis)

Specification‑driven discovery of novel molecules with: candidate generation, fast descriptors and ML predictions (with uncertainty where available), high‑level and strictly non‑actionable synthesis ideas, and reproducible, research‑grade artifacts and reports.

Safety first: This project must never produce harmful or illegal content and must not emit actionable synthesis instructions. All synthesis outputs are coarse‑grained concepts only. See “Safety & Compliance”.

---

## What’s Inside

- Orchestrated pipeline: spec → generate → validate → predict → score/select → geometry → high‑level synth → report
- CLI tooling: `moldis run`, `report`, `eval/score`, `conformers`, `synth`, `safety-lint`, `schema`, `train`
- Deterministic runs with run IDs, config snapshotting, and caching
- Artifacts: CSVs, JSON, Markdown report, SDF and summaries for structures

---

## Requirements

- Python 3.11 (or 3.12)
- macOS or Linux (tested); Windows likely works with WSL
- For chemistry features (RDKit): NumPy < 2 is required for current wheels

---

## Installation

Option A — Dev environment with extras

```bash
# Create venv, install dev tools
make setup

# Add chemistry + ML extras
.venv/bin/pip install -e ".[chem,ml]"

# Or use helper that pins numpy<2 and RDKit
make chem-setup
```

Option B — Minimal install

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e .[chem,ml]   # or just . if you won’t use RDKit/ML
```

Without installing the package, you can also run with `PYTHONPATH=src`:

```bash
PYTHONPATH=src python -m moldis --help
```

Troubleshooting RDKit/NumPy ABI

- If you see “_ARRAY_API not found” or similar, ensure NumPy < 2:
  `pip install --no-cache-dir "numpy<2" "rdkit-pypi==2022.9.5"`

---

## Quickstart

1) Run the end‑to‑end pipeline on the example spec

```bash
# Creates a new run under artifacts/run-<timestamp>-<id>
make run SPEC=configs/spec.example.yaml
# or
moldis run --spec configs/spec.example.yaml
```

2) Resume a run to reuse intermediates and skip completed stages

```bash
moldis run --spec configs/spec.example.yaml --resume <run_id>
```

3) Build a simple geometry report from an existing geometry dir (subset of P9)

```bash
moldis report --geom-dir artifacts/<run_id>/geom \
  --out-md artifacts/<run_id>/report.md \
  --out-json artifacts/<run_id>/report.json
```

Artifacts produced per run (under `artifacts/<run_id>/`):

- `config.spec.json`: spec snapshot
- `logs/run.jsonl`: structured run log
- `candidates.csv`, `selected.csv`: IDs, SMILES, properties, and selected subset
- `geom/summaries.jsonl` (+ SDF files if configured): conformer summaries
- `synth/feasibility.json`, `synth/routes.json`: feasibility metrics + non‑actionable ideas
- `routes/routes.json`: convenience copy of routes
- `report.md`, `report.json`: report outputs

---

## CLI Reference

See full help: `moldis --help`

- `moldis run --spec <path> [--resume <run_id>]`
  - Validate spec, pass safety gate, execute the pipeline, write artifacts.
  - `--resume` reuses existing artifacts to skip completed stages.

- `moldis report --geom-dir <dir> --out-md <path> --out-json <path>`
  - Generate a simple geometry summary report from conformer outputs.

- `moldis eval --input candidates.csv --objectives 'QED:max,MW:min' --k 20` (alias of `score`)
  - Evaluate/Select from a CSV of candidates using Pareto or weighted sum methods.
  - Common flags: `--method pareto|weighted`, `--weights w1,w2,...`, `--penalty-k 1.0`, `--sigmas col1,col2,...`.

- `moldis conformers --smiles 'CCO' --out artifacts/geom` (or `--input candidates.csv`)
  - Generate conformers and energy summaries; optional `--xtb` if xTB is available.

- `moldis synth --smiles 'O=C(Nc1ccccc1)C' --out artifacts/synth`
  - Compute synthesizability metrics and output non‑actionable, high‑level route ideas.

- `moldis safety-lint --routes-file artifacts/<run_id>/synth/routes.json`
  - Screen a routes JSON for banned/actionable terms (lint only).

- `moldis schema --out artifacts/spec.schema.json`
  - Export the JSON Schema for the Spec interface.

- `moldis train --task esol --out artifacts/models/esol_rf --model rf`
  - Train a simple baseline regressor on a toy dataset; saves metrics and a plot.

---

## The Spec (configs/spec.example.yaml)

```yaml
request_id: EXAMPLE-0001
use_case: pharma
objectives:
  - name: logP
    target: 2.0
    tolerance: 0.5
    weight: 0.5
  - name: TPSA
    target: 75
    tolerance: 15
    weight: 0.5
constraints:
  allowed_elements: [C, H, N, O, F, S, Cl]
  max_heavy_atoms: 40
  max_rings: 4
  novelty_threshold: medium
  synthesizability_min: 0.6
  toxicology_profile: low_risk_only
compute:
  budget: standard
  max_candidates: 1000
  physics_tier: none
ethics:
  safety_mode: strict
  prohibited_motifs: []
notes: "Benign example for testing."
```

Notes
- Objectives can be target+tolerance (scored to [0,1]) or raw (max/min goals in selection).
- Constraints limit chemical space and synthesizability.
- Compute controls candidate budget and optional physics tier.
- Ethics enforces strict safety posture; unsafe requests get blocked.

---

## Reproducibility

- Deterministic seeds throughout generation/selection; run IDs include UTC timestamp.
- Config snapshot saved at `artifacts/<run_id>/config.spec.json`.
- Repeat runs with the same spec/seed produce identical checksums for core artifacts (excluding timestamps).
- `--resume <run_id>` reuses intermediates and skips completed stages.

---

## Make Targets

```bash
make setup     # venv + dev deps
make test      # run unit/integration tests
make lint      # static checks
make fix       # auto‑format + quick fixes
make type      # mypy (pydantic plugin enabled)
make chem-setup  # install numpy<2 and RDKit wheels
make run SPEC=configs/spec.example.yaml
make report GEOM=artifacts/<run_id>/geom
```

---

## Programmatic Usage

```python
from moldis.spec.models import validate_spec_payload
from moldis.orchestrator.pipeline import run_pipeline
from moldis.utils.io import load_yaml_or_json

spec = validate_spec_payload(load_yaml_or_json("configs/spec.example.yaml"))
run_id = run_pipeline(spec)  # or run_pipeline(spec, resume="run-...")
print("Artifacts in", f"artifacts/{run_id}")
```

---

## Safety & Compliance

- High‑level synthesis only: no quantities, temperatures, times, or operational steps.
- Explicitly refuse or block harmful/illegal substances and requests.
- Safety lint checks prevent actionable content in route suggestions.
- Use case posture and blocks are logged; see `AGENTS.md` for details.

This software is for conceptual design and research only. Always consult qualified experts and primary literature before any laboratory work.

---

## Troubleshooting

- RDKit + NumPy errors like “_ARRAY_API not found”
  - Ensure `numpy<2` and install `rdkit-pypi==2022.9.5`.
  - `make chem-setup` applies pinned constraints from `constraints/chem.txt`.

- `ModuleNotFoundError: moldis`
  - Make sure the package is installed (`pip install -e .`) or run with `PYTHONPATH=src`.

- macOS security or code‑sign issues with optional external tools (e.g., xTB)
  - These features are optional; keep physics tier off or follow tool‑specific install guides.

---

## Contributing

See `CONTRIBUTING.md` and `AGENTS.md` for guidelines and safety policies. Prefer small, focused PRs with tests. Do not introduce actionable synthesis content.

---

## License

UNLICENSED — for internal evaluation and development only.

---

## References

- RDKit — Open‑source cheminformatics
- OpenFF — Force fields and toolkits
- scikit‑learn — ML baselines
- MOSES/GuacaMol — Generation benchmarks
- SCScore/SA Score — Synthesizability metrics
