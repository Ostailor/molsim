# Vendored helpers

This directory is for optional, single-file third‑party helpers that are convenient to vendor.

SA Score (Ertl)
- What: RDKit's synthesizability score (1..10). We map it to 0..1 where higher = easier.
- Source: https://raw.githubusercontent.com/rdkit/rdkit/master/Contrib/SA_Score/sascorer.py
- How to enable: download and place the file as:
  - `src/moldis/third_party/sascorer.py`
- Usage in code: the feasibility module auto‑detects it and prefers it when present; otherwise it falls back to a proxy score.

Notes
- Keep these files unmodified where possible; if changes are needed, document them here.
- We do not publish these files to PyPI automatically; they are optional.
