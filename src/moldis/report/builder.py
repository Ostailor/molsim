from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _read_summaries(path: str | Path) -> list[dict[str, Any]]:
    p = Path(path)
    if p.is_dir():
        p = p / "summaries.jsonl"
    if not p.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        try:
            rows.append(json.loads(line))
        except Exception:
            continue
    return rows


def build_geom_report(
    geom_dir: str | Path, out_md: str | Path, out_json: str | Path | None = None
) -> Path:
    rows = _read_summaries(geom_dir)
    total = len(rows)
    succ = sum(1 for r in rows if r.get("success"))

    def _fnum(x: Any) -> float:
        try:
            return float(x)  # type: ignore[arg-type]
        except Exception:
            return 0.0

    min_energies = [_fnum(r.get("min_energy")) for r in rows if r.get("min_energy") is not None]
    dipoles = [
        _fnum(r.get("dipole_magnitude")) for r in rows if r.get("dipole_magnitude") is not None
    ]
    xtb_e = [r.get("xtb_energy") for r in rows if r.get("xtb_energy") is not None]

    md = []
    md.append("# Conformer Summary\n")
    md.append(f"Total molecules: {total}\n")
    md.append(f"Success: {succ} ({(succ/total*100.0 if total else 0):.1f}%)\n")
    if min_energies:
        md.append(
            f"Min energy: mean={sum(min_energies)/len(min_energies):.3f}, n={len(min_energies)}\n"
        )
    if dipoles:
        md.append(
            f"Dipole magnitude (approx): mean={sum(dipoles)/len(dipoles):.3f}, n={len(dipoles)}\n"
        )
    if xtb_e:
        md.append(f"xTB energy (if available): n={len(xtb_e)}\n")

    # Top-5 by min energy
    top = sorted(
        (r for r in rows if r.get("min_energy") is not None), key=lambda r: r["min_energy"]
    )[:5]
    if top:
        md.append("\n## Lowest-energy examples\n")
        md.append("id/smiles | min_energy | forcefield\n")
        md.append("---|---:|---\n")
        for r in top:
            label = r.get("inchikey") or r.get("smiles")
            md.append(f"{label} | {r.get('min_energy'):.3f} | {r.get('forcefield')}\n")

    out_md = Path(out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(md), encoding="utf-8")

    if out_json is not None:
        summary = {
            "total": total,
            "success": succ,
            "min_energy_count": len(min_energies),
            "dipole_count": len(dipoles),
            "xtb_energy_count": len(xtb_e),
        }
        Path(out_json).write_text(
            json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2), encoding="utf-8"
        )

    return out_md
