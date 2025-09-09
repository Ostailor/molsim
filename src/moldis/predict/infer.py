from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .features import featurize_smiles
from .linear import LinearModel


@dataclass
class Prediction:
    mean: float
    sigma: float
    ok: bool


def _hash_config(config: dict[str, Any]) -> str:
    blob = json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def batch_predict(smiles_list: list[str], model: LinearModel) -> list[Prediction]:
    feats = featurize_smiles(smiles_list)
    preds = model.predict(feats.X)
    out: list[Prediction] = []
    for i, m in enumerate(preds):
        out.append(Prediction(mean=float(m), sigma=float(model.sigma), ok=feats.ok[i]))
    return out


def batch_predict_cached(
    smiles_list: list[str],
    model: LinearModel,
    cache_dir: str | Path = "artifacts/cache",
    config: dict[str, Any] | None = None,
) -> list[Prediction]:
    cache_conf = config or {"model": "linear", "features": model.feature_names}
    cache_key = _hash_config(cache_conf)
    cache_path = Path(cache_dir) / f"pred_esol_{cache_key}.json"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache: dict[str, dict[str, Any]] = {}
    if cache_path.exists():
        try:
            cache = json.loads(cache_path.read_text(encoding="utf-8"))
        except Exception:
            cache = {}

    feats = featurize_smiles(smiles_list)
    preds = model.predict(feats.X)
    out: list[Prediction] = []
    updated = False
    for i, s in enumerate(smiles_list):
        if s in cache:
            rec = cache[s]
            out.append(
                Prediction(
                    mean=float(rec["mean"]),
                    sigma=float(rec["sigma"]),
                    ok=bool(rec["ok"]),
                )
            )
        else:
            pred = Prediction(mean=float(preds[i]), sigma=float(model.sigma), ok=feats.ok[i])
            cache[s] = {"mean": pred.mean, "sigma": pred.sigma, "ok": pred.ok}
            out.append(pred)
            updated = True
    if updated:
        cache_path.write_text(
            json.dumps(cache, ensure_ascii=False, sort_keys=True), encoding="utf-8"
        )
    return out
