from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path


def plot_interval_calibration(
    nominal: Sequence[float],
    coverage: Sequence[float],
    out_path: str | Path,
    title: str = "Calibration",
) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        raise RuntimeError("matplotlib not installed. Install extras: '.[ml]'") from e

    x = list(nominal)
    y = list(coverage)
    fig, ax = plt.subplots(figsize=(4, 4), dpi=150)
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="ideal")
    ax.plot(x, y, marker="o", label="empirical")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("nominal interval level")
    ax.set_ylabel("empirical coverage")
    ax.set_title(title)
    ax.legend()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
