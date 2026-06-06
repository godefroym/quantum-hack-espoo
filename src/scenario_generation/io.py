from __future__ import annotations

from pathlib import Path
import csv
import numpy as np

from systemic_risk.spec import SystemSpec
from systemic_risk.data.loaders import load_system_spec as _load_spec


def load_system_spec(path: str | Path) -> SystemSpec:
    return _load_spec(path)


def save_scenarios(path: str | Path, samples: np.ndarray, node_names: list[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    samples = np.asarray(samples, dtype=int)
    n_samples, n = samples.shape
    if len(node_names) != n:
        raise ValueError("node_names length must match number of columns in samples")
    # Save as CSV with header of node names, one scenario per row
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(node_names)
        for row in samples:
            writer.writerow(row.tolist())
