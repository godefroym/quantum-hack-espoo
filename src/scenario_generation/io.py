from __future__ import annotations

from pathlib import Path
import csv
import numpy as np

from systemic_risk.spec import SystemSpec


def load_system_spec(path: str | Path) -> SystemSpec:
    """Load a `SystemSpec` from a JSON or NPZ file.

    Falls back to `SystemSpec.load_json` for other file extensions.
    """
    p = Path(path)
    if p.suffix == ".npz":
        return SystemSpec.load_npz(p)
    # default: attempt JSON load
    return SystemSpec.load_json(p)


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
