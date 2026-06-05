from __future__ import annotations

from pathlib import Path

from systemic_risk.spec import SystemSpec


def load_system_spec(path: str | Path) -> SystemSpec:
    path = Path(path)
    if path.suffix.lower() == ".json":
        return SystemSpec.load_json(path)
    if path.suffix.lower() == ".npz":
        return SystemSpec.load_npz(path)
    raise ValueError("Supported SystemSpec formats are .json and .npz")


def save_system_spec(spec: SystemSpec, path: str | Path) -> None:
    path = Path(path)
    if path.suffix.lower() == ".json":
        spec.save_json(path)
        return
    if path.suffix.lower() == ".npz":
        spec.save_npz(path)
        return
    raise ValueError("Supported SystemSpec formats are .json and .npz")
