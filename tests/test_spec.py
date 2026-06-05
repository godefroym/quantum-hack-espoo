from __future__ import annotations

import numpy as np

from systemic_risk.data import make_synthetic_system
from systemic_risk.spec import SystemSpec


def test_synthetic_system_is_deterministic() -> None:
    first = make_synthetic_system(n=16, seed=42)
    second = make_synthetic_system(n=16, seed=42)

    assert first.node_names == second.node_names
    assert np.allclose(first.exposure_matrix, second.exposure_matrix)
    assert np.allclose(first.marginal_default_probs, second.marginal_default_probs)


def test_spec_json_roundtrip(tmp_path) -> None:
    spec = make_synthetic_system(n=12, seed=3)
    path = tmp_path / "spec.json"
    spec.save_json(path)

    loaded = SystemSpec.load_json(path)

    assert loaded.node_names == spec.node_names
    assert np.allclose(loaded.capital_buffers, spec.capital_buffers)
    assert np.allclose(loaded.target_pairwise_corr, spec.target_pairwise_corr)
