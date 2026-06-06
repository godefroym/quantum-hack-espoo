from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

from systemic_risk.simulator import simulate_many

SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "compare_real_institutions_quantum.py"
)
SCRIPT_SPEC = importlib.util.spec_from_file_location(
    "compare_real_institutions_quantum",
    SCRIPT_PATH,
)
assert SCRIPT_SPEC is not None and SCRIPT_SPEC.loader is not None
SCRIPT_MODULE = importlib.util.module_from_spec(SCRIPT_SPEC)
SCRIPT_SPEC.loader.exec_module(SCRIPT_MODULE)


def test_stressed_real_bank_spec_contains_all_banks_and_no_corporates() -> None:
    spec = SCRIPT_MODULE.stressed_real_bank_spec(0.05)

    assert spec.n == 28
    assert set(spec.node_types) == {"bank"}
    assert abs(float(spec.marginal_default_probs.mean()) - 0.05) < 1e-12


def test_stressed_real_institution_spec_contains_banks_and_corporates() -> None:
    spec = SCRIPT_MODULE.stressed_real_institution_spec(0.05)

    assert spec.n == 38
    assert spec.node_types.count("bank") == 28
    assert spec.node_types.count("corporate") == 10
    assert abs(float(spec.marginal_default_probs.mean()) - 0.05) < 1e-12


def test_real_bank_chain_has_two_entanglement_layers_and_calibrates() -> None:
    spec = SCRIPT_MODULE.stressed_real_bank_spec(0.05)
    block = SCRIPT_MODULE.fitted_chain(spec)

    assert len(block.edges) == 27
    assert block.entanglement_depth == 2
    marginals, _ = SCRIPT_MODULE.mps_backend.block_moments(
        block.ry, block.edges, block.cry
    )
    assert np.max(np.abs(marginals - spec.marginal_default_probs)) < 1e-3


def test_sample_calibrated_block_is_reproducible() -> None:
    spec = SCRIPT_MODULE.stressed_real_bank_spec(0.05)
    edges = [(0, 1), (1, 2), (2, 3)]

    first = SCRIPT_MODULE._sample_calibrated_block(
        spec, edges, iterations=1, shots=256, seed=8
    )
    second = SCRIPT_MODULE._sample_calibrated_block(
        spec, edges, iterations=1, shots=256, seed=8
    )

    assert np.allclose(first.ry, second.ry)
    assert np.allclose(first.cry, second.cry)


def test_vectorized_cascade_matches_reference_engine() -> None:
    spec = SCRIPT_MODULE.stressed_real_bank_spec(0.05)
    rng = np.random.default_rng(4)
    samples = (rng.random((20, spec.n)) < spec.marginal_default_probs).astype(int)

    counts, depths = SCRIPT_MODULE.vectorized_cascade(samples, spec)
    reference = simulate_many(samples, spec)

    assert np.array_equal(counts, [result.failure_count for result in reference])
    assert np.array_equal(depths, [result.cascade_depth for result in reference])
