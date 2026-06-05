from __future__ import annotations

import numpy as np

from systemic_risk.simulator import run_cascade
from systemic_risk.spec import SystemSpec


def _spec(W: np.ndarray, c: np.ndarray) -> SystemSpec:
    n = len(c)
    return SystemSpec(
        node_names=[f"Node {i}" for i in range(n)],
        node_types=["bank"] * n,
        exposure_matrix=W,
        capital_buffers=c,
        marginal_default_probs=np.full(n, 0.05),
        target_pairwise_corr=np.eye(n),
        clusters=["banks"] * n,
    )


def test_no_edge_graph_keeps_initial_failures() -> None:
    spec = _spec(np.zeros((3, 3)), np.ones(3))
    result = run_cascade(np.array([1, 0, 1]), spec)

    assert result.failure_count == 2
    assert np.array_equal(result.final_defaults, np.array([1, 0, 1]))


def test_chain_contagion_triggers_known_cascade() -> None:
    W = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    )
    spec = _spec(W, np.array([2.0, 0.5, 0.5]))

    result = run_cascade(np.array([1, 0, 0]), spec)

    assert result.failure_count == 3
    assert result.rounds_to_convergence == 2


def test_cascade_is_monotone() -> None:
    W = np.array(
        [
            [0.0, 0.4, 0.4, 0.0],
            [0.0, 0.0, 0.0, 0.7],
            [0.0, 0.0, 0.0, 0.7],
            [0.0, 0.0, 0.0, 0.0],
        ]
    )
    spec = _spec(W, np.array([0.5, 0.5, 0.5, 2.0]))

    smaller = run_cascade(np.array([0, 1, 0, 0]), spec)
    larger = run_cascade(np.array([0, 1, 1, 0]), spec)

    assert smaller.failure_count <= larger.failure_count
