from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from systemic_risk.spec import SystemSpec


@dataclass
class CascadeResult:
    initial_defaults: np.ndarray
    final_defaults: np.ndarray
    failure_count: int
    rounds_to_convergence: int
    states_by_round: list[np.ndarray]
    systemic_collapse: bool


def run_cascade(
    initial_defaults: np.ndarray,
    spec: SystemSpec,
    max_rounds: int = 30,
    collapse_threshold: float = 0.5,
) -> CascadeResult:
    """Run deterministic fixed-point contagion from an initial default vector."""
    state = np.asarray(initial_defaults, dtype=bool)
    if state.shape != (spec.n,):
        raise ValueError(f"initial_defaults must have shape ({spec.n},)")
    if not 0 < collapse_threshold <= 1:
        raise ValueError("collapse_threshold must lie in (0, 1]")

    states = [state.copy()]
    for _ in range(max_rounds):
        losses = spec.exposure_matrix @ state.astype(float)
        threshold_failures = losses > spec.capital_buffers
        next_state = state | threshold_failures
        if np.array_equal(next_state, state):
            break
        state = next_state
        states.append(state.copy())

    failure_count = int(state.sum())
    return CascadeResult(
        initial_defaults=np.asarray(initial_defaults, dtype=int),
        final_defaults=state.astype(int),
        failure_count=failure_count,
        rounds_to_convergence=len(states) - 1,
        states_by_round=[round_state.astype(int) for round_state in states],
        systemic_collapse=failure_count >= int(np.ceil(collapse_threshold * spec.n)),
    )


def simulate_many(
    scenarios: np.ndarray,
    spec: SystemSpec,
    max_rounds: int = 30,
    collapse_threshold: float = 0.5,
) -> list[CascadeResult]:
    scenarios = np.asarray(scenarios, dtype=int)
    if scenarios.ndim != 2 or scenarios.shape[1] != spec.n:
        raise ValueError(f"scenarios must have shape (n_samples, {spec.n})")
    return [
        run_cascade(row, spec, max_rounds=max_rounds, collapse_threshold=collapse_threshold)
        for row in scenarios
    ]
