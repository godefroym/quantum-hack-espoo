from __future__ import annotations

import numpy as np

from systemic_risk.generators.base import sample_diagnostics
from systemic_risk.simulator.cascade import CascadeResult
from systemic_risk.spec import SystemSpec


def compute_metrics(
    samples: np.ndarray,
    cascade_results: list[CascadeResult],
    spec: SystemSpec,
    severe_threshold: int,
) -> dict[str, float]:
    """Compute scenario-generation and contagion tail metrics."""
    samples = np.asarray(samples, dtype=int)
    failures = np.array([result.failure_count for result in cascade_results], dtype=float)
    diagnostics = sample_diagnostics(samples)
    target_joint = spec.target_pairwise_joint_probs()
    pairwise_mask = ~np.eye(spec.n, dtype=bool)

    severe = failures >= severe_threshold
    severe_samples = samples[severe]
    unique_severe = 0 if len(severe_samples) == 0 else np.unique(severe_samples, axis=0).shape[0]

    return {
        "mean_cascade_size": float(failures.mean()),
        "max_cascade_size": float(failures.max()),
        "p_severe_cascade": float(severe.mean()),
        "tail_mean_1pct": _tail_mean(failures, 0.01),
        "tail_mean_5pct": _tail_mean(failures, 0.05),
        "tail_mean_10pct": _tail_mean(failures, 0.10),
        "unique_severe_scenarios": float(unique_severe),
        "marginal_rmse": float(
            np.sqrt(np.mean((diagnostics.sampled_marginals - spec.marginal_default_probs) ** 2))
        ),
        "pairwise_joint_rmse": float(
            np.sqrt(
                np.mean(
                    (
                        diagnostics.sampled_pairwise_joint[pairwise_mask]
                        - target_joint[pairwise_mask]
                    )
                    ** 2
                )
            )
        ),
        "scenario_diversity": float(diagnostics.n_unique_scenarios / max(len(samples), 1)),
    }


def _tail_mean(values: np.ndarray, fraction: float) -> float:
    if len(values) == 0:
        return 0.0
    k = max(1, int(np.ceil(fraction * len(values))))
    return float(np.sort(values)[-k:].mean())
