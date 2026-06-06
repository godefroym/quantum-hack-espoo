from __future__ import annotations

from typing import Any

import numpy as np

from systemic_risk.evaluation.joint_structure import (
    cascade_count_cvar,
    higher_order_structure,
    tail_dependence,
)
from systemic_risk.generators.base import sample_diagnostics
from systemic_risk.simulator.cascade import CascadeOutcome, CascadeResult
from systemic_risk.spec import SystemSpec


def compute_metrics(
    samples: np.ndarray,
    cascade_results: list[CascadeOutcome],
    spec: SystemSpec,
    severe_threshold: int,
    *,
    include_joint_structure: bool = True,
) -> dict[str, float]:
    """Compute scenario-generation and contagion tail metrics."""
    samples = np.asarray(samples, dtype=int)
    failures = np.array([result.failure_count for result in cascade_results], dtype=float)
    diagnostics = sample_diagnostics(samples)
    target_joint = spec.target_pairwise_joint_probs()
    pairwise_mask = ~np.eye(spec.n, dtype=bool)

    severe = failures >= severe_threshold
    severe_samples = samples[severe]
    unique_severe = (
        0
        if len(severe_samples) == 0
        else np.unique(severe_samples, axis=0).shape[0]
    )
    depths = np.array([result.cascade_depth for result in cascade_results], dtype=float)
    collapse = np.array(
        [result.systemic_collapse for result in cascade_results],
        dtype=float,
    )

    metrics = {
        "mean_cascade_size": float(failures.mean()),
        "mean_failure_fraction": float(failures.mean() / spec.n),
        "mean_cascade_depth": float(depths.mean()),
        "max_cascade_depth": float(depths.max()),
        "systemic_collapse_frequency": float(collapse.mean()),
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
        "cascade_count_cvar_95": cascade_count_cvar(failures, alpha=0.95),
        "cascade_count_cvar_99": cascade_count_cvar(failures, alpha=0.99),
    }
    if include_joint_structure:
        structure = higher_order_structure(samples)
        dependence = tail_dependence(samples)
        metrics.update(
            {
                "coskewness_rms": structure.coskewness_rms,
                "coskewness_max": structure.coskewness_max,
                "excess_coskewness_rms": structure.excess_coskewness_rms,
                "excess_coskewness_max": structure.excess_coskewness_max,
                "aggregate_tail_dependence": dependence.aggregate_tail_dependence,
                "pairwise_lower_tail_dependence": dependence.pairwise_lower_tail_dependence,
                "excess_pairwise_lower_tail_dependence": (
                    dependence.excess_pairwise_lower_tail_dependence
                ),
                "joint_tail_excess": dependence.joint_tail_excess,
            }
        )
    return metrics


def result_summary(result: CascadeOutcome) -> dict[str, Any]:
    """Compact, JSON-compatible cascade summary for reports and APIs."""
    return {
        "scenario_id": result.scenario_id,
        "node_count": result.node_count,
        "final_failure_count": result.failure_count,
        "failure_fraction": result.failure_fraction,
        "cascade_depth": result.cascade_depth,
        "systemic_collapse": result.systemic_collapse,
        "converged": result.converged,
        "failed_nodes": result.failed_nodes,
    }


def batch_summary(results: list[CascadeOutcome]) -> list[dict[str, Any]]:
    return [result_summary(result) for result in results]


def aggregate_results(results: list[CascadeOutcome]) -> dict[str, float | int]:
    """Aggregate cascade outcomes independently of the scenario generator."""
    if not results:
        return {
            "num_scenarios": 0,
            "mean_final_failure_count": 0.0,
            "mean_failure_fraction": 0.0,
            "mean_cascade_depth": 0.0,
            "systemic_collapse_frequency": 0.0,
            "max_final_failure_count": 0,
            "max_cascade_depth": 0,
            "convergence_frequency": 0.0,
        }

    failures = np.array([result.failure_count for result in results], dtype=float)
    fractions = np.array([result.failure_fraction for result in results], dtype=float)
    depths = np.array([result.cascade_depth for result in results], dtype=float)
    collapse = np.array([result.systemic_collapse for result in results], dtype=float)
    converged = np.array([result.converged for result in results], dtype=float)
    return {
        "num_scenarios": len(results),
        "mean_final_failure_count": float(failures.mean()),
        "mean_failure_fraction": float(fractions.mean()),
        "mean_cascade_depth": float(depths.mean()),
        "systemic_collapse_frequency": float(collapse.mean()),
        "max_final_failure_count": int(failures.max()),
        "max_cascade_depth": int(depths.max()),
        "convergence_frequency": float(converged.mean()),
    }


def failure_round_distribution(result: CascadeResult) -> dict[int, int]:
    """Return ``round -> number of institutions first failing in that round``."""
    rounds = result.failure_round[result.failure_round >= 0]
    if len(rounds) == 0:
        return {}
    counts = np.bincount(rounds)
    return {round_index: int(count) for round_index, count in enumerate(counts) if count}


def node_failure_frequencies(results: list[CascadeOutcome]) -> dict[str, float]:
    """Return each institution's observed failure frequency over a result batch."""
    if not results:
        return {}
    node_names = results[0].node_names
    if not node_names:
        node_names = tuple(str(i) for i in range(results[0].node_count))
    if any(
        result.node_count != len(node_names)
        or (result.node_names and result.node_names != results[0].node_names)
        for result in results
    ):
        raise ValueError("all cascade results must use the same node set")
    failures = np.stack([result.final_defaults for result in results])
    return {
        name: float(frequency)
        for name, frequency in zip(node_names, failures.mean(axis=0))
    }


def tail_failure_probability(
    results: list[CascadeOutcome],
    *,
    min_failure_fraction: float,
) -> float:
    """Estimate ``P(final failure fraction >= threshold)``."""
    if not 0 <= min_failure_fraction <= 1:
        raise ValueError("min_failure_fraction must lie in [0, 1]")
    if not results:
        return 0.0
    return float(
        np.mean(
            [
                result.failure_fraction >= min_failure_fraction
                for result in results
            ]
        )
    )


def _tail_mean(values: np.ndarray, fraction: float) -> float:
    if len(values) == 0:
        return 0.0
    k = max(1, int(np.ceil(fraction * len(values))))
    return float(np.sort(values)[-k:].mean())
