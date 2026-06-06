from __future__ import annotations

from collections import Counter
from statistics import mean
from typing import Any

from contagion.simulator import CascadeResult


def result_summary(result: CascadeResult) -> dict[str, Any]:
    """
    Compact result dictionary for E's evaluation harness.
    """

    return {
        "system_id": result.system_id,
        "scenario_id": result.scenario_id,
        "node_count": result.node_count,
        "final_failure_count": result.final_failure_count,
        "failure_fraction": result.failure_fraction,
        "cascade_depth": result.cascade_depth,
        "systemic_collapse": result.systemic_collapse,
        "failed_nodes": result.failed_nodes,
    }


def batch_summary(results: list[CascadeResult]) -> list[dict[str, Any]]:
    return [result_summary(result) for result in results]


def aggregate_results(results: list[CascadeResult]) -> dict[str, Any]:
    """
    Aggregate metrics for comparing B vs C in E.

    This intentionally does not inspect generator metadata.
    E can group results externally by generator.
    """

    if not results:
        return {
            "num_scenarios": 0,
            "mean_final_failure_count": 0.0,
            "mean_failure_fraction": 0.0,
            "mean_cascade_depth": 0.0,
            "systemic_collapse_frequency": 0.0,
            "max_final_failure_count": 0,
            "max_cascade_depth": 0,
        }

    systemic_count = sum(1 for result in results if result.systemic_collapse)

    return {
        "num_scenarios": len(results),
        "mean_final_failure_count": mean(
            result.final_failure_count for result in results
        ),
        "mean_failure_fraction": mean(result.failure_fraction for result in results),
        "mean_cascade_depth": mean(result.cascade_depth for result in results),
        "systemic_collapse_frequency": systemic_count / len(results),
        "max_final_failure_count": max(
            result.final_failure_count for result in results
        ),
        "max_cascade_depth": max(result.cascade_depth for result in results),
    }


def failure_round_distribution(result: CascadeResult) -> dict[int, int]:
    """
    Returns a map:
        round_number -> number of failures in that round
    """

    counter = Counter(result.failure_round.values())

    return {round_number: counter[round_number] for round_number in sorted(counter)}


def node_failure_frequencies(results: list[CascadeResult]) -> dict[str, float]:
    """
    For a batch of scenarios, returns:
        node_id -> fraction of scenarios in which the node failed
    """

    if not results:
        return {}

    counts: Counter[str] = Counter()

    for result in results:
        counts.update(result.failed_nodes)

    node_ids = sorted(
        {node_id for result in results for node_id in result.cumulative_losses}
    )

    return {node_id: counts[node_id] / len(results) for node_id in node_ids}


def tail_failure_probability(
    results: list[CascadeResult],
    *,
    min_failure_fraction: float,
) -> float:
    """
    Probability that a scenario causes at least min_failure_fraction failures.
    Useful for comparing tail risk surfaced by B and C.
    """

    if not results:
        return 0.0

    tail_count = sum(
        1 for result in results if result.failure_fraction >= min_failure_fraction
    )

    return tail_count / len(results)
