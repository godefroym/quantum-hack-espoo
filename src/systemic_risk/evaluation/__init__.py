"""Evaluation metrics and comparison harness."""

from systemic_risk.evaluation.harness import EvaluationHarness, GeneratorRunResult
from systemic_risk.evaluation.metrics import (
    aggregate_results,
    batch_summary,
    compute_metrics,
    failure_round_distribution,
    node_failure_frequencies,
    result_summary,
    tail_failure_probability,
)

__all__ = [
    "EvaluationHarness",
    "GeneratorRunResult",
    "aggregate_results",
    "batch_summary",
    "compute_metrics",
    "failure_round_distribution",
    "node_failure_frequencies",
    "result_summary",
    "tail_failure_probability",
]
