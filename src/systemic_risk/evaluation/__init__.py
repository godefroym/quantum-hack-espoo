"""Evaluation metrics and comparison harness."""

from systemic_risk.evaluation.channels import (
    ContagionChannel,
    ExposureCascadeChannel,
    HuangFireSaleChannel,
    as_channel,
)
from systemic_risk.evaluation.harness import EvaluationHarness, GeneratorRunResult
from systemic_risk.evaluation.joint_structure import (
    HigherOrderStructure,
    TailDependence,
    aggregate_tail_dependence,
    cascade_count_cvar,
    connected_third_cumulants,
    gaussian_copula_reference_coskewness,
    higher_order_structure,
    joint_tail_excess,
    pairwise_lower_tail_dependence,
    tail_dependence,
)
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
    "ContagionChannel",
    "EvaluationHarness",
    "ExposureCascadeChannel",
    "GeneratorRunResult",
    "HigherOrderStructure",
    "HuangFireSaleChannel",
    "TailDependence",
    "aggregate_results",
    "as_channel",
    "aggregate_tail_dependence",
    "batch_summary",
    "cascade_count_cvar",
    "compute_metrics",
    "connected_third_cumulants",
    "failure_round_distribution",
    "gaussian_copula_reference_coskewness",
    "higher_order_structure",
    "joint_tail_excess",
    "node_failure_frequencies",
    "pairwise_lower_tail_dependence",
    "result_summary",
    "tail_dependence",
    "tail_failure_probability",
]
