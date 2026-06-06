"""Criteria 2 & 3 — higher-order/tail structure and its effect on the cascade tail.

Criterion 2 (genuine higher-order structure)
    The entangled generator must carry joint-tail dependence the best **moment-matched** classical
    model cannot reproduce. The Gaussian copula is the structural foil: matched to the same
    marginals and pairwise correlations, it has *provably zero* lower-tail dependence and a
    co-skewness pinned by its first two moments. The discriminators are the **excess** co-skewness
    (sampled minus the closed-form Gaussian-copula reference) and the tail-dependence statistics.
    A Gaussian copula's excess co-skewness is finite-sample noise that vanishes as N grows; a
    genuinely higher-order joint keeps it large and stable. :func:`excess_coskewness_convergence`
    makes that contrast explicit.

Criterion 3 (material to risk)
    The extra structure must measurably *move the systemic-risk outcome* — the contagion-cascade
    tail — not merely show up as a static distributional difference. We compare, on the same spec
    and the same cascade engine, the entangled generator against the moment-matched Gaussian foil:
    ``p_severe_cascade``, the upper tail-means of the cascade size, and deep-tail co-default mass.

    One honest subtlety surfaced and is reported: at tiny credit marginals the systemic mode is
    *rare but catastrophic* (mostly zero defaults, occasionally the whole block). A fixed-α CVaR
    (95/99%) can place its VaR level at a zero count and so understate the move; the deeper
    ``P(K ≥ half)`` and a 99.9% count-CVaR are the faithful deep-tail statistics here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from systemic_risk.evaluation import cascade_count_cvar, compute_metrics
from systemic_risk.evaluation.joint_structure import higher_order_structure
from systemic_risk.generators.base import ScenarioGenerator
from systemic_risk.simulator.cascade import simulate_many
from systemic_risk.spec import SystemSpec


# --- the metric keys each criterion reports (kept here so tables and CSV stay in sync) --------- #
SECOND_ORDER_KEYS = ("marginal_rmse", "pairwise_joint_rmse")
HIGHER_ORDER_KEYS = (
    "excess_coskewness_rms",
    "excess_coskewness_max",
    "excess_pairwise_lower_tail_dependence",
    "aggregate_tail_dependence",
    "joint_tail_excess",
)
CASCADE_TAIL_KEYS = (
    "p_severe_cascade",
    "mean_cascade_size",
    "max_cascade_size",
    "tail_mean_1pct",
    "tail_mean_5pct",
    "cascade_count_cvar_95",
    "cascade_count_cvar_99",
)


@dataclass
class GeneratorEvaluation:
    """Everything one generator produced on a spec: samples, cascades, and the metric dict."""

    name: str
    samples: np.ndarray
    failure_counts: np.ndarray
    metrics: dict[str, float]


def evaluate_generator(
    generator: ScenarioGenerator,
    spec: SystemSpec,
    n_samples: int,
    seed: int,
    severe_threshold: int | None = None,
) -> GeneratorEvaluation:
    """Fit, sample, run the cascade, and compute the full metric dict for one generator."""
    generator.fit(spec)
    samples = generator.sample(n_samples, seed=seed)
    cascades = simulate_many(samples, spec)
    failure_counts = np.array([c.failure_count for c in cascades], dtype=float)
    threshold = severe_threshold if severe_threshold is not None else int(np.ceil(0.5 * spec.n))
    metrics = compute_metrics(samples, cascades, spec, severe_threshold=threshold)
    metrics.update(_deep_tail_metrics(samples, failure_counts, spec))
    return GeneratorEvaluation(generator.name, samples, failure_counts, metrics)


def _deep_tail_metrics(
    samples: np.ndarray, failure_counts: np.ndarray, spec: SystemSpec
) -> dict[str, float]:
    """Add deep-tail co-default statistics the fixed-α CVaRs can miss on spiky systemic modes.

    ``p_half_or_more_default`` is the probability the *initial* shock already defaults at least
    half the block together (the common-shock mode); ``p_cascade_half_or_more`` is the same after
    contagion; ``cascade_count_cvar_999`` is the 99.9% expected shortfall of the cascade size.
    """
    half = int(np.ceil(0.5 * spec.n))
    initial_counts = np.asarray(samples, dtype=int).sum(axis=1)
    return {
        "p_initial_half_or_more": float(np.mean(initial_counts >= half)),
        "p_cascade_half_or_more": float(np.mean(failure_counts >= half)),
        "cascade_count_cvar_999": cascade_count_cvar(failure_counts, alpha=0.999),
    }


DEEP_TAIL_KEYS = ("p_initial_half_or_more", "p_cascade_half_or_more", "cascade_count_cvar_999")


def excess_coskewness_convergence(
    generator: ScenarioGenerator,
    spec: SystemSpec,
    sample_sizes: tuple[int, ...],
    seed: int,
) -> list[tuple[int, float]]:
    """Return ``[(N, excess_coskewness_rms), ...]`` for a fitted generator across sample sizes.

    For a Gaussian copula this sequence decays toward zero (its genuine higher-order structure is
    zero — the signal is sampling noise); for the entangled generator it plateaus at a large value
    (genuine structure). Showing the trajectory is the rigorous way to prove the excess is real.
    """
    generator.fit(spec)
    out: list[tuple[int, float]] = []
    for n_samples in sample_sizes:
        samples = generator.sample(n_samples, seed=seed)
        out.append((n_samples, higher_order_structure(samples).excess_coskewness_rms))
    return out
