"""Discrimination tests for higher-order joint-structure and tail-dependence statistics.

Each statistic is validated against sample matrices with KNOWN structure, constructed here so
the tests are self-contained. Four reference joints are used:

* **independent Bernoulli** -- second order is trivial, so higher-order and tail-dependence
  statistics must sit at their zero baselines;
* **Gaussian copula** (threshold an equicorrelated normal) -- a genuinely second-order /
  elliptical model: it carries substantial pairwise correlation yet, being elliptical, has a
  third-order structure *pinned* by its marginals and correlation and provably zero lower-tail
  dependence at any correlation ``< 1``;
* **GHZ-like common-shock mixture** -- a rare "everyone defaults at once" event mixed with an
  independent idiosyncratic background. This is the crucial foil: it can be tuned to the *same*
  marginals and the *same* pairwise correlation as a Gaussian copula while remaining
  non-elliptical, so its connected third cumulant and joint-tail co-default exceed anything a
  correlation-matched Gaussian copula (or any second-order model) can reproduce;
* **comonotone** -- the rank-1 "all-default-together" limit, used for tail dependence (where it
  is maximal). It is deliberately *not* used as the excess-coskewness foil: comonotonicity is
  the degenerate correlation-one limit of the Gaussian copula itself, so a Gaussian copula does
  reproduce its co-skewness -- which is exactly why the GHZ mixture, not comonotonicity, is the
  referee-proof beyond-second-order test.

The assertions check separation (zero/near-baseline for the second-order and independent cases,
large for the genuinely higher-order case), not merely that the functions run.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

from systemic_risk.evaluation.joint_structure import (
    aggregate_tail_dependence,
    cascade_count_cvar,
    connected_third_cumulants,
    gaussian_copula_reference_coskewness,
    higher_order_structure,
    joint_tail_excess,
    pairwise_lower_tail_dependence,
    tail_dependence,
)

N_SAMPLES = 120_000
N_INSTITUTIONS = 8
MARGINAL = 0.15


def _independent_bernoulli(rng: np.random.Generator, marginal: float = MARGINAL) -> np.ndarray:
    probs = np.full(N_INSTITUTIONS, marginal)
    return (rng.random((N_SAMPLES, N_INSTITUTIONS)) < probs).astype(int)


def _comonotone(rng: np.random.Generator, marginal: float = MARGINAL) -> np.ndarray:
    """All institutions default together driven by one common uniform (rank-comonotone)."""
    common = rng.random(N_SAMPLES)
    probs = np.full(N_INSTITUTIONS, marginal)
    return (common[:, None] < probs[None, :]).astype(int)


def _gaussian_copula(
    rng: np.random.Generator, rho: float, marginal: float = MARGINAL
) -> np.ndarray:
    """Threshold an equicorrelated standard normal -- a second-order (elliptical) model."""
    covariance = np.full((N_INSTITUTIONS, N_INSTITUTIONS), rho)
    np.fill_diagonal(covariance, 1.0)
    chol = np.linalg.cholesky(covariance)
    latent = rng.standard_normal((N_SAMPLES, N_INSTITUTIONS)) @ chol.T
    threshold = norm.ppf(marginal)
    return (latent < threshold).astype(int)


def _ghz_mixture(
    rng: np.random.Generator, common_prob: float, idiosyncratic_prob: float
) -> np.ndarray:
    """Rare 'everyone defaults at once' shock OR an independent idiosyncratic default.

    A fraction ``common_prob`` of scenarios default every institution together (the GHZ-like
    all-ones lump); the rest carry independent ``idiosyncratic_prob`` defaults. The common shock
    injects three-way co-default and joint-tail mass that no elliptical model can reproduce at
    matched marginals and correlation.
    """
    common = (rng.random(N_SAMPLES) < common_prob).astype(int)
    idiosyncratic = (rng.random((N_SAMPLES, N_INSTITUTIONS)) < idiosyncratic_prob).astype(int)
    return np.maximum(common[:, None], idiosyncratic)


def _average_pairwise_correlation(samples: np.ndarray) -> float:
    marginals = samples.mean(axis=0)
    scales = np.sqrt(marginals * (1.0 - marginals))
    centered = samples - marginals
    cov = (centered.T @ centered) / samples.shape[0]
    n = samples.shape[1]
    off_diagonal = [
        cov[i, j] / (scales[i] * scales[j])
        for i in range(n)
        for j in range(i + 1, n)
        if scales[i] > 0 and scales[j] > 0
    ]
    return float(np.mean(off_diagonal))


def _gaussian_copula_matching(samples: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Draw a Gaussian-copula sample matched to ``samples``' marginals and correlation.

    This builds the explicit second-order foil for an arbitrary joint: same per-institution
    default probabilities, same pairwise default correlations, elliptical dependence.
    """
    from systemic_risk.evaluation.joint_structure import _latent_correlation

    marginals = samples.mean(axis=0).astype(float)
    n = samples.shape[1]
    scales = np.sqrt(marginals * (1.0 - marginals))
    centered = samples - marginals
    cov = (centered.T @ centered) / samples.shape[0]
    correlation = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            correlation[i, j] = correlation[j, i] = cov[i, j] / (scales[i] * scales[j])

    thresholds = norm.ppf(np.clip(marginals, 1e-9, 1.0 - 1e-9))
    latent = _latent_correlation(marginals, correlation, thresholds)
    eigenvalues, eigenvectors = np.linalg.eigh(latent)
    latent = (eigenvectors * np.clip(eigenvalues, 1e-6, None)) @ eigenvectors.T
    diag = np.sqrt(np.diag(latent))
    latent = latent / np.outer(diag, diag)
    chol = np.linalg.cholesky(latent)
    draws = rng.standard_normal((N_SAMPLES, n)) @ chol.T
    return (draws < thresholds).astype(int)


# --------------------------------------------------------------- third-order structure
def test_third_cumulant_is_negligible_for_independent() -> None:
    rng = np.random.default_rng(0)
    independent = higher_order_structure(_independent_bernoulli(rng))

    assert independent.coskewness_rms < 0.02
    assert independent.excess_coskewness_rms < 0.02
    assert independent.excess_coskewness_max < 0.05


def test_third_cumulant_separates_structured_from_independent() -> None:
    rng = np.random.default_rng(1)
    independent = higher_order_structure(_independent_bernoulli(rng))
    ghz = higher_order_structure(_ghz_mixture(rng, common_prob=0.05, idiosyncratic_prob=0.105))

    assert ghz.coskewness_rms > 0.3
    assert ghz.coskewness_rms > 25.0 * independent.coskewness_rms


def test_excess_coskewness_is_near_zero_for_gaussian_copula() -> None:
    rng = np.random.default_rng(2)
    samples = _gaussian_copula(rng, rho=0.6)

    assert _average_pairwise_correlation(samples) > 0.25

    structure = higher_order_structure(samples)
    # The raw co-skewness of a thresholded Gaussian is non-trivial at asymmetric marginals...
    assert structure.coskewness_rms > 0.1
    # ...but it is pinned by the marginals + correlation, so the beyond-second-order excess
    # collapses to near zero.
    assert structure.excess_coskewness_rms < 0.05
    assert structure.excess_coskewness_rms < 0.2 * structure.coskewness_rms


def test_excess_coskewness_flags_ghz_above_correlation_matched_gaussian() -> None:
    rng = np.random.default_rng(3)
    ghz_samples = _ghz_mixture(rng, common_prob=0.05, idiosyncratic_prob=0.105)
    matched_gaussian = _gaussian_copula_matching(ghz_samples, rng)

    # The Gaussian foil is matched on the first two orders: same marginals, same correlation.
    assert abs(ghz_samples.mean() - matched_gaussian.mean()) < 0.01
    assert abs(
        _average_pairwise_correlation(ghz_samples)
        - _average_pairwise_correlation(matched_gaussian)
    ) < 0.02

    ghz = higher_order_structure(ghz_samples)
    gaussian = higher_order_structure(matched_gaussian)

    # Despite identical first/second order, the GHZ mixture carries large beyond-second-order
    # co-skewness while the correlation-matched Gaussian foil cannot fake it.
    assert ghz.excess_coskewness_rms > 0.3
    assert gaussian.excess_coskewness_rms < 0.05
    assert ghz.excess_coskewness_rms > 10.0 * gaussian.excess_coskewness_rms


def test_gaussian_copula_reference_matches_its_own_sample_coskewness() -> None:
    rng = np.random.default_rng(4)
    samples = _gaussian_copula(rng, rho=0.5).astype(float)
    marginals = samples.mean(axis=0)
    centered = samples - marginals
    scales = np.sqrt(marginals * (1.0 - marginals))
    n = samples.shape[1]
    correlation = np.eye(n)
    cov = (centered.T @ centered) / samples.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            correlation[i, j] = correlation[j, i] = cov[i, j] / (scales[i] * scales[j])

    sampled = connected_third_cumulants(samples)
    reference = gaussian_copula_reference_coskewness(marginals, correlation)

    # The closed-form Gaussian-copula reference reproduces the sampled co-skewness within
    # Monte-Carlo error, confirming it is the correct second-order baseline to subtract.
    assert np.max(np.abs(sampled - reference)) < 0.05


# ----------------------------------------------------------------- tail dependence
def test_pairwise_lower_tail_dependence_orders_independent_gaussian_comonotone() -> None:
    rng = np.random.default_rng(5)
    independent = pairwise_lower_tail_dependence(_independent_bernoulli(rng))[1]
    gaussian = pairwise_lower_tail_dependence(_gaussian_copula(rng, rho=0.6))[1]
    comonotone = pairwise_lower_tail_dependence(_comonotone(rng))[1]

    # Excess over the independence baseline: ~0 when defaults do not cluster pairwise.
    assert abs(independent) < 0.02
    # A comonotone joint means defaulting alongside another is near-certain: P ~ 1 vs p = 0.15.
    assert comonotone > 0.7
    # Gaussian copula carries pairwise correlation, so it sits strictly between the two.
    assert independent < gaussian < comonotone


def test_pairwise_conditional_default_rate_is_one_for_comonotone() -> None:
    rng = np.random.default_rng(6)
    conditional, _ = pairwise_lower_tail_dependence(_comonotone(rng))

    # Given any institution defaults, every other defaults too in the comonotone joint.
    assert conditional > 0.95


def test_gaussian_copula_tail_dependence_decays_as_the_tail_deepens() -> None:
    # The defining property of the Gaussian copula: lower-tail dependence is zero for rho < 1.
    # Empirically, the conditional co-default rate's excess over independence shrinks toward
    # zero as the marginal threshold (and hence the conditioning event) is pushed deeper.
    excess_by_depth = []
    for marginal in (0.20, 0.05, 0.01):
        rng = np.random.default_rng(7)
        excess = pairwise_lower_tail_dependence(
            _gaussian_copula(rng, rho=0.6, marginal=marginal)
        )[1]
        excess_by_depth.append(excess)

    assert excess_by_depth[0] > excess_by_depth[1] > excess_by_depth[2]
    # Comonotone, by contrast, keeps full tail dependence no matter how deep the tail.
    rng = np.random.default_rng(7)
    deep_comonotone = pairwise_lower_tail_dependence(_comonotone(rng, marginal=0.01))[0]
    assert deep_comonotone > 0.95


def test_joint_tail_excess_separates_clustered_from_independent() -> None:
    rng = np.random.default_rng(8)
    independent = joint_tail_excess(_independent_bernoulli(rng))
    ghz = joint_tail_excess(_ghz_mixture(rng, common_prob=0.05, idiosyncratic_prob=0.105))
    comonotone = joint_tail_excess(_comonotone(rng))

    # Independence is the zero baseline by construction (Monte-Carlo noise only).
    assert abs(independent) < 0.005
    # A rare common shock makes many-at-once defaults far more likely than the marginals imply.
    assert ghz > 0.02
    assert ghz > 10.0 * abs(independent)
    # A comonotone joint puts the whole upper tail together: maximal joint-tail excess.
    assert comonotone > 0.1
    assert comonotone > ghz


def test_joint_tail_excess_flags_ghz_above_correlation_matched_gaussian_deep_tail() -> None:
    rng = np.random.default_rng(9)
    ghz_samples = _ghz_mixture(rng, common_prob=0.05, idiosyncratic_prob=0.105)
    matched_gaussian = _gaussian_copula_matching(ghz_samples, rng)

    # Deep in the tail (almost everyone defaulting together) the Gaussian copula's clustering
    # has decayed, while the GHZ common shock still puts the whole system down together.
    ghz = joint_tail_excess(ghz_samples, fraction=1.0)
    gaussian = joint_tail_excess(matched_gaussian, fraction=1.0)

    assert ghz > 0.02
    assert ghz > 3.0 * gaussian


def test_aggregate_tail_dependence_separates_clustered_from_gaussian() -> None:
    rng = np.random.default_rng(10)
    gaussian = aggregate_tail_dependence(_gaussian_copula(rng, rho=0.6))
    comonotone = aggregate_tail_dependence(_comonotone(rng))

    # Gaussian copula has provably zero tail dependence: its far tail does not cluster beyond
    # the bulk, so the conditional far-tail ratio stays low...
    assert gaussian < 0.3
    # ...while a comonotone joint puts the whole far tail together.
    assert comonotone > 0.9
    assert comonotone > 3.0 * gaussian


def test_tail_dependence_summary_matches_component_functions() -> None:
    rng = np.random.default_rng(11)
    samples = _gaussian_copula(rng, rho=0.4)
    summary = tail_dependence(samples)

    assert summary.aggregate_tail_dependence == aggregate_tail_dependence(samples)
    conditional, excess = pairwise_lower_tail_dependence(samples)
    assert summary.pairwise_lower_tail_dependence == conditional
    assert summary.excess_pairwise_lower_tail_dependence == excess
    assert summary.joint_tail_excess == joint_tail_excess(samples)


# ------------------------------------------------------------------ cascade CVaR
def test_cascade_cvar_is_tail_average_and_orders_correctly() -> None:
    # 90% of scenarios are mild (10 defaults), 10% are severe (50 defaults).
    counts = np.array([10] * 9 + [50], dtype=float)

    cvar_90 = cascade_count_cvar(counts, alpha=0.90)
    cvar_50 = cascade_count_cvar(counts, alpha=0.50)

    # The worst 10% is exactly the severe scenario, so CVaR_90 reads its loss.
    assert cvar_90 == 50.0
    # A wider tail averages in the mild bulk, so it sits between the mean and the worst case.
    assert counts.mean() <= cvar_50 <= cvar_90
    assert cvar_50 < cvar_90


def test_cascade_cvar_matches_loss_distribution_definition() -> None:
    from systemic_risk.models.ising import LossDistribution

    rng = np.random.default_rng(12)
    n = 12
    counts = rng.integers(0, n + 1, size=40_000)
    pmf = np.bincount(counts, minlength=n + 1).astype(float)
    pmf /= pmf.sum()
    loss = LossDistribution(pmf=pmf, exact=False)

    for alpha in (0.90, 0.95, 0.99):
        sample_cvar = cascade_count_cvar(counts.astype(float), alpha=alpha)
        # The estimator forms the same default-count pmf and defers to the shared CVaR
        # definition, so it reproduces it exactly.
        assert sample_cvar == loss.cvar(alpha=alpha)


def test_cascade_cvar_handles_empty_input() -> None:
    assert cascade_count_cvar(np.array([]), alpha=0.95) == 0.0
