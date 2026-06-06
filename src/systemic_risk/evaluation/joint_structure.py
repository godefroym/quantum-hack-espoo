"""Higher-order joint-structure and tail-dependence statistics for default samples.

These statistics exist to substantiate (or falsify) the project's core claim: that an
entangled scenario generator carries genuine *higher-order joint-tail structure* that a
second-order classical foil -- one calibrated to the same per-institution default
probabilities and the same pairwise default correlations -- structurally cannot reproduce.

First/second-order summaries (marginals, pairwise correlation) cannot distinguish a generator
from a genuinely moment-matched foil. This module therefore reports a Gaussian-reference
third-order statistic plus descriptive tail summaries:

* **Connected third cumulant (co-skewness).** ``C_ijk = E[(x_i-p_i)(x_j-p_j)(x_k-p_k)]`` is
  the part of three-way co-default not carried by the marginals or pairwise correlations. It
  is exactly zero under independence, and for a Gaussian copula it is *pinned* by the
  marginals and the correlation matrix (a threshold-nonlinearity term, recovered in closed
  form by :func:`gaussian_copula_reference_coskewness`). Excess over that reference isolates
  structure beyond this specific Gaussian foil.

* **Tail summaries.** Pairwise conditional default rates, aggregate default-count concentration,
  and excess many-default probability describe clustering. Their baselines must be stated:
  pairwise conditional default is second order, while aggregate and joint-tail summaries should
  only be compared after checking realized lower-order moments.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
from scipy.stats import multivariate_normal, norm

from systemic_risk.models.ising import LossDistribution
from systemic_risk.utils.validation import ensure_binary_samples


@dataclass
class HigherOrderStructure:
    """Connected third-order co-default structure of a binary sample matrix.

    ``coskewness_rms`` / ``coskewness_max`` summarise the normalised connected third
    cumulant over all institution triples. ``excess_coskewness_rms`` /
    ``excess_coskewness_max`` summarise the part of that co-skewness *not* explained by a
    Gaussian copula matched to the sample's own marginals and pairwise correlations; these
    are the beyond-Gaussian discriminators (near zero for a Gaussian copula, potentially large
    for a non-Gaussian joint).
    """

    coskewness_rms: float
    coskewness_max: float
    excess_coskewness_rms: float
    excess_coskewness_max: float
    n_triples: int


@dataclass
class TailDependence:
    """Lower-tail (joint-extreme co-default) dependence of a binary sample matrix.

    ``aggregate_tail_dependence`` is the empirical concentration ratio
    ``P(K >= upper) / P(K >= inner)`` for aggregate default count ``K`` at two high quantiles.
    ``pairwise_lower_tail_dependence`` is the mean over ordered pairs of
    ``P(x_i = 1 | x_j = 1)``, the chance a second institution defaults given the first does;
    ``excess_pairwise_lower_tail_dependence`` subtracts the independence baseline (the mean
    marginal). For binary indicators this is a second-order diagnostic fixed by the marginals and
    pairwise joint probabilities; it is not higher-order evidence by itself.
    ``joint_tail_excess`` is the excess probability that at least a fixed fraction of
    institutions default *together* over the same-marginal independence baseline -- the
    many-defaults-at-once statistic that is exactly zero in expectation under independence and
    strongly positive for clustered / common-shock joints.
    """

    aggregate_tail_dependence: float
    pairwise_lower_tail_dependence: float
    excess_pairwise_lower_tail_dependence: float
    joint_tail_excess: float


def _marginals_and_scales(samples: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    marginals = samples.mean(axis=0)
    scales = np.sqrt(np.clip(marginals * (1.0 - marginals), 0.0, None))
    return marginals, scales


def _pairwise_correlation(samples: np.ndarray) -> np.ndarray:
    marginals, scales = _marginals_and_scales(samples)
    centered = samples - marginals
    cov = (centered.T @ centered) / samples.shape[0]
    denom = np.outer(scales, scales)
    with np.errstate(divide="ignore", invalid="ignore"):
        corr = np.where(denom > 0, cov / denom, 0.0)
    np.fill_diagonal(corr, 1.0)
    return np.clip(corr, -1.0, 1.0)


def connected_third_cumulants(samples: np.ndarray) -> np.ndarray:
    """Return the normalised connected third cumulant for every institution triple.

    For a triple ``(i, j, k)`` the value is
    ``E[(x_i - p_i)(x_j - p_j)(x_k - p_k)] / (sigma_i sigma_j sigma_k)`` (the co-skewness),
    evaluated over the sample. Triples that include a constant institution
    (``sigma == 0``) contribute zero. The result is a length ``C(n, 3)`` array ordered by
    :func:`itertools.combinations`.
    """
    samples = ensure_binary_samples(samples).astype(float)
    n = samples.shape[1]
    marginals, scales = _marginals_and_scales(samples)
    centered = samples - marginals
    values = np.empty(max(n * (n - 1) * (n - 2) // 6, 0), dtype=float)
    for index, (i, j, k) in enumerate(combinations(range(n), 3)):
        denom = scales[i] * scales[j] * scales[k]
        if denom == 0.0:
            values[index] = 0.0
            continue
        central = float(np.mean(centered[:, i] * centered[:, j] * centered[:, k]))
        values[index] = central / denom
    return values


def gaussian_copula_reference_coskewness(
    marginals: np.ndarray, correlation: np.ndarray
) -> np.ndarray:
    """Return the connected third cumulant a Gaussian copula would produce per triple.

    Given target ``marginals`` and a pairwise ``correlation`` matrix (Pearson correlation of
    the binary indicators), this is the normalised connected third cumulant of the
    second-order model that thresholds a standard multivariate normal: each ``E[x_a x_b x_c]``
    is the trivariate-normal orthant probability with the latent correlations implied by the
    binary correlations. Subtracting this from the sampled co-skewness isolates structure
    beyond second order. Triple ordering matches :func:`itertools.combinations`.
    """
    marginals = np.asarray(marginals, dtype=float)
    correlation = np.asarray(correlation, dtype=float)
    n = marginals.shape[0]
    thresholds = norm.ppf(np.clip(marginals, 1e-12, 1.0 - 1e-12))
    scales = np.sqrt(np.clip(marginals * (1.0 - marginals), 0.0, None))
    latent = _latent_correlation(marginals, correlation, thresholds)

    values = np.empty(max(n * (n - 1) * (n - 2) // 6, 0), dtype=float)
    for index, (i, j, k) in enumerate(combinations(range(n), 3)):
        denom = scales[i] * scales[j] * scales[k]
        if denom == 0.0:
            values[index] = 0.0
            continue
        triple_corr = np.array(
            [
                [1.0, latent[i, j], latent[i, k]],
                [latent[i, j], 1.0, latent[j, k]],
                [latent[i, k], latent[j, k], 1.0],
            ]
        )
        e_ijk = _trivariate_orthant(thresholds[[i, j, k]], triple_corr)
        e_ij = _bivariate_orthant(thresholds[i], thresholds[j], latent[i, j])
        e_ik = _bivariate_orthant(thresholds[i], thresholds[k], latent[i, k])
        e_jk = _bivariate_orthant(thresholds[j], thresholds[k], latent[j, k])
        pi, pj, pk = marginals[i], marginals[j], marginals[k]
        central = e_ijk - pi * e_jk - pj * e_ik - pk * e_ij + 2.0 * pi * pj * pk
        values[index] = central / denom
    return values


def higher_order_structure(samples: np.ndarray) -> HigherOrderStructure:
    """Summarise connected third-order structure and its excess over a second-order foil."""
    samples = ensure_binary_samples(samples)
    coskewness = connected_third_cumulants(samples)
    marginals, _ = _marginals_and_scales(samples.astype(float))
    correlation = _pairwise_correlation(samples.astype(float))
    reference = gaussian_copula_reference_coskewness(marginals, correlation)
    excess = coskewness - reference
    return HigherOrderStructure(
        coskewness_rms=_rms(coskewness),
        coskewness_max=_abs_max(coskewness),
        excess_coskewness_rms=_rms(excess),
        excess_coskewness_max=_abs_max(excess),
        n_triples=int(coskewness.size),
    )


def aggregate_tail_dependence(
    samples: np.ndarray, inner_quantile: float = 0.90, upper_quantile: float = 0.99
) -> float:
    """Return ``P(K >= upper | K >= inner)`` for the aggregate default count ``K``.

    ``K`` is the per-scenario number of defaults. The two thresholds are the ``inner`` and
    ``upper`` empirical quantiles of ``K``. This is a finite-sample, discrete-count summary,
    not the asymptotic copula tail-dependence coefficient. It becomes uninformative when the
    two quantile thresholds coincide.
    """
    samples = ensure_binary_samples(samples)
    if not 0.0 < inner_quantile < upper_quantile < 1.0:
        raise ValueError("require 0 < inner_quantile < upper_quantile < 1")
    counts = samples.sum(axis=1)
    inner_threshold = np.quantile(counts, inner_quantile)
    upper_threshold = np.quantile(counts, upper_quantile)
    inner_mass = float(np.mean(counts >= inner_threshold))
    if inner_mass == 0.0:
        return 0.0
    upper_mass = float(np.mean(counts >= upper_threshold))
    return upper_mass / inner_mass


def pairwise_lower_tail_dependence(samples: np.ndarray) -> tuple[float, float]:
    """Return ``(mean P(x_i=1 | x_j=1), excess over the independence baseline)``.

    The first value is the mean over ordered institution pairs of the chance that ``i``
    defaults given ``j`` defaults -- the empirical lower-tail co-default rate. The second
    subtracts the mean marginal (what that rate would be under independence), so it is near
    zero for independent samples and positive whenever defaults cluster pairwise. Generators
    matched on first and second moments should agree on it.
    """
    samples = ensure_binary_samples(samples).astype(float)
    n_samples, n = samples.shape
    if n < 2:
        return 0.0, 0.0
    marginals = samples.mean(axis=0)
    joint = (samples.T @ samples) / n_samples
    conditionals = []
    baselines = []
    for i in range(n):
        for j in range(n):
            if i == j or marginals[j] == 0.0:
                continue
            conditionals.append(joint[i, j] / marginals[j])
            baselines.append(marginals[i])
    if not conditionals:
        return 0.0, 0.0
    mean_conditional = float(np.mean(conditionals))
    excess = mean_conditional - float(np.mean(baselines))
    return mean_conditional, excess


def joint_tail_excess(samples: np.ndarray, fraction: float = 0.5) -> float:
    """Return ``P(K >= ceil(fraction * n)) - P_independent(K >= ceil(fraction * n))``.

    ``K`` is the per-scenario number of defaults. The baseline is the same probability for an
    independent model carrying the *same marginals* -- computed exactly by convolving the
    per-institution Bernoulli pmfs -- so the difference isolates joint clustering from the
    marginal default level. It is zero in expectation under independence, positive whenever
    many institutions default together more often than their marginals alone would imply, and
    large for common-shock / comonotone joints. Unlike a pairwise correlation it captures the
    many-at-once co-default that a correlation summary misses.
    """
    samples = ensure_binary_samples(samples)
    n = samples.shape[1]
    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must lie in (0, 1]")
    threshold = int(np.ceil(fraction * n))
    counts = samples.sum(axis=1)
    empirical = float(np.mean(counts >= threshold))
    baseline = _independent_count_tail(samples.mean(axis=0), threshold)
    return empirical - baseline


def tail_dependence(
    samples: np.ndarray,
    inner_quantile: float = 0.90,
    upper_quantile: float = 0.99,
    fraction: float = 0.5,
) -> TailDependence:
    """Summarise aggregate, pairwise, and joint lower-tail co-default dependence."""
    aggregate = aggregate_tail_dependence(samples, inner_quantile, upper_quantile)
    pairwise, excess = pairwise_lower_tail_dependence(samples)
    return TailDependence(
        aggregate_tail_dependence=aggregate,
        pairwise_lower_tail_dependence=pairwise,
        excess_pairwise_lower_tail_dependence=excess,
        joint_tail_excess=joint_tail_excess(samples, fraction),
    )


def cascade_count_cvar(failure_counts: np.ndarray, alpha: float = 0.95) -> float:
    """Return the CVaR (expected shortfall) of the cascade default-count distribution.

    Defers to :meth:`systemic_risk.models.ising.LossDistribution.cvar` for one consistent
    CVaR definition across the project: the empirical default-count distribution is formed
    into a pmf and ``CVaR_alpha = E[K | K >= VaR_alpha]`` is read off it, with the VaR level
    chosen so the conditioning tail carries mass ``>= 1 - alpha``. ``failure_counts`` is the
    per-scenario number of defaults after contagion (the ``failure_count`` of each cascade
    result).
    """
    counts = np.rint(np.asarray(failure_counts, dtype=float)).astype(int)
    if counts.size == 0:
        return 0.0
    pmf = np.bincount(np.clip(counts, 0, None)).astype(float)
    pmf /= pmf.sum()
    return LossDistribution(pmf=pmf, exact=False).cvar(alpha=alpha)


def _latent_correlation(
    marginals: np.ndarray, correlation: np.ndarray, thresholds: np.ndarray
) -> np.ndarray:
    """Solve for the latent normal correlation matching each binary Pearson correlation."""
    n = marginals.shape[0]
    scales = np.sqrt(np.clip(marginals * (1.0 - marginals), 0.0, None))
    latent = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            if scales[i] == 0.0 or scales[j] == 0.0:
                value = 0.0
            else:
                target_joint = (
                    correlation[i, j] * scales[i] * scales[j] + marginals[i] * marginals[j]
                )
                value = _solve_latent_correlation(
                    thresholds[i], thresholds[j], target_joint
                )
            latent[i, j] = latent[j, i] = value
    return latent


def _solve_latent_correlation(
    threshold_i: float, threshold_j: float, target_joint: float
) -> float:
    """Bisection for the latent correlation whose orthant probability is ``target_joint``."""
    lower_bound = max(0.0, _bivariate_orthant(threshold_i, threshold_j, -1.0))
    upper_bound = _bivariate_orthant(threshold_i, threshold_j, 1.0)
    target = float(np.clip(target_joint, lower_bound, upper_bound))
    low, high = -0.999, 0.999
    for _ in range(60):
        mid = 0.5 * (low + high)
        if _bivariate_orthant(threshold_i, threshold_j, mid) < target:
            low = mid
        else:
            high = mid
    return 0.5 * (low + high)


def _bivariate_orthant(threshold_i: float, threshold_j: float, rho: float) -> float:
    """Return ``P(Z_i <= threshold_i, Z_j <= threshold_j)`` for standard bivariate normal."""
    rho = float(np.clip(rho, -0.999999, 0.999999))
    cov = np.array([[1.0, rho], [rho, 1.0]])
    return float(multivariate_normal.cdf([threshold_i, threshold_j], mean=[0.0, 0.0], cov=cov))


def _trivariate_orthant(thresholds: np.ndarray, correlation: np.ndarray) -> float:
    """Return the standard trivariate-normal orthant probability below ``thresholds``.

    The triple correlation is assembled from independently solved pairwise latent correlations
    and so need not be jointly positive semidefinite (three strong pairwise correlations can be
    mutually infeasible). It is projected to the nearest positive-definite correlation before
    evaluating the CDF.
    """
    psd = _nearest_positive_definite_correlation(correlation)
    return float(
        multivariate_normal.cdf(thresholds, mean=np.zeros(3), cov=psd, allow_singular=True)
    )


def _nearest_positive_definite_correlation(matrix: np.ndarray, floor: float = 1e-6) -> np.ndarray:
    """Project a symmetric matrix to a positive-definite correlation matrix.

    Eigenvalues are floored to ``floor`` and the result rescaled to unit diagonal; a final
    eigenvalue floor (without re-clipping the off-diagonals) keeps it strictly positive
    definite so the normal CDF accepts it.
    """
    symmetric = 0.5 * (matrix + matrix.T)
    for _ in range(2):
        eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
        eigenvalues = np.maximum(eigenvalues, floor)
        symmetric = (eigenvectors * eigenvalues) @ eigenvectors.T
        scale = np.sqrt(np.diag(symmetric))
        symmetric = symmetric / np.outer(scale, scale)
    return symmetric


def _independent_count_tail(marginals: np.ndarray, threshold: int) -> float:
    """Return ``P(K >= threshold)`` for independent Bernoullis with the given marginals.

    The default-count pmf of independent (non-identical) Bernoullis is the convolution of the
    per-institution pmfs ``[1 - p_i, p_i]``; the upper tail is summed from ``threshold``.
    """
    if threshold <= 0:
        return 1.0
    pmf = np.array([1.0])
    for p in marginals:
        pmf = np.convolve(pmf, [1.0 - p, p])
    return float(pmf[threshold:].sum())


def _rms(values: np.ndarray) -> float:
    return 0.0 if values.size == 0 else float(np.sqrt(np.mean(values**2)))


def _abs_max(values: np.ndarray) -> float:
    return 0.0 if values.size == 0 else float(np.max(np.abs(values)))
