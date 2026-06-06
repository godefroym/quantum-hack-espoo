"""Higher-order joint-structure and tail-dependence statistics for default samples.

Marginals and pairwise correlation cannot separate a genuinely higher-order joint from a
second-order foil matched to the same marginals and correlations, so two discriminators a
second-order model provably cannot fake are used:

* **Excess connected third cumulant (co-skewness).** The sampled co-skewness
  ``E[(x_i-p_i)(x_j-p_j)(x_k-p_k)]`` minus the value a Gaussian copula with the same
  marginals and correlation would produce (closed form in
  :func:`gaussian_copula_reference_coskewness`). Near zero for any elliptical model, large
  for non-elliptical joints (e.g. a rare common shock).

* **Lower-tail dependence.** A Gaussian copula has zero lower-tail dependence for correlation
  ``< 1``. The pairwise/aggregate tail ratios and :func:`joint_tail_excess` (excess
  probability of many defaults at once over the independence baseline) read near the
  independence baseline for a Gaussian copula but clearly positive for clustered joints.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
from scipy.optimize import brentq
from scipy.stats import multivariate_normal, norm

from systemic_risk.models.ising import LossDistribution
from systemic_risk.utils.validation import ensure_binary_samples, nearest_psd_correlation


@dataclass
class HigherOrderStructure:
    """Connected third-order co-default structure over all institution triples.

    The ``excess_*`` fields subtract the correlation-matched Gaussian-copula reference,
    isolating structure beyond second order.
    """

    coskewness_rms: float
    coskewness_max: float
    excess_coskewness_rms: float
    excess_coskewness_max: float
    n_triples: int


@dataclass
class TailDependence:
    """Lower-tail (joint-extreme co-default) dependence of a binary sample matrix.

    See the per-statistic functions (:func:`aggregate_tail_dependence`,
    :func:`pairwise_lower_tail_dependence`, :func:`joint_tail_excess`) for definitions.
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
    co_moment = np.einsum("ti,tj,tk->ijk", centered, centered, centered) / samples.shape[0]
    denom = np.einsum("i,j,k->ijk", scales, scales, scales)
    with np.errstate(divide="ignore", invalid="ignore"):
        normalized = np.where(denom > 0, co_moment / denom, 0.0)
    return normalized[_triple_indices(n)]


def gaussian_copula_reference_coskewness(
    marginals: np.ndarray, correlation: np.ndarray
) -> np.ndarray:
    """Return the connected third cumulant a Gaussian copula would produce per triple.

    Each ``E[x_a x_b x_c]`` is the trivariate-normal orthant probability with the latent
    correlations implied by the binary ``correlation``; subtracting the result from the
    sampled co-skewness isolates structure beyond second order. Triple ordering matches
    :func:`itertools.combinations`.

    The per-pair latent correlations are solved once and reused for the pairwise orthants
    ``E[x_i x_j]``; the trivariate orthant needs a jointly PSD 3x3 block, so it reads from a
    once-projected PSD copy of the latent matrix (every 3x3 principal submatrix of a PSD
    matrix is itself PSD), avoiding a per-triple PSD repair.
    """
    marginals = np.asarray(marginals, dtype=float)
    correlation = np.asarray(correlation, dtype=float)
    n = marginals.shape[0]
    thresholds = norm.ppf(np.clip(marginals, 1e-12, 1.0 - 1e-12))
    scales = np.sqrt(np.clip(marginals * (1.0 - marginals), 0.0, None))
    latent = _latent_correlation(marginals, correlation, thresholds)
    latent_psd = nearest_psd_correlation(latent)
    pairwise_e = _pairwise_orthant_matrix(thresholds, scales, latent)

    values = np.empty(max(n * (n - 1) * (n - 2) // 6, 0), dtype=float)
    for index, (i, j, k) in enumerate(combinations(range(n), 3)):
        denom = scales[i] * scales[j] * scales[k]
        if denom == 0.0:
            values[index] = 0.0
            continue
        idx = [i, j, k]
        e_ijk = _trivariate_orthant(thresholds[idx], latent_psd[np.ix_(idx, idx)])
        pi, pj, pk = marginals[i], marginals[j], marginals[k]
        central = (
            e_ijk
            - pi * pairwise_e[j, k]
            - pj * pairwise_e[i, k]
            - pk * pairwise_e[i, j]
            + 2.0 * pi * pj * pk
        )
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
    ``upper`` empirical quantiles of ``K``. The ratio measures whether the far tail clusters
    beyond the merely-large tail: it tends to zero for a Gaussian copula (no joint-extreme
    co-default) and to one for a comonotone / all-together joint.
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
    zero for independent or Gaussian-copula samples and clearly positive when defaults
    cluster.
    """
    samples = ensure_binary_samples(samples).astype(float)
    n_samples, n = samples.shape
    if n < 2:
        return 0.0, 0.0
    marginals = samples.mean(axis=0)
    joint = (samples.T @ samples) / n_samples
    valid = ~np.eye(n, dtype=bool) & (marginals[None, :] > 0.0)
    if not valid.any():
        return 0.0, 0.0
    conditionals = joint[valid] / np.broadcast_to(marginals, (n, n))[valid]
    baselines = np.broadcast_to(marginals[:, None], (n, n))[valid]
    mean_conditional = float(conditionals.mean())
    return mean_conditional, mean_conditional - float(baselines.mean())


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
    """Return ``CVaR_alpha = E[K | K >= VaR_alpha]`` of the cascade default-count ``K``.

    ``failure_counts`` is the per-scenario number of defaults after contagion. The empirical
    pmf is formed and the shared :meth:`LossDistribution.cvar` definition is reused.
    """
    counts = np.rint(np.asarray(failure_counts, dtype=float)).astype(int)
    if counts.size == 0:
        return 0.0
    pmf = np.bincount(np.clip(counts, 0, None)).astype(float)
    pmf /= pmf.sum()
    return LossDistribution(pmf=pmf, exact=False).cvar(alpha=alpha)


def _triple_indices(n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the ``(i, j, k)`` index arrays for every triple in combinations order."""
    if n < 3:
        empty = np.empty(0, dtype=int)
        return empty, empty, empty
    triples = np.fromiter(
        (idx for combo in combinations(range(n), 3) for idx in combo),
        dtype=int,
        count=n * (n - 1) * (n - 2) // 2,
    ).reshape(-1, 3)
    return triples[:, 0], triples[:, 1], triples[:, 2]


def _latent_correlation(
    marginals: np.ndarray, correlation: np.ndarray, thresholds: np.ndarray
) -> np.ndarray:
    """Solve for the latent normal correlation matching each binary Pearson correlation.

    Identical ``(threshold_i, threshold_j, target_joint)`` triples (common when marginals
    repeat) are solved once and reused.
    """
    n = marginals.shape[0]
    scales = np.sqrt(np.clip(marginals * (1.0 - marginals), 0.0, None))
    latent = np.eye(n)
    cache: dict[tuple[float, float, float], float] = {}
    for i in range(n):
        for j in range(i + 1, n):
            if scales[i] == 0.0 or scales[j] == 0.0:
                latent[i, j] = latent[j, i] = 0.0
                continue
            target_joint = (
                correlation[i, j] * scales[i] * scales[j] + marginals[i] * marginals[j]
            )
            key = (
                round(float(thresholds[i]), 9),
                round(float(thresholds[j]), 9),
                round(float(target_joint), 12),
            )
            value = cache.get(key)
            if value is None:
                value = _solve_latent_correlation(
                    thresholds[i], thresholds[j], target_joint
                )
                cache[key] = value
            latent[i, j] = latent[j, i] = value
    return latent


def _solve_latent_correlation(
    threshold_i: float, threshold_j: float, target_joint: float
) -> float:
    """Latent correlation whose bivariate orthant probability equals ``target_joint``.

    The orthant is monotone in the correlation, so a bracketed root find converges in far
    fewer CDF evaluations than fixed-iteration bisection.
    """
    low, high = -0.999, 0.999
    lower_bound = _bivariate_orthant(threshold_i, threshold_j, low)
    upper_bound = _bivariate_orthant(threshold_i, threshold_j, high)
    target = float(np.clip(target_joint, lower_bound, upper_bound))
    if target <= lower_bound:
        return low
    if target >= upper_bound:
        return high
    return float(
        brentq(
            lambda rho: _bivariate_orthant(threshold_i, threshold_j, rho) - target,
            low,
            high,
            xtol=1e-6,
        )
    )


def _bivariate_orthant(threshold_i: float, threshold_j: float, rho: float) -> float:
    """Return ``P(Z_i <= threshold_i, Z_j <= threshold_j)`` for standard bivariate normal."""
    rho = float(np.clip(rho, -0.999999, 0.999999))
    cov = np.array([[1.0, rho], [rho, 1.0]])
    return float(
        multivariate_normal.cdf(
            [threshold_i, threshold_j],
            mean=[0.0, 0.0],
            cov=cov,
            rng=np.random.default_rng(0),
        )
    )


def _pairwise_orthant_matrix(
    thresholds: np.ndarray, scales: np.ndarray, latent: np.ndarray
) -> np.ndarray:
    """Return ``E[x_i x_j] = P(Z_i <= t_i, Z_j <= t_j)`` for all pairs (diagonal unused)."""
    n = thresholds.shape[0]
    pairwise = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            if scales[i] == 0.0 or scales[j] == 0.0:
                continue
            pairwise[i, j] = pairwise[j, i] = _bivariate_orthant(
                thresholds[i], thresholds[j], latent[i, j]
            )
    return pairwise


def _trivariate_orthant(thresholds: np.ndarray, correlation: np.ndarray) -> float:
    """Return the standard trivariate-normal orthant probability below ``thresholds``.

    ``correlation`` is a 3x3 principal submatrix of an already-PSD latent matrix.
    """
    return float(
        multivariate_normal.cdf(
            thresholds,
            mean=np.zeros(3),
            cov=correlation,
            allow_singular=True,
            rng=np.random.default_rng(0),
        )
    )


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
