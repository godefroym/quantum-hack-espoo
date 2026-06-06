"""Criterion 1 — first- and second-order match (marginals + pairwise correlation).

The claim under test: the entangled generator's marginals and pairwise default correlations
match the strongest classical generator within tolerance, so they are genuinely interchangeable
*at that level*. Two honesty rules apply (see :mod:`._specs`):

* match is scored against the **achievable (Fréchet) ceiling**, not the nominal target, because a
  large fraction of the real targets are infeasible for any binary model at these tiny marginals;
* the entangled generator must be a single fully-simulated block on the chosen spec, otherwise its
  cross-cluster correlation is silently lost — which is why the criterion spec is one community.

Everything here is read off the empirical samples (not the analytic moments), so a sceptic sees
exactly what the sampler produces. RMSE and worst-edge error are reported against both the nominal
target and the achievable ceiling, with the generator's mean correlation alongside both means.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from systemic_risk.generators.moments import targets_from_spec
from systemic_risk.spec import SystemSpec


@dataclass(frozen=True)
class SecondOrderMatch:
    """Empirical 1st/2nd-order fidelity of one generator's samples to a spec."""

    generator: str
    marginal_rmse: float
    marginal_max_abs_err: float
    mean_corr_generated: float
    corr_rmse_vs_nominal: float
    corr_rmse_vs_achievable: float
    corr_max_abs_err_vs_achievable: float


def empirical_marginals_and_corr(samples: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(marginals, pearson_correlation)`` of a binary sample matrix."""
    samples = np.asarray(samples, dtype=float)
    n_samples, n = samples.shape
    marginals = samples.mean(axis=0)
    scale = np.sqrt(np.clip(marginals * (1.0 - marginals), 0.0, None))
    joint = (samples.T @ samples) / max(n_samples, 1)
    denom = np.outer(scale, scale)
    with np.errstate(divide="ignore", invalid="ignore"):
        corr = np.where(denom > 0, (joint - np.outer(marginals, marginals)) / denom, 0.0)
    np.fill_diagonal(corr, 1.0)
    return marginals, np.clip(corr, -1.0, 1.0)


def second_order_match(
    generator: str,
    samples: np.ndarray,
    spec: SystemSpec,
    achievable: np.ndarray,
) -> SecondOrderMatch:
    """Score one generator's empirical binary moments against the canonical target."""
    marginals, corr = empirical_marginals_and_corr(samples)
    iu = np.triu_indices(spec.n, k=1)
    nominal = targets_from_spec(spec).pairwise_corr
    marg_err = marginals - spec.marginal_default_probs
    corr_err_nominal = corr[iu] - nominal[iu]
    corr_err_achievable = corr[iu] - achievable[iu]
    return SecondOrderMatch(
        generator=generator,
        marginal_rmse=float(np.sqrt(np.mean(marg_err**2))),
        marginal_max_abs_err=float(np.max(np.abs(marg_err))) if marg_err.size else 0.0,
        mean_corr_generated=float(corr[iu].mean()) if iu[0].size else 0.0,
        corr_rmse_vs_nominal=float(np.sqrt(np.mean(corr_err_nominal**2))) if iu[0].size else 0.0,
        corr_rmse_vs_achievable=(
            float(np.sqrt(np.mean(corr_err_achievable**2))) if iu[0].size else 0.0
        ),
        corr_max_abs_err_vs_achievable=(
            float(np.max(np.abs(corr_err_achievable))) if iu[0].size else 0.0
        ),
    )
