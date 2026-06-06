"""Closed-form homogeneous (mean-field / infinite-range) Ising loss distribution.

For a *homogeneous* portfolio -- uniform field ``h`` on every node and uniform coupling ``J``
on every pair -- the log-weight of a configuration with ``k`` defaults depends only on ``k``::

    Pi(x) = h k + J * k (k - 1) / 2,

and there are ``C(n, k)`` configurations with exactly ``k`` defaults, so the number-of-defaults
distribution is closed form::

    P(K = k) propto C(n, k) exp(h k + J k (k - 1) / 2),   k = 0, ..., n.

This is the long-range / mean-field Ising credit model of **Molins & Vives (2005)**
(arXiv:cond-mat/0401378) and **Kitsukawa, Mori & Hisakado (2006)** (arXiv:physics/0603040):
the field sets the marginal default probability and the coupling sets the default
correlation. Crucially it is **exact at any ``n``, including 54**, because the sum runs over
only ``n + 1`` default-count states rather than ``2^n`` configurations. We use it as the
ground-truth validation oracle for the MCMC sampler at ``n = 54``.

This module is the ``{0, 1}``-basis (default-indicator) statement; it is equivalent to the
``+/-1`` Hamiltonian ``H = -(J'/N) sum_{i<j} sigma_i sigma_j - H' sum_i sigma_i`` used in the
cited papers, up to the standard field/coupling reparameterisation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq
from scipy.special import gammaln

from systemic_risk.models.ising import LossDistribution


def _log_binom(n: int, k: np.ndarray) -> np.ndarray:
    return gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1)


@dataclass
class MeanFieldIsingOracle:
    """Homogeneous Ising loss distribution, exact at any ``n``.

    Parameters
    ----------
    n:
        Number of institutions.
    field:
        Uniform ``{0,1}``-basis field ``h``.
    coupling:
        Uniform ``{0,1}``-basis pairwise coupling ``J``.
    """

    n: int
    field: float
    coupling: float

    def loss_distribution(self) -> LossDistribution:
        """Return the exact distribution of the number of defaults ``K``."""
        k = np.arange(self.n + 1)
        log_w = _log_binom(self.n, k) + self.field * k + self.coupling * k * (k - 1) / 2.0
        log_w -= log_w.max()
        w = np.exp(log_w)
        pmf = w / w.sum()
        return LossDistribution(pmf=pmf, exact=True)

    def marginal_default_prob(self) -> float:
        """Return ``P(default_i) = E[K] / n`` (identical for every node)."""
        return self.loss_distribution().mean() / self.n

    def co_default_prob(self) -> float:
        """Return ``P(default_i and default_j) = E[K (K - 1)] / (n (n - 1))``."""
        dist = self.loss_distribution()
        k = dist.counts
        e_k_km1 = float(np.dot(k * (k - 1), dist.pmf))
        return e_k_km1 / (self.n * (self.n - 1))

    def default_correlation(self) -> float:
        """Return the pairwise default (event) correlation ``rho_d``."""
        p = self.marginal_default_prob()
        q = self.co_default_prob()
        denom = p * (1.0 - p)
        if denom <= 0:
            return 0.0
        return (q - p * p) / denom

    @classmethod
    def from_targets(
        cls,
        n: int,
        target_marginal: float,
        target_default_corr: float,
        *,
        max_iter: int = 100,
        tol: float = 1e-10,
    ) -> "MeanFieldIsingOracle":
        """Solve for ``(h, J)`` reproducing a target marginal and default correlation.

        Uses a nested 1-D root find: an outer bracketed solve on ``J`` (monotone in the
        induced default correlation) and, for each candidate ``J``, an inner solve on ``h``
        (monotone in the induced marginal). Both are exact given the closed-form distribution.
        """
        target_marginal = float(np.clip(target_marginal, 1e-9, 1.0 - 1e-9))

        def field_for_marginal(coupling: float) -> float:
            def marginal_residual(h: float) -> float:
                return cls(n=n, field=h, coupling=coupling).marginal_default_prob() - target_marginal

            # The marginal is monotone increasing in h; [-60, 60] brackets every feasible
            # target (at h = +-60 the marginal is already 0 / 1 to machine precision).
            lo, hi = -60.0, 60.0
            f_lo, f_hi = marginal_residual(lo), marginal_residual(hi)
            if f_lo > 0:
                return lo
            if f_hi < 0:
                return hi
            return brentq(marginal_residual, lo, hi, xtol=tol, maxiter=max_iter)

        if abs(target_default_corr) < 1e-12:
            h = field_for_marginal(0.0)
            return cls(n=n, field=h, coupling=0.0)

        def corr_residual(coupling: float) -> float:
            h = field_for_marginal(coupling)
            return cls(n=n, field=h, coupling=coupling).default_correlation() - target_default_corr

        # Default correlation increases with the coupling. Bracket J, expanding the open end
        # until the residual changes sign, then solve.
        lo, hi = 0.0, 0.01
        if target_default_corr < 0:
            lo, hi = -5.0, 0.0
            while corr_residual(lo) > 0 and lo > -200.0:
                lo *= 2.0
        else:
            while corr_residual(hi) < 0 and hi < 200.0:
                hi *= 1.7
        coupling = brentq(corr_residual, lo, hi, xtol=tol, maxiter=max_iter)
        h = field_for_marginal(coupling)
        return cls(n=n, field=h, coupling=coupling)


def total_variation_distance(a: LossDistribution, b: LossDistribution) -> float:
    """Total-variation distance between two default-count distributions on the same ``n``."""
    if a.n != b.n:
        raise ValueError("loss distributions must share the same support size n")
    return 0.5 * float(np.sum(np.abs(a.pmf - b.pmf)))
