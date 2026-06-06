"""Pairwise Ising / Boltzmann scenario generator.

Wraps :class:`systemic_risk.models.ising.IsingModel` in the
:class:`~systemic_risk.generators.base.ScenarioGenerator` interface. Fitting a
:class:`SystemSpec` produces the plausibility distribution of
``scenario_generation.md``::

    Pi(x) = sum_i h_i x_i + sum_{i<j} J_ij x_i x_j,    P(x) propto exp(Pi(x)),

with ``x in {0, 1}^n`` (``x_i = 1`` -> institution ``i`` initially defaults). Calibration
follows the recipe in ``research/sections/03_statistical_mechanics_ising.md`` (G1-G3):

1. **Couplings** ``J_ij`` come from :func:`couplings_from_spec` -- inverse-Ising from the
   spec's target correlation matrix, or gravity-weighted from its exposure matrix.
2. **Fields** ``h_i`` are initialised at ``logit(p_i)`` and **refit by Boltzmann learning**
   (``delta h_i propto p_i - <x_i>_model``) so the model marginals match the target
   ``p_i`` even though the couplings drift them. For small ``n`` the model marginals in the
   refit come from exact enumeration; for large ``n`` from the naive mean-field estimate.
3. **Sampling** is by an automatic, size-keyed method (the same choice
   :meth:`IsingModel.method_for` makes): exact enumeration for ``n <= 20``, single-spin-flip
   Gibbs MCMC for ``20 < n < 40``, and Gibbs + parallel tempering at ``n >= 40`` (e.g. the
   ``n = 54`` hardware target), where exact enumeration of ``2^n`` is impossible and the
   correlation-driven first-order transition (Molins & Vives 2005) causes critical slowing
   down.

References: Filiz, Guo, Morton & Sturmfels (2012) arXiv:0809.1393; Schneidman et al. (2006)
Nature 440; Bury (2013) arXiv:1210.8380; Nguyen, Zecchina & Berg (2017) arXiv:1702.01522;
Cimini et al. (2015) Sci. Rep. 5, 15758; Molins & Vives (2005) arXiv:cond-mat/0401378.
"""

from __future__ import annotations

import numpy as np

from systemic_risk.generators.base import ScenarioGenerator, require_fitted
from systemic_risk.models.calibration import (
    couplings_from_spec,
    fit_fields_boltzmann,
    logit_fields,
)
from systemic_risk.models.ising import MAX_EXACT_N, IsingModel
from systemic_risk.spec import SystemSpec


class IsingBoltzmannGenerator(ScenarioGenerator):
    """Correlated-default generator backed by a calibrated pairwise Ising model."""

    name = "Ising Boltzmann"

    def __init__(
        self,
        *,
        route: str = "auto",
        coupling_scale: float | str = "auto",
        correlation_method: str = "tap",
        coupling_cap: float = 6.0,
        refit_fields: bool = True,
        field_learning_rate: float = 0.5,
        field_max_iter: int = 500,
        field_tol: float = 1e-3,
        sample_method: str | None = None,
        burn_in: int = 1_000,
        thin: int = 10,
        n_replicas: int = 8,
        beta_min: float = 0.2,
    ) -> None:
        self.route = route
        # ``coupling_scale="auto"`` calibrates the overall coupling strength to the spec's
        # target mean default correlation via the closed-form homogeneous oracle (so the
        # generated co-default rate lands near target instead of overshooting through the
        # correlation-driven first-order transition). A float pins the scale directly.
        self.coupling_scale = coupling_scale
        self.correlation_method = correlation_method
        self.coupling_cap = coupling_cap
        self.refit_fields = refit_fields
        self.field_learning_rate = field_learning_rate
        self.field_max_iter = field_max_iter
        self.field_tol = field_tol
        self.sample_method = sample_method
        self.burn_in = burn_in
        self.thin = thin
        self.n_replicas = n_replicas
        self.beta_min = beta_min

        self.spec_: SystemSpec | None = None
        self.model_: IsingModel | None = None
        self.fields_: np.ndarray | None = None
        self.couplings_: np.ndarray | None = None
        self.route_used_: str | None = None
        self.coupling_scale_used_: float | None = None
        self.fit_info_: dict[str, float] = {}

    # ------------------------------------------------------------------ fitting
    def fit(self, spec: SystemSpec) -> None:
        self.spec_ = spec
        target_p = np.clip(spec.marginal_default_probs.copy(), 1e-9, 1.0 - 1e-9)

        # Build the *structural* coupling matrix (which pairs couple, and how strongly
        # relative to each other) at unit scale, then set the overall magnitude.
        base_scale = 1.0
        couplings, route_used = couplings_from_spec(
            spec,
            route=self._resolve_route(spec),
            coupling_scale=base_scale,
            correlation_method=self.correlation_method,
            coupling_cap=self.coupling_cap,
        )
        self.route_used_ = route_used

        scale = self._resolve_coupling_scale(spec, couplings)
        self.coupling_scale_used_ = scale
        couplings = np.clip(scale * couplings, -self.coupling_cap, self.coupling_cap)
        self.couplings_ = couplings

        fields = logit_fields(target_p)
        if self.refit_fields:
            # For small n the Boltzmann-learning refit uses exact model marginals;
            # for large n it falls back to the cheap mean-field estimate (the default).
            moment_fn = self._exact_marginals if spec.n <= MAX_EXACT_N else None
            fields, info = fit_fields_boltzmann(
                target_p,
                couplings,
                initial_fields=fields,
                learning_rate=self.field_learning_rate,
                max_iter=self.field_max_iter,
                tol=self.field_tol,
                moment_fn=moment_fn,
            )
            self.fit_info_ = {**info, "field_refit": True}
        else:
            self.fit_info_ = {"field_refit": False}

        self.fields_ = fields
        self.model_ = IsingModel(fields, couplings)

    @staticmethod
    def _exact_marginals(fields: np.ndarray, couplings: np.ndarray) -> np.ndarray:
        """Exact model marginals via enumeration (used in the small-n field refit)."""
        marginals, _ = IsingModel(fields, couplings).exact_moments()
        return marginals

    # ---------------------------------------------------------------- calibration
    def _resolve_route(self, spec: SystemSpec) -> str:
        """Pick the coupling route.

        Unlike :func:`couplings_from_spec`'s ``"auto"`` (which prefers the correlation
        route), we prefer the **exposure** route whenever a non-empty exposure matrix is
        present: the inverse-Ising correlation route is numerically unstable in the
        very-small-marginal / very-small-correlation credit regime (tiny ``+/-1`` variances
        ``4 p (1 - p)`` make ``C^-1`` blow up), whereas the exposure graph is the project's
        load-bearing source of couplings (Cimini et al. 2015).
        """
        if self.route != "auto":
            return self.route
        if float(np.sum(spec.exposure_matrix)) > 0.0:
            return "exposure"
        if spec.target_pairwise_corr is not None:
            return "correlation"
        return "exposure"

    def _resolve_coupling_scale(self, spec: SystemSpec, couplings: np.ndarray) -> float:
        """Resolve the overall coupling magnitude.

        A float ``coupling_scale`` is returned as-is. ``"auto"`` bisects a single global
        multiplier ``s`` so the **mean-field-predicted** mean off-diagonal default
        correlation of ``s * couplings`` matches the spec's target mean default correlation.
        The prediction uses the Gaussian linear-response (susceptibility) formula, which is
        ``O(n^3)`` and therefore affordable at every size including ``n = 54`` (unlike exact
        enumeration). It accounts for the heterogeneous, sparse coupling structure of the
        exposure graph -- the homogeneous oracle alone would mis-scale because the exposure
        graph couples only a fraction of pairs. Bisecting in ``s`` also keeps the sampler on
        the low-correlation branch, clear of the first-order transition (Molins & Vives 2005).
        """
        if not isinstance(self.coupling_scale, str):
            return float(self.coupling_scale)
        if self.coupling_scale != "auto":
            raise ValueError(f"unknown coupling_scale {self.coupling_scale!r}")

        if not np.any(couplings != 0.0):
            return 1.0

        target_corr = self._target_mean_default_corr(spec)
        if abs(target_corr) < 1e-9:
            return 0.0

        fields0 = logit_fields(np.clip(spec.marginal_default_probs, 1e-9, 1.0 - 1e-9))

        def predicted_corr(scale: float) -> float:
            return self._mean_field_mean_corr(fields0, scale * couplings)

        # Bisect on scale (predicted correlation is monotone increasing in scale on the
        # stable branch). Cap the upper bracket well below the transition blow-up.
        lo, hi = 0.0, 1.0
        f_hi = predicted_corr(hi)
        guard = 0
        while f_hi < target_corr and hi < 50.0 and guard < 60:
            hi *= 1.5
            f_hi = predicted_corr(hi)
            guard += 1
        if f_hi < target_corr:  # cannot reach target without blowing up; use the cap
            return float(hi)
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if predicted_corr(mid) < target_corr:
                lo = mid
            else:
                hi = mid
        return float(0.5 * (lo + hi))

    @staticmethod
    def _mean_field_mean_corr(fields: np.ndarray, couplings: np.ndarray) -> float:
        """Mean off-diagonal default correlation predicted by Gaussian linear response.

        Solves the mean-field marginals ``m_i = sigmoid(h_i + sum_j J_ij m_j)`` then uses the
        susceptibility (connected-correlation) matrix ``C = (D^-1 - J)^-1`` with
        ``D = diag(m_i (1 - m_i))``, giving the pairwise Pearson default correlations
        ``rho_ij = C_ij / sqrt(C_ii C_jj)``. (Standard inverse-Ising linear response; Nguyen,
        Zecchina & Berg 2017.) Returns the mean over ``i < j``.
        """
        from systemic_risk.models.calibration import mean_field_marginals

        n = len(fields)
        m = mean_field_marginals(fields, couplings)
        var = np.clip(m * (1.0 - m), 1e-12, None)
        a = np.diag(1.0 / var) - couplings
        try:
            cov = np.linalg.inv(a)
        except np.linalg.LinAlgError:
            cov = np.linalg.pinv(a)
        d = np.sqrt(np.clip(np.diag(cov), 1e-18, None))
        corr = cov / np.outer(d, d)
        iu = np.triu_indices(n, k=1)
        return float(np.mean(corr[iu]))

    @staticmethod
    def _target_mean_default_corr(spec: SystemSpec) -> float:
        """Target mean off-diagonal default (event) correlation implied by the spec."""
        if spec.target_pairwise_corr is not None:
            iu = np.triu_indices(spec.n, k=1)
            return float(spec.target_pairwise_corr[iu].mean())
        # Fall back to the correlation implied by the target joint default probabilities.
        joint = spec.target_pairwise_joint_probs()
        p = spec.marginal_default_probs
        iu = np.triu_indices(spec.n, k=1)
        denom = np.sqrt(p * (1.0 - p))
        rho = (joint[iu] - np.outer(p, p)[iu]) / (denom[iu[0]] * denom[iu[1]] + 1e-12)
        return float(np.mean(rho))

    # ----------------------------------------------------------------- sampling
    def _sample_kwargs(self) -> dict[str, object]:
        return {
            "burn_in": self.burn_in,
            "thin": self.thin,
            "n_replicas": self.n_replicas,
            "beta_min": self.beta_min,
        }

    @property
    def sampler(self) -> str:
        """Return the sampling method that will be used for the fitted model's size."""
        require_fitted(self.model_, self.name)
        return self.sample_method or self.model_.method_for()

    def sample(self, n_samples: int, seed: int | None = None) -> np.ndarray:
        require_fitted(self.model_, self.name)
        return self.model_.sample(
            n_samples,
            seed=seed,
            method=self.sample_method,
            **self._sample_kwargs(),  # type: ignore[arg-type]
        )
