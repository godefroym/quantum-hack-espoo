from __future__ import annotations

import numpy as np
from scipy.optimize import brentq
from scipy.stats import multivariate_normal, norm

from systemic_risk.generators.base import ScenarioGenerator, require_fitted
from systemic_risk.generators.moments import MomentTargets, targets_from_spec
from systemic_risk.spec import SystemSpec
from systemic_risk.utils.validation import nearest_psd_correlation


class GaussianCopulaGenerator(ScenarioGenerator):
    """Latent Gaussian copula baseline with target marginals and dependencies."""

    name = "Gaussian copula"

    def __init__(self) -> None:
        self.p_: np.ndarray | None = None
        self.corr_: np.ndarray | None = None
        self.thresholds_: np.ndarray | None = None
        self.targets_: MomentTargets | None = None
        self.max_projection_change_: float | None = None

    def fit(self, spec: SystemSpec) -> None:
        self.targets_ = targets_from_spec(spec)
        self.p_ = np.clip(self.targets_.marginals, 1e-9, 1 - 1e-9)
        self.thresholds_ = norm.ppf(self.p_)
        raw_latent_corr = _calibrate_latent_correlation(
            self.thresholds_,
            self.targets_.pairwise_joint,
        )
        self.corr_ = nearest_psd_correlation(raw_latent_corr)
        self.max_projection_change_ = float(
            np.max(np.abs(self.corr_ - raw_latent_corr))
        )

    def sample(self, n_samples: int, seed: int | None = None) -> np.ndarray:
        require_fitted(self.p_, self.name)
        require_fitted(self.corr_, self.name)
        require_fitted(self.thresholds_, self.name)
        rng = np.random.default_rng(seed)
        latent = rng.multivariate_normal(
            mean=np.zeros(len(self.p_)),
            cov=self.corr_,
            size=n_samples,
            check_valid="ignore",
        )
        return (latent <= self.thresholds_).astype(int)


def _calibrate_latent_correlation(
    thresholds: np.ndarray,
    target_joint: np.ndarray,
) -> np.ndarray:
    """Invert binary co-default probabilities into latent Gaussian correlations."""
    n = len(thresholds)
    latent_corr = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            rho = _solve_latent_pair(
                thresholds[i],
                thresholds[j],
                target_joint[i, j],
            )
            latent_corr[i, j] = latent_corr[j, i] = rho
    return latent_corr


def _solve_latent_pair(
    threshold_i: float,
    threshold_j: float,
    target_joint: float,
) -> float:
    def joint_probability(rho: float) -> float:
        return float(
            multivariate_normal.cdf(
                [threshold_i, threshold_j],
                mean=[0.0, 0.0],
                cov=[[1.0, rho], [rho, 1.0]],
                maxpts=20_000,
                abseps=1e-9,
                releps=1e-9,
                rng=np.random.default_rng(0),
            )
        )

    lower_rho, upper_rho = -0.999, 0.999
    lower_joint = joint_probability(lower_rho)
    upper_joint = joint_probability(upper_rho)
    if target_joint <= lower_joint + 1e-12:
        return lower_rho
    if target_joint >= upper_joint - 1e-12:
        return upper_rho
    return float(
        brentq(
            lambda rho: joint_probability(rho) - target_joint,
            lower_rho,
            upper_rho,
            xtol=1e-10,
            rtol=1e-10,
        )
    )
