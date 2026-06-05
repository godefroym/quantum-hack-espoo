from __future__ import annotations

import numpy as np
from scipy.stats import norm

from systemic_risk.generators.base import ScenarioGenerator, require_fitted
from systemic_risk.spec import SystemSpec
from systemic_risk.utils.validation import nearest_psd_correlation


class GaussianCopulaGenerator(ScenarioGenerator):
    """Latent Gaussian copula baseline with target marginals and dependencies."""

    name = "Gaussian copula"

    def __init__(self) -> None:
        self.p_: np.ndarray | None = None
        self.corr_: np.ndarray | None = None
        self.thresholds_: np.ndarray | None = None

    def fit(self, spec: SystemSpec) -> None:
        self.p_ = np.clip(spec.marginal_default_probs.copy(), 1e-9, 1 - 1e-9)
        raw_corr = spec.target_pairwise_corr
        if raw_corr is None:
            raw_corr = np.eye(spec.n)
        self.corr_ = nearest_psd_correlation(raw_corr)
        self.thresholds_ = norm.ppf(self.p_)

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
