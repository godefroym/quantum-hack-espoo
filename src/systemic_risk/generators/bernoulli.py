from __future__ import annotations

import numpy as np

from systemic_risk.generators.base import ScenarioGenerator, require_fitted
from systemic_risk.spec import SystemSpec


class BernoulliGenerator(ScenarioGenerator):
    """Independent default generator using the target marginal probabilities."""

    name = "Bernoulli"

    def __init__(self) -> None:
        self.p_: np.ndarray | None = None

    def fit(self, spec: SystemSpec) -> None:
        self.p_ = spec.marginal_default_probs.copy()

    def sample(self, n_samples: int, seed: int | None = None) -> np.ndarray:
        require_fitted(self.p_, self.name)
        rng = np.random.default_rng(seed)
        return (rng.random((n_samples, len(self.p_))) < self.p_).astype(int)
