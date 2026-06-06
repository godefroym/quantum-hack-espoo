from __future__ import annotations

import numpy as np
from typing import Optional

from systemic_risk.spec import SystemSpec
from systemic_risk.generators.base import ScenarioGenerator
from systemic_risk.generators.gaussian_copula import GaussianCopulaGenerator


class GAMGenerator(ScenarioGenerator):
    """Generative Augmentation Model (GAM).

    Strategy (simple, reproducible):
    - Use the Gaussian copula baseline to match target marginals and latent correlations
    - Apply a single-step exposure-aware augmentation: if the initial sampled defaults
      imply losses to an institution that exceed its buffer, mark it defaulted.

    This preserves schema and approximately preserves marginal / pairwise structure
    while injecting exposure-aware stress behaviour.
    """

    name = "GAM generator"

    def __init__(self, augmentation_strength: float = 0.10) -> None:
        """augmentation_strength: probability in [0,1] that an exposure-triggered
        buffer-breach converts into an actual default in the augmented sample.

        Setting this low preserves the Gaussian-copula dependency structure while
        allowing exposure-aware stress to appear occasionally. Default=0.10.
        """
        if not (0.0 <= augmentation_strength <= 1.0):
            raise ValueError("augmentation_strength must lie in [0, 1]")
        self.augmentation_strength = float(augmentation_strength)
        self._copula: Optional[GaussianCopulaGenerator] = None
        self._spec: Optional[SystemSpec] = None

    def fit(self, spec: SystemSpec) -> None:
        self._spec = spec
        gc = GaussianCopulaGenerator()
        gc.fit(spec)
        self._copula = gc

    def sample(self, n_samples: int, seed: int | None = None) -> np.ndarray:
        if self._copula is None or self._spec is None:
            raise RuntimeError("GAMGenerator must be fit before sampling")
        rng = np.random.default_rng(seed)
        base = self._copula.sample(n_samples, seed=seed)

        # compute diagnostics before augmentation
        try:
            pre_diag = self._copula.diagnostics(base)
        except Exception:
            pre_diag = None

        # exposure-aware probabilistic augmentation (reduces correlation drift)
        W = np.asarray(self._spec.exposure_matrix, dtype=float)
        buffers = np.asarray(self._spec.capital_buffers, dtype=float)
        # losses to each institution i for each sample: losses[s,i] = sum_j base[s,j]*W[i,j]
        losses = base @ W.T
        trigger = losses > buffers
        if self.augmentation_strength <= 0.0:
            augmented_mask = np.zeros_like(trigger, dtype=bool)
        elif self.augmentation_strength >= 1.0:
            augmented_mask = trigger.copy()
        else:
            coin = rng.random(size=trigger.shape)
            augmented_mask = trigger & (coin < self.augmentation_strength)

        final = np.clip(base | augmented_mask.astype(int), 0, 1).astype(int)

        # compute diagnostics after augmentation
        try:
            post_diag = self.diagnostics(final)
        except Exception:
            post_diag = None

        # attach diagnostics to object for inspection (not part of ScenarioGenerator API)
        self._last_diagnostics = {
            "pre": pre_diag,
            "post": post_diag,
            "augmentation_strength": self.augmentation_strength,
        }
        return final
