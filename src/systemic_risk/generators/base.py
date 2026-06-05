from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from systemic_risk.spec import SystemSpec
from systemic_risk.utils.validation import ensure_binary_samples


@dataclass
class GeneratorDiagnostics:
    sampled_marginals: np.ndarray
    sampled_pairwise_joint: np.ndarray
    sampled_pairwise_corr: np.ndarray
    n_unique_scenarios: int


class ScenarioGenerator(ABC):
    """Common interface for binary initial-default scenario generators."""

    name = "scenario_generator"

    @abstractmethod
    def fit(self, spec: SystemSpec) -> None:
        raise NotImplementedError

    @abstractmethod
    def sample(self, n_samples: int, seed: int | None = None) -> np.ndarray:
        raise NotImplementedError

    def diagnostics(self, samples: np.ndarray) -> GeneratorDiagnostics:
        samples = ensure_binary_samples(samples)
        return sample_diagnostics(samples)


def sample_diagnostics(samples: np.ndarray) -> GeneratorDiagnostics:
    samples = ensure_binary_samples(samples)
    n_samples, n = samples.shape
    marginals = samples.mean(axis=0)
    pairwise_joint = (samples.T @ samples) / max(n_samples, 1)
    pairwise_corr = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            denom = np.sqrt(
                marginals[i]
                * (1 - marginals[i])
                * marginals[j]
                * (1 - marginals[j])
            )
            corr = 0.0 if denom == 0 else (pairwise_joint[i, j] - marginals[i] * marginals[j]) / denom
            pairwise_corr[i, j] = pairwise_corr[j, i] = float(np.clip(corr, -1.0, 1.0))
    unique = np.unique(samples, axis=0).shape[0]
    return GeneratorDiagnostics(
        sampled_marginals=marginals,
        sampled_pairwise_joint=pairwise_joint,
        sampled_pairwise_corr=pairwise_corr,
        n_unique_scenarios=unique,
    )


def require_fitted(value: object | None, generator_name: str) -> None:
    if value is None:
        raise RuntimeError(f"{generator_name} must be fit before sampling")
