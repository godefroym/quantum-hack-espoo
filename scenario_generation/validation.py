from __future__ import annotations

import numpy as np
from typing import Tuple

from systemic_risk.spec import SystemSpec


def validate_marginals(samples: np.ndarray, spec: SystemSpec) -> Tuple[np.ndarray, np.ndarray]:
    samples = np.asarray(samples, dtype=float)
    sampled = samples.mean(axis=0)
    target = np.asarray(spec.marginal_default_probs, dtype=float)
    return sampled, target


def validate_pairwise_corr(samples: np.ndarray, spec: SystemSpec) -> Tuple[np.ndarray, np.ndarray]:
    samples = np.asarray(samples, dtype=float)
    n_samples, n = samples.shape
    sampled_joint = (samples.T @ samples) / max(n_samples, 1)
    sampled_corr = np.eye(n)
    sampled_marginals = samples.mean(axis=0)
    for i in range(n):
        for j in range(i + 1, n):
            denom = (sampled_marginals[i] * (1 - sampled_marginals[i]) * sampled_marginals[j] * (1 - sampled_marginals[j])) ** 0.5
            corr = 0.0 if denom == 0 else (sampled_joint[i, j] - sampled_marginals[i] * sampled_marginals[j]) / denom
            sampled_corr[i, j] = sampled_corr[j, i] = float(np.clip(corr, -1.0, 1.0))

    # Convert the spec's targets into the Bernoulli pairwise correlation space
    # (the spec may carry a latent Gaussian correlation; use the joint probs API
    # which already handles conversion based on metadata).
    from systemic_risk.spec import joint_to_corr

    target_joint = spec.target_pairwise_joint_probs()
    target = joint_to_corr(target_joint, np.asarray(spec.marginal_default_probs))

    return sampled_corr, target


def validate_schema(samples: np.ndarray, spec: SystemSpec) -> bool:
    samples = np.asarray(samples)
    if samples.ndim != 2:
        return False
    if samples.shape[1] != spec.n:
        return False
    if not np.all(np.isin(samples, [0, 1])):
        return False
    return True


def validate_scenarios(samples: np.ndarray, spec: SystemSpec) -> dict:
    sampled_marginals, target_marginals = validate_marginals(samples, spec)
    sampled_corr, target_corr = validate_pairwise_corr(samples, spec)
    schema_ok = validate_schema(samples, spec)
    return {
        "schema_ok": schema_ok,
        "sampled_marginals": sampled_marginals,
        "target_marginals": target_marginals,
        "sampled_corr": sampled_corr,
        "target_corr": target_corr,
    }
