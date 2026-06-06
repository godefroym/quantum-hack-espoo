from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from systemic_risk.spec import SystemSpec, joint_to_corr
from systemic_risk.utils.validation import ensure_binary_samples


@dataclass(frozen=True)
class MomentTargets:
    """Shared first- and second-order targets for every scenario generator."""

    marginals: np.ndarray
    pairwise_joint: np.ndarray
    pairwise_corr: np.ndarray

    @property
    def n(self) -> int:
        return len(self.marginals)

    @property
    def off_diagonal_mask(self) -> np.ndarray:
        return ~np.eye(self.n, dtype=bool)


@dataclass(frozen=True)
class MomentErrors:
    marginal_rmse: float
    pairwise_joint_rmse: float
    pairwise_corr_rmse: float


def targets_from_spec(spec: SystemSpec) -> MomentTargets:
    """Resolve one canonical moment target representation from a SystemSpec."""
    marginals = spec.marginal_default_probs.copy()
    pairwise_joint = spec.target_pairwise_joint_probs()
    pairwise_corr = joint_to_corr(pairwise_joint, marginals)
    np.fill_diagonal(pairwise_corr, 1.0)
    return MomentTargets(
        marginals=marginals,
        pairwise_joint=pairwise_joint,
        pairwise_corr=pairwise_corr,
    )


def empirical_moments(samples: np.ndarray) -> MomentTargets:
    samples = ensure_binary_samples(samples)
    n_samples = len(samples)
    marginals = samples.mean(axis=0)
    pairwise_joint = (samples.T @ samples) / max(n_samples, 1)
    pairwise_corr = joint_to_corr(pairwise_joint, marginals)
    np.fill_diagonal(pairwise_corr, 1.0)
    return MomentTargets(
        marginals=marginals,
        pairwise_joint=pairwise_joint,
        pairwise_corr=pairwise_corr,
    )


def moment_errors(samples: np.ndarray, targets: MomentTargets) -> MomentErrors:
    observed = empirical_moments(samples)
    if observed.n != targets.n:
        raise ValueError("samples and targets must have the same number of entities")
    mask = targets.off_diagonal_mask
    return MomentErrors(
        marginal_rmse=float(
            np.sqrt(np.mean((observed.marginals - targets.marginals) ** 2))
        ),
        pairwise_joint_rmse=float(
            np.sqrt(
                np.mean(
                    (
                        observed.pairwise_joint[mask]
                        - targets.pairwise_joint[mask]
                    )
                    ** 2
                )
            )
        ),
        pairwise_corr_rmse=float(
            np.sqrt(
                np.mean(
                    (
                        observed.pairwise_corr[mask]
                        - targets.pairwise_corr[mask]
                    )
                    ** 2
                )
            )
        ),
    )
