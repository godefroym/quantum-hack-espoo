from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import norm

from systemic_risk.utils.validation import nearest_psd_correlation


CORRELATION_SPACE_BINARY_DEFAULT = "binary_default"
CORRELATION_SPACE_LATENT_GAUSSIAN = "latent_gaussian"
VALID_CORRELATION_SPACES = {
    CORRELATION_SPACE_BINARY_DEFAULT,
    CORRELATION_SPACE_LATENT_GAUSSIAN,
}


def _as_float_array(value: Any, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


@dataclass
class SystemSpec:
    """Canonical financial system specification.

    W[i, j] is the loss to institution i if institution j defaults.
    """

    node_names: list[str]
    node_types: list[str]
    exposure_matrix: np.ndarray
    capital_buffers: np.ndarray
    marginal_default_probs: np.ndarray
    target_pairwise_corr: np.ndarray | None = None
    target_joint_probs: np.ndarray | None = None
    clusters: list[str] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.node_names = list(self.node_names)
        self.node_types = list(self.node_types)
        self.exposure_matrix = _as_float_array(self.exposure_matrix, "exposure_matrix")
        self.capital_buffers = _as_float_array(self.capital_buffers, "capital_buffers")
        self.marginal_default_probs = _as_float_array(
            self.marginal_default_probs, "marginal_default_probs"
        )
        if self.target_pairwise_corr is not None:
            self.target_pairwise_corr = _as_float_array(
                self.target_pairwise_corr, "target_pairwise_corr"
            )
        if self.target_joint_probs is not None:
            self.target_joint_probs = _as_float_array(
                self.target_joint_probs, "target_joint_probs"
            )
        if self.clusters is not None:
            self.clusters = list(self.clusters)
        self.metadata = dict(self.metadata)
        self.validate()

    @property
    def n(self) -> int:
        return len(self.node_names)

    @property
    def correlation_space(self) -> str:
        """Meaning of ``target_pairwise_corr``.

        Specs written before the contract was introduced default to binary
        default-event correlation for backward compatibility.
        """
        return str(
            self.metadata.get(
                "correlation_space",
                CORRELATION_SPACE_BINARY_DEFAULT,
            )
        )

    def validate(self) -> None:
        n = len(self.node_names)
        if n == 0:
            raise ValueError("SystemSpec must contain at least one node")
        if len(self.node_types) != n:
            raise ValueError("node_types must have the same length as node_names")
        if self.clusters is not None and len(self.clusters) != n:
            raise ValueError("clusters must have the same length as node_names")
        if self.exposure_matrix.shape != (n, n):
            raise ValueError("exposure_matrix must have shape (n, n)")
        if self.capital_buffers.shape != (n,):
            raise ValueError("capital_buffers must have shape (n,)")
        if self.marginal_default_probs.shape != (n,):
            raise ValueError("marginal_default_probs must have shape (n,)")
        if np.any(self.exposure_matrix < 0):
            raise ValueError("exposure_matrix must be nonnegative")
        if np.any(np.diag(self.exposure_matrix) != 0):
            raise ValueError("exposure_matrix diagonal must be zero")
        if np.any(self.capital_buffers < 0):
            raise ValueError("capital_buffers must be nonnegative")
        if np.any((self.marginal_default_probs < 0) | (self.marginal_default_probs > 1)):
            raise ValueError("marginal_default_probs must lie in [0, 1]")
        if self.correlation_space not in VALID_CORRELATION_SPACES:
            raise ValueError(
                "metadata['correlation_space'] must be one of "
                f"{sorted(VALID_CORRELATION_SPACES)}"
            )
        if self.target_pairwise_corr is not None:
            if self.target_pairwise_corr.shape != (n, n):
                raise ValueError("target_pairwise_corr must have shape (n, n)")
            if not np.allclose(self.target_pairwise_corr, self.target_pairwise_corr.T):
                raise ValueError("target_pairwise_corr must be symmetric")
            if np.any((self.target_pairwise_corr < -1) | (self.target_pairwise_corr > 1)):
                raise ValueError("target_pairwise_corr values must lie in [-1, 1]")
            if not np.allclose(np.diag(self.target_pairwise_corr), 1):
                raise ValueError("target_pairwise_corr diagonal must be one")
        if self.target_joint_probs is not None:
            if self.target_joint_probs.shape != (n, n):
                raise ValueError("target_joint_probs must have shape (n, n)")
            if np.any((self.target_joint_probs < 0) | (self.target_joint_probs > 1)):
                raise ValueError("target_joint_probs must lie in [0, 1]")
            if not np.allclose(np.diag(self.target_joint_probs), self.marginal_default_probs):
                raise ValueError("target_joint_probs diagonal must equal marginal probabilities")

    def dependency_matrix(self) -> np.ndarray:
        """Return a symmetric dependency score matrix with diagonal zero."""
        if self.target_pairwise_corr is not None:
            if self.correlation_space == CORRELATION_SPACE_LATENT_GAUSSIAN:
                dep = joint_to_corr(
                    self.target_pairwise_joint_probs(),
                    self.marginal_default_probs,
                )
            else:
                dep = self.target_pairwise_corr.copy()
        elif self.target_joint_probs is not None:
            dep = joint_to_corr(self.target_joint_probs, self.marginal_default_probs)
        else:
            dep = np.eye(self.n)
        np.fill_diagonal(dep, 0.0)
        return dep

    def target_pairwise_joint_probs(self) -> np.ndarray:
        """Return target P(default_i and default_j) with diagonal P(default_i)."""
        if self.target_joint_probs is not None:
            return self.target_joint_probs.copy()
        if self.target_pairwise_corr is None:
            joint = np.outer(
                self.marginal_default_probs,
                self.marginal_default_probs,
            )
            np.fill_diagonal(joint, self.marginal_default_probs)
            return joint
        if self.correlation_space == CORRELATION_SPACE_LATENT_GAUSSIAN:
            return latent_corr_to_joint(
                self.target_pairwise_corr,
                self.marginal_default_probs,
            )
        return corr_to_joint(self.target_pairwise_corr, self.marginal_default_probs)

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_names": self.node_names,
            "node_types": self.node_types,
            "exposure_matrix": self.exposure_matrix.tolist(),
            "capital_buffers": self.capital_buffers.tolist(),
            "marginal_default_probs": self.marginal_default_probs.tolist(),
            "target_pairwise_corr": None
            if self.target_pairwise_corr is None
            else self.target_pairwise_corr.tolist(),
            "target_joint_probs": None
            if self.target_joint_probs is None
            else self.target_joint_probs.tolist(),
            "clusters": self.clusters,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SystemSpec:
        return cls(**data)

    def save_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load_json(cls, path: str | Path) -> SystemSpec:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def save_npz(self, path: str | Path) -> None:
        np.savez_compressed(path, spec_json=json.dumps(self.to_dict()))

    @classmethod
    def load_npz(cls, path: str | Path) -> SystemSpec:
        with np.load(path, allow_pickle=False) as data:
            return cls.from_dict(json.loads(str(data["spec_json"])))


def corr_to_joint(corr: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Convert Bernoulli correlations to feasible co-default probabilities."""
    corr = np.asarray(corr, dtype=float)
    p = np.asarray(p, dtype=float)
    scale = np.sqrt(np.outer(p * (1 - p), p * (1 - p)))
    candidate = np.outer(p, p) + corr * scale
    lower = np.maximum(0.0, p[:, None] + p[None, :] - 1.0)  # Frechet bounds
    upper = np.minimum(p[:, None], p[None, :])
    joint = np.clip(candidate, lower, upper)
    np.fill_diagonal(joint, p)
    return joint


def latent_corr_to_joint(corr: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Threshold a latent Gaussian correlation into co-default probabilities."""
    p = np.asarray(p, dtype=float)
    corr = nearest_psd_correlation(np.asarray(corr, dtype=float))
    thresholds = norm.ppf(np.clip(p, 1e-12, 1.0 - 1e-12))
    rho = np.clip(corr, -0.999, 0.999)
    joint = _bivariate_normal_cdf(
        thresholds[:, None], thresholds[None, :], rho
    )
    # Degenerate marginals: collapse to the deterministic Frechet value.
    zero = (p <= 0.0)[:, None] | (p <= 0.0)[None, :]
    joint = np.where(zero, 0.0, joint)
    one_i = (p >= 1.0)[:, None] & np.ones_like(joint, dtype=bool)
    joint = np.where(one_i, np.broadcast_to(p[None, :], joint.shape), joint)
    one_j = np.ones_like(joint, dtype=bool) & (p >= 1.0)[None, :]
    joint = np.where(one_j, np.broadcast_to(p[:, None], joint.shape), joint)
    joint = (joint + joint.T) / 2.0
    np.fill_diagonal(joint, p)
    return joint


def joint_to_corr(joint: np.ndarray, p: np.ndarray) -> np.ndarray:
    joint = np.asarray(joint, dtype=float)
    p = np.asarray(p, dtype=float)
    denom = np.sqrt(np.outer(p * (1 - p), p * (1 - p)))
    corr = np.divide(
        joint - np.outer(p, p), denom, out=np.zeros_like(joint), where=denom > 0
    )
    corr = np.clip(corr, -1.0, 1.0)
    np.fill_diagonal(corr, 1.0)
    return corr


def _bivariate_normal_cdf(h: np.ndarray, k: np.ndarray, r: np.ndarray, n_nodes: int = 60) -> np.ndarray:
    """Vectorized standard bivariate normal CDF ``P(X<=h, Y<=k)`` with correlation ``r``.

    Gauss-Legendre quadrature of the Drezner-Wesolowsky identity
    ``Phi2(h,k;r) = Phi(h) Phi(k) + integral_0^r phi2(h,k;t) dt``; matches
    ``scipy.stats.multivariate_normal.cdf`` to machine precision.
    """
    h, k, r = np.broadcast_arrays(h, k, np.asarray(r, dtype=float))
    nodes, weights = np.polynomial.legendre.leggauss(n_nodes)
    t = r[..., None] * (nodes + 1.0) / 2.0
    one_minus_t2 = 1.0 - t**2
    density = np.exp(
        -(h[..., None] ** 2 + k[..., None] ** 2 - 2.0 * t * h[..., None] * k[..., None])
        / (2.0 * one_minus_t2)
    ) / (2.0 * np.pi * np.sqrt(one_minus_t2))
    integral = np.sum(weights * density * (r[..., None] / 2.0), axis=-1)
    return norm.cdf(h) * norm.cdf(k) + integral
