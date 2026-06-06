"""Synthetic financial-system test instances with PLANTED cluster structure.

The mixture roadmap (partition institutions into clusters that are strongly correlated
WITHIN and weakly correlated ACROSS, sample each cluster independently -- eventually on
separate quantum devices -- and stitch back classically) needs controllable test data
to develop and validate against. This module produces it.

Construction (a multi-factor Gaussian / Vasicek-CreditMetrics threshold model). Each
institution ``i`` belonging to cluster ``c(i)`` carries a latent standard-normal
variable::

    X_i = sqrt(beta_global) * Z_global
        + sqrt(beta_within) * Z_{c(i)}
        + sqrt(1 - beta_global - beta_within) * eps_i

with ``Z_global`` one GLOBAL systemic factor shared across all clusters, ``Z_c`` one
PER-CLUSTER factor, and ``eps_i`` idiosyncratic noise -- all independent standard
normals. Institution ``i`` defaults when ``X_i <= ppf(p_i)`` (the marginal threshold),
so the marginals are exactly ``p_i`` by construction.

The resulting latent (Gaussian) correlation is, for ``i != j``::

    rho_ij = beta_global + beta_within   if c(i) == c(j)   (within-cluster)
           = beta_global                 otherwise          (cross-cluster)

So:

- ``beta_global`` is the SINGLE TUNABLE cross-cluster dial. The cross-cluster block is
  a rank-1 function of this one loading: ``beta_global = 0`` gives fully separable
  clusters (block-diagonal latent correlation -> independent clusters), and raising it
  drives shared systemic co-movement up to a strongly-coupled regime.
- ``beta_within`` is the within-cluster excess loading; within-cluster correlation
  exceeds cross-cluster correlation by exactly ``beta_within > 0`` by construction.

The latent correlation is recorded on a :class:`~systemic_risk.spec.SystemSpec` in the
``latent_gaussian`` correlation space (so the repo's copula/QCBM baselines and the
co-default thresholding utilities interpret it correctly), with the ground-truth cluster
labels stored in ``spec.clusters``. :func:`reference_default_samples` draws ground-truth
binary default vectors from the exact factor model to serve as the reference distribution
later validation steps compare against.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

from systemic_risk.spec import CORRELATION_SPACE_LATENT_GAUSSIAN, SystemSpec


__all__ = [
    "ClusteredSystemConfig",
    "make_clustered_system",
    "planted_latent_correlation",
    "reference_default_samples",
    "cluster_block_means",
]


@dataclass(frozen=True)
class ClusteredSystemConfig:
    """Configuration for a planted-cluster synthetic system.

    Attributes:
        cluster_sizes: Number of members per cluster, e.g. ``[8, 8, 6]``. Each entry
            should be ``<= 20`` (the quantum-device qubit budget per cluster).
        cross_coupling: The cross-cluster dial ``beta_global`` in ``[0, 1]``. The latent
            cross-cluster correlation equals this value; ``0`` -> fully separable clusters.
        within_coupling: The within-cluster excess loading ``beta_within`` in ``(0, 1)``.
            Within-cluster latent correlation is ``cross_coupling + within_coupling`` and
            so always exceeds the cross-cluster correlation by ``within_coupling``.
        marginal_default_prob: Baseline marginal default probability ``p_i``. A small
            amount of per-node heterogeneity is applied deterministically from ``seed``
            unless ``heterogeneous_marginals`` is ``False``.
        heterogeneous_marginals: If ``True`` (default) marginals are jittered per node so
            the instance is not degenerate; if ``False`` every node uses
            ``marginal_default_prob`` exactly.
        seed: Determinism seed for marginal jitter and node naming.
    """

    cluster_sizes: tuple[int, ...]
    cross_coupling: float = 0.05
    within_coupling: float = 0.45
    marginal_default_prob: float = 0.05
    heterogeneous_marginals: bool = True
    seed: int = 0

    def __post_init__(self) -> None:
        if len(self.cluster_sizes) == 0:
            raise ValueError("cluster_sizes must contain at least one cluster")
        if any(size <= 0 for size in self.cluster_sizes):
            raise ValueError("every cluster size must be positive")
        if any(size > 20 for size in self.cluster_sizes):
            raise ValueError("cluster sizes must be <= 20 (the per-device qubit budget)")
        if not 0.0 <= self.cross_coupling <= 1.0:
            raise ValueError("cross_coupling (the dial) must lie in [0, 1]")
        if not 0.0 < self.within_coupling < 1.0:
            raise ValueError("within_coupling must lie in (0, 1)")
        if self.cross_coupling + self.within_coupling >= 1.0:
            raise ValueError(
                "cross_coupling + within_coupling must be < 1 so idiosyncratic "
                "variance stays positive (it equals 1 - cross - within)"
            )
        if not 0.0 < self.marginal_default_prob < 1.0:
            raise ValueError("marginal_default_prob must lie in (0, 1)")

    @property
    def n(self) -> int:
        return int(sum(self.cluster_sizes))

    @property
    def n_clusters(self) -> int:
        return len(self.cluster_sizes)

    def cluster_labels(self) -> np.ndarray:
        """Ground-truth integer cluster label per node, in node order."""
        labels = np.concatenate(
            [np.full(size, c, dtype=int) for c, size in enumerate(self.cluster_sizes)]
        )
        return labels


def planted_latent_correlation(
    labels: np.ndarray,
    cross_coupling: float,
    within_coupling: float,
) -> np.ndarray:
    """Build the planted latent Gaussian correlation matrix from cluster labels.

    Off-diagonal entry is ``cross_coupling + within_coupling`` for same-cluster pairs and
    ``cross_coupling`` for cross-cluster pairs; the diagonal is 1. This is exactly the
    correlation of the two-factor model in the module docstring and is positive
    semidefinite by construction (it is a nonnegative combination of an all-ones global
    block and per-cluster all-ones blocks plus a positive diagonal).
    """
    labels = np.asarray(labels, dtype=int)
    same = labels[:, None] == labels[None, :]
    corr = np.full((labels.size, labels.size), float(cross_coupling))
    corr[same] = cross_coupling + within_coupling
    np.fill_diagonal(corr, 1.0)
    return corr


def _marginals(config: ClusteredSystemConfig) -> np.ndarray:
    p = np.full(config.n, config.marginal_default_prob, dtype=float)
    if config.heterogeneous_marginals:
        rng = np.random.default_rng(config.seed)
        # Multiplicative jitter in roughly [0.7, 1.4]; clipped to a sane PD band.
        factor = rng.uniform(0.7, 1.4, size=config.n)
        p = np.clip(p * factor, 1e-3, 0.5)
    return p


def make_clustered_system(config: ClusteredSystemConfig) -> SystemSpec:
    """Build a :class:`SystemSpec` with planted cluster structure.

    The returned spec carries:

    - ``clusters``: ground-truth cluster label strings (``"cluster_0"`` ...), one per node;
    - ``target_pairwise_corr``: the planted latent Gaussian correlation matrix, in the
      ``latent_gaussian`` correlation space (set via metadata) so the repo's copula / QCBM
      generators and the ``latent_corr_to_joint`` thresholding utilities read it correctly;
    - ``marginal_default_probs``: the per-node marginals (jittered unless disabled);
    - planting parameters and the ground-truth integer labels recorded in ``metadata``.

    There is no exposure graph here (this is a generation/correlation test instance, not a
    contagion instance): ``exposure_matrix`` is zero and ``capital_buffers`` are unit. The
    object is still a first-class, fully-valid ``SystemSpec``.
    """
    labels = config.cluster_labels()
    corr = planted_latent_correlation(labels, config.cross_coupling, config.within_coupling)
    p = _marginals(config)

    n = config.n
    cluster_names = [f"cluster_{int(c)}" for c in labels]
    node_names = [
        f"inst_{i:02d}_c{int(labels[i])}" for i in range(n)
    ]
    node_types = ["bank"] * n

    metadata = {
        "name": "Planted-cluster synthetic system",
        "correlation_space": CORRELATION_SPACE_LATENT_GAUSSIAN,
        "generator": "clustered_synthetic.make_clustered_system",
        "seed": int(config.seed),
        "n": int(n),
        "n_clusters": int(config.n_clusters),
        "cluster_sizes": list(config.cluster_sizes),
        "cluster_labels": labels.tolist(),
        "cross_coupling": float(config.cross_coupling),
        "within_coupling": float(config.within_coupling),
        "within_cluster_latent_corr": float(config.cross_coupling + config.within_coupling),
        "cross_cluster_latent_corr": float(config.cross_coupling),
        "factor_model": (
            "two-factor Gaussian / Vasicek: X_i = sqrt(beta_global) Z_global + "
            "sqrt(beta_within) Z_cluster + sqrt(1-beta_global-beta_within) eps_i, "
            "default when X_i <= ppf(p_i). beta_global is the cross-cluster dial."
        ),
        "description": (
            "Controllable planted-cluster test instance for the classical+quantum "
            "mixture roadmap: strong within-cluster, weak cross-cluster latent "
            "correlation with a single tunable cross-cluster dial."
        ),
    }

    return SystemSpec(
        node_names=node_names,
        node_types=node_types,
        exposure_matrix=np.zeros((n, n), dtype=float),
        capital_buffers=np.ones(n, dtype=float),
        marginal_default_probs=p,
        target_pairwise_corr=corr,
        clusters=cluster_names,
        metadata=metadata,
    )


def reference_default_samples(
    spec: SystemSpec,
    n_samples: int,
    seed: int | None = None,
) -> np.ndarray:
    """Draw ground-truth correlated binary default vectors from a planted-cluster spec.

    Samples the exact two-factor Gaussian model implied by the spec's ground-truth cluster
    labels and planting parameters (read from ``spec.metadata``), thresholding the latent
    variable at each node's marginal. Returns an ``(n_samples, n)`` 0/1 array where a 1 is
    a default. This is the REFERENCE distribution later validation steps (clustering
    recovery, classical stitching of independently-sampled clusters) compare against.

    The model is sampled directly from its factor decomposition rather than via a full
    multivariate normal draw, so the per-cluster and global factors are explicit and the
    same routine extends to sampling clusters independently.
    """
    labels = _labels_from_spec(spec)
    beta_global = float(spec.metadata["cross_coupling"])
    beta_within = float(spec.metadata["within_coupling"])
    p = spec.marginal_default_probs
    thresholds = norm.ppf(np.clip(p, 1e-12, 1.0 - 1e-12))

    rng = np.random.default_rng(seed)
    n = spec.n
    n_clusters = int(labels.max()) + 1 if labels.size else 0

    z_global = rng.standard_normal(size=(n_samples, 1))
    z_cluster = rng.standard_normal(size=(n_samples, n_clusters))
    eps = rng.standard_normal(size=(n_samples, n))

    idio = max(1.0 - beta_global - beta_within, 0.0)
    latent = (
        np.sqrt(beta_global) * z_global
        + np.sqrt(beta_within) * z_cluster[:, labels]
        + np.sqrt(idio) * eps
    )
    return (latent <= thresholds[None, :]).astype(int)


def _labels_from_spec(spec: SystemSpec) -> np.ndarray:
    """Recover ground-truth integer cluster labels from a planted-cluster spec."""
    if "cluster_labels" in spec.metadata:
        return np.asarray(spec.metadata["cluster_labels"], dtype=int)
    if spec.clusters is None:
        raise ValueError("spec has no cluster labels; not a planted-cluster system")
    # Fall back to factorising the cluster name strings, preserving first-seen order.
    order: dict[str, int] = {}
    labels = []
    for name in spec.clusters:
        if name not in order:
            order[name] = len(order)
        labels.append(order[name])
    return np.asarray(labels, dtype=int)


def cluster_block_means(
    matrix: np.ndarray,
    labels: np.ndarray,
) -> tuple[float, float]:
    """Mean of off-diagonal within-cluster and cross-cluster entries of ``matrix``.

    A small diagnostic helper: returns ``(within_mean, cross_mean)`` over off-diagonal
    pairs, used to verify that within-cluster co-movement exceeds cross-cluster co-movement
    and that the dial moves the cross-cluster block. ``matrix`` is typically a measured
    co-default correlation matrix from :func:`reference_default_samples`.
    """
    labels = np.asarray(labels, dtype=int)
    n = matrix.shape[0]
    off = ~np.eye(n, dtype=bool)
    same = (labels[:, None] == labels[None, :]) & off
    cross = (labels[:, None] != labels[None, :]) & off
    within_mean = float(matrix[same].mean()) if np.any(same) else float("nan")
    cross_mean = float(matrix[cross].mean()) if np.any(cross) else float("nan")
    return within_mean, cross_mean
