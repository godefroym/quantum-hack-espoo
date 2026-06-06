"""Classical reconciliation of independent per-cluster samples into a global joint.

Independent per-cluster samples carry the EXACT within-cluster default law but ZERO
cross-cluster co-movement. The systemically dangerous tail -- many clusters defaulting at once
-- lives entirely in that missing co-movement. This module rebuilds it classically.

Mechanism: a **one-factor common-shock rank coupler**.

For every global scenario ``s`` and every cluster ``c`` we draw a latent *stress score*::

    S_c[s] = sqrt(beta) * Z[s] + sqrt(1 - beta) * eps_c[s]

with one shared standard-normal crisis factor ``Z`` and independent per-cluster noise
``eps_c`` (all standard normals). The shared ``Z`` is the "crisis latent" that makes the
clusters co-move; ``beta in [0, 1)`` is the single coupling loading. We then pick, for cluster
``c`` in scenario ``s``, one of that cluster's OWN samples at the empirical quantile
``Phi(S_c[s])`` of its default-count ordering -- a more stressed score selects a
more-defaults sample.

Two honest properties of this construction:

* **Within-cluster structure is untouched / exact.** Each cluster scenario is one of the
  cluster's own measured samples, chosen by quantile. Over many scenarios the selection is
  uniform over the pool, so the cluster's marginal joint default law is reproduced exactly --
  whatever the quantum loader (or hardware) produced.
* **Cross-cluster structure is a classical approximation.** Co-movement enters only through the
  shared scalar ``Z`` acting on each cluster's *aggregate* severity (a one-factor Gaussian
  copula on cluster-aggregate stress), so it is a rank-one, second-order reconstruction of the
  dropped cross-cluster boundary -- it cannot, and does not claim to, reproduce cross-cluster
  *entanglement* (LOCC no-go). The common shock is exactly the mechanism that can CREATE an
  all-clusters-default tail atom that never appears in any independent sample set.

The single loading ``beta`` is fit from the system's own cross-cluster co-default *target*
(or from measured cross-cluster moments) -- never read off any ground-truth planting -- by a
monotone bisection that drives the reconciled mean cross-cluster co-default correlation onto
the target.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.stats import norm

from systemic_risk.mixture.cluster_loader import ClusterSamples
from systemic_risk.spec import SystemSpec
from systemic_risk.utils.validation import ensure_binary_samples


def cross_cluster_corr_target(
    spec: SystemSpec,
    labels: np.ndarray,
) -> float:
    """Mean off-diagonal CROSS-cluster co-default (binary) correlation the spec targets.

    Reads the spec's own target pairwise co-default correlation
    (:meth:`SystemSpec.dependency_matrix`, which thresholds the latent-Gaussian target into the
    binary-default space) and averages the entries that straddle two different clusters under
    ``labels``. This is the system's *stated* cross-cluster co-movement target -- the quantity
    the classical coupler is fit to -- and is computed from the spec, not from any planted
    ground-truth sample set. ``labels`` are the DISCOVERED cluster labels (one per node).
    """
    dep = spec.dependency_matrix()
    return _mean_cross_block(dep, np.asarray(labels, dtype=int))


def independent_global_samples(clusters: list[ClusterSamples], n: int) -> np.ndarray:
    """The naive baseline: concatenate the independent per-cluster samples, no coupling.

    Builds an ``(n_samples, n)`` global 0/1 matrix by placing each cluster's own samples into
    its global columns, row-aligned. Cross-cluster co-movement is whatever the independent draws
    happened to share -- i.e. essentially zero. This is the baseline the reconciliation must
    beat. All clusters must carry the same number of samples.
    """
    n_samples = _common_sample_count(clusters)
    out = np.zeros((n_samples, n), dtype=int)
    for cluster in clusters:
        out[:, list(cluster.members)] = cluster.samples
    return out


@dataclass
class ReconciliationResult:
    """Output of :meth:`CommonShockReconciler.reconcile`."""

    samples: np.ndarray
    beta: float
    target_cross_corr: float
    achieved_cross_corr: float
    n_clusters: int
    cluster_sources: list[str] = field(default_factory=list)

    @property
    def n_samples(self) -> int:
        return int(self.samples.shape[0])


class CommonShockReconciler:
    """Stitch independent per-cluster samples into a coupled global joint default matrix.

    ``n`` is the global institution count and ``labels`` the discovered cluster label per node.
    Call :meth:`fit` with a cross-cluster co-default *target* (a scalar correlation, e.g. from
    :func:`cross_cluster_corr_target`) to solve the one-factor loading ``beta``; then
    :meth:`reconcile` to produce coupled global samples. :meth:`fit_reconcile` does both.

    The reconciler is agnostic to where the per-cluster samples came from -- statevector loader
    or real device -- because it only ever touches :class:`ClusterSamples`.
    """

    def __init__(self, n: int, labels: np.ndarray) -> None:
        self.n = int(n)
        self.labels = np.asarray(labels, dtype=int)
        if self.labels.shape != (self.n,):
            raise ValueError("labels must have one entry per global node")
        self.beta_: float | None = None
        self.target_cross_corr_: float | None = None

    # ------------------------------------------------------------------- fitting
    def fit(
        self,
        clusters: list[ClusterSamples],
        target_cross_corr: float,
        *,
        seed: int | None = 0,
        fit_samples: int = 20_000,
        tol: float = 1e-3,
        max_iter: int = 40,
    ) -> "CommonShockReconciler":
        """Solve the one-factor loading ``beta`` matching ``target_cross_corr``.

        The reconciled mean cross-cluster co-default correlation is monotone increasing in
        ``beta`` (more shared crisis factor -> more co-movement), so a bisection on
        ``beta in [0, 0.999]`` converges. Each trial reconciles a modest ``fit_samples``-row
        subsample and measures the achieved cross-cluster correlation. ``target_cross_corr``
        comes from the spec's own target or from measured cross-cluster moments -- not from any
        planted ground truth.
        """
        self.target_cross_corr_ = float(target_cross_corr)
        if len(self._active_clusters(clusters)) < 2 or target_cross_corr <= 0.0:
            # Nothing to couple (single cluster) or no positive co-movement targeted.
            self.beta_ = 0.0
            return self

        rng = np.random.default_rng(seed)

        def achieved(beta: float) -> float:
            samples = self._reconcile(clusters, beta, rng, fit_samples)
            return _mean_cross_block(_binary_corr(samples), self.labels)

        lo, hi = 0.0, 0.999
        a_lo = achieved(lo)
        a_hi = achieved(hi)
        if target_cross_corr <= a_lo + tol:
            self.beta_ = lo
            return self
        if target_cross_corr >= a_hi - tol:
            self.beta_ = hi
            return self
        for _ in range(max_iter):
            mid = 0.5 * (lo + hi)
            a_mid = achieved(mid)
            if abs(a_mid - target_cross_corr) <= tol:
                self.beta_ = mid
                return self
            if a_mid < target_cross_corr:
                lo = mid
            else:
                hi = mid
        self.beta_ = 0.5 * (lo + hi)
        return self

    # ---------------------------------------------------------------- reconciling
    def reconcile(
        self,
        clusters: list[ClusterSamples],
        n_samples: int,
        *,
        seed: int | None = None,
    ) -> ReconciliationResult:
        """Produce ``n_samples`` coupled global default scenarios. Requires :meth:`fit` first."""
        if self.beta_ is None:
            raise RuntimeError("call fit (or fit_reconcile) before reconcile")
        rng = np.random.default_rng(seed)
        samples = self._reconcile(clusters, self.beta_, rng, n_samples)
        achieved = _mean_cross_block(_binary_corr(samples), self.labels)
        return ReconciliationResult(
            samples=samples,
            beta=float(self.beta_),
            target_cross_corr=float(self.target_cross_corr_ or 0.0),
            achieved_cross_corr=float(achieved),
            n_clusters=len(self._active_clusters(clusters)),
            cluster_sources=[c.source for c in clusters],
        )

    def fit_reconcile(
        self,
        clusters: list[ClusterSamples],
        target_cross_corr: float,
        n_samples: int,
        *,
        seed: int | None = 0,
        fit_samples: int = 20_000,
    ) -> ReconciliationResult:
        """Convenience: :meth:`fit` then :meth:`reconcile` with the same target."""
        self.fit(clusters, target_cross_corr, seed=seed, fit_samples=fit_samples)
        return self.reconcile(clusters, n_samples, seed=seed)

    # --------------------------------------------------------------------- engine
    def _reconcile(
        self,
        clusters: list[ClusterSamples],
        beta: float,
        rng: np.random.Generator,
        n_samples: int,
    ) -> np.ndarray:
        """Core common-shock rank coupling (see module docstring).

        Draw a shared crisis factor ``Z`` and per-cluster stress ``S_c = sqrt(beta) Z +
        sqrt(1-beta) eps_c``; for each cluster pick, per scenario, one of its own samples at the
        default-count quantile ``Phi(S_c)``. The within-cluster law of each picked column block
        is exactly the cluster's measured law.
        """
        beta = float(np.clip(beta, 0.0, 0.999))
        out = np.zeros((n_samples, self.n), dtype=int)
        z = rng.standard_normal(n_samples)
        sqrt_b = np.sqrt(beta)
        sqrt_1mb = np.sqrt(1.0 - beta)
        for cluster in clusters:
            pool = cluster.samples
            m = pool.shape[0]
            if m == 0:
                continue
            order = _severity_order(pool, rng)
            sorted_pool = pool[order]
            eps = rng.standard_normal(n_samples)
            stress = sqrt_b * z + sqrt_1mb * eps
            # Map the standard-normal stress to a position in [0, m) via its Gaussian quantile.
            # High stress -> high severity rank -> more defaults; shared Z couples clusters.
            quantile = norm.cdf(stress)
            picks = np.clip((quantile * m).astype(int), 0, m - 1)
            out[:, list(cluster.members)] = sorted_pool[picks]
        return out

    def _active_clusters(self, clusters: list[ClusterSamples]) -> list[ClusterSamples]:
        return [c for c in clusters if c.size > 0 and c.n_samples > 0]


# --------------------------------------------------------------------- helpers
def _severity_order(pool: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Indices sorting a cluster's samples ascending by default count, ties broken randomly.

    Random (not stable) tie-breaking matters: many samples share a default count, and a stable
    sort would map equal-count strings to a fixed sub-order, biasing WHICH equal-count string a
    given quantile selects. A random shuffle within a count band keeps the quantile->sample map
    a faithful (measure-preserving) reshuffle of the cluster's own law.
    """
    counts = pool.sum(axis=1)
    jitter = rng.random(counts.shape[0])
    return np.lexsort((jitter, counts))


def _mean_cross_block(matrix: np.ndarray, labels: np.ndarray) -> float:
    """Mean of entries of ``matrix`` whose row/col belong to different clusters (off-diagonal)."""
    n = matrix.shape[0]
    cross = (labels[:, None] != labels[None, :]) & ~np.eye(n, dtype=bool)
    return float(matrix[cross].mean()) if np.any(cross) else 0.0


def _binary_corr(samples: np.ndarray) -> np.ndarray:
    """Pearson co-default correlation of a 0/1 sample matrix (diagonal = 1)."""
    samples = ensure_binary_samples(samples).astype(float)
    n_samples = samples.shape[0]
    marg = samples.mean(axis=0)
    cov = (samples.T @ samples) / max(n_samples, 1) - np.outer(marg, marg)
    scale = np.sqrt(np.clip(marg * (1.0 - marg), 0.0, None))
    denom = np.outer(scale, scale)
    with np.errstate(divide="ignore", invalid="ignore"):
        corr = np.where(denom > 0, cov / denom, 0.0)
    np.fill_diagonal(corr, 1.0)
    return np.clip(corr, -1.0, 1.0)


def _common_sample_count(clusters: list[ClusterSamples]) -> int:
    counts = {c.n_samples for c in clusters if c.size > 0}
    if len(counts) > 1:
        raise ValueError("all clusters must carry the same number of samples")
    return counts.pop() if counts else 0
