"""Classical + quantum MIXTURE pipeline: sample clusters independently, stitch globally.

The roadmap this package implements: partition institutions into clusters that each fit a
small quantum device (``<= max_cluster_size`` qubits), load and sample every cluster's
correlated default distribution **independently** (one small circuit per cluster, eventually
one real device per cluster), then **reconcile classically** -- because independent
per-cluster samples carry *no* cross-cluster co-movement, and the systemically dangerous
"many clusters default together" tail lives precisely in that co-movement.

Honest division of labour:

* **Within-cluster** structure comes from the (quantum) per-cluster sampler and is exact --
  the per-cluster joint default law is whatever the loader / hardware produced, untouched.
* **Cross-cluster** structure is a CLASSICAL reconstruction: a one-factor common-shock
  coupler whose single loading is fit to the system's *own* cross-cluster co-default target
  (or to measured cross-cluster moments), never read off any ground-truth planting. The error
  is second order in the dropped cross-cluster covariance and is reported honestly. No
  cross-cluster entanglement is created (LOCC no-go); we do not claim any.

Public surface:

* :class:`ClusterSamples` -- a source-agnostic container of one cluster's bitstrings. Build it
  from the statevector loader (:func:`sample_cluster_statevector`) now, or from measured
  hardware bitstrings (:func:`cluster_samples_from_bitstrings`) later -- the reconciler does
  not care which.
* :class:`CommonShockReconciler` -- fits the one-factor coupling to a cross-cluster co-default
  target and stitches independent :class:`ClusterSamples` into a global joint default matrix.
* :func:`independent_global_samples` -- the naive baseline (concatenate independent clusters,
  zero cross-cluster co-movement) the reconciliation is measured against.
"""

from systemic_risk.mixture.cluster_loader import (
    ClusterSamples,
    cluster_samples_from_bitstrings,
    sample_cluster_statevector,
    sample_clusters_statevector,
)
from systemic_risk.mixture.pipeline import (
    attach_cluster_exposures,
    cascade_loss_cvar,
    default_count_distribution,
    kl_divergence,
    reconciliation_diagnostics,
    tail_count_l1,
    total_variation,
)
from systemic_risk.mixture.reconcile import (
    CommonShockReconciler,
    ReconciliationResult,
    cross_cluster_corr_target,
    independent_global_samples,
)

__all__ = [
    "ClusterSamples",
    "CommonShockReconciler",
    "ReconciliationResult",
    "attach_cluster_exposures",
    "cascade_loss_cvar",
    "cluster_samples_from_bitstrings",
    "cross_cluster_corr_target",
    "default_count_distribution",
    "independent_global_samples",
    "kl_divergence",
    "reconciliation_diagnostics",
    "sample_cluster_statevector",
    "sample_clusters_statevector",
    "tail_count_l1",
    "total_variation",
]
