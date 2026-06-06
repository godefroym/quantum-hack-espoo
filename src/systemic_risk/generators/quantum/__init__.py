"""Quantum Born-machine internals and entanglement-layout utilities.

Also hosts the amplitude-estimation *calculation* surface (:mod:`.amplitude_estimation`):
the Grover/Q operator and Maximum-Likelihood Amplitude Estimation, simulated exactly on the
same numpy statevector engine, with hardware-relevant oracle-query accounting.
"""

from systemic_risk.generators.quantum.amplitude_estimation import (
    AmplitudeEstimate,
    GroverOperator,
    QueryComplexityPoint,
    mc_queries_for_relative_error,
    mlae_queries,
    qae_queries_for_relative_error,
    query_complexity_curve,
    run_mlae,
)
from systemic_risk.generators.quantum.layout import (
    ClusterResult,
    PairLink,
    build_clustering_layout,
    build_clustering_layout_from_spec,
    build_dependency_matrix,
    build_entanglement_layers,
    classify_pairs,
    sparsify_entanglement_pairs,
    threshold_connected_components,
)

__all__ = [
    "AmplitudeEstimate",
    "ClusterResult",
    "GroverOperator",
    "PairLink",
    "QueryComplexityPoint",
    "build_clustering_layout",
    "build_clustering_layout_from_spec",
    "build_dependency_matrix",
    "build_entanglement_layers",
    "classify_pairs",
    "mc_queries_for_relative_error",
    "mlae_queries",
    "qae_queries_for_relative_error",
    "query_complexity_curve",
    "run_mlae",
    "sparsify_entanglement_pairs",
    "threshold_connected_components",
]
