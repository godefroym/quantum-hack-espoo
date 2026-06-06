"""Public package interface for contagion."""

from contagion.clustering import (
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
from contagion.simulator import CascadeResult, run_cascade
from contagion.spec import (
    Scenario,
    SystemSpec,
    validate_scenario,
    validate_system_spec,
)
from contagion.toy_networks import (
    create_no_exposure_network,
    create_star_network,
    create_toy_chain_network,
)

__all__ = [
    "ClusterResult",
    "PairLink",
    "build_clustering_layout",
    "build_clustering_layout_from_spec",
    "build_dependency_matrix",
    "build_entanglement_layers",
    "classify_pairs",
    "sparsify_entanglement_pairs",
    "threshold_connected_components",
    "CascadeResult",
    "run_cascade",
    "Scenario",
    "SystemSpec",
    "validate_scenario",
    "validate_system_spec",
    "create_no_exposure_network",
    "create_toy_chain_network",
    "create_star_network",
]
