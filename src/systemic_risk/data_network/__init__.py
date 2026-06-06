"""Task A — data and exposure network.

Owns sourcing and shaping the real-world input into the canonical system spec:
nodes (institutions), weighted edges (reconstructed exposures), per-node marginals,
the pairwise correlation matrix, community structure, a documented feature schema,
and provenance.

Pipeline (see the per-module docstrings):

    sources/  -> clean -> estimate -> reconstruct -> cluster -> assemble -> validate

The layered, frozen :class:`NetworkSpec` is the source-of-truth object. It round-trips
losslessly to/from JSON and *assembles down* into the flat
:class:`systemic_risk.spec.SystemSpec` that parts B/C/D (generators, simulator,
evaluation) already consume — so this module blends in without breaking them.
"""

from systemic_risk.data_network.spec import (
    EmpiricalLayer,
    FeatureField,
    FeatureSchema,
    NetworkSpec,
    Provenance,
    ReconstructedLayer,
)
from systemic_risk.data_network.assemble import build_network_spec, build_system_spec

__all__ = [
    "EmpiricalLayer",
    "ReconstructedLayer",
    "FeatureSchema",
    "FeatureField",
    "Provenance",
    "NetworkSpec",
    "build_network_spec",
    "build_system_spec",
]
