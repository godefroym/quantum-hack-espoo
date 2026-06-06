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

Quick start for B/C/D (return a ready-to-use flat ``SystemSpec``)::

    from systemic_risk.data_network import build_system_spec, build_synthetic_system_spec

    spec = build_system_spec()                 # the REAL 28-bank exposure network
    spec = build_synthetic_system_spec(n=54)   # calibrated-synthetic (scales to 54 qubits)
"""

from systemic_risk.data_network.spec import (
    EmpiricalLayer,
    FeatureField,
    FeatureSchema,
    NetworkSpec,
    Provenance,
    ReconstructedLayer,
)
from systemic_risk.data_network.assemble import (
    build_network_spec,
    build_synthetic_system_spec,
    build_system_spec,
)

__all__ = [
    "EmpiricalLayer",
    "ReconstructedLayer",
    "FeatureSchema",
    "FeatureField",
    "Provenance",
    "NetworkSpec",
    # --- the two B/C/D entrypoints (return a flat SystemSpec) ---
    "build_system_spec",            # real bank network
    "build_synthetic_system_spec",  # calibrated-synthetic, scales to n=54
    # --- the layered builder (returns a NetworkSpec) ---
    "build_network_spec",
]
