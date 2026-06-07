"""The real 28-bank exposure network pipeline.

Sources and shapes real-world input into the canonical system spec: institutions,
reconstructed bilateral exposures, per-node marginals, the equity correlation matrix,
community structure, a documented feature schema, and provenance.

Pipeline (see the per-module docstrings)::

    sources/  -> clean -> estimate -> reconstruct -> cluster -> assemble -> validate

The layered, frozen :class:`NetworkSpec` is the source-of-truth object; it round-trips
losslessly to/from JSON and assembles down into the flat
:class:`systemic_risk.spec.SystemSpec`. Quick start::

    from systemic_risk.data_network import build_system_spec, build_synthetic_system_spec

    spec = build_system_spec()                 # the real 28-bank exposure network
    spec = build_synthetic_system_spec(n=54)   # calibrated-synthetic, scales to 54 qubits
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
from systemic_risk.data_network.stress import (
    QPU_NOISE_FLOOR,
    StressCalibration,
    apply_stress,
    stressed_marginals,
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
    # --- 2008 stress transform (baseline SystemSpec -> crisis SystemSpec) ---
    "apply_stress",
    "stressed_marginals",
    "StressCalibration",
    "QPU_NOISE_FLOOR",
]
