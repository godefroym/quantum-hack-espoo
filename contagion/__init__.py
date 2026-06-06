"""Public package interface for contagion."""

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
