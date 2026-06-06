"""Classical contagion simulator."""

from systemic_risk.simulator.cascade import (
    CascadeOutcome,
    CascadeResult,
    CascadeScenario,
    run_cascade,
    scenario_from_binary_vector,
    scenario_from_loss_vector,
    scenario_from_named_shocks,
    simulate_many,
)
from systemic_risk.simulator.huang import (
    HuangCascadeResult,
    huang_failure_probability,
    run_huang_cascade,
    simulate_huang_scenarios,
)

__all__ = [
    "CascadeOutcome",
    "CascadeResult",
    "CascadeScenario",
    "HuangCascadeResult",
    "huang_failure_probability",
    "run_cascade",
    "run_huang_cascade",
    "scenario_from_binary_vector",
    "scenario_from_loss_vector",
    "scenario_from_named_shocks",
    "simulate_huang_scenarios",
    "simulate_many",
]
