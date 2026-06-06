"""Classical contagion simulator."""

from systemic_risk.simulator.cascade import CascadeResult, run_cascade, simulate_many
from systemic_risk.simulator.huang import (
    HuangCascadeResult,
    huang_failure_probability,
    run_huang_cascade,
    simulate_huang_scenarios,
)

__all__ = [
    "CascadeResult",
    "HuangCascadeResult",
    "huang_failure_probability",
    "run_cascade",
    "run_huang_cascade",
    "simulate_huang_scenarios",
    "simulate_many",
]
