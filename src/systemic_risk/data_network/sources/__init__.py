"""Input sources for the data-and-network layer.

- ``roster``         — the real anchor: a curated roster of real systemically important
                       banks (public ratings + balance-sheet scale).
- ``equity_returns`` — equity co-movement correlation from daily price history (Yahoo),
                       with a committed snapshot for reproducible offline runs.
- ``synthetic``      — calibrated-synthetic generator for scaling beyond the real roster
                       (wraps the existing ``make_scalable_system`` up to the 54-qubit target).
"""

from systemic_risk.data_network.sources.roster import RosterRow, load_roster
from systemic_risk.data_network.sources.equity_returns import (
    EquityCorrelation,
    load_or_fetch_correlation,
)
from systemic_risk.data_network.sources.synthetic import synthetic_network_spec

__all__ = [
    "RosterRow",
    "load_roster",
    "EquityCorrelation",
    "load_or_fetch_correlation",
    "synthetic_network_spec",
]
