"""Input sources for the data-and-network layer.

- ``roster``         — the real anchor: a curated roster of real systemically important
                       banks (public ratings + balance-sheet scale).
- ``equity_returns`` — equity co-movement correlation from daily price history (Yahoo),
                       with a committed snapshot for reproducible offline runs.
- ``synthetic``      — calibrated-synthetic generator for scaling beyond the real roster
                       (wraps the existing ``make_scalable_system`` up to the 54-qubit target).
- ``holdings_13f``   — 13F institutional holdings -> portfolio-overlap (common-asset / fire-sale)
                       network: the real-data common-asset contagion channel.
"""

from systemic_risk.data_network.sources.roster import RosterRow, load_roster
from systemic_risk.data_network.sources.equity_returns import (
    EquityCorrelation,
    load_or_fetch_correlation,
)
from systemic_risk.data_network.sources.synthetic import synthetic_network_spec
from systemic_risk.data_network.sources.holdings_13f import (
    ColumnSpec,
    HoldingsMatrix,
    HoldingsPanel,
    cosine_overlap,
    crsp_illiquidity,
    directed_fire_sale_matrix,
    holdings_matrix,
    liquidity_weighted_overlap,
    load_holdings,
    load_illiquidity,
    load_or_sample_holdings,
    rdate_to_yearqtr,
    sample_holdings_csv,
    synthetic_holdings_panel,
    validated_overlap_network,
)

__all__ = [
    "RosterRow",
    "load_roster",
    "EquityCorrelation",
    "load_or_fetch_correlation",
    "synthetic_network_spec",
    # --- 13F portfolio-overlap (common-asset / fire-sale) source ---
    "ColumnSpec",
    "HoldingsPanel",
    "HoldingsMatrix",
    "load_holdings",
    "holdings_matrix",
    "cosine_overlap",
    "validated_overlap_network",
    "liquidity_weighted_overlap",
    "directed_fire_sale_matrix",
    "sample_holdings_csv",
    "load_or_sample_holdings",
    "rdate_to_yearqtr",
    "crsp_illiquidity",
    "load_illiquidity",
    "synthetic_holdings_panel",
]
