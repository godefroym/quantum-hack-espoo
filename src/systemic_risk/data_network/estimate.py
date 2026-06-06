"""Raw clean nodes -> empirical marginals, correlation, and balance-sheet constraints.

Part A owns the empirical layer:

- **marginals ``p_i``** from the whole-letter rating bucket via the committed Moody's
  1-year PD table (``data/external/ratings/moodys_pd_by_rating.csv``, Exhibit 17);
- **correlation matrix** from the real equity-return co-movements (projected to the nearest
  PSD correlation so the copula baselines can use it as a latent covariance);
- **interbank totals** (asset row-sums and liability col-sums) and **capital buffers**
  from total assets via Basel-style ratios — the constraints the reconstruction consumes.
"""

from __future__ import annotations

import numpy as np

from systemic_risk.data_network.clean import CleanNode
from systemic_risk.data_network.sources.equity_returns import EquityCorrelation
from systemic_risk.utils.ratings import RATING_PD_DEFAULT, load_rating_pd
from systemic_risk.utils.validation import nearest_psd_correlation


def marginals_from_ratings(nodes: tuple[CleanNode, ...]) -> tuple[np.ndarray, str]:
    """Map each node's rating bucket to its 1-year PD. Returns ``(p, pd_source)``."""
    table, source = load_rating_pd()
    p = np.array([table.get(node.rating_bucket, RATING_PD_DEFAULT["BBB"]) for node in nodes],
                 dtype=float)
    return np.clip(p, 1e-5, 1.0 - 1e-9), source


def correlation_from_equity(
    nodes: tuple[CleanNode, ...], ec: EquityCorrelation
) -> np.ndarray:
    """Reorder the equity correlation to node order and project to the nearest PSD corr."""
    tickers = [node.ticker for node in nodes]
    sub = ec.reordered(tickers)
    corr = nearest_psd_correlation(sub)
    # nearest_psd_correlation clips to +/-0.999 last, nicking the diagonal; restore it
    # exactly so the flat SystemSpec's unit-diagonal contract holds.
    np.fill_diagonal(corr, 1.0)
    return corr


def interbank_totals(
    nodes: tuple[CleanNode, ...], interbank_share: float = 0.20
) -> tuple[np.ndarray, np.ndarray]:
    """Interbank asset (row) and liability (col) totals from total assets.

    Each node's interbank assets are ``interbank_share`` of its total assets (Gai-Kapadia /
    Basel: interbank ~20% of the balance sheet). Liabilities are set to the same per-node
    scale and then rescaled so the system asset total equals the liability total (required
    for a feasible bilateral reconstruction).
    """
    ta = np.array([node.total_assets_usd_bn for node in nodes], dtype=float)
    assets = interbank_share * ta
    liabilities = interbank_share * ta
    total = assets.sum()
    if liabilities.sum() > 0:
        liabilities = liabilities * (total / liabilities.sum())
    return assets, liabilities


def capital_buffers(
    nodes: tuple[CleanNode, ...], buffer_ratio: float = 0.06
) -> np.ndarray:
    """Tier-1-style loss-absorbing buffers (~4-8% of total assets; ~6% central)."""
    ta = np.array([node.total_assets_usd_bn for node in nodes], dtype=float)
    return buffer_ratio * ta
