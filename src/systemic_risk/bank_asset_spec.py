from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class BankAssetSystemSpec:
    """Bipartite bank-asset balance-sheet system.

    holdings[i, m] is the initial book value of asset class m held by bank i.
    """

    bank_names: list[str]
    asset_names: list[str]
    holdings: np.ndarray
    liabilities: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.bank_names = list(self.bank_names)
        self.asset_names = list(self.asset_names)
        self.holdings = np.asarray(self.holdings, dtype=float)
        self.liabilities = np.asarray(self.liabilities, dtype=float)
        self.metadata = dict(self.metadata)
        self.validate()

    @property
    def n_banks(self) -> int:
        return len(self.bank_names)

    @property
    def n_assets(self) -> int:
        return len(self.asset_names)

    @property
    def total_assets(self) -> np.ndarray:
        return self.holdings.sum(axis=1)

    @property
    def equity(self) -> np.ndarray:
        return self.total_assets - self.liabilities

    @property
    def market_values(self) -> np.ndarray:
        return self.holdings.sum(axis=0)

    @property
    def portfolio_weights(self) -> np.ndarray:
        totals = self.total_assets[:, None]
        return np.divide(
            self.holdings,
            totals,
            out=np.zeros_like(self.holdings),
            where=totals > 0,
        )

    @property
    def market_shares(self) -> np.ndarray:
        totals = self.market_values[None, :]
        return np.divide(
            self.holdings,
            totals,
            out=np.zeros_like(self.holdings),
            where=totals > 0,
        )

    def validate(self) -> None:
        if not self.bank_names:
            raise ValueError("bank_names must not be empty")
        if not self.asset_names:
            raise ValueError("asset_names must not be empty")
        if len(set(self.bank_names)) != len(self.bank_names):
            raise ValueError("bank_names must be unique")
        if len(set(self.asset_names)) != len(self.asset_names):
            raise ValueError("asset_names must be unique")
        if self.holdings.shape != (self.n_banks, self.n_assets):
            raise ValueError("holdings must have shape (n_banks, n_assets)")
        if self.liabilities.shape != (self.n_banks,):
            raise ValueError("liabilities must have shape (n_banks,)")
        if not np.all(np.isfinite(self.holdings)):
            raise ValueError("holdings must contain only finite values")
        if not np.all(np.isfinite(self.liabilities)):
            raise ValueError("liabilities must contain only finite values")
        if np.any(self.holdings < 0):
            raise ValueError("holdings must be nonnegative")
        if np.any(self.liabilities < 0):
            raise ValueError("liabilities must be nonnegative")
        if np.any(self.total_assets <= 0):
            raise ValueError("each bank must hold a positive amount of assets")
        if np.any(self.market_values <= 0):
            raise ValueError("each asset class must be held by at least one bank")
        if np.any(self.liabilities >= self.total_assets):
            raise ValueError("banks must be solvent before the simulated shock")
