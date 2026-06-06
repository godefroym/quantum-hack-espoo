from __future__ import annotations

import numpy as np

from systemic_risk.bank_asset_spec import BankAssetSystemSpec


HUANG_ASSET_NAMES = [
    "construction_and_land_development",
    "farmland_secured_loans",
    "residential_1_to_4_family",
    "multifamily_residential",
    "nonfarm_nonresidential",
    "agricultural_loans",
    "commercial_and_industrial_loans",
    "consumer_loans",
    "other_loans",
    "state_and_municipal_obligations",
    "held_to_maturity_securities",
    "available_for_sale_securities",
    "premises_and_fixed_assets",
]


_PAPER_AVERAGE_WEIGHTS = np.array(
    [
        0.082,
        0.038,
        0.167,
        0.013,
        0.150,
        0.041,
        0.031,
        0.097,
        0.171,
        0.046,
        0.003,
        0.004,
        0.020,
    ],
    dtype=float,
)


def make_huang_2008_style_system(
    n_banks: int = 16,
    seed: int = 2007,
) -> BankAssetSystemSpec:
    """Create synthetic 2007-style bank balance sheets.

    Asset categories and average portfolio weights come from Huang et al.
    The generated bank-level portfolios are synthetic and must not be treated
    as reconstructed historical observations.
    """
    if n_banks < 6:
        raise ValueError("n_banks must be at least 6")

    rng = np.random.default_rng(seed)
    base = _PAPER_AVERAGE_WEIGHTS / _PAPER_AVERAGE_WEIGHTS.sum()
    profiles = [
        ("CRE Specialist", {0: 3.6, 4: 2.8}, (0.035, 0.055)),
        ("Mortgage Lender", {2: 3.2, 3: 2.0}, (0.045, 0.070)),
        ("Agricultural Bank", {1: 2.8, 5: 3.4}, (0.080, 0.120)),
        ("Consumer Bank", {7: 3.0, 8: 1.5}, (0.060, 0.090)),
        ("Commercial Bank", {6: 3.2, 9: 1.5}, (0.055, 0.085)),
        ("Diversified Bank", {}, (0.070, 0.105)),
    ]

    bank_names: list[str] = []
    portfolio_weights = np.zeros((n_banks, len(HUANG_ASSET_NAMES)))
    equity_ratios = np.zeros(n_banks)

    for i in range(n_banks):
        profile_name, multipliers, equity_range = profiles[i % len(profiles)]
        center = base.copy()
        for asset_idx, multiplier in multipliers.items():
            center[asset_idx] *= multiplier
        center /= center.sum()

        portfolio_weights[i] = rng.dirichlet(180.0 * center)
        equity_ratios[i] = rng.uniform(*equity_range)
        bank_names.append(f"{profile_name} {i + 1}")

    total_assets = rng.lognormal(mean=np.log(100.0), sigma=0.55, size=n_banks)
    holdings = total_assets[:, None] * portfolio_weights
    liabilities = total_assets * (1 - equity_ratios)

    return BankAssetSystemSpec(
        bank_names=bank_names,
        asset_names=HUANG_ASSET_NAMES,
        holdings=holdings,
        liabilities=liabilities,
        metadata={
            "name": "Synthetic Huang 2008-style bank-asset system",
            "seed": seed,
            "source": "Huang et al. (2013), Scientific Reports 3, 1219",
            "doi": "10.1038/srep01219",
            "historical_data": False,
        },
    )
