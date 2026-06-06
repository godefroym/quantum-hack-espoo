from __future__ import annotations

import numpy as np

from systemic_risk.bank_asset_spec import BankAssetSystemSpec
from systemic_risk.data import bank_asset_to_system_spec


def _spec() -> BankAssetSystemSpec:
    return BankAssetSystemSpec(
        bank_names=["Bank A", "Bank B", "Bank C"],
        asset_names=["Real estate", "Securities"],
        holdings=np.array(
            [
                [80.0, 20.0],
                [50.0, 50.0],
                [10.0, 90.0],
            ]
        ),
        liabilities=np.array([92.0, 90.0, 85.0]),
    )


def test_adapter_builds_first_order_huang_losses() -> None:
    bank_assets = _spec()

    adapted = bank_asset_to_system_spec(
        bank_assets,
        alpha=np.array([0.10, 0.20]),
        marginal_default_probs=np.array([0.08, 0.05, 0.03]),
    )

    market_values = bank_assets.market_values
    expected_a_from_b = (
        80.0 * 0.10 * 50.0 / market_values[0]
        + 20.0 * 0.20 * 50.0 / market_values[1]
    )
    assert np.isclose(adapted.exposure_matrix[0, 1], expected_a_from_b)
    assert np.allclose(np.diag(adapted.exposure_matrix), 0.0)
    assert np.allclose(adapted.capital_buffers, bank_assets.equity)


def test_adapter_targets_are_psd_and_deterministic() -> None:
    first = bank_asset_to_system_spec(_spec(), mean_default_probability=0.06)
    second = bank_asset_to_system_spec(_spec(), mean_default_probability=0.06)

    assert np.allclose(first.marginal_default_probs, second.marginal_default_probs)
    assert np.isclose(first.marginal_default_probs.mean(), 0.06)
    assert np.allclose(first.target_pairwise_corr, first.target_pairwise_corr.T)
    assert np.allclose(np.diag(first.target_pairwise_corr), 1.0)
    assert np.linalg.eigvalsh(first.target_pairwise_corr).min() >= -1e-10


def test_lower_equity_ratio_gets_higher_heuristic_default_probability() -> None:
    adapted = bank_asset_to_system_spec(_spec(), mean_default_probability=0.06)

    equity_ratios = _spec().equity / _spec().total_assets
    lowest_equity_bank = int(np.argmin(equity_ratios))
    highest_equity_bank = int(np.argmax(equity_ratios))

    assert (
        adapted.marginal_default_probs[lowest_equity_bank]
        > adapted.marginal_default_probs[highest_equity_bank]
    )
