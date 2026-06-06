from __future__ import annotations

import numpy as np
import pytest

from systemic_risk.bank_asset_spec import BankAssetSystemSpec
from systemic_risk.data import HUANG_ASSET_NAMES, make_huang_2008_style_system
from systemic_risk.simulator import (
    huang_failure_probability,
    run_huang_cascade,
    simulate_huang_scenarios,
)


def _two_bank_cascade_spec() -> BankAssetSystemSpec:
    return BankAssetSystemSpec(
        bank_names=["CRE Bank", "Diversified Bank"],
        asset_names=["commercial_real_estate", "other_assets"],
        holdings=np.array(
            [
                [80.0, 20.0],
                [20.0, 80.0],
            ]
        ),
        liabilities=np.array([70.0, 88.0]),
    )


def test_huang_failure_probability_matches_piecewise_formula() -> None:
    liabilities = np.full(4, 100.0)
    assets = np.array([105.0, 97.5, 92.5, 85.0])

    probabilities = huang_failure_probability(assets, liabilities, eta=0.1)

    assert np.allclose(probabilities, np.array([0.0, 0.25, 0.75, 1.0]))


def test_no_shock_produces_no_failures() -> None:
    result = run_huang_cascade(_two_bank_cascade_spec(), alpha=0.5, eta=0.0)

    assert result.failure_count == 0
    assert result.rounds_to_convergence == 0
    assert np.allclose(result.final_asset_price_factors, 1.0)


def test_asset_shock_and_fire_sale_trigger_two_round_cascade() -> None:
    result = run_huang_cascade(
        _two_bank_cascade_spec(),
        asset_price_shocks={"commercial_real_estate": 0.6},
        alpha=0.5,
        eta=0.0,
    )

    assert result.rounds_to_convergence == 2
    assert np.array_equal(result.new_failures_by_round[0], np.array([1, 0]))
    assert np.array_equal(result.new_failures_by_round[1], np.array([0, 1]))
    assert np.array_equal(result.final_bank_defaults, np.array([1, 1]))
    assert np.allclose(result.final_asset_price_factors, np.array([0.1, 0.5]))


def test_zero_price_impact_stops_contagion_after_direct_failure() -> None:
    result = run_huang_cascade(
        _two_bank_cascade_spec(),
        asset_price_shocks={"commercial_real_estate": 0.6},
        alpha=0.0,
        eta=0.0,
    )

    assert result.rounds_to_convergence == 1
    assert np.array_equal(result.final_bank_defaults, np.array([1, 0]))
    assert np.allclose(result.final_asset_price_factors, np.array([0.6, 1.0]))


def test_initial_bank_default_can_seed_the_same_fire_sale_engine() -> None:
    result = run_huang_cascade(
        _two_bank_cascade_spec(),
        initial_bank_defaults=np.array([1, 0]),
        alpha=0.8,
        eta=0.0,
    )

    assert np.array_equal(result.final_bank_defaults, np.array([1, 1]))
    assert result.rounds_to_convergence == 1


def test_random_tolerances_are_reproducible() -> None:
    spec = _two_bank_cascade_spec()

    first = run_huang_cascade(
        spec,
        asset_price_shocks={"commercial_real_estate": 0.6},
        alpha=0.2,
        eta=0.2,
        seed=42,
    )
    second = run_huang_cascade(
        spec,
        asset_price_shocks={"commercial_real_estate": 0.6},
        alpha=0.2,
        eta=0.2,
        seed=42,
    )

    assert np.array_equal(first.bank_tolerances, second.bank_tolerances)
    assert np.array_equal(first.final_bank_defaults, second.final_bank_defaults)
    assert np.allclose(first.final_asset_price_factors, second.final_asset_price_factors)


def test_huang_style_system_is_well_formed_and_deterministic() -> None:
    first = make_huang_2008_style_system(n_banks=12, seed=7)
    second = make_huang_2008_style_system(n_banks=12, seed=7)

    assert first.holdings.shape == (12, 13)
    assert first.asset_names == HUANG_ASSET_NAMES
    assert np.all(first.equity > 0)
    assert np.allclose(first.holdings, second.holdings)
    assert np.allclose(first.portfolio_weights.sum(axis=1), 1.0)


def test_invalid_shock_asset_name_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown asset"):
        run_huang_cascade(
            _two_bank_cascade_spec(),
            asset_price_shocks={"not_an_asset": 0.5},
        )


def test_shared_binary_scenario_batch_is_supported() -> None:
    scenarios = np.array([[0, 0], [1, 0], [0, 1]])

    results = simulate_huang_scenarios(
        scenarios,
        _two_bank_cascade_spec(),
        alpha=0.8,
        eta=0.0,
        seed=5,
    )

    assert len(results) == 3
    assert results[0].failure_count == 0
    assert results[1].failure_count == 2
    assert results[2].failure_count == 1


def test_larger_price_impact_does_not_reduce_failures() -> None:
    spec = _two_bank_cascade_spec()
    shocks = {"commercial_real_estate": 0.6}

    liquid = run_huang_cascade(spec, shocks, alpha=0.0, eta=0.0)
    illiquid = run_huang_cascade(spec, shocks, alpha=0.5, eta=0.0)

    assert liquid.failure_count <= illiquid.failure_count
