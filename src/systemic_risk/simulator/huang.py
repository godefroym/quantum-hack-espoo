from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from systemic_risk.bank_asset_spec import BankAssetSystemSpec
from systemic_risk.simulator.cascade import (
    is_systemic_collapse,
    validate_contagion_limits,
)


@dataclass
class HuangCascadeResult:
    """Bank-asset fire-sale outcome, conforming to ``CascadeOutcome``."""

    initial_bank_defaults: np.ndarray
    final_bank_defaults: np.ndarray
    initial_asset_price_factors: np.ndarray
    final_asset_price_factors: np.ndarray
    bank_tolerances: np.ndarray
    new_failures_by_round: list[np.ndarray]
    states_by_round: list[np.ndarray]
    price_factors_by_round: list[np.ndarray]
    bank_asset_values_by_round: list[np.ndarray]
    failure_probabilities_by_round: list[np.ndarray]
    rounds_to_convergence: int
    failure_count: int
    systemic_collapse: bool
    node_names: tuple[str, ...] = ()
    converged: bool = True
    scenario_id: str = ""

    @property
    def final_defaults(self) -> np.ndarray:
        """Shared-protocol alias for the post-cascade failed-bank vector."""
        return self.final_bank_defaults

    @property
    def initial_defaults(self) -> np.ndarray:
        return self.initial_bank_defaults

    @property
    def node_count(self) -> int:
        return len(self.final_bank_defaults)

    @property
    def failure_fraction(self) -> float:
        return self.failure_count / self.node_count if self.node_count else 0.0

    @property
    def cascade_depth(self) -> int:
        return self.rounds_to_convergence

    @property
    def failed_nodes(self) -> list[str]:
        names = self.node_names or tuple(
            str(i) for i in range(self.node_count)
        )
        return [name for name, failed in zip(names, self.final_bank_defaults) if failed]


def huang_failure_probability(
    current_assets: np.ndarray,
    liabilities: np.ndarray,
    eta: float,
) -> np.ndarray:
    """Failure probability implied by Huang et al.'s random distress barrier."""
    assets = np.asarray(current_assets, dtype=float)
    liabilities = np.asarray(liabilities, dtype=float)
    if assets.shape != liabilities.shape:
        raise ValueError("current_assets and liabilities must have the same shape")
    if not 0 <= eta <= 0.5:
        raise ValueError("eta must lie in [0, 0.5]")

    probabilities = np.zeros_like(assets)
    below_liabilities = assets < liabilities
    if eta == 0:
        probabilities[below_liabilities] = 1.0
        return probabilities

    certain_failure = assets <= (1 - eta) * liabilities
    transition = below_liabilities & ~certain_failure
    probabilities[certain_failure] = 1.0
    probabilities[transition] = (
        liabilities[transition] - assets[transition]
    ) / (eta * liabilities[transition])
    return np.clip(probabilities, 0.0, 1.0)


def run_huang_cascade(
    spec: BankAssetSystemSpec,
    asset_price_shocks: np.ndarray | Mapping[str, float] | None = None,
    initial_bank_defaults: np.ndarray | None = None,
    *,
    alpha: float | np.ndarray = 0.1,
    eta: float = 0.0,
    seed: int | None = None,
    bank_tolerances: np.ndarray | None = None,
    max_rounds: int = 100,
    collapse_threshold: float = 0.5,
) -> HuangCascadeResult:
    """Run the Huang bank-asset fire-sale cascade.

    Asset price factors are relative to pre-shock unit prices. A newly failed
    bank liquidates its original holdings and causes an alpha-scaled deduction
    from the corresponding asset class's total market value.
    """
    validate_contagion_limits(max_rounds, collapse_threshold)
    if not 0 <= eta <= 0.5:
        raise ValueError("eta must lie in [0, 0.5]")

    shock_factors = _resolve_asset_price_shocks(spec, asset_price_shocks)
    alpha_by_asset = _resolve_alpha(spec, alpha)
    initial_defaults = _resolve_initial_defaults(spec, initial_bank_defaults)
    tolerances = _resolve_tolerances(spec, eta, seed, bank_tolerances)

    failed = initial_defaults.astype(bool).copy()
    price_factors = _price_factors_after_liquidation(
        spec,
        shock_factors,
        failed,
        alpha_by_asset,
    )

    new_failures_by_round: list[np.ndarray] = []
    states_by_round: list[np.ndarray] = []
    price_factors_by_round: list[np.ndarray] = []
    bank_asset_values_by_round: list[np.ndarray] = []
    failure_probabilities_by_round: list[np.ndarray] = []
    converged = False

    for _ in range(max_rounds):
        current_values = spec.holdings @ price_factors
        probabilities = huang_failure_probability(
            current_values,
            spec.liabilities,
            eta,
        )
        distress_barriers = (1 - tolerances) * spec.liabilities
        new_failures = (~failed) & (current_values < distress_barriers)

        price_factors_by_round.append(price_factors.copy())
        bank_asset_values_by_round.append(current_values.copy())
        failure_probabilities_by_round.append(probabilities)

        if not np.any(new_failures):
            converged = True
            break

        failed |= new_failures
        new_failures_by_round.append(new_failures.astype(int))
        states_by_round.append(failed.astype(int).copy())
        price_factors = _price_factors_after_liquidation(
            spec,
            shock_factors,
            failed,
            alpha_by_asset,
        )

    failure_count = int(failed.sum())
    return HuangCascadeResult(
        initial_bank_defaults=initial_defaults,
        final_bank_defaults=failed.astype(int),
        initial_asset_price_factors=shock_factors,
        final_asset_price_factors=price_factors,
        bank_tolerances=tolerances,
        new_failures_by_round=new_failures_by_round,
        states_by_round=states_by_round,
        price_factors_by_round=price_factors_by_round,
        bank_asset_values_by_round=bank_asset_values_by_round,
        failure_probabilities_by_round=failure_probabilities_by_round,
        rounds_to_convergence=len(new_failures_by_round),
        failure_count=failure_count,
        systemic_collapse=is_systemic_collapse(
            failure_count, spec.n_banks, collapse_threshold
        ),
        node_names=tuple(spec.bank_names),
        converged=converged,
    )


def simulate_huang_scenarios(
    scenarios: np.ndarray,
    spec: BankAssetSystemSpec,
    asset_price_shocks: np.ndarray | Mapping[str, float] | None = None,
    *,
    alpha: float | np.ndarray = 0.1,
    eta: float = 0.0,
    seed: int | None = None,
    max_rounds: int = 100,
    collapse_threshold: float = 0.5,
) -> list[HuangCascadeResult]:
    """Evaluate shared-format binary bank-default scenarios with Huang's engine."""
    samples = np.asarray(scenarios, dtype=int)
    if samples.ndim != 2 or samples.shape[1] != spec.n_banks:
        raise ValueError("scenarios must have shape (n_scenarios, n_banks)")
    if not np.all((samples == 0) | (samples == 1)):
        raise ValueError("scenarios must contain only 0/1 values")

    child_seeds = np.random.SeedSequence(seed).spawn(len(samples))
    results = []
    for index, (scenario, child_seed) in enumerate(zip(samples, child_seeds)):
        result = run_huang_cascade(
            spec,
            asset_price_shocks=asset_price_shocks,
            initial_bank_defaults=scenario,
            alpha=alpha,
            eta=eta,
            seed=int(child_seed.generate_state(1)[0]),
            max_rounds=max_rounds,
            collapse_threshold=collapse_threshold,
        )
        result.scenario_id = str(index)
        results.append(result)
    return results


def _resolve_asset_price_shocks(
    spec: BankAssetSystemSpec,
    shocks: np.ndarray | Mapping[str, float] | None,
) -> np.ndarray:
    if shocks is None:
        return np.ones(spec.n_assets, dtype=float)
    if isinstance(shocks, Mapping):
        unknown = set(shocks) - set(spec.asset_names)
        if unknown:
            raise ValueError(f"unknown asset names: {sorted(unknown)}")
        factors = np.ones(spec.n_assets, dtype=float)
        asset_index = {name: idx for idx, name in enumerate(spec.asset_names)}
        for name, factor in shocks.items():
            factors[asset_index[name]] = factor
    else:
        factors = np.asarray(shocks, dtype=float)
    if factors.shape != (spec.n_assets,):
        raise ValueError("asset_price_shocks must have shape (n_assets,)")
    if not np.all(np.isfinite(factors)):
        raise ValueError("asset_price_shocks must contain only finite values")
    if np.any((factors < 0) | (factors > 1)):
        raise ValueError("asset price factors must lie in [0, 1]")
    return factors.copy()


def _resolve_alpha(
    spec: BankAssetSystemSpec,
    alpha: float | np.ndarray,
) -> np.ndarray:
    values = np.asarray(alpha, dtype=float)
    if values.ndim == 0:
        values = np.full(spec.n_assets, float(values))
    if values.shape != (spec.n_assets,):
        raise ValueError("alpha must be a scalar or have shape (n_assets,)")
    if not np.all(np.isfinite(values)):
        raise ValueError("alpha must contain only finite values")
    if np.any((values < 0) | (values > 1)):
        raise ValueError("alpha values must lie in [0, 1]")
    return values


def _resolve_initial_defaults(
    spec: BankAssetSystemSpec,
    initial_defaults: np.ndarray | None,
) -> np.ndarray:
    if initial_defaults is None:
        return np.zeros(spec.n_banks, dtype=int)
    defaults = np.asarray(initial_defaults, dtype=int)
    if defaults.shape != (spec.n_banks,):
        raise ValueError("initial_bank_defaults must have shape (n_banks,)")
    if not np.all((defaults == 0) | (defaults == 1)):
        raise ValueError("initial_bank_defaults must contain only 0/1 values")
    return defaults


def _resolve_tolerances(
    spec: BankAssetSystemSpec,
    eta: float,
    seed: int | None,
    tolerances: np.ndarray | None,
) -> np.ndarray:
    if tolerances is None:
        return np.random.default_rng(seed).uniform(0.0, eta, size=spec.n_banks)
    values = np.asarray(tolerances, dtype=float)
    if values.shape != (spec.n_banks,):
        raise ValueError("bank_tolerances must have shape (n_banks,)")
    if not np.all(np.isfinite(values)):
        raise ValueError("bank_tolerances must contain only finite values")
    if np.any((values < 0) | (values > eta)):
        raise ValueError("bank_tolerances must lie in [0, eta]")
    return values.copy()


def _price_factors_after_liquidation(
    spec: BankAssetSystemSpec,
    initial_shock_factors: np.ndarray,
    failed: np.ndarray,
    alpha_by_asset: np.ndarray,
) -> np.ndarray:
    liquidated_holdings = spec.holdings[failed].sum(axis=0)
    liquidated_market_share = liquidated_holdings / spec.market_values
    return np.clip(
        initial_shock_factors - alpha_by_asset * liquidated_market_share,
        0.0,
        1.0,
    )
