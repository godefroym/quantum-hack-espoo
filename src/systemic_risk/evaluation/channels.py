from __future__ import annotations

from typing import Mapping, Protocol, runtime_checkable

import numpy as np

from systemic_risk.bank_asset_spec import BankAssetSystemSpec
from systemic_risk.data.bank_asset_adapter import bank_asset_to_system_spec
from systemic_risk.simulator.cascade import CascadeOutcome, simulate_many
from systemic_risk.simulator.huang import simulate_huang_scenarios
from systemic_risk.spec import SystemSpec


@runtime_checkable
class ContagionChannel(Protocol):
    """A generator-target spec plus the engine that turns samples into outcomes.

    The harness fits generators against :attr:`spec` and evaluates their binary
    default samples through :meth:`simulate`, which returns objects satisfying
    :class:`~systemic_risk.simulator.cascade.CascadeOutcome`. This is the seam
    that lets one evaluation spine drive either the exposure cascade or the
    Huang bank-asset fire-sale.
    """

    name: str

    @property
    def spec(self) -> SystemSpec: ...

    def simulate(self, samples: np.ndarray) -> list[CascadeOutcome]: ...


class ExposureCascadeChannel:
    """Deterministic exposure-cascade channel (the harness default)."""

    name = "exposure_cascade"

    def __init__(
        self,
        spec: SystemSpec,
        *,
        max_rounds: int | None = None,
        collapse_threshold: float = 0.5,
        lgd: float | np.ndarray = 1.0,
        fail_on_equal: bool = False,
    ) -> None:
        self._spec = spec
        self.max_rounds = max_rounds
        self.collapse_threshold = collapse_threshold
        self.lgd = lgd
        self.fail_on_equal = fail_on_equal

    @property
    def spec(self) -> SystemSpec:
        return self._spec

    def simulate(self, samples: np.ndarray) -> list[CascadeOutcome]:
        return simulate_many(
            samples,
            self._spec,
            max_rounds=self.max_rounds,
            collapse_threshold=self.collapse_threshold,
            lgd=self.lgd,
            fail_on_equal=self.fail_on_equal,
        )


class HuangFireSaleChannel:
    """Bank-asset fire-sale channel.

    Generators are fit against the flat :class:`SystemSpec` produced by
    :func:`bank_asset_to_system_spec`, while contagion is evaluated on the
    original :class:`BankAssetSystemSpec` with the Huang price-impact engine.
    """

    name = "huang_fire_sale"

    def __init__(
        self,
        bank_asset_spec: BankAssetSystemSpec,
        *,
        asset_price_shocks: np.ndarray | Mapping[str, float] | None = None,
        alpha: float | np.ndarray = 0.08,
        eta: float = 0.0,
        collapse_threshold: float = 0.5,
        max_rounds: int = 100,
        seed: int = 0,
        generator_spec: SystemSpec | None = None,
        mean_default_probability: float = 0.04,
    ) -> None:
        self.bank_asset_spec = bank_asset_spec
        self.asset_price_shocks = asset_price_shocks
        self.alpha = alpha
        self.eta = eta
        self.collapse_threshold = collapse_threshold
        self.max_rounds = max_rounds
        self.seed = seed
        self._spec = generator_spec or bank_asset_to_system_spec(
            bank_asset_spec,
            alpha=alpha,
            mean_default_probability=mean_default_probability,
        )

    @property
    def spec(self) -> SystemSpec:
        return self._spec

    def simulate(self, samples: np.ndarray) -> list[CascadeOutcome]:
        return simulate_huang_scenarios(
            samples,
            self.bank_asset_spec,
            asset_price_shocks=self.asset_price_shocks,
            alpha=self.alpha,
            eta=self.eta,
            seed=self.seed,
            max_rounds=self.max_rounds,
            collapse_threshold=self.collapse_threshold,
        )


def as_channel(target: SystemSpec | ContagionChannel, **kwargs) -> ContagionChannel:
    """Wrap a bare :class:`SystemSpec` in the default exposure channel."""
    if isinstance(target, SystemSpec):
        return ExposureCascadeChannel(target, **kwargs)
    return target
