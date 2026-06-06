from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

import numpy as np

from systemic_risk.spec import SystemSpec


@runtime_checkable
class CascadeOutcome(Protocol):
    """Shared contract for any contagion result (exposure cascade or fire-sale).

    Evaluation, aggregation, and visualization treat both contagion channels
    through this interface, so each result type must expose the post-contagion
    failed set, a per-round trace, and systemic-collapse status. The exact
    round-zero convention of the trace lists is channel-specific.
    """

    final_defaults: np.ndarray
    failure_count: int
    rounds_to_convergence: int
    systemic_collapse: bool
    converged: bool
    scenario_id: str
    node_names: tuple[str, ...]
    new_failures_by_round: list[np.ndarray]
    states_by_round: list[np.ndarray]

    @property
    def node_count(self) -> int: ...

    @property
    def failure_fraction(self) -> float: ...

    @property
    def cascade_depth(self) -> int: ...

    @property
    def failed_nodes(self) -> list[str]: ...


@dataclass(frozen=True)
class CascadeScenario:
    """One generator-agnostic stress scenario in canonical node order."""

    initial_defaults: np.ndarray
    exogenous_losses: np.ndarray | None = None
    scenario_id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        raw_defaults = np.asarray(self.initial_defaults)
        if raw_defaults.ndim != 1 or not np.all(
            (raw_defaults == 0) | (raw_defaults == 1)
        ):
            raise ValueError("initial_defaults must be a one-dimensional binary vector")
        defaults = raw_defaults.astype(int)
        losses = (
            np.zeros(len(defaults), dtype=float)
            if self.exogenous_losses is None
            else np.asarray(self.exogenous_losses, dtype=float)
        )
        if losses.shape != defaults.shape:
            raise ValueError("exogenous_losses must match initial_defaults")
        if not np.all(np.isfinite(losses)) or np.any(losses < 0):
            raise ValueError("exogenous_losses must be finite and nonnegative")
        object.__setattr__(self, "initial_defaults", defaults.copy())
        object.__setattr__(self, "exogenous_losses", losses.copy())
        object.__setattr__(self, "scenario_id", str(self.scenario_id))
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass
class CascadeResult:
    initial_defaults: np.ndarray
    final_defaults: np.ndarray
    failure_count: int
    rounds_to_convergence: int
    states_by_round: list[np.ndarray]
    systemic_collapse: bool
    scenario_id: str = ""
    scenario_metadata: dict[str, Any] = field(default_factory=dict)
    node_names: tuple[str, ...] = ()
    exogenous_losses: np.ndarray = field(
        default_factory=lambda: np.array([], dtype=float)
    )
    cumulative_losses: np.ndarray = field(
        default_factory=lambda: np.array([], dtype=float)
    )
    failure_round: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))
    new_failures_by_round: list[np.ndarray] = field(default_factory=list)
    losses_by_round: list[np.ndarray] = field(default_factory=list)
    converged: bool = True

    @property
    def node_count(self) -> int:
        return len(self.final_defaults)

    @property
    def failure_fraction(self) -> float:
        return self.failure_count / self.node_count if self.node_count else 0.0

    @property
    def cascade_depth(self) -> int:
        failed_rounds = self.failure_round[self.failure_round >= 0]
        return int(failed_rounds.max()) if len(failed_rounds) else 0

    @property
    def failed_nodes(self) -> list[str]:
        if not self.node_names:
            return [str(i) for i in np.flatnonzero(self.final_defaults)]
        return [
            name
            for name, failed in zip(self.node_names, self.final_defaults)
            if failed
        ]

    @property
    def failure_round_by_node(self) -> dict[str, int]:
        names = self.node_names or tuple(str(i) for i in range(self.node_count))
        return {
            name: int(round_index)
            for name, round_index in zip(names, self.failure_round)
            if round_index >= 0
        }

    @property
    def round_failures(self) -> list[list[str]]:
        names = self.node_names or tuple(str(i) for i in range(self.node_count))
        return [
            [name for name, failed in zip(names, round_defaults) if failed]
            for round_defaults in self.new_failures_by_round
        ]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible audit record."""
        data = asdict(self)
        for key in (
            "initial_defaults",
            "final_defaults",
            "exogenous_losses",
            "cumulative_losses",
            "failure_round",
        ):
            data[key] = data[key].tolist()
        data["states_by_round"] = [state.tolist() for state in self.states_by_round]
        data["new_failures_by_round"] = [
            state.tolist() for state in self.new_failures_by_round
        ]
        data["losses_by_round"] = [losses.tolist() for losses in self.losses_by_round]
        data["node_names"] = list(self.node_names)
        data["failure_fraction"] = self.failure_fraction
        data["cascade_depth"] = self.cascade_depth
        data["failed_nodes"] = self.failed_nodes
        data["failure_round_by_node"] = self.failure_round_by_node
        data["round_failures"] = self.round_failures
        return data


def scenario_from_binary_vector(
    failure_vector: Sequence[int | bool],
    *,
    scenario_id: str = "",
    metadata: Mapping[str, Any] | None = None,
) -> CascadeScenario:
    """Convert one B/C binary sample into the shared scenario type."""
    return CascadeScenario(
        initial_defaults=np.asarray(failure_vector),
        scenario_id=scenario_id,
        metadata=dict(metadata or {}),
    )


def scenario_from_loss_vector(
    loss_vector: Sequence[float],
    *,
    scenario_id: str = "",
    metadata: Mapping[str, Any] | None = None,
) -> CascadeScenario:
    """Create a scenario driven only by direct exogenous losses."""
    losses = np.asarray(loss_vector, dtype=float)
    return CascadeScenario(
        initial_defaults=np.zeros(len(losses), dtype=int),
        exogenous_losses=losses,
        scenario_id=scenario_id,
        metadata=dict(metadata or {}),
    )


def scenario_from_named_shocks(
    spec: SystemSpec,
    *,
    initial_failed: Sequence[str] = (),
    exogenous_losses: Mapping[str, float] | None = None,
    scenario_id: str = "",
    metadata: Mapping[str, Any] | None = None,
) -> CascadeScenario:
    """Build a scenario from institution names while preserving canonical order."""
    if len(set(spec.node_names)) != spec.n:
        raise ValueError("node_names must be unique for named scenarios")
    index = {name: i for i, name in enumerate(spec.node_names)}
    unknown = (set(initial_failed) | set(exogenous_losses or {})) - set(index)
    if unknown:
        raise ValueError(f"unknown institution names: {sorted(unknown)}")

    defaults = np.zeros(spec.n, dtype=int)
    for name in initial_failed:
        defaults[index[name]] = 1
    losses = np.zeros(spec.n, dtype=float)
    for name, loss in (exogenous_losses or {}).items():
        losses[index[name]] = float(loss)
    return CascadeScenario(
        initial_defaults=defaults,
        exogenous_losses=losses,
        scenario_id=scenario_id,
        metadata=dict(metadata or {}),
    )


def run_cascade(
    initial_defaults: np.ndarray | CascadeScenario,
    spec: SystemSpec,
    max_rounds: int | None = None,
    collapse_threshold: float = 0.5,
    *,
    exogenous_losses: np.ndarray | None = None,
    lgd: float | np.ndarray = 1.0,
    fail_on_equal: bool = False,
) -> CascadeResult:
    """Run deterministic fixed-point contagion from one shared-format scenario.

    ``W[i, j]`` is the exposure loss received by institution ``i`` when
    institution ``j`` defaults. Each newly failed institution transmits its loss
    once. The default threshold remains strict by default for compatibility with
    the original simulator; set ``fail_on_equal=True`` for ``loss >= capital``.
    """
    scenario = _resolve_scenario(initial_defaults, spec, exogenous_losses)
    round_limit = spec.n if max_rounds is None else int(max_rounds)
    validate_contagion_limits(round_limit, collapse_threshold)

    effective_exposure = spec.exposure_matrix * _resolve_lgd(lgd, spec.n)
    cumulative_losses = scenario.exogenous_losses.copy()

    def crosses_threshold(losses: np.ndarray) -> np.ndarray:
        if fail_on_equal:
            return losses >= spec.capital_buffers
        return losses > spec.capital_buffers

    forced_defaults = scenario.initial_defaults.astype(bool)
    state = forced_defaults | crosses_threshold(cumulative_losses)
    initial_failures = state.astype(int)
    failure_round = np.full(spec.n, -1, dtype=int)
    failure_round[state] = 0
    states = [state.astype(int).copy()]
    new_failures_by_round = [initial_failures.copy()]
    losses_by_round = [scenario.exogenous_losses.copy()]
    frontier = state.copy()
    converged = not np.any(frontier) or bool(np.all(state))

    for round_index in range(1, round_limit + 1):
        if converged:
            break
        round_losses = effective_exposure @ frontier.astype(float)
        losses_by_round.append(round_losses.copy())
        cumulative_losses += round_losses
        new_failures = (~state) & crosses_threshold(cumulative_losses)
        if not np.any(new_failures):
            converged = True
            break

        state |= new_failures
        failure_round[new_failures] = round_index
        states.append(state.astype(int).copy())
        new_failures_by_round.append(new_failures.astype(int))
        frontier = new_failures
        if np.all(state):
            converged = True
            break

    failure_count = int(state.sum())
    return CascadeResult(
        initial_defaults=scenario.initial_defaults.copy(),
        final_defaults=state.astype(int),
        failure_count=failure_count,
        rounds_to_convergence=len(states) - 1,
        states_by_round=states,
        systemic_collapse=is_systemic_collapse(failure_count, spec.n, collapse_threshold),
        scenario_id=scenario.scenario_id,
        scenario_metadata=dict(scenario.metadata),
        node_names=tuple(spec.node_names),
        exogenous_losses=scenario.exogenous_losses.copy(),
        cumulative_losses=cumulative_losses,
        failure_round=failure_round,
        new_failures_by_round=new_failures_by_round,
        losses_by_round=losses_by_round,
        converged=converged,
    )


def simulate_many(
    scenarios: np.ndarray,
    spec: SystemSpec,
    max_rounds: int | None = None,
    collapse_threshold: float = 0.5,
    *,
    exogenous_losses: np.ndarray | None = None,
    lgd: float | np.ndarray = 1.0,
    fail_on_equal: bool = False,
) -> list[CascadeResult]:
    raw_scenarios = np.asarray(scenarios)
    if raw_scenarios.ndim != 2 or raw_scenarios.shape[1] != spec.n:
        raise ValueError(f"scenarios must have shape (n_samples, {spec.n})")
    if not np.all((raw_scenarios == 0) | (raw_scenarios == 1)):
        raise ValueError("scenarios must contain only 0/1 values")
    scenarios = raw_scenarios.astype(int)

    if exogenous_losses is None:
        loss_rows = np.zeros_like(scenarios, dtype=float)
    else:
        loss_rows = np.asarray(exogenous_losses, dtype=float)
        if loss_rows.shape == (spec.n,):
            loss_rows = np.broadcast_to(loss_rows, scenarios.shape)
        if loss_rows.shape != scenarios.shape:
            raise ValueError(
                "exogenous_losses must have shape (n,) or (n_samples, n)"
            )

    return [
        run_cascade(
            CascadeScenario(
                initial_defaults=row,
                exogenous_losses=loss_row,
                scenario_id=str(index),
            ),
            spec,
            max_rounds=max_rounds,
            collapse_threshold=collapse_threshold,
            lgd=lgd,
            fail_on_equal=fail_on_equal,
        )
        for index, (row, loss_row) in enumerate(zip(scenarios, loss_rows))
    ]


def _resolve_scenario(
    value: np.ndarray | CascadeScenario,
    spec: SystemSpec,
    exogenous_losses: np.ndarray | None,
) -> CascadeScenario:
    if isinstance(value, CascadeScenario):
        if exogenous_losses is not None:
            raise ValueError(
                "exogenous_losses must be stored on CascadeScenario when one is provided"
            )
        scenario = value
    else:
        scenario = CascadeScenario(
            initial_defaults=np.asarray(value),
            exogenous_losses=exogenous_losses,
        )
    if scenario.initial_defaults.shape != (spec.n,):
        raise ValueError(f"initial_defaults must have shape ({spec.n},)")
    return scenario


def _resolve_lgd(value: float | np.ndarray, n: int) -> np.ndarray:
    lgd = np.asarray(value, dtype=float)
    if lgd.ndim == 0:
        lgd = np.full((n, n), float(lgd))
    if lgd.shape != (n, n):
        raise ValueError("lgd must be a scalar or have shape (n, n)")
    if not np.all(np.isfinite(lgd)) or np.any(lgd < 0):
        raise ValueError("lgd must be finite and nonnegative")
    return lgd


def validate_contagion_limits(max_rounds: int, collapse_threshold: float) -> None:
    """Shared guardrails for any contagion engine's run controls."""
    if max_rounds <= 0:
        raise ValueError("max_rounds must be positive")
    if not 0 < collapse_threshold <= 1:
        raise ValueError("collapse_threshold must lie in (0, 1]")


def is_systemic_collapse(failure_count: int, node_count: int, collapse_threshold: float) -> bool:
    """Whether ``failure_count`` reaches the systemic-collapse share of ``node_count``."""
    return failure_count >= int(np.ceil(collapse_threshold * node_count))
