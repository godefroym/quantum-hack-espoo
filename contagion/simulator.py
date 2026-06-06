from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Any

from contagion.spec import (
    Scenario,
    SystemSpec,
    build_outgoing_edges,
    get_capital_map,
    get_node_order,
    ordered_subset,
    validate_scenario,
    validate_system_spec,
)


@dataclass(frozen=True)
class CascadeResult:
    system_id: str
    scenario_id: str

    node_count: int
    failure_round: dict[str, int]
    round_failures: list[list[str]]
    round_loss_contributions: list[dict[str, Any]]
    cumulative_losses: dict[str, float]

    failed_nodes: list[str]
    final_failure_count: int
    failure_fraction: float
    cascade_depth: int

    systemic_threshold_fraction: float
    systemic_collapse: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def run_cascade(
    spec: SystemSpec,
    scenario: Scenario,
    *,
    default_lgd: float = 1.0,
    fail_on_equal: bool = True,
) -> CascadeResult:
    """
    Deterministic, generator-agnostic contagion simulator.

    Edge convention:
        source -> target means:
        if source fails, target suffers exposure * lgd as a loss.

    Cascade rule:
        1. Nodes in scenario["initial_failed"] fail at round 0.
        2. Nodes with exogenous loss >= capital also fail at round 0.
        3. Newly failed nodes transmit losses to outgoing neighbors.
        4. A node fails once cumulative losses breach its capital.
        5. Repeat synchronously until no new failures occur.

    Determinism guarantees:
        - no randomness
        - no generator-specific branches
        - node processing follows spec["nodes"] order
        - edge processing is canonicalized
    """

    validate_system_spec(spec)
    validate_scenario(spec, scenario)

    if default_lgd < 0:
        raise ValueError("default_lgd must be non-negative")

    system_id = str(spec.get("system_id", ""))
    scenario_id = str(scenario.get("scenario_id", ""))

    node_order = get_node_order(spec)
    node_count = len(node_order)

    capital = get_capital_map(spec)
    outgoing = build_outgoing_edges(spec)

    systemic_threshold_fraction = float(spec.get("systemic_threshold_fraction", 0.3))

    if not 0 <= systemic_threshold_fraction <= 1:
        raise ValueError("systemic_threshold_fraction must be in [0, 1]")

    exogenous_losses = scenario.get("exogenous_losses", {})

    cumulative_losses: dict[str, float] = {
        node_id: float(exogenous_losses.get(node_id, 0.0)) for node_id in node_order
    }

    forced_initial_failures = set(scenario.get("initial_failed", []))

    failure_round: dict[str, int] = {}
    initial_failures: set[str] = set()

    for node_id in node_order:
        direct_failure = node_id in forced_initial_failures

        if fail_on_equal:
            loss_failure = cumulative_losses[node_id] >= capital[node_id]
        else:
            loss_failure = cumulative_losses[node_id] > capital[node_id]

        if direct_failure or loss_failure:
            initial_failures.add(node_id)
            failure_round[node_id] = 0

    round_failures: list[list[str]] = [ordered_subset(node_order, initial_failures)]

    round_loss_contributions: list[dict[str, Any]] = []

    frontier = initial_failures
    current_round = 0

    while frontier:
        next_round = current_round + 1
        losses_this_round: defaultdict[str, float] = defaultdict(float)

        for failed_source in ordered_subset(node_order, frontier):
            for edge in outgoing[failed_source]:
                target = edge["target"]

                if target in failure_round:
                    continue

                lgd = float(edge.get("lgd", default_lgd))
                loss = float(edge["exposure"]) * lgd

                if loss != 0.0:
                    losses_this_round[target] += loss

        ordered_losses = {
            node_id: losses_this_round[node_id]
            for node_id in node_order
            if losses_this_round[node_id] != 0.0
        }

        if ordered_losses:
            round_loss_contributions.append(
                {
                    "round": next_round,
                    "losses": ordered_losses,
                }
            )

        for node_id, loss in ordered_losses.items():
            cumulative_losses[node_id] += loss

        new_failures: set[str] = set()

        for node_id in node_order:
            if node_id in failure_round:
                continue

            if fail_on_equal:
                has_failed = cumulative_losses[node_id] >= capital[node_id]
            else:
                has_failed = cumulative_losses[node_id] > capital[node_id]

            if has_failed:
                new_failures.add(node_id)
                failure_round[node_id] = next_round

        if not new_failures:
            break

        while len(round_failures) <= next_round:
            round_failures.append([])

        round_failures[next_round] = ordered_subset(node_order, new_failures)

        frontier = new_failures
        current_round = next_round

    failed_nodes = [node_id for node_id in node_order if node_id in failure_round]

    final_failure_count = len(failed_nodes)
    failure_fraction = final_failure_count / node_count if node_count else 0.0
    cascade_depth = max(failure_round.values(), default=0)

    systemic_collapse = failure_fraction >= systemic_threshold_fraction

    return CascadeResult(
        system_id=system_id,
        scenario_id=scenario_id,
        node_count=node_count,
        failure_round={node_id: failure_round[node_id] for node_id in failed_nodes},
        round_failures=round_failures,
        round_loss_contributions=round_loss_contributions,
        cumulative_losses={node_id: cumulative_losses[node_id] for node_id in node_order},
        failed_nodes=failed_nodes,
        final_failure_count=final_failure_count,
        failure_fraction=failure_fraction,
        cascade_depth=cascade_depth,
        systemic_threshold_fraction=systemic_threshold_fraction,
        systemic_collapse=systemic_collapse,
    )
