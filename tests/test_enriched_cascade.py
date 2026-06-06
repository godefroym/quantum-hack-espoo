from __future__ import annotations

import numpy as np
import pytest

from systemic_risk.evaluation import (
    aggregate_results,
    failure_round_distribution,
    node_failure_frequencies,
    result_summary,
    tail_failure_probability,
)
from systemic_risk.simulator import (
    CascadeScenario,
    run_cascade,
    scenario_from_binary_vector,
    scenario_from_loss_vector,
    scenario_from_named_shocks,
    simulate_many,
)
from systemic_risk.spec import SystemSpec
from systemic_risk.visualization import plot_cascade


def _spec(
    names: list[str],
    capital: list[float],
    edges: list[tuple[str, str, float]],
) -> SystemSpec:
    index = {name: i for i, name in enumerate(names)}
    exposure = np.zeros((len(names), len(names)))
    for source, target, loss in edges:
        exposure[index[target], index[source]] = loss
    return SystemSpec(
        node_names=names,
        node_types=["bank"] * len(names),
        exposure_matrix=exposure,
        capital_buffers=np.asarray(capital, dtype=float),
        marginal_default_probs=np.full(len(names), 0.05),
        target_pairwise_corr=np.eye(len(names)),
        clusters=["test"] * len(names),
    )


def _chain_spec() -> SystemSpec:
    return _spec(
        ["A", "B", "C", "D", "E"],
        [999.0, 5.0, 4.0, 8.0, 4.0],
        [
            ("A", "B", 6.0),
            ("A", "C", 5.0),
            ("B", "D", 4.0),
            ("C", "D", 5.0),
            ("D", "E", 5.0),
        ],
    )


def test_known_chain_exposes_round_level_diagnostics() -> None:
    spec = _chain_spec()
    scenario = scenario_from_named_shocks(
        spec,
        initial_failed=["A"],
        scenario_id="A-fails",
        metadata={"generator": "manual"},
    )

    result = run_cascade(scenario, spec)

    assert result.failure_round_by_node == {
        "A": 0,
        "B": 1,
        "C": 1,
        "D": 2,
        "E": 3,
    }
    assert result.round_failures == [["A"], ["B", "C"], ["D"], ["E"]]
    assert result.failure_count == 5
    assert result.failure_fraction == 1.0
    assert result.cascade_depth == 3
    assert result.converged
    assert result.scenario_id == "A-fails"
    assert result.scenario_metadata == {"generator": "manual"}


def test_identical_inputs_produce_identical_audit_records() -> None:
    spec = _chain_spec()
    scenario = scenario_from_named_shocks(spec, initial_failed=["A"])

    first = run_cascade(scenario, spec)
    second = run_cascade(scenario, spec)

    assert first.to_dict() == second.to_dict()


def test_exogenous_losses_can_trigger_round_zero_failure() -> None:
    spec = _spec(["A", "B"], [10.0, 10.0], [])
    scenario = scenario_from_loss_vector(
        [0.0, 11.0],
        scenario_id="direct-shock",
    )

    result = run_cascade(scenario, spec)

    assert result.failed_nodes == ["B"]
    assert result.failure_round_by_node == {"B": 0}
    assert np.allclose(result.cumulative_losses, [0.0, 11.0])


def test_equal_to_capital_is_opt_in_failure_rule() -> None:
    spec = _spec(["A"], [10.0], [])
    scenario = scenario_from_loss_vector([10.0])

    strict = run_cascade(scenario, spec)
    inclusive = run_cascade(scenario, spec, fail_on_equal=True)

    assert strict.failure_count == 0
    assert inclusive.failure_count == 1


def test_lgd_controls_transmitted_losses() -> None:
    spec = _spec(["A", "B"], [100.0, 7.0], [("A", "B", 10.0)])
    scenario = scenario_from_named_shocks(spec, initial_failed=["A"])

    full_loss = run_cascade(scenario, spec, lgd=1.0)
    reduced_loss = run_cascade(scenario, spec, lgd=0.5)
    edge_lgd = np.ones((2, 2))
    edge_lgd[1, 0] = 0.5
    edge_reduced_loss = run_cascade(scenario, spec, lgd=edge_lgd)

    assert full_loss.failed_nodes == ["A", "B"]
    assert reduced_loss.failed_nodes == ["A"]
    assert reduced_loss.cumulative_losses[1] == pytest.approx(5.0)
    assert edge_reduced_loss.failed_nodes == ["A"]


def test_max_rounds_reports_non_convergence_without_hiding_partial_state() -> None:
    result = run_cascade(
        scenario_from_binary_vector([1, 0, 0, 0, 0]),
        _chain_spec(),
        max_rounds=1,
    )

    assert not result.converged
    assert result.failed_nodes == ["A", "B", "C"]
    assert result.rounds_to_convergence == 1


def test_named_scenario_rejects_unknown_institutions() -> None:
    with pytest.raises(ValueError, match="unknown institution"):
        scenario_from_named_shocks(_chain_spec(), initial_failed=["UNKNOWN"])


def test_non_binary_scenarios_are_rejected_before_integer_cast() -> None:
    spec = _spec(["A", "B"], [10.0, 10.0], [])

    with pytest.raises(ValueError, match="binary"):
        run_cascade(np.array([0.5, 0.0]), spec)
    with pytest.raises(ValueError, match="0/1"):
        simulate_many(np.array([[0.5, 0.0]]), spec)


def test_batch_simulation_accepts_shared_exogenous_loss_vector() -> None:
    spec = _spec(["A", "B"], [10.0, 10.0], [])
    results = simulate_many(
        np.array([[0, 0], [1, 0]]),
        spec,
        exogenous_losses=np.array([0.0, 11.0]),
    )

    assert results[0].failed_nodes == ["B"]
    assert results[1].failed_nodes == ["A", "B"]
    assert [result.scenario_id for result in results] == ["0", "1"]


def test_cascade_metrics_are_generator_agnostic() -> None:
    spec = _chain_spec()
    severe = run_cascade(
        CascadeScenario(
            initial_defaults=np.array([1, 0, 0, 0, 0]),
            metadata={"generator": "B"},
        ),
        spec,
    )
    quiet = run_cascade(
        CascadeScenario(
            initial_defaults=np.zeros(5, dtype=int),
            metadata={"generator": "C"},
        ),
        spec,
    )
    results = [severe, quiet]

    aggregate = aggregate_results(results)
    assert aggregate["num_scenarios"] == 2
    assert aggregate["max_final_failure_count"] == 5
    assert aggregate["max_cascade_depth"] == 3
    assert aggregate["systemic_collapse_frequency"] == 0.5
    assert failure_round_distribution(severe) == {0: 1, 1: 2, 2: 1, 3: 1}
    assert tail_failure_probability(results, min_failure_fraction=0.8) == 0.5
    assert node_failure_frequencies(results)["A"] == 0.5
    assert result_summary(severe)["failed_nodes"] == ["A", "B", "C", "D", "E"]


def test_cascade_plot_uses_canonical_spec(tmp_path) -> None:
    spec = _chain_spec()
    result = run_cascade(
        scenario_from_named_shocks(spec, initial_failed=["A"]),
        spec,
    )
    output = tmp_path / "cascade.png"

    figure = plot_cascade(spec, result, output)

    assert output.exists()
    assert output.stat().st_size > 0
    figure.clear()
