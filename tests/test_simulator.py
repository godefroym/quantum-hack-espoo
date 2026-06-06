import pytest

from contagion.metrics import (
    aggregate_results,
    failure_round_distribution,
    tail_failure_probability,
)
from contagion.simulator import run_cascade
from contagion.spec import scenario_from_binary_vector
from contagion.toy_networks import (
    create_no_exposure_network,
    create_star_network,
    create_toy_chain_network,
)


def test_no_exposure_means_no_cascade():
    spec = create_no_exposure_network()

    scenario = {
        "scenario_id": "no_edges_A_fails",
        "initial_failed": ["A"],
        "exogenous_losses": {},
        "metadata": {
            "generator": "manual",
        },
    }

    result = run_cascade(spec, scenario)

    assert result.failed_nodes == ["A"]
    assert result.failure_round == {"A": 0}
    assert result.final_failure_count == 1
    assert result.cascade_depth == 0
    assert result.systemic_collapse is False


def test_toy_chain_produces_known_cascade():
    spec = create_toy_chain_network()

    scenario = {
        "scenario_id": "toy_A_fails",
        "initial_failed": ["A"],
        "exogenous_losses": {},
    }

    result = run_cascade(spec, scenario)

    assert result.failure_round == {
        "A": 0,
        "B": 1,
        "C": 1,
        "D": 2,
        "E": 3,
    }

    assert result.round_failures == [
        ["A"],
        ["B", "C"],
        ["D"],
        ["E"],
    ]

    assert result.final_failure_count == 5
    assert result.failure_fraction == 1.0
    assert result.cascade_depth == 3
    assert result.systemic_collapse is True


def test_identical_inputs_produce_identical_outputs():
    spec = create_toy_chain_network()

    scenario = {
        "scenario_id": "determinism_check",
        "initial_failed": ["A"],
        "exogenous_losses": {},
    }

    result_1 = run_cascade(spec, scenario)
    result_2 = run_cascade(spec, scenario)

    assert result_1.to_dict() == result_2.to_dict()


def test_generator_metadata_does_not_affect_cascade():
    spec = create_toy_chain_network()

    scenario_gaussian = {
        "scenario_id": "same_initial_failure",
        "initial_failed": ["A"],
        "exogenous_losses": {},
        "metadata": {
            "generator": "gaussian_copula",
        },
    }

    scenario_quantum = {
        "scenario_id": "same_initial_failure",
        "initial_failed": ["A"],
        "exogenous_losses": {},
        "metadata": {
            "generator": "quantum_entanglement",
        },
    }

    result_gaussian = run_cascade(spec, scenario_gaussian)
    result_quantum = run_cascade(spec, scenario_quantum)

    assert result_gaussian.to_dict() == result_quantum.to_dict()


def test_exogenous_loss_can_trigger_round_zero_failure():
    spec = create_no_exposure_network()

    scenario = {
        "scenario_id": "shock_B_directly",
        "initial_failed": [],
        "exogenous_losses": {
            "B": 10.0,
        },
    }

    result = run_cascade(spec, scenario)

    assert result.failed_nodes == ["B"]
    assert result.failure_round == {"B": 0}
    assert result.final_failure_count == 1


def test_partial_loss_does_not_fail_node():
    spec = {
        "system_id": "partial_loss_test",
        "systemic_threshold_fraction": 0.5,
        "nodes": [
            {"id": "A", "capital": 10.0},
            {"id": "B", "capital": 10.0},
        ],
        "edges": [
            {"source": "A", "target": "B", "exposure": 5.0, "lgd": 1.0},
        ],
    }

    scenario = {
        "scenario_id": "A_fails_B_survives",
        "initial_failed": ["A"],
        "exogenous_losses": {},
    }

    result = run_cascade(spec, scenario)

    assert result.failed_nodes == ["A"]
    assert result.cumulative_losses["B"] == 5.0
    assert result.final_failure_count == 1
    assert result.cascade_depth == 0


def test_star_network_hub_failure_causes_leaf_failures():
    spec = create_star_network()

    scenario = {
        "scenario_id": "hub_fails",
        "initial_failed": ["HUB"],
        "exogenous_losses": {},
    }

    result = run_cascade(spec, scenario)

    assert result.failure_round == {
        "HUB": 0,
        "A": 1,
        "B": 1,
        "C": 1,
        "D": 1,
    }

    assert result.final_failure_count == 5
    assert result.systemic_collapse is True


def test_star_network_single_leaf_failure_does_not_fail_hub():
    spec = create_star_network()

    scenario = {
        "scenario_id": "leaf_A_fails",
        "initial_failed": ["A"],
        "exogenous_losses": {},
    }

    result = run_cascade(spec, scenario)

    assert result.failed_nodes == ["A"]
    assert result.cumulative_losses["HUB"] == 2.0
    assert result.final_failure_count == 1
    assert result.systemic_collapse is False


def test_scenario_from_binary_vector_uses_shared_format():
    node_order = ["A", "B", "C"]

    scenario = scenario_from_binary_vector(
        node_order,
        [1, 0, 1],
        scenario_id="binary_sample",
        metadata={
            "generator": "test_generator",
        },
    )

    assert scenario == {
        "scenario_id": "binary_sample",
        "initial_failed": ["A", "C"],
        "exogenous_losses": {},
        "metadata": {
            "generator": "test_generator",
        },
    }


def test_unknown_initial_failed_node_raises_error():
    spec = create_no_exposure_network()

    scenario = {
        "scenario_id": "bad_node",
        "initial_failed": ["UNKNOWN"],
        "exogenous_losses": {},
    }

    with pytest.raises(ValueError):
        run_cascade(spec, scenario)


def test_metrics_helpers():
    spec = create_toy_chain_network()

    result_1 = run_cascade(
        spec,
        {
            "scenario_id": "A_fails",
            "initial_failed": ["A"],
            "exogenous_losses": {},
        },
    )

    result_2 = run_cascade(
        spec,
        {
            "scenario_id": "no_failures",
            "initial_failed": [],
            "exogenous_losses": {},
        },
    )

    aggregate = aggregate_results([result_1, result_2])

    assert aggregate["num_scenarios"] == 2
    assert aggregate["max_final_failure_count"] == 5
    assert aggregate["max_cascade_depth"] == 3
    assert aggregate["systemic_collapse_frequency"] == 0.5

    assert failure_round_distribution(result_1) == {
        0: 1,
        1: 2,
        2: 1,
        3: 1,
    }

    assert tail_failure_probability(
        [result_1, result_2],
        min_failure_fraction=0.8,
    ) == 0.5
