from pprint import pprint

from contagion.metrics import failure_round_distribution, result_summary
from contagion.simulator import run_cascade
from contagion.toy_networks import create_toy_chain_network
from contagion.visualization import plot_cascade


def main() -> None:
    spec = create_toy_chain_network()

    scenario = {
        "scenario_id": "toy_A_fails",
        "initial_failed": ["A"],
        "exogenous_losses": {},
        "metadata": {
            "generator": "manual_example",
        },
    }

    result = run_cascade(spec, scenario)

    important_result = {
        "failure_round": result.failure_round,
        "failed_nodes": result.failed_nodes,
        "final_failure_count": result.final_failure_count,
        "failure_fraction": result.failure_fraction,
        "cascade_depth": result.cascade_depth,
        "systemic_collapse": result.systemic_collapse,
    }

    print("\nImportant cascade result:")
    pprint(important_result)

    print("\nCompact summary:")
    pprint(result_summary(result))

    print("\nFailure round distribution:")
    pprint(failure_round_distribution(result))

    plot_cascade(
        spec,
        result,
        save_path="outputs/toy_cascade.png",
    )

    print("\nSaved visualization to outputs/toy_cascade.png")


if __name__ == "__main__":
    main()
