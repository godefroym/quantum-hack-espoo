from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, TypeAlias


SystemSpec: TypeAlias = dict[str, Any]
Scenario: TypeAlias = dict[str, Any]


def load_json(path: str | Path) -> dict[str, Any]:
    path = Path(path)

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def get_node_order(spec: SystemSpec) -> list[str]:
    return [node["id"] for node in spec["nodes"]]


def get_node_set(spec: SystemSpec) -> set[str]:
    return set(get_node_order(spec))


def get_capital_map(spec: SystemSpec) -> dict[str, float]:
    return {node["id"]: float(node["capital"]) for node in spec["nodes"]}


def ordered_subset(node_order: list[str], node_set: set[str]) -> list[str]:
    return [node_id for node_id in node_order if node_id in node_set]


def build_outgoing_edges(spec: SystemSpec) -> dict[str, list[dict[str, Any]]]:
    """
    Builds outgoing adjacency lists.

    Edge convention:
        source -> target means:
        if source fails, target receives exposure * lgd as a loss.
    """

    node_order = get_node_order(spec)
    node_index = {node_id: i for i, node_id in enumerate(node_order)}

    outgoing: dict[str, list[dict[str, Any]]] = {node_id: [] for node_id in node_order}

    for edge in spec["edges"]:
        outgoing[edge["source"]].append(edge)

    for source in outgoing:
        outgoing[source] = sorted(
            outgoing[source],
            key=lambda edge: (
                node_index[edge["target"]],
                edge["source"],
                edge["target"],
                float(edge["exposure"]),
                float(edge.get("lgd", 1.0)),
            ),
        )

    return outgoing


def validate_system_spec(spec: SystemSpec) -> None:
    if not isinstance(spec, dict):
        raise TypeError("System spec must be a dictionary")

    if "nodes" not in spec:
        raise ValueError("System spec must contain 'nodes'")

    if "edges" not in spec:
        raise ValueError("System spec must contain 'edges'")

    if not isinstance(spec["nodes"], list):
        raise TypeError("'nodes' must be a list")

    if not isinstance(spec["edges"], list):
        raise TypeError("'edges' must be a list")

    node_ids: list[str] = []

    for node in spec["nodes"]:
        if not isinstance(node, dict):
            raise TypeError("Each node must be a dictionary")

        if "id" not in node:
            raise ValueError("Each node must contain 'id'")

        if "capital" not in node:
            raise ValueError(f"Node {node['id']} must contain 'capital'")

        node_id = node["id"]

        if not isinstance(node_id, str) or not node_id:
            raise ValueError("Node id must be a non-empty string")

        capital = float(node["capital"])

        if capital < 0:
            raise ValueError(f"Node {node_id} has negative capital")

        node_ids.append(node_id)

    if len(node_ids) != len(set(node_ids)):
        raise ValueError("Node ids must be unique")

    node_set = set(node_ids)

    for edge in spec["edges"]:
        if not isinstance(edge, dict):
            raise TypeError("Each edge must be a dictionary")

        for key in ("source", "target", "exposure"):
            if key not in edge:
                raise ValueError(f"Each edge must contain '{key}'")

        source = edge["source"]
        target = edge["target"]

        if source not in node_set:
            raise ValueError(f"Edge source references unknown node: {source}")

        if target not in node_set:
            raise ValueError(f"Edge target references unknown node: {target}")

        exposure = float(edge["exposure"])

        if exposure < 0:
            raise ValueError(f"Edge {source}->{target} has negative exposure")

        lgd = float(edge.get("lgd", 1.0))

        if lgd < 0:
            raise ValueError(f"Edge {source}->{target} has negative lgd")


def validate_scenario(spec: SystemSpec, scenario: Scenario) -> None:
    if not isinstance(scenario, dict):
        raise TypeError("Scenario must be a dictionary")

    node_set = get_node_set(spec)

    initial_failed = scenario.get("initial_failed", [])

    if not isinstance(initial_failed, list):
        raise TypeError("'initial_failed' must be a list")

    unknown_initial_failures = set(initial_failed) - node_set

    if unknown_initial_failures:
        raise ValueError(
            "Scenario contains unknown initial failed nodes: "
            f"{sorted(unknown_initial_failures)}"
        )

    exogenous_losses = scenario.get("exogenous_losses", {})

    if not isinstance(exogenous_losses, dict):
        raise TypeError("'exogenous_losses' must be a dictionary")

    for node_id, loss in exogenous_losses.items():
        if node_id not in node_set:
            raise ValueError(f"Scenario contains unknown shocked node: {node_id}")

        if float(loss) < 0:
            raise ValueError(f"Scenario contains negative exogenous loss for {node_id}")


def scenario_from_binary_vector(
    node_order: list[str],
    failure_vector: list[int | bool],
    *,
    scenario_id: str = "",
    metadata: dict[str, Any] | None = None,
) -> Scenario:
    """
    Helper for B/C.

    Converts a binary generator sample into D's shared scenario format.

    Example:
        node_order = ["A", "B", "C"]
        failure_vector = [1, 0, 1]

        returns initial_failed = ["A", "C"]
    """

    if len(node_order) != len(failure_vector):
        raise ValueError("node_order and failure_vector must have the same length")

    initial_failed = [
        node_id for node_id, failed in zip(node_order, failure_vector) if bool(failed)
    ]

    return {
        "scenario_id": scenario_id,
        "initial_failed": initial_failed,
        "exogenous_losses": {},
        "metadata": metadata or {},
    }


def scenario_from_loss_vector(
    node_order: list[str],
    loss_vector: list[float],
    *,
    scenario_id: str = "",
    metadata: dict[str, Any] | None = None,
) -> Scenario:
    """
    Helper for B/C.

    Converts a vector of direct exogenous losses into D's shared scenario format.
    """

    if len(node_order) != len(loss_vector):
        raise ValueError("node_order and loss_vector must have the same length")

    exogenous_losses = {
        node_id: float(loss)
        for node_id, loss in zip(node_order, loss_vector)
        if float(loss) != 0.0
    }

    return {
        "scenario_id": scenario_id,
        "initial_failed": [],
        "exogenous_losses": exogenous_losses,
        "metadata": metadata or {},
    }


def aggregate_edge_exposures(spec: SystemSpec) -> dict[tuple[str, str], float]:
    """
    Optional audit helper.

    Returns total exposure by directed pair.
    """

    exposures: defaultdict[tuple[str, str], float] = defaultdict(float)

    for edge in spec["edges"]:
        exposures[(edge["source"], edge["target"])] += float(edge["exposure"])

    return dict(exposures)
