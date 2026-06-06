from __future__ import annotations

from pathlib import Path
from typing import Any

from contagion.simulator import CascadeResult
from contagion.spec import SystemSpec, get_node_order


def plot_cascade(
    spec: SystemSpec,
    result: CascadeResult,
    *,
    save_path: str | Path | None = None,
    show_edge_labels: bool = True,
) -> Any:
    """
    Minimal network visualization.

    Requires:
        pip install matplotlib networkx

    Node color indicates failure round.
    Healthy nodes are placed after the final failure round on the color scale.
    """

    import matplotlib.pyplot as plt
    import networkx as nx

    graph = nx.DiGraph()

    node_order = get_node_order(spec)

    for node_id in node_order:
        graph.add_node(node_id)

    for edge in spec["edges"]:
        graph.add_edge(
            edge["source"],
            edge["target"],
            weight=float(edge["exposure"]),
        )

    pos = nx.spring_layout(graph, seed=42)

    max_round = max(result.failure_round.values(), default=0)
    healthy_value = max_round + 1

    node_values = [result.failure_round.get(node_id, healthy_value) for node_id in node_order]

    node_labels = {
        node_id: (
            f"{node_id}\nr={result.failure_round[node_id]}"
            if node_id in result.failure_round
            else f"{node_id}\nhealthy"
        )
        for node_id in node_order
    }

    fig, ax = plt.subplots(figsize=(8, 6))

    nx.draw_networkx_edges(
        graph,
        pos,
        ax=ax,
        arrows=True,
        arrowstyle="->",
        width=1.5,
    )

    nodes = nx.draw_networkx_nodes(
        graph,
        pos,
        ax=ax,
        node_color=node_values,
        cmap="viridis",
        node_size=1500,
    )

    nx.draw_networkx_labels(
        graph,
        pos,
        labels=node_labels,
        ax=ax,
        font_size=9,
    )

    if show_edge_labels:
        edge_labels = {
            (edge["source"], edge["target"]): str(edge["exposure"])
            for edge in spec["edges"]
        }

        nx.draw_networkx_edge_labels(
            graph,
            pos,
            edge_labels=edge_labels,
            ax=ax,
            font_size=8,
        )

    title = (
        f"Scenario: {result.scenario_id or 'unnamed'} | "
        f"Cascade size: {result.final_failure_count}/{result.node_count} | "
        f"Depth: {result.cascade_depth} | "
        f"Systemic: {result.systemic_collapse}"
    )

    ax.set_title(title)
    ax.axis("off")

    colorbar = fig.colorbar(nodes, ax=ax)
    colorbar.set_label("Failure round; healthy nodes shown after final round")

    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=160)

    return fig, ax
