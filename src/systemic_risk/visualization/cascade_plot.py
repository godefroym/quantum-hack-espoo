from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from systemic_risk.simulator import CascadeResult
from systemic_risk.spec import SystemSpec


def plot_cascade(
    spec: SystemSpec,
    result: CascadeResult,
    path: str | Path | None = None,
    *,
    show_edge_labels: bool = False,
) -> plt.Figure:
    """Plot failure rounds on the canonical directed exposure network."""
    graph = nx.DiGraph()
    graph.add_nodes_from(range(spec.n))
    for target, source in zip(*np.nonzero(spec.exposure_matrix)):
        graph.add_edge(
            int(source),
            int(target),
            weight=float(spec.exposure_matrix[target, source]),
        )

    positions = nx.spring_layout(graph, seed=42, weight="weight")
    healthy_value = result.cascade_depth + 1
    node_values = [
        int(round_index) if round_index >= 0 else healthy_value
        for round_index in result.failure_round
    ]
    labels = {
        i: (
            f"{name}\nr={result.failure_round[i]}"
            if result.failure_round[i] >= 0
            else f"{name}\nhealthy"
        )
        for i, name in enumerate(spec.node_names)
    }

    fig, ax = plt.subplots(figsize=(10, 7))
    nx.draw_networkx_edges(
        graph,
        positions,
        ax=ax,
        arrows=True,
        arrowstyle="->",
        alpha=0.45,
        width=1.2,
    )
    nodes = nx.draw_networkx_nodes(
        graph,
        positions,
        ax=ax,
        node_color=node_values,
        cmap="viridis",
        node_size=1200,
    )
    nx.draw_networkx_labels(
        graph,
        positions,
        labels=labels,
        ax=ax,
        font_size=8,
    )
    if show_edge_labels:
        edge_labels = {
            (source, target): f"{attributes['weight']:.2g}"
            for source, target, attributes in graph.edges(data=True)
        }
        nx.draw_networkx_edge_labels(
            graph,
            positions,
            edge_labels=edge_labels,
            ax=ax,
            font_size=7,
        )

    title = (
        f"Cascade {result.scenario_id or 'scenario'}: "
        f"{result.failure_count}/{spec.n} failed, depth {result.cascade_depth}"
    )
    ax.set_title(title)
    ax.axis("off")
    colorbar = fig.colorbar(nodes, ax=ax)
    colorbar.set_label("Failure round; healthy nodes follow the final round")
    fig.tight_layout()

    if path is not None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=180)
    return fig
