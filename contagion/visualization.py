"""
Visualization helpers for contagion and clustering outputs.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from contagion.clustering import ClusterResult, PairLink
from contagion.simulator import CascadeResult
from contagion.spec import SystemSpec, get_node_order


def cluster_layout_positions(
    result: ClusterResult,
    *,
    cluster_radius: float = 4.0,
    node_radius: float = 0.9,
) -> Dict[int, Tuple[float, float]]:
    """
    Deterministically place clusters far apart and nodes inside each cluster.

    This is intentionally simple and stable.
    It avoids force-directed randomness.
    """

    cluster_count = len(result.clusters)
    positions: Dict[int, Tuple[float, float]] = {}

    if cluster_count == 1:
        cluster_centres = [(0.0, 0.0)]
    else:
        cluster_centres = []
        for cluster_id in range(cluster_count):
            angle = 2.0 * math.pi * cluster_id / cluster_count
            cluster_centres.append(
                (
                    cluster_radius * math.cos(angle),
                    cluster_radius * math.sin(angle),
                )
            )

    for cluster_id, cluster in enumerate(result.clusters):
        centre_x, centre_y = cluster_centres[cluster_id]
        size = len(cluster)

        if size == 1:
            positions[cluster[0]] = (centre_x, centre_y)
            continue

        local_radius = node_radius * max(1.0, math.sqrt(size / 3.0))

        for local_idx, node_idx in enumerate(cluster):
            angle = 2.0 * math.pi * local_idx / size
            positions[node_idx] = (
                centre_x + local_radius * math.cos(angle),
                centre_y + local_radius * math.sin(angle),
            )

    return positions


def plot_entanglement_layout(
    result: ClusterResult,
    *,
    title: str = "Financial dependency clusters and entanglement layout",
    show_classical: bool = True,
    max_classical_edges: int = 40,
    save_path: Optional[str] = None,
):
    """
    Plot clusters with entanglement and classical relationships.

    Solid thick lines:
        entangled pairs

    Thin dashed lines:
        classical non-entangled dependencies
    """

    positions = cluster_layout_positions(result)

    fig, ax = plt.subplots(figsize=(10, 8))

    if show_classical:
        classical_edges = sorted(
            result.classical_pairs,
            key=lambda p: (-p.strength, p.i, p.j),
        )[:max_classical_edges]

        for pair in classical_edges:
            _draw_edge(
                ax,
                pair,
                positions,
                linewidth=0.8,
                alpha=0.25,
                linestyle="--",
            )

    for pair in result.entangled_pairs:
        _draw_edge(
            ax,
            pair,
            positions,
            linewidth=2.4,
            alpha=0.85,
            linestyle="-",
        )

    cluster_labels = result.cluster_labels

    for idx, name in enumerate(result.institutions):
        x, y = positions[idx]
        ax.scatter(x, y, s=350)
        ax.text(
            x,
            y,
            name,
            ha="center",
            va="center",
            fontsize=9,
            fontweight="bold",
        )

        ax.text(
            x,
            y - 0.32,
            f"C{cluster_labels[idx]}",
            ha="center",
            va="center",
            fontsize=7,
        )

    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()

    if save_path is not None:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=180, bbox_inches="tight")

    return fig, ax


def plot_dependency_matrix(
    result: ClusterResult,
    *,
    title: str = "Dependency matrix",
    save_path: Optional[str] = None,
):
    """
    Heatmap of the final dependency matrix used for clustering.
    """

    fig, ax = plt.subplots(figsize=(8, 7))

    image = ax.imshow(result.dependency_matrix, vmin=0.0, vmax=1.0)

    ax.set_xticks(np.arange(len(result.institutions)))
    ax.set_yticks(np.arange(len(result.institutions)))

    ax.set_xticklabels(result.institutions, rotation=90)
    ax.set_yticklabels(result.institutions)

    ax.set_title(title)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout()

    if save_path is not None:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=180, bbox_inches="tight")

    return fig, ax


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


def _draw_edge(
    ax,
    pair: PairLink,
    positions: Dict[int, Tuple[float, float]],
    *,
    linewidth: float,
    alpha: float,
    linestyle: str,
) -> None:
    x1, y1 = positions[pair.i]
    x2, y2 = positions[pair.j]

    ax.plot(
        [x1, x2],
        [y1, y2],
        linewidth=linewidth,
        alpha=alpha,
        linestyle=linestyle,
    )

    mid_x = 0.5 * (x1 + x2)
    mid_y = 0.5 * (y1 + y2)

    if pair.relation == "entangled":
        ax.text(
            mid_x,
            mid_y,
            f"{pair.strength:.2f}",
            fontsize=7,
            ha="center",
            va="center",
        )
