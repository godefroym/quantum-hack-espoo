from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from systemic_risk.spec import SystemSpec


_TYPE_COLORS = {
    "bank": "#2f6f9f",
    "insurer": "#6f8f3a",
    "fund": "#9f6b2f",
    "corporate": "#8f4d6f",
    "sovereign": "#4f4f4f",
    "CCP": "#b24a3c",
}


def build_financial_graph(spec: SystemSpec) -> nx.DiGraph:
    graph = nx.DiGraph()
    for idx, name in enumerate(spec.node_names):
        graph.add_node(
            name,
            node_type=spec.node_types[idx],
            cluster=None if spec.clusters is None else spec.clusters[idx],
            p_default=float(spec.marginal_default_probs[idx]),
            capital=float(spec.capital_buffers[idx]),
        )
    for i, borrower in enumerate(spec.node_names):
        for j, defaulting in enumerate(spec.node_names):
            weight = spec.exposure_matrix[i, j]
            if weight > 0:
                graph.add_edge(defaulting, borrower, weight=float(weight))
    return graph


def plot_financial_network(
    spec: SystemSpec,
    path: str | Path | None = None,
    seed: int = 11,
) -> plt.Figure:
    graph = build_financial_graph(spec)
    pos = nx.spring_layout(graph, seed=seed, weight="weight")
    fig, ax = plt.subplots(figsize=(11, 7))
    node_colors = [_TYPE_COLORS.get(spec.node_types[i], "#777777") for i in range(spec.n)]
    node_sizes = 550 + 450 * spec.marginal_default_probs / max(spec.marginal_default_probs.max(), 1e-9)
    widths = np.array([graph[u][v]["weight"] for u, v in graph.edges()])
    if len(widths) > 0:
        widths = 0.4 + 2.8 * widths / widths.max()

    nx.draw_networkx_edges(
        graph,
        pos,
        ax=ax,
        width=widths,
        arrowsize=12,
        alpha=0.35,
        edge_color="#555555",
    )
    nx.draw_networkx_nodes(
        graph,
        pos,
        ax=ax,
        node_color=node_colors,
        node_size=node_sizes,
        linewidths=0.8,
        edgecolors="#1f1f1f",
    )
    nx.draw_networkx_labels(graph, pos, ax=ax, font_size=8)
    ax.set_title("Synthetic Financial Exposure Network")
    ax.axis("off")
    fig.tight_layout()
    if path is not None:
        fig.savefig(path, dpi=180)
    return fig
