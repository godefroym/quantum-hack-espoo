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


_COMMUNITY_PALETTE = [
    "#2f6f9f", "#b24a3c", "#6f8f3a", "#9f6b2f", "#8f4d6f",
    "#3a8f8f", "#7a5fb0", "#b08a2f", "#4f4f4f", "#c0507a",
]


def plot_community_network(
    spec: SystemSpec,
    path: str | Path | None = None,
    seed: int = 11,
    title: str = "Financial System Exposure Network — Detected Communities",
    max_edges: int | None = 90,
) -> plt.Figure:
    """Render the network coloured by detected community, with a community-aware layout.

    Nodes are coloured by their ``spec.clusters`` label (the community-detection output),
    sized by marginal default probability, and laid out so community structure is visible.
    This is the legibility deliverable for part A: one clean plot that makes the cluster
    story obvious. Falls back to type-colouring if no clusters are attached.

    ``max_edges`` keeps only that many heaviest exposures for drawing (dense max-entropy
    reconstructions are near-complete graphs and would otherwise render as a hairball). Set to
    ``None`` to draw every edge.
    """
    if spec.clusters is None:
        return plot_financial_network(spec, path=path, seed=seed)

    graph = build_financial_graph(spec)
    labels_sorted = sorted(set(spec.clusters))
    color_of = {lab: _COMMUNITY_PALETTE[i % len(_COMMUNITY_PALETTE)]
                for i, lab in enumerate(labels_sorted)}

    # Community-aware layout computed directly from spec.clusters (not a spring relaxation):
    # well-separated community centres on a circle, members ringed around their own centre.
    # Deterministic, so a dense max-entropy graph's heavy central hub never collapses to a blob.
    n_comm = len(labels_sorted)
    centre_angles = np.linspace(0, 2 * np.pi, n_comm, endpoint=False)
    centres = {
        lab: np.array([np.cos(a), np.sin(a)]) * (3.2 if n_comm > 1 else 0.0)
        for lab, a in zip(labels_sorted, centre_angles)
    }
    members: dict[object, list[int]] = {lab: [] for lab in labels_sorted}
    for i in range(spec.n):
        members[spec.clusters[i]].append(i)
    pos: dict[str, np.ndarray] = {}
    for lab, idxs in members.items():
        m = len(idxs)
        radius = 0.0 if m == 1 else 0.55 + 0.12 * m
        for k, i in enumerate(idxs):
            a = 2 * np.pi * k / max(m, 1)
            pos[spec.node_names[i]] = centres[lab] + radius * np.array([np.cos(a), np.sin(a)])

    fig, ax = plt.subplots(figsize=(11, 7.5))
    node_colors = [color_of[spec.clusters[i]] for i in range(spec.n)]
    node_sizes = 550 + 600 * spec.marginal_default_probs / max(
        spec.marginal_default_probs.max(), 1e-9
    )

    # For drawing, keep only the heaviest edges so the community structure is legible;
    # colour an edge by its endpoints' community (grey if it crosses communities).
    edges = sorted(graph.edges(data=True), key=lambda e: e[2]["weight"], reverse=True)
    if max_edges is not None:
        edges = edges[:max_edges]
    cluster_of = {spec.node_names[i]: spec.clusters[i] for i in range(spec.n)}
    edge_list = [(u, v) for u, v, _ in edges]
    weights = np.array([d["weight"] for _, _, d in edges])
    widths = 0.3 + 2.6 * weights / weights.max() if len(weights) else weights
    edge_colors = [
        color_of[cluster_of[u]] if cluster_of[u] == cluster_of[v] else "#9a9a9a"
        for u, v in edge_list
    ]

    nx.draw_networkx_edges(graph, pos, ax=ax, edgelist=edge_list, width=widths,
                           arrowsize=8, alpha=0.30, edge_color=edge_colors)

    # Node colour = community; node shape = class (circle = financial, square = corporate),
    # so the non-financial companies are visually distinct from the banks/insurers/funds.
    is_corp = [t == "corporate" for t in spec.node_types]
    for shape, want_corp in (("o", False), ("s", True)):
        idxs = [i for i in range(spec.n) if is_corp[i] == want_corp]
        if not idxs:
            continue
        nx.draw_networkx_nodes(
            graph, pos, ax=ax,
            nodelist=[spec.node_names[i] for i in idxs],
            node_color=[node_colors[i] for i in idxs],
            node_size=[node_sizes[i] for i in idxs],
            node_shape=shape, linewidths=0.8, edgecolors="#1f1f1f",
        )
    nx.draw_networkx_labels(graph, pos, ax=ax, font_size=7.5)

    community_handles = [
        plt.Line2D([0], [0], marker="o", linestyle="", markersize=10,
                   markerfacecolor=color_of[lab], markeredgecolor="#1f1f1f",
                   label=str(lab))
        for lab in labels_sorted
    ]
    community_legend = ax.legend(handles=community_handles, title="Community",
                                 loc="upper left", fontsize=8, title_fontsize=9,
                                 framealpha=0.9)
    ax.add_artist(community_legend)
    if any(is_corp):
        class_handles = [
            plt.Line2D([0], [0], marker="o", linestyle="", markersize=10,
                       markerfacecolor="#bbbbbb", markeredgecolor="#1f1f1f",
                       label="financial"),
            plt.Line2D([0], [0], marker="s", linestyle="", markersize=10,
                       markerfacecolor="#bbbbbb", markeredgecolor="#1f1f1f",
                       label="corporate"),
        ]
        ax.legend(handles=class_handles, title="Node class", loc="upper right",
                  fontsize=8, title_fontsize=9, framealpha=0.9)
    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()
    if path is not None:
        fig.savefig(path, dpi=180)
    return fig


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
