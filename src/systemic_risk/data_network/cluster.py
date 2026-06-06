"""Community detection + stability.

Communities are detected on the real equity-return correlation graph (the genuine
co-movement signal — banks that move together under common shocks fall in the same
community, which tends to surface regional / business-model structure). Greedy modularity
maximization (Clauset-Newman-Moore, via networkx) gives integer community labels.

Stability is checked by perturbing the correlation matrix with small symmetric noise,
re-detecting, and measuring the adjusted Rand index against the unperturbed labels,
averaged over several perturbations. A high mean ARI means the community structure is a
property of the data, not of the algorithm's tie-breaking.
"""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np


def _graph_from_correlation(corr: np.ndarray, threshold: float) -> nx.Graph:
    """Undirected weighted graph: edge (i,j) with weight = positive correlation above cut."""
    n = corr.shape[0]
    graph = nx.Graph()
    graph.add_nodes_from(range(n))
    for i in range(n):
        for j in range(i + 1, n):
            w = float(corr[i, j])
            if w > threshold:
                graph.add_edge(i, j, weight=w)
    return graph


def detect_communities(
    corr: np.ndarray,
    threshold: float = 0.0,
    resolution: float = 1.0,
) -> tuple[np.ndarray, float]:
    """Greedy-modularity communities on the correlation graph.

    Returns ``(labels, modularity)`` where ``labels[i]`` is node ``i``'s integer community.
    Community ids are assigned in descending community-size order for determinism.
    """
    n = corr.shape[0]
    graph = _graph_from_correlation(corr, threshold)
    if graph.number_of_edges() == 0:
        return np.zeros(n, dtype=int), 0.0

    communities = nx.community.greedy_modularity_communities(
        graph, weight="weight", resolution=resolution
    )
    # Deterministic labelling: larger communities first, then by smallest member index.
    ordered = sorted(communities, key=lambda c: (-len(c), min(c)))
    labels = np.full(n, -1, dtype=int)
    for label, members in enumerate(ordered):
        for node in members:
            labels[node] = label
    # Any isolated nodes (no qualifying edges) form singleton communities.
    next_label = len(ordered)
    for i in range(n):
        if labels[i] == -1:
            labels[i] = next_label
            next_label += 1
    modularity = float(nx.community.modularity(graph, ordered, weight="weight")) if ordered else 0.0
    return labels, modularity


def adjusted_rand_index(a: np.ndarray, b: np.ndarray) -> float:
    """Adjusted Rand index between two label vectors (no scikit-learn dependency)."""
    a = np.asarray(a)
    b = np.asarray(b)
    n = a.size
    if n < 2:
        return 1.0
    a_labels = {v: i for i, v in enumerate(np.unique(a))}
    b_labels = {v: i for i, v in enumerate(np.unique(b))}
    contingency = np.zeros((len(a_labels), len(b_labels)), dtype=float)
    for x, y in zip(a, b):
        contingency[a_labels[x], b_labels[y]] += 1

    def comb2(x: np.ndarray) -> np.ndarray:
        return x * (x - 1.0) / 2.0

    sum_comb_c = comb2(contingency.sum(axis=1)).sum()
    sum_comb_k = comb2(contingency.sum(axis=0)).sum()
    sum_comb = comb2(contingency).sum()
    total_comb = n * (n - 1.0) / 2.0
    expected = sum_comb_c * sum_comb_k / total_comb if total_comb > 0 else 0.0
    maximum = 0.5 * (sum_comb_c + sum_comb_k)
    denom = maximum - expected
    if denom == 0:
        return 1.0
    return float((sum_comb - expected) / denom)


@dataclass(frozen=True)
class ClusterReport:
    labels: np.ndarray
    modularity: float
    n_communities: int
    mean_ari: float
    min_ari: float
    stable: bool


def cluster_stability(
    corr: np.ndarray,
    base_labels: np.ndarray,
    n_perturb: int = 8,
    noise: float = 0.05,
    threshold: float = 0.0,
    seed: int = 0,
    stable_ari: float = 0.6,
) -> ClusterReport:
    """Re-detect communities under small correlation perturbations; report mean/min ARI."""
    rng = np.random.default_rng(seed)
    n = corr.shape[0]
    base_mod = float("nan")
    aris: list[float] = []
    for _ in range(n_perturb):
        pert = rng.normal(0.0, noise, size=(n, n))
        pert = (pert + pert.T) / 2.0
        np.fill_diagonal(pert, 0.0)
        noisy = np.clip(corr + pert, -1.0, 1.0)
        np.fill_diagonal(noisy, 1.0)
        labels, _ = detect_communities(noisy, threshold=threshold)
        aris.append(adjusted_rand_index(base_labels, labels))
    mean_ari = float(np.mean(aris)) if aris else 1.0
    min_ari = float(np.min(aris)) if aris else 1.0
    return ClusterReport(
        labels=np.asarray(base_labels, dtype=int),
        modularity=base_mod,
        n_communities=int(len(np.unique(base_labels))),
        mean_ari=mean_ari,
        min_ari=min_ari,
        stable=mean_ari >= stable_ari,
    )


def cluster_with_stability(
    corr: np.ndarray,
    threshold: float = 0.0,
    n_perturb: int = 8,
    noise: float = 0.05,
    seed: int = 0,
    stable_ari: float = 0.6,
) -> ClusterReport:
    """Detect communities and immediately assess their stability."""
    labels, modularity = detect_communities(corr, threshold=threshold)
    report = cluster_stability(
        corr, labels, n_perturb=n_perturb, noise=noise,
        threshold=threshold, seed=seed, stable_ari=stable_ari,
    )
    return ClusterReport(
        labels=labels,
        modularity=modularity,
        n_communities=int(len(np.unique(labels))),
        mean_ari=report.mean_ari,
        min_ari=report.min_ari,
        stable=report.stable,
    )


__all__ = [
    "detect_communities",
    "adjusted_rand_index",
    "cluster_stability",
    "cluster_with_stability",
    "ClusterReport",
]
