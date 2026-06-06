"""Deterministic clustering and entanglement-layout utilities.

This module is intentionally deterministic and generator-agnostic.

It does NOT run the contagion simulator.
It decides which institutions are close enough to justify entanglement structure.

Core idea:
    dependency(i, j)
        = weighted positive correlation
        + weighted symmetrised exposure strength

Strong intra-cluster dependencies become entanglement candidates.
Weak or cross-cluster dependencies remain classical.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Iterable, List, Literal, Optional, Sequence, Tuple

import numpy as np

from systemic_risk.spec import SystemSpec


CorrelationMode = Literal["positive", "absolute"]


@dataclass(frozen=True)
class PairLink:
    """
    A relationship between two institutions.

    relation:
        "entangled" means this pair is close enough to encode using entanglement.
        "classical" means this pair should remain classical / non-entangled.
    """

    i: int
    j: int
    institution_i: str
    institution_j: str
    strength: float
    distance: float
    cluster_i: int
    cluster_j: int
    relation: Literal["entangled", "classical"]

    def as_tuple(self) -> Tuple[str, str, float, str]:
        return (
            self.institution_i,
            self.institution_j,
            self.strength,
            self.relation,
        )


@dataclass(frozen=True)
class ClusterResult:
    """
    Full result of the clustering + entanglement layout pass.
    """

    institutions: Tuple[str, ...]
    dependency_matrix: np.ndarray
    cluster_labels: Tuple[int, ...]
    clusters: Tuple[Tuple[int, ...], ...]
    entangled_pairs: Tuple[PairLink, ...]
    classical_pairs: Tuple[PairLink, ...]
    entanglement_layers: Tuple[Tuple[PairLink, ...], ...]
    cluster_threshold: float
    entangle_threshold: float
    classical_threshold: float

    def cluster_names(self) -> List[List[str]]:
        return [[self.institutions[i] for i in cluster] for cluster in self.clusters]

    def summary(self) -> str:
        lines = []
        lines.append("Clustering / entanglement layout summary")
        lines.append("=" * 48)
        lines.append(f"Institution count:      {len(self.institutions)}")
        lines.append(f"Cluster count:          {len(self.clusters)}")
        lines.append(f"Entangled pair count:   {len(self.entangled_pairs)}")
        lines.append(f"Classical pair count:   {len(self.classical_pairs)}")
        lines.append(f"Entanglement layers:    {len(self.entanglement_layers)}")
        lines.append("")
        lines.append("Clusters:")
        for cluster_id, cluster in enumerate(self.clusters):
            names = ", ".join(self.institutions[i] for i in cluster)
            lines.append(f"  Cluster {cluster_id}: {names}")
        lines.append("")
        lines.append("Entangled pairs:")
        if not self.entangled_pairs:
            lines.append("  None")
        else:
            for pair in self.entangled_pairs:
                lines.append(
                    f"  {pair.institution_i} -- {pair.institution_j} "
                    f"(strength={pair.strength:.3f})"
                )
        return "\n".join(lines)


def build_clustering_layout(
    *,
    institutions: Sequence[str],
    correlation_matrix: Sequence[Sequence[float]],
    exposure_matrix: Optional[Sequence[Sequence[float]]] = None,
    corr_weight: float = 0.75,
    exposure_weight: float = 0.25,
    correlation_mode: CorrelationMode = "positive",
    cluster_threshold: float = 0.55,
    entangle_threshold: float = 0.65,
    classical_threshold: float = 0.10,
    max_entangled_degree: Optional[int] = 3,
) -> ClusterResult:
    """
    Build clusters and an entanglement layout from financial dependency data.
    """

    names = tuple(str(x) for x in institutions)
    if len(names) == 0:
        raise ValueError("institutions must not be empty")

    corr = _as_square_matrix(correlation_matrix, "correlation_matrix")
    n = corr.shape[0]

    if len(names) != n:
        raise ValueError(
            "Number of institutions must match correlation_matrix size: "
            f"{len(names)} institutions but matrix is {n}x{n}"
        )

    if exposure_matrix is None:
        exposure = np.zeros((n, n), dtype=float)
    else:
        exposure = _as_square_matrix(exposure_matrix, "exposure_matrix")
        if exposure.shape != corr.shape:
            raise ValueError(
                "exposure_matrix must have same shape as correlation_matrix"
            )

    dependency = build_dependency_matrix(
        correlation_matrix=corr,
        exposure_matrix=exposure,
        corr_weight=corr_weight,
        exposure_weight=exposure_weight,
        correlation_mode=correlation_mode,
    )

    clusters, labels = threshold_connected_components(
        dependency,
        threshold=cluster_threshold,
    )

    entangled_candidates, classical_pairs = classify_pairs(
        institutions=names,
        dependency_matrix=dependency,
        cluster_labels=labels,
        entangle_threshold=entangle_threshold,
        classical_threshold=classical_threshold,
    )

    entangled_pairs = sparsify_entanglement_pairs(
        entangled_candidates,
        n_nodes=n,
        max_degree=max_entangled_degree,
    )
    kept = {(pair.i, pair.j) for pair in entangled_pairs}
    classical_pairs.extend(
        replace(pair, relation="classical")
        for pair in entangled_candidates
        if (pair.i, pair.j) not in kept
    )
    classical_pairs.sort(key=lambda pair: (-pair.strength, pair.i, pair.j))

    layers = build_entanglement_layers(entangled_pairs, n_nodes=n)

    return ClusterResult(
        institutions=names,
        dependency_matrix=dependency,
        cluster_labels=tuple(labels),
        clusters=tuple(tuple(cluster) for cluster in clusters),
        entangled_pairs=tuple(entangled_pairs),
        classical_pairs=tuple(classical_pairs),
        entanglement_layers=tuple(tuple(layer) for layer in layers),
        cluster_threshold=cluster_threshold,
        entangle_threshold=entangle_threshold,
        classical_threshold=classical_threshold,
    )


def build_dependency_matrix(
    *,
    correlation_matrix: Sequence[Sequence[float]],
    exposure_matrix: Optional[Sequence[Sequence[float]]] = None,
    corr_weight: float = 0.75,
    exposure_weight: float = 0.25,
    correlation_mode: CorrelationMode = "positive",
) -> np.ndarray:
    """
    Convert correlations and exposures into a symmetric dependency matrix.

    Output is in [0, 1], with zero diagonal.
    """

    corr = _as_square_matrix(correlation_matrix, "correlation_matrix")
    n = corr.shape[0]
    has_exposure_input = exposure_matrix is not None

    if not has_exposure_input:
        exposure = np.zeros((n, n), dtype=float)
    else:
        exposure = _as_square_matrix(exposure_matrix, "exposure_matrix")
        if exposure.shape != corr.shape:
            raise ValueError(
                "exposure_matrix must have same shape as correlation_matrix"
            )

    if not has_exposure_input:
        exposure_weight = 0.0

    if corr_weight < 0 or exposure_weight < 0:
        raise ValueError("corr_weight and exposure_weight must be non-negative")

    if correlation_mode == "positive":
        corr_score = np.clip(corr, 0.0, 1.0)
    elif correlation_mode == "absolute":
        corr_score = np.abs(np.clip(corr, -1.0, 1.0))
    else:
        raise ValueError("correlation_mode must be either 'positive' or 'absolute'")

    np.fill_diagonal(corr_score, 0.0)

    exposure_score = _normalise_exposure_matrix(exposure)

    if not np.any(exposure_score > 0.0):
        exposure_weight = 0.0

    if not np.any(corr_score > 0.0):
        corr_weight = 0.0

    total_weight = corr_weight + exposure_weight
    if total_weight == 0:
        return np.zeros_like(corr_score)

    corr_weight = corr_weight / total_weight
    exposure_weight = exposure_weight / total_weight

    dependency = corr_weight * corr_score + exposure_weight * exposure_score
    dependency = np.clip(dependency, 0.0, 1.0)
    dependency = 0.5 * (dependency + dependency.T)
    np.fill_diagonal(dependency, 0.0)

    return dependency


def threshold_connected_components(
    dependency_matrix: Sequence[Sequence[float]],
    *,
    threshold: float,
) -> Tuple[List[List[int]], List[int]]:
    """
    Deterministically cluster nodes using threshold-connected components.
    """

    matrix = _as_square_matrix(dependency_matrix, "dependency_matrix")
    n = matrix.shape[0]

    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be in [0, 1]")

    visited = [False] * n
    clusters: List[List[int]] = []

    for start in range(n):
        if visited[start]:
            continue

        stack = [start]
        visited[start] = True
        component = []

        while stack:
            node = stack.pop()
            component.append(node)

            neighbours = [
                other
                for other in range(n)
                if other != node
                and not visited[other]
                and matrix[node, other] >= threshold
            ]

            for other in sorted(neighbours, reverse=True):
                visited[other] = True
                stack.append(other)

        clusters.append(sorted(component))

    clusters.sort(key=lambda cluster: (cluster[0], len(cluster)))

    labels = [-1] * n
    for cluster_id, cluster in enumerate(clusters):
        for node in cluster:
            labels[node] = cluster_id

    return clusters, labels


def classify_pairs(
    *,
    institutions: Sequence[str],
    dependency_matrix: Sequence[Sequence[float]],
    cluster_labels: Sequence[int],
    entangle_threshold: float,
    classical_threshold: float,
) -> Tuple[List[PairLink], List[PairLink]]:
    """
    Split pairwise relationships into entangled and classical.
    """

    names = tuple(str(x) for x in institutions)
    matrix = _as_square_matrix(dependency_matrix, "dependency_matrix")
    n = matrix.shape[0]

    if len(names) != n:
        raise ValueError("institutions length must match dependency_matrix size")

    if len(cluster_labels) != n:
        raise ValueError("cluster_labels length must match dependency_matrix size")

    if not 0 <= entangle_threshold <= 1:
        raise ValueError("entangle_threshold must be in [0, 1]")

    if not 0 <= classical_threshold <= 1:
        raise ValueError("classical_threshold must be in [0, 1]")

    entangled: List[PairLink] = []
    classical: List[PairLink] = []

    for i in range(n):
        for j in range(i + 1, n):
            strength = float(matrix[i, j])
            distance = 1.0 - strength
            same_cluster = cluster_labels[i] == cluster_labels[j]

            if same_cluster and strength >= entangle_threshold:
                entangled.append(
                    PairLink(
                        i=i,
                        j=j,
                        institution_i=names[i],
                        institution_j=names[j],
                        strength=strength,
                        distance=distance,
                        cluster_i=int(cluster_labels[i]),
                        cluster_j=int(cluster_labels[j]),
                        relation="entangled",
                    )
                )
            elif strength >= classical_threshold:
                classical.append(
                    PairLink(
                        i=i,
                        j=j,
                        institution_i=names[i],
                        institution_j=names[j],
                        strength=strength,
                        distance=distance,
                        cluster_i=int(cluster_labels[i]),
                        cluster_j=int(cluster_labels[j]),
                        relation="classical",
                    )
                )

    entangled.sort(key=lambda p: (-p.strength, p.i, p.j))
    classical.sort(key=lambda p: (-p.strength, p.i, p.j))

    return entangled, classical


def sparsify_entanglement_pairs(
    pairs: Sequence[PairLink],
    *,
    n_nodes: int,
    max_degree: Optional[int],
) -> List[PairLink]:
    """
    Keep entanglement layout sparse by limiting each node's entangled degree.
    """

    _validate_pair_indices(pairs, n_nodes)
    sorted_pairs = sorted(pairs, key=lambda p: (-p.strength, p.i, p.j))

    if max_degree is None:
        return sorted_pairs

    if max_degree < 0:
        raise ValueError("max_degree must be None or non-negative")

    degree = [0] * n_nodes
    kept: List[PairLink] = []

    for pair in sorted_pairs:
        if degree[pair.i] >= max_degree:
            continue
        if degree[pair.j] >= max_degree:
            continue

        kept.append(pair)
        degree[pair.i] += 1
        degree[pair.j] += 1

    kept.sort(key=lambda p: (-p.strength, p.i, p.j))
    return kept


def build_entanglement_layers(
    pairs: Sequence[PairLink],
    *,
    n_nodes: int,
    max_edges_per_layer: Optional[int] = None,
) -> List[List[PairLink]]:
    """
    Greedily schedule entanglement edges into non-overlapping layers.
    """

    _validate_pair_indices(pairs, n_nodes)
    if max_edges_per_layer is not None and max_edges_per_layer <= 0:
        raise ValueError("max_edges_per_layer must be positive or None")

    sorted_pairs = sorted(pairs, key=lambda p: (-p.strength, p.i, p.j))

    layers: List[List[PairLink]] = []
    used_nodes_per_layer: List[set[int]] = []

    for pair in sorted_pairs:
        placed = False

        for layer, used_nodes in zip(layers, used_nodes_per_layer):
            if pair.i in used_nodes or pair.j in used_nodes:
                continue

            if max_edges_per_layer is not None and len(layer) >= max_edges_per_layer:
                continue

            layer.append(pair)
            used_nodes.add(pair.i)
            used_nodes.add(pair.j)
            placed = True
            break

        if not placed:
            layers.append([pair])
            used_nodes_per_layer.append({pair.i, pair.j})

    return layers


def extract_arrays_from_spec(
    spec: Any,
) -> Tuple[Tuple[str, ...], np.ndarray, Optional[np.ndarray]]:
    """Extract names, binary-default dependency, and exposures from a system spec."""
    if isinstance(spec, SystemSpec):
        return (
            tuple(spec.node_names),
            spec.dependency_matrix(),
            spec.exposure_matrix.copy(),
        )

    nodes = _get_field(spec, "node_names")
    if nodes is None:
        nodes = _get_field(spec, "nodes")
    if nodes is None:
        raise ValueError("Could not find 'node_names' or 'nodes' on spec.")

    dependency_method = getattr(spec, "dependency_matrix", None)
    corr = dependency_method() if callable(dependency_method) else None
    if corr is None:
        corr = _get_field(spec, "target_pairwise_corr")
    if corr is None:
        corr = _get_field(spec, "correlation_matrix")

    if corr is None:
        corr = _get_field(spec, "correlations")

    if corr is None:
        raise ValueError(
            "Could not find a dependency or correlation matrix on spec."
        )

    institutions = tuple(_node_name(node) for node in nodes)
    correlation_matrix = _as_square_matrix(corr, "correlation_matrix")

    exposure_matrix = _get_field(spec, "exposure_matrix")

    if exposure_matrix is not None:
        exposure = _as_square_matrix(exposure_matrix, "exposure_matrix")
        return institutions, correlation_matrix, exposure

    edges = _get_field(spec, "edges")
    if edges is None:
        return institutions, correlation_matrix, None

    exposure = _edges_to_exposure_matrix(edges, institutions)
    return institutions, correlation_matrix, exposure


def build_clustering_layout_from_spec(
    spec: Any,
    **kwargs: Any,
) -> ClusterResult:
    """
    Build clustering layout directly from your existing system spec.
    """

    institutions, corr, exposure = extract_arrays_from_spec(spec)

    return build_clustering_layout(
        institutions=institutions,
        correlation_matrix=corr,
        exposure_matrix=exposure,
        **kwargs,
    )


def _as_square_matrix(value: Sequence[Sequence[float]], name: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=float)

    if matrix.ndim != 2:
        raise ValueError(f"{name} must be 2-dimensional")

    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"{name} must be square")

    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} contains non-finite values")

    return matrix.copy()


def _normalise_exposure_matrix(exposure: np.ndarray) -> np.ndarray:
    exposure = np.asarray(exposure, dtype=float).copy()

    if np.any(exposure < 0):
        raise ValueError("exposure_matrix must not contain negative values")

    np.fill_diagonal(exposure, 0.0)

    sym = exposure + exposure.T
    max_value = float(np.max(sym))

    if max_value <= 0:
        return np.zeros_like(sym)

    score = sym / max_value
    score = np.clip(score, 0.0, 1.0)
    np.fill_diagonal(score, 0.0)

    return score


def _validate_pair_indices(pairs: Sequence[PairLink], n_nodes: int) -> None:
    if n_nodes < 0:
        raise ValueError("n_nodes must be non-negative")
    for pair in pairs:
        if pair.i == pair.j or min(pair.i, pair.j) < 0 or max(pair.i, pair.j) >= n_nodes:
            raise ValueError("pair indices must reference two distinct nodes")


def _get_field(obj: Any, name: str) -> Any:
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


def _node_name(node: Any) -> str:
    if isinstance(node, str):
        return node

    if isinstance(node, dict):
        for key in ("id", "name", "ticker", "institution"):
            if key in node:
                return str(node[key])

    for attr in ("id", "name", "ticker", "institution"):
        if hasattr(node, attr):
            return str(getattr(node, attr))

    return str(node)


def _edge_field(edge: Any, names: Sequence[str]) -> Any:
    if isinstance(edge, dict):
        for name in names:
            if name in edge:
                return edge[name]

    for name in names:
        if hasattr(edge, name):
            return getattr(edge, name)

    return None


def _edges_to_exposure_matrix(
    edges: Iterable[Any],
    institutions: Sequence[str],
) -> np.ndarray:
    index = {name: i for i, name in enumerate(institutions)}
    n = len(institutions)
    matrix = np.zeros((n, n), dtype=float)

    for edge in edges:
        source = _edge_field(edge, ("source", "src", "from_node", "from_id", "lender"))
        target = _edge_field(edge, ("target", "dst", "to_node", "to_id", "borrower"))
        weight = _edge_field(edge, ("weight", "exposure", "amount", "value"))

        if source is None or target is None or weight is None:
            raise ValueError("Each edge must expose source/target/weight-style fields")

        source_name = str(source)
        target_name = str(target)

        if source_name not in index:
            raise ValueError(f"Unknown edge source: {source_name}")

        if target_name not in index:
            raise ValueError(f"Unknown edge target: {target_name}")

        matrix[index[source_name], index[target_name]] += float(weight)

    return matrix
