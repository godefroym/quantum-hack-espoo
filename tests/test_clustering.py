from __future__ import annotations

import numpy as np

from contagion.clustering import (
    build_clustering_layout,
    build_dependency_matrix,
    build_entanglement_layers,
    threshold_connected_components,
)


def test_dependency_matrix_is_symmetric_and_zero_diagonal() -> None:
    corr = np.array(
        [
            [1.0, 0.8, -0.4],
            [0.8, 1.0, 0.2],
            [-0.4, 0.2, 1.0],
        ]
    )

    exposure = np.array(
        [
            [0.0, 10.0, 0.0],
            [5.0, 0.0, 0.0],
            [0.0, 3.0, 0.0],
        ]
    )

    dependency = build_dependency_matrix(
        correlation_matrix=corr,
        exposure_matrix=exposure,
        correlation_mode="positive",
    )

    assert dependency.shape == (3, 3)
    assert np.allclose(dependency, dependency.T)
    assert np.allclose(np.diag(dependency), 0.0)
    assert np.all(dependency >= 0.0)
    assert np.all(dependency <= 1.0)


def test_threshold_connected_components() -> None:
    dependency = np.array(
        [
            [0.0, 0.8, 0.1, 0.0],
            [0.8, 0.0, 0.2, 0.0],
            [0.1, 0.2, 0.0, 0.9],
            [0.0, 0.0, 0.9, 0.0],
        ]
    )

    clusters, labels = threshold_connected_components(
        dependency,
        threshold=0.7,
    )

    assert clusters == [[0, 1], [2, 3]]
    assert labels == [0, 0, 1, 1]


def test_clustering_layout_finds_entangled_pairs() -> None:
    institutions = ["A", "B", "C", "D"]

    corr = np.array(
        [
            [1.0, 0.9, 0.1, 0.1],
            [0.9, 1.0, 0.2, 0.1],
            [0.1, 0.2, 1.0, 0.85],
            [0.1, 0.1, 0.85, 1.0],
        ]
    )

    result = build_clustering_layout(
        institutions=institutions,
        correlation_matrix=corr,
        cluster_threshold=0.7,
        entangle_threshold=0.8,
        classical_threshold=0.1,
        max_entangled_degree=None,
    )

    cluster_names = result.cluster_names()

    assert ["A", "B"] in cluster_names
    assert ["C", "D"] in cluster_names

    entangled = {
        (pair.institution_i, pair.institution_j) for pair in result.entangled_pairs
    }

    assert ("A", "B") in entangled
    assert ("C", "D") in entangled


def test_entanglement_layers_have_no_node_collision() -> None:
    institutions = ["A", "B", "C", "D"]

    corr = np.array(
        [
            [1.0, 0.9, 0.8, 0.7],
            [0.9, 1.0, 0.85, 0.75],
            [0.8, 0.85, 1.0, 0.95],
            [0.7, 0.75, 0.95, 1.0],
        ]
    )

    result = build_clustering_layout(
        institutions=institutions,
        correlation_matrix=corr,
        cluster_threshold=0.6,
        entangle_threshold=0.7,
        classical_threshold=0.1,
        max_entangled_degree=None,
    )

    layers = build_entanglement_layers(
        result.entangled_pairs,
        n_nodes=len(institutions),
    )

    for layer in layers:
        used = set()

        for pair in layer:
            assert pair.i not in used
            assert pair.j not in used

            used.add(pair.i)
            used.add(pair.j)


def test_max_entangled_degree_is_respected() -> None:
    institutions = ["A", "B", "C", "D", "E"]

    corr = np.array(
        [
            [1.0, 0.95, 0.94, 0.93, 0.92],
            [0.95, 1.0, 0.91, 0.90, 0.89],
            [0.94, 0.91, 1.0, 0.88, 0.87],
            [0.93, 0.90, 0.88, 1.0, 0.86],
            [0.92, 0.89, 0.87, 0.86, 1.0],
        ]
    )

    result = build_clustering_layout(
        institutions=institutions,
        correlation_matrix=corr,
        cluster_threshold=0.8,
        entangle_threshold=0.85,
        classical_threshold=0.1,
        max_entangled_degree=2,
    )

    degree = {i: 0 for i in range(len(institutions))}

    for pair in result.entangled_pairs:
        degree[pair.i] += 1
        degree[pair.j] += 1

    assert all(value <= 2 for value in degree.values())
