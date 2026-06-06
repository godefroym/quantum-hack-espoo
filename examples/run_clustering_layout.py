from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from contagion.clustering import build_clustering_layout
from contagion.visualization import (
    plot_dependency_matrix,
    plot_entanglement_layout,
)


def main() -> None:
    institutions = [
        "JPM",
        "BAC",
        "CITI",
        "GS",
        "MS",
        "BARC",
        "HSBC",
        "DB",
    ]

    correlation_matrix = np.array(
        [
            [1.00, 0.82, 0.78, 0.35, 0.32, 0.20, 0.18, 0.15],
            [0.82, 1.00, 0.80, 0.31, 0.30, 0.22, 0.16, 0.14],
            [0.78, 0.80, 1.00, 0.28, 0.26, 0.18, 0.17, 0.13],
            [0.35, 0.31, 0.28, 1.00, 0.84, 0.38, 0.24, 0.20],
            [0.32, 0.30, 0.26, 0.84, 1.00, 0.34, 0.22, 0.21],
            [0.20, 0.22, 0.18, 0.38, 0.34, 1.00, 0.77, 0.73],
            [0.18, 0.16, 0.17, 0.24, 0.22, 0.77, 1.00, 0.76],
            [0.15, 0.14, 0.13, 0.20, 0.21, 0.73, 0.76, 1.00],
        ],
        dtype=float,
    )

    exposure_matrix = np.array(
        [
            [0, 80, 60, 10, 5, 0, 0, 0],
            [75, 0, 65, 5, 5, 0, 0, 0],
            [55, 60, 0, 4, 2, 0, 0, 0],
            [8, 4, 3, 0, 70, 15, 0, 0],
            [5, 5, 2, 68, 0, 12, 0, 0],
            [0, 0, 0, 10, 12, 0, 55, 50],
            [0, 0, 0, 0, 0, 60, 0, 58],
            [0, 0, 0, 0, 0, 52, 56, 0],
        ],
        dtype=float,
    )

    result = build_clustering_layout(
        institutions=institutions,
        correlation_matrix=correlation_matrix,
        exposure_matrix=exposure_matrix,
        corr_weight=0.75,
        exposure_weight=0.25,
        correlation_mode="positive",
        cluster_threshold=0.55,
        entangle_threshold=0.65,
        classical_threshold=0.15,
        max_entangled_degree=3,
    )

    print(result.summary())

    print("\nEntanglement layers:")
    for layer_id, layer in enumerate(result.entanglement_layers):
        pairs = [f"{pair.institution_i}-{pair.institution_j}" for pair in layer]
        print(f"  Layer {layer_id}: {pairs}")

    plot_entanglement_layout(
        result,
        save_path="outputs/entanglement_layout.png",
    )

    plot_dependency_matrix(
        result,
        save_path="outputs/dependency_matrix.png",
    )


if __name__ == "__main__":
    main()
