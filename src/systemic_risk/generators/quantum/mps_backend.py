"""Qiskit Aer matrix-product-state backend for shallow, sparse Born machines."""

from __future__ import annotations

import numpy as np


def block_moments(
    ry: np.ndarray,
    edges: list[tuple[int, int]],
    cry: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return exact marginals and pairwise joints without forming a dense statevector."""
    from qiskit import transpile
    from qiskit.quantum_info import Pauli
    from qiskit_aer import AerSimulator

    from systemic_risk.generators.quantum.qiskit_backend import build_circuit

    n = len(ry)
    circuit = build_circuit(ry, edges, cry)
    for i in range(n):
        circuit.save_expectation_value(Pauli("Z"), [i], label=f"z{i}")
    for i in range(n):
        for j in range(i + 1, n):
            circuit.save_expectation_value(Pauli("ZZ"), [i, j], label=f"zz{i}_{j}")

    simulator = AerSimulator(method="matrix_product_state")
    compiled = transpile(circuit, simulator, optimization_level=0)
    data = simulator.run(compiled).result().data(0)

    z = np.array([float(data[f"z{i}"]) for i in range(n)])
    marginals = (1.0 - z) / 2.0
    pairwise = np.diag(marginals)
    for i in range(n):
        for j in range(i + 1, n):
            zz = float(data[f"zz{i}_{j}"])
            joint = (1.0 - z[i] - z[j] + zz) / 4.0
            pairwise[i, j] = pairwise[j, i] = joint
    return marginals, pairwise


def sample_block(
    ry: np.ndarray,
    edges: list[tuple[int, int]],
    cry: np.ndarray,
    n_samples: int,
    *,
    seed: int | None = None,
) -> np.ndarray:
    """Sample a sparse circuit with Aer MPS; output column ``i`` is logical qubit ``i``."""
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")

    from qiskit import transpile
    from qiskit_aer import AerSimulator

    from systemic_risk.generators.quantum.ibm_runtime import _bitstrings_to_samples
    from systemic_risk.generators.quantum.qiskit_backend import build_circuit

    simulator = AerSimulator(method="matrix_product_state")
    circuit = build_circuit(ry, edges, cry, measure=True)
    compiled = transpile(circuit, simulator, optimization_level=1)
    result = simulator.run(
        compiled,
        shots=n_samples,
        memory=True,
        seed_simulator=seed,
    ).result()
    return _bitstrings_to_samples(result.get_memory(), len(ry))

