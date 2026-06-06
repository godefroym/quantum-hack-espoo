"""Optional Qiskit-backed statevector path for the entangled Born machine.

Builds the *same* RY + CRY ansatz as :mod:`systemic_risk.generators.quantum.statevector`
as a real :class:`qiskit.QuantumCircuit`, then reads exact Born probabilities from
:class:`qiskit.quantum_info.Statevector`. Used only when the ``quantum`` extra is installed;
the numpy statevector is the default so the suite runs without Qiskit.

Qiskit indexes qubit ``i`` as the ``i``-th (little-endian) bit, the opposite of the numpy
engine's C-order grid, so the bit-unpacking here mirrors that convention. Both backends
therefore return identical per-qubit marginals, pairwise joints, and column-ordered samples.
"""

from __future__ import annotations

import numpy as np


def build_circuit(
    ry: np.ndarray,
    edges: list[tuple[int, int]],
    cry: np.ndarray,
    *,
    measure: bool = False,
):
    """Build the RY + CRY circuit, optionally with a full measurement register."""
    from qiskit import QuantumCircuit

    n = len(ry)
    qc = QuantumCircuit(n)
    for qubit, theta in enumerate(ry):
        qc.ry(float(theta), qubit)
    for (control, target), alpha in zip(edges, cry):
        qc.cry(float(alpha), control, target)
    if measure:
        qc.measure_all()
    return qc


def _probabilities(ry: np.ndarray, edges: list[tuple[int, int]], cry: np.ndarray) -> np.ndarray:
    from qiskit.quantum_info import Statevector

    return np.asarray(Statevector(build_circuit(ry, edges, cry)).probabilities(), dtype=float)


def block_moments(
    ry: np.ndarray, edges: list[tuple[int, int]], cry: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return exact ``(marginals, pairwise_joint)`` of the Qiskit RY + CRY block."""
    n = len(ry)
    prob = _probabilities(ry, edges, cry)
    index = np.arange(prob.size, dtype=np.uint64)
    # Qiskit little-endian: qubit i is bit i of the state index.
    bits = ((index[:, None] >> np.arange(n, dtype=np.uint64)[None, :]) & np.uint64(1)).astype(float)
    marginals = prob @ bits
    pairwise = np.einsum("s,si,sj->ij", prob, bits, bits)
    return marginals, pairwise


def sample_block(
    ry: np.ndarray,
    edges: list[tuple[int, int]],
    cry: np.ndarray,
    n_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw ``n_samples`` bitstrings (column ``i`` = qubit ``i``) from the Qiskit block."""
    n = len(ry)
    prob = _probabilities(ry, edges, cry)
    picks = rng.choice(prob.size, size=n_samples, p=prob)
    bit_positions = np.arange(n, dtype=np.uint64)
    bits = (picks.astype(np.uint64)[:, None] >> bit_positions[None, :]) & np.uint64(1)
    return bits.astype(int)


def is_available() -> bool:
    """Return ``True`` when Qiskit can be imported."""
    try:
        import qiskit  # noqa: F401
    except ImportError:
        return False
    return True
