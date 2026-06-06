"""Exact numpy statevector simulator for the scenario-generator Born machines.

A minimal, dependency-free statevector engine specialised to the gate set the
entangled generator needs: ``RY`` single-qubit rotations and the amplitude-mixing
entanglers (``CRY``, ``CNOT``) that create *measurable* computational-basis
correlations. It is exact -- the returned ``probabilities`` are the Born
probabilities :math:`P(x) = |\\langle x | \\psi \\rangle|^2`, with no sampling noise
-- and fits in memory to roughly ``n = 24`` qubits (``2^24`` complex amplitudes
:math:`\\approx` 0.27 GB).

Qubit ``i`` is tensor axis ``i`` of a ``(2,) * n`` array; bit ``i`` of a
computational-basis index is the value of qubit ``i`` (little-endian over
institutions, matching :class:`systemic_risk.models.ising.IsingModel`). The
encoding is ``|0> = survives``, ``|1> = initial default``.

Phase-only (``Z``-diagonal) gates such as ``RZZ`` are intentionally absent: they
leave every ``Z``-basis marginal and covariance unchanged, so they cannot carry
default correlations. Correlations here come exclusively from amplitude mixing.
"""

from __future__ import annotations

import numpy as np


class StateVector:
    """Mutable ``n``-qubit statevector initialised to ``|0...0>``."""

    def __init__(self, n: int) -> None:
        if n < 1:
            raise ValueError("n must be positive")
        self.n = n
        self.amplitudes = np.zeros((2,) * n, dtype=np.complex128)
        self.amplitudes[(0,) * n] = 1.0

    # ------------------------------------------------------- entangled builders
    @classmethod
    def product_blend(cls, n: int, weight: float, benign: float, systemic: float) -> "StateVector":
        """Build the GHZ-style blend ``sqrt(w)|A>^{(x)n} + sqrt(1-w)|B>^{(x)n}``.

        ``|A>`` / ``|B>`` are the per-qubit ``RY`` states with default probabilities
        ``systemic`` / ``benign``. The result is a genuinely entangled (non-product) state
        whose computational-basis amplitudes are real; the small-``n`` exact statevector
        confirms the closed-form number-of-defaults law used at ``n = 54``.
        """
        a = np.array([np.sqrt(1.0 - systemic), np.sqrt(systemic)], dtype=np.complex128)
        b = np.array([np.sqrt(1.0 - benign), np.sqrt(benign)], dtype=np.complex128)
        psi_a, psi_b = a, b
        for _ in range(n - 1):
            psi_a = np.kron(psi_a, a)
            psi_b = np.kron(psi_b, b)
        psi = np.sqrt(weight) * psi_a + np.sqrt(1.0 - weight) * psi_b
        psi = psi / np.linalg.norm(psi)
        state = cls(n)
        state.amplitudes = psi.reshape((2,) * n)
        return state

    @classmethod
    def symmetric_shells(cls, shell_amplitudes: np.ndarray) -> "StateVector":
        """Build the permutation-symmetric state with amplitude ``a_k`` on every weight-``k`` string.

        ``shell_amplitudes[k]`` is the amplitude assigned to *each* computational-basis string of
        Hamming weight ``k`` (``k = 0..n``). The result is an exchangeable, generally entangled
        state; small-``n`` use only (it forms the ``2^n`` grid), to verify the closed-form
        loss-count law against the exact statevector.
        """
        a = np.asarray(shell_amplitudes, dtype=np.complex128)
        n = a.size - 1
        index = np.arange(1 << n, dtype=np.uint64)
        weight = np.zeros(1 << n, dtype=int)
        for bit in range(n):
            weight += ((index >> np.uint64(bit)) & np.uint64(1)).astype(int)
        flat = a[weight]
        norm = np.linalg.norm(flat)
        if norm > 0:
            flat = flat / norm
        state = cls(n)
        state.amplitudes = flat.reshape((2,) * n)
        return state

    # ------------------------------------------------------------------- gates
    def ry(self, qubit: int, theta: float) -> "StateVector":
        """Apply ``RY(theta)`` to ``qubit`` (``P(qubit=1) = sin^2(theta/2)`` from |0>)."""
        c, s = np.cos(theta / 2.0), np.sin(theta / 2.0)
        gate = np.array([[c, -s], [s, c]], dtype=np.complex128)
        self.amplitudes = np.moveaxis(
            np.tensordot(gate, self.amplitudes, axes=([1], [qubit])), 0, qubit
        )
        return self

    def cry(self, control: int, target: int, alpha: float) -> "StateVector":
        """Apply ``RY(alpha)`` to ``target`` on the ``control = 1`` branch (controlled-RY)."""
        if control == target:
            raise ValueError("control and target must differ")
        c, s = np.cos(alpha / 2.0), np.sin(alpha / 2.0)
        gate = np.array([[c, -s], [s, c]], dtype=np.complex128)
        index = [slice(None)] * self.n
        index[control] = 1
        branch = self.amplitudes[tuple(index)]
        axis = target if target < control else target - 1
        self.amplitudes[tuple(index)] = np.moveaxis(
            np.tensordot(gate, branch, axes=([1], [axis])), 0, axis
        )
        return self

    def cnot(self, control: int, target: int) -> "StateVector":
        """Apply a CNOT flipping ``target`` on the ``control = 1`` branch."""
        if control == target:
            raise ValueError("control and target must differ")
        index = [slice(None)] * self.n
        index[control] = 1
        axis = target if target < control else target - 1
        self.amplitudes[tuple(index)] = np.flip(self.amplitudes[tuple(index)], axis=axis)
        return self

    def rxx(self, a: int, b: int, phi: float) -> "StateVector":
        """Apply ``RXX(phi) = exp(-i phi/2 * X_a X_b)``, a symmetric amplitude entangler.

        Unlike :meth:`cry` this is permutation symmetric in ``(a, b)``, so a uniform
        ``RXX`` layer over a complete graph yields an exchangeable state (every qubit
        keeps the same marginal). It mixes amplitude between basis states, so it does
        move ``Z``-basis correlations -- to leading order ``cov_ab`` grows like ``phi^2``.
        """
        if a == b:
            raise ValueError("a and b must differ")
        c, s = np.cos(phi / 2.0), -1j * np.sin(phi / 2.0)
        x = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
        flipped = np.tensordot(x, self.amplitudes, axes=([1], [a]))
        flipped = np.tensordot(x, np.moveaxis(flipped, 0, a), axes=([1], [b]))
        self.amplitudes = c * self.amplitudes + s * np.moveaxis(flipped, 0, b)
        return self

    # --------------------------------------------------------------- readout
    def probabilities(self) -> np.ndarray:
        """Return the exact Born probabilities as a flat ``(2^n,)`` array."""
        return (np.abs(self.amplitudes) ** 2).reshape(-1)

    def marginals(self) -> np.ndarray:
        """Return ``P(qubit_i = 1)`` for every qubit (exact)."""
        prob = np.abs(self.amplitudes) ** 2
        out = np.empty(self.n)
        for i in range(self.n):
            index = [slice(None)] * self.n
            index[i] = 1
            out[i] = prob[tuple(index)].sum()
        return out

    def pairwise_joint(self) -> np.ndarray:
        """Return ``P(qubit_i = 1 and qubit_j = 1)`` (exact), diagonal = marginals."""
        prob = np.abs(self.amplitudes) ** 2
        out = np.empty((self.n, self.n))
        for i in range(self.n):
            for j in range(i, self.n):
                index = [slice(None)] * self.n
                index[i] = 1
                index[j] = 1
                out[i, j] = out[j, i] = prob[tuple(index)].sum()
        return out


def sample_bitstrings(
    probabilities: np.ndarray, n: int, n_samples: int, rng: np.random.Generator
) -> np.ndarray:
    """Draw ``n_samples`` computational-basis bitstrings from a probability vector.

    Returns an ``(n_samples, n)`` int array of 0/1 values. The probability vector is the
    flattened ``(2,) * n`` amplitude grid (C-order), so qubit ``i`` lives at bit position
    ``n - 1 - i`` of each drawn index; column ``i`` of the output is qubit ``i``, matching
    :meth:`StateVector.marginals`.
    """
    picks = rng.choice(probabilities.size, size=n_samples, p=probabilities)
    bit_positions = (n - 1 - np.arange(n)).astype(np.uint64)
    bits = (picks.astype(np.uint64)[:, None] >> bit_positions[None, :]) & np.uint64(1)
    return bits.astype(int)
