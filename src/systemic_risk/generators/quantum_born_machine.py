"""Entangled, quantum-native scenario generator (a computational-basis Born machine).

One qubit per institution; a measured computational-basis bitstring is one correlated
default scenario (``1`` = the institution initially defaults). The output distribution is
``P(x) = |<x|U(theta)|0>|^2`` for a parameterised circuit ``U(theta)`` whose angles are set
analytically from the spec's marginals and dependency graph -- the QCBM state-loader of the
project's quantum-advantage plan. It is an honest drop-in for the strongest classical
baseline (matching marginals and pairwise co-defaults), but its correlations and joint-tail
structure come from genuine amplitude entanglement, not a classical sampler.

Two ansatz families share one calibration + interface (set ``ansatz=``):

``"entangled"`` (default)
    Hardware-efficient ``RY`` marginals + amplitude-mixing controlled-``RY`` entanglers on the
    strongest dependency-graph edges. Per-edge angles come from the closed-form single-pair
    covariance inversion (:mod:`.quantum.ansatz`); a light Newton calibration against the
    *exact statevector* moments then cleans up the interference where edges share a qubit.
    The entangler graph is restricted to within-community edges once the system is too large to
    simulate whole, so the circuit is block-separable by :attr:`SystemSpec.clusters` and every
    simulated block stays small -- the construction extrapolates to 54 qubits without ever
    forming the ``2^54`` statevector.

``"ghz_systemic"``
    A GHZ-style superposition of a benign and a systemic product state. The coherent
    "everyone defaults together" component IS the systemic mode, so lower-tail / joint
    dependence is built in by construction; the number-of-defaults law is closed form at any
    ``n`` (no ``2^n`` sum), which is what makes it exact at 54 qubits and the cleanest
    "entanglement carries tail dependence" demonstration.

The misleading ``RZZ`` story of the previous placeholder is corrected: a ``Z``-diagonal gate is
inert in the measured (``Z``) basis, so the correlations here are produced by amplitude mixing.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from systemic_risk.generators.base import ScenarioGenerator, require_fitted
from systemic_risk.generators.quantum import ansatz as A
from systemic_risk.generators.quantum.statevector import StateVector, sample_bitstrings
from systemic_risk.spec import SystemSpec


@dataclass
class BornMachineDiagnostics:
    backend: str
    ansatz: str
    n_qubits: int
    n_edges: int
    n_blocks: int
    max_block_size: int


class EntangledBornMachineGenerator(ScenarioGenerator):
    """Entangled computational-basis Born machine for correlated default scenarios."""

    name = "Entangled Born machine"

    def __init__(
        self,
        *,
        ansatz: str = "entangled",
        backend: str = "statevector",
        edge_threshold: float = 0.02,
        max_degree: int | None = None,
        max_block_qubits: int = 22,
        calibrate: bool = True,
        calibration_iterations: int = 30,
        benign_fraction: float = 0.3,
    ) -> None:
        if ansatz not in {"entangled", "ghz_systemic"}:
            raise ValueError(f"unknown ansatz {ansatz!r}")
        if backend not in {"statevector", "qiskit"}:
            raise ValueError(f"unknown backend {backend!r}")
        self.ansatz = ansatz
        self.backend = backend
        self.edge_threshold = edge_threshold
        self.max_degree = max_degree
        self.max_block_qubits = max_block_qubits
        self.calibrate = calibrate
        self.calibration_iterations = calibration_iterations
        self.benign_fraction = benign_fraction

        self.spec_: SystemSpec | None = None
        self.edges_: list[tuple[int, int]] = []
        self.blocks_: list[A.EntangledCircuit] = []
        self.ghz_: A.GHZBlend | None = None
        self.symmetric_: A.SymmetricIsingLoader | None = None
        self.backend_used_: str | None = None

    # ------------------------------------------------------------------ fitting
    def fit(self, spec: SystemSpec) -> None:
        self.spec_ = spec
        self.edges_ = []
        self.blocks_ = []
        self.ghz_ = None
        self.symmetric_ = None
        self.backend_used_ = self._resolve_backend()
        if self.ansatz == "ghz_systemic":
            self._fit_ghz(spec)
        else:
            self._fit_entangled(spec)

    def _resolve_backend(self) -> str:
        if self.backend == "qiskit":
            from systemic_risk.generators.quantum import qiskit_backend

            if not qiskit_backend.is_available():
                raise RuntimeError("backend='qiskit' requires the 'quantum' extra (qiskit)")
            return "qiskit"
        return "statevector"

    def _fit_ghz(self, spec: SystemSpec) -> None:
        p = np.clip(spec.marginal_default_probs, 1e-6, 1.0 - 1e-6)
        iu = np.triu_indices(spec.n, k=1)
        corr = spec.dependency_matrix()
        mean_corr = float(np.clip(corr[iu].mean(), 0.0, 0.999)) if spec.n > 1 else 0.0
        self.ghz_ = A.GHZBlend.from_targets(
            spec.n,
            target_marginal=float(p.mean()),
            target_default_corr=mean_corr,
            benign_fraction=self.benign_fraction,
        )

    def _fit_entangled(self, spec: SystemSpec) -> None:
        # Homogeneous (uniform marginal + equicorrelated) targets are exchangeable and admit an
        # exact entangled state-loader: it reproduces the mean-field Ising loss-count law to
        # machine precision at any n (the n = 54 oracle validation), with no 2^n state.
        if A.is_homogeneous(spec):
            p = np.clip(spec.marginal_default_probs, 1e-6, 1.0 - 1e-6)
            iu = np.triu_indices(spec.n, k=1)
            corr = spec.dependency_matrix()
            mean_corr = float(corr[iu].mean()) if spec.n > 1 else 0.0
            self.symmetric_ = A.SymmetricIsingLoader.from_targets(
                spec.n, float(p.mean()), mean_corr
            )
            return

        p = np.clip(spec.marginal_default_probs, 1e-6, 1.0 - 1e-6)
        cov = A.target_covariance(spec)

        # Keep the whole system in one block while it is small enough to simulate exactly;
        # otherwise restrict to within-community edges so blocks stay small.
        whole_system = spec.n <= self.max_block_qubits
        self.edges_ = A.dependency_edges(
            spec,
            threshold=self.edge_threshold,
            within_clusters_only=not whole_system,
            max_degree=self.max_degree,
        )
        block_qubits = A.partition_blocks(spec, self.edges_, max_block=self.max_block_qubits)

        moments_fn = self._block_moments_fn()
        self.blocks_ = []
        for qubits in block_qubits:
            circuit = A._block_circuit(qubits, p, cov, self.edges_)
            if self.calibrate and circuit.size > 1 and circuit.edges:
                circuit = A.calibrate_block(
                    circuit, moments_fn, iterations=self.calibration_iterations
                )
            self.blocks_.append(circuit)

    def _block_moments_fn(self):
        """Return the exact-moment callback used by the calibration loop, per backend."""
        if self.backend_used_ == "qiskit":
            from systemic_risk.generators.quantum import qiskit_backend

            return qiskit_backend.block_moments
        return _statevector_block_moments

    # ----------------------------------------------------------------- sampling
    def sample(self, n_samples: int, seed: int | None = None) -> np.ndarray:
        require_fitted(self.spec_, self.name)
        rng = np.random.default_rng(seed)
        if self.ansatz == "ghz_systemic":
            return self._sample_ghz(n_samples, rng)
        return self._sample_entangled(n_samples, rng)

    def _sample_entangled(self, n_samples: int, rng: np.random.Generator) -> np.ndarray:
        if self.symmetric_ is not None:
            return self._sample_exchangeable(self.symmetric_.loss_count_pmf(), n_samples, rng)
        samples = np.zeros((n_samples, self.spec_.n), dtype=int)
        for block in self.blocks_:
            cols = block.qubits
            if block.size == 1:
                p = np.sin(block.ry[0] / 2.0) ** 2
                samples[:, cols[0]] = (rng.random(n_samples) < p).astype(int)
                continue
            samples[:, cols] = self._sample_block(block, n_samples, rng)
        return samples

    def _sample_block(
        self, block: A.EntangledCircuit, n_samples: int, rng: np.random.Generator
    ) -> np.ndarray:
        if self.backend_used_ == "qiskit":
            from systemic_risk.generators.quantum import qiskit_backend

            return qiskit_backend.sample_block(block.ry, block.edges, block.cry, n_samples, rng)
        state = _build_statevector(block.ry, block.edges, block.cry)
        return sample_bitstrings(state.probabilities(), block.size, n_samples, rng)

    def _sample_ghz(self, n_samples: int, rng: np.random.Generator) -> np.ndarray:
        return self._sample_exchangeable(self.ghz_.loss_count_pmf(), n_samples, rng)

    def _sample_exchangeable(
        self, loss_pmf: np.ndarray, n_samples: int, rng: np.random.Generator
    ) -> np.ndarray:
        """Sample an exchangeable state from its closed-form number-of-defaults law.

        Draw the default count ``k ~ P(K = k)`` (exact, no ``2^n`` state), then scatter ``k``
        defaults uniformly across institutions -- the exact conditional for a permutation-symmetric
        state. This is what lets the exchangeable ansaetze sample at ``n = 54``.
        """
        n = self.spec_.n
        counts = rng.choice(n + 1, size=n_samples, p=loss_pmf)
        samples = np.zeros((n_samples, n), dtype=int)
        for row, k in enumerate(counts):
            if k:
                samples[row, rng.choice(n, size=int(k), replace=False)] = 1
        return samples

    # --------------------------------------------------------------- exact view
    def _full_state(self) -> StateVector | None:
        """Return the exact full-system statevector when one is materialisable, else ``None``.

        Defined for the GHZ and homogeneous-symmetric states (any small ``n``) and for a
        single-block entangled fit; ``None`` when the entangled fit is split across cluster
        blocks (``n`` too large to form the ``2^n`` state).
        """
        if self.ghz_ is not None:
            return self._ghz_state()
        if self.symmetric_ is not None:
            return StateVector.symmetric_shells(self.symmetric_.shell_amplitudes())
        if len(self.blocks_) == 1 and self.blocks_[0].size == self.spec_.n:
            block = self.blocks_[0]
            return _build_statevector(block.ry, block.edges, block.cry)
        return None

    def exact_probabilities(self) -> np.ndarray:
        """Return the full exact Born probability vector (small ``n`` only).

        The flat layout matches :class:`StateVector`: qubit ``i`` is bit ``n - 1 - i`` of the
        index. Raises when the system was fit across multiple cluster blocks.
        """
        require_fitted(self.spec_, self.name)
        state = self._full_state()
        if state is None:
            raise RuntimeError(
                "exact_probabilities requires a materialisable full-system state "
                f"(n={self.spec_.n} exceeds max_block_qubits={self.max_block_qubits})"
            )
        return state.probabilities()

    def exact_moments(self) -> tuple[np.ndarray, np.ndarray]:
        """Return exact ``(marginals, pairwise_joint)`` from the statevector (small ``n``)."""
        require_fitted(self.spec_, self.name)
        state = self._full_state()
        if state is None:
            raise RuntimeError("exact_moments requires a materialisable full-system state")
        return state.marginals(), state.pairwise_joint()

    def loss_count_pmf(self) -> np.ndarray:
        """Return the exact number-of-defaults distribution where a closed form exists.

        Closed form (exact at *any* ``n``) for the exchangeable ansaetze -- the homogeneous
        symmetric loader and the GHZ blend; otherwise from the full statevector (small ``n``).
        Used for the ``n = 54`` oracle validation and tail diagnostics.
        """
        require_fitted(self.spec_, self.name)
        if self.symmetric_ is not None:
            return self.symmetric_.loss_count_pmf()
        if self.ghz_ is not None:
            return self.ghz_.loss_count_pmf()
        probs = self.exact_probabilities()
        counts = np.array([bin(i).count("1") for i in range(probs.size)])
        return np.bincount(counts, weights=probs, minlength=self.spec_.n + 1)

    def _ghz_state(self) -> StateVector:
        return StateVector.product_blend(
            self.spec_.n, self.ghz_.weight, self.ghz_.benign, self.ghz_.systemic
        )

    # ------------------------------------------------------------- description
    def circuit_description(self) -> dict[str, object]:
        require_fitted(self.spec_, self.name)
        common = {
            "qubits": self.spec_.node_names,
            "encoding": "|0> = survives, |1> = initial default",
            "readout": "computational (Z) basis; P(x) = |<x|U|0>|^2",
            "backend": self.backend_used_,
            "ansatz": self.ansatz,
        }
        if self.ghz_ is not None:
            common.update(
                {
                    "state": "sqrt(w)|systemic>^(x)n + sqrt(1-w)|benign>^(x)n (GHZ-style)",
                    "single_qubit_layers": "RY rotations setting the benign/systemic default levels",
                    "entangling_mechanism": (
                        "coherent superposition of two homogeneous product states: amplitude on "
                        "the all-default string carries the systemic lower-tail dependence"
                    ),
                    "weight": self.ghz_.weight,
                    "benign_default_prob": self.ghz_.benign,
                    "systemic_default_prob": self.ghz_.systemic,
                }
            )
            return common
        if self.symmetric_ is not None:
            common.update(
                {
                    "state": "permutation-symmetric superposition over Hamming-weight shells",
                    "single_qubit_layers": (
                        "homogeneous: amplitude a_k = sqrt(w_k / C(n,k)) on every weight-k string"
                    ),
                    "entangling_mechanism": (
                        "coherent amplitude across all default-count shells reproduces the "
                        "homogeneous mean-field Ising loss-count law exactly at any n; genuine "
                        "entanglement (non-product), not phase"
                    ),
                    "field": self.symmetric_.field,
                    "coupling": self.symmetric_.coupling,
                }
            )
            return common
        common.update(
            {
                "single_qubit_layers": (
                    "RY rotations, theta_i = 2 arcsin(sqrt(p_i)), reproducing the marginals"
                ),
                "entangling_mechanism": (
                    "amplitude-mixing controlled-RY (CRY) on dependency-graph edges; the "
                    "covariance is cov_ij = p_i(1-p_i)(sin^2((theta_j+alpha)/2) - p_j). A "
                    "Z-diagonal gate (e.g. RZZ) would be inert in the measured basis, so the "
                    "correlations come from amplitude entanglement, not phase."
                ),
                "edges": [
                    (self.spec_.node_names[i], self.spec_.node_names[j]) for i, j in self.edges_
                ],
                "blocks": [[self.spec_.node_names[q] for q in b.qubits] for b in self.blocks_],
            }
        )
        return common

    def diagnostics_summary(self) -> BornMachineDiagnostics:
        require_fitted(self.spec_, self.name)
        # The exchangeable states (GHZ / homogeneous symmetric) are one global block over all
        # qubits; the heterogeneous CRY fit splits into per-cluster blocks.
        if self.blocks_:
            n_blocks = len(self.blocks_)
            max_block = max(b.size for b in self.blocks_)
        else:
            n_blocks = 1
            max_block = self.spec_.n
        return BornMachineDiagnostics(
            backend=self.backend_used_ or self.backend,
            ansatz=self.ansatz,
            n_qubits=self.spec_.n,
            n_edges=len(self.edges_),
            n_blocks=n_blocks,
            max_block_size=max_block,
        )


# --------------------------------------------------------------- numpy backend hooks
def _build_statevector(
    ry: np.ndarray, edges: list[tuple[int, int]], cry: np.ndarray
) -> StateVector:
    state = StateVector(len(ry))
    for qubit, theta in enumerate(ry):
        state.ry(qubit, float(theta))
    for (control, target), alpha in zip(edges, cry):
        state.cry(control, target, float(alpha))
    return state


def _statevector_block_moments(
    ry: np.ndarray, edges: list[tuple[int, int]], cry: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    state = _build_statevector(ry, edges, cry)
    return state.marginals(), state.pairwise_joint()


class EntangledPQCGenerator(EntangledBornMachineGenerator):
    """Backwards-compatible alias replacing the old Gibbs-sampler placeholder.

    Accepts (and ignores) the placeholder's former ``layers`` / ``gibbs_sweeps`` /
    ``burn_in`` / ``coupling_scale`` keywords so existing callers keep working; the real
    sampler is the entangled Born machine above.
    """

    name = "Entangled PQC"

    def __init__(
        self,
        *,
        layers: int | None = None,
        gibbs_sweeps: int | None = None,
        burn_in: int | None = None,
        coupling_scale: float | None = None,
        **kwargs: object,
    ) -> None:
        super().__init__(**kwargs)  # type: ignore[arg-type]
