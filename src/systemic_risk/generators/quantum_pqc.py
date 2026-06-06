from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from systemic_risk.generators.base import ScenarioGenerator, require_fitted
from systemic_risk.generators.moments import MomentTargets, targets_from_spec
from systemic_risk.spec import SystemSpec


@dataclass
class PQCDiagnostics:
    backend: str
    n_qubits: int
    n_edges: int
    layers: int


class EntangledPQCGenerator(ScenarioGenerator):
    """Entanglement-structured PQC scenario generator.

    The MVP keeps the public interface quantum-native while using a deterministic
    Born-inspired fallback sampler when a quantum SDK is unavailable. The fallback
    preserves the intended role: it samples a non-factorized joint distribution
    over initial default scenarios and exposes moment-matching training hooks.
    """

    name = "Entangled PQC"

    def __init__(
        self,
        layers: int = 2,
        coupling_scale: float = 1.2,
        gibbs_sweeps: int = 18,
        burn_in: int = 30,
    ) -> None:
        self.layers = layers
        self.coupling_scale = coupling_scale
        self.gibbs_sweeps = gibbs_sweeps
        self.burn_in = burn_in
        self.spec_: SystemSpec | None = None
        self.bias_: np.ndarray | None = None
        self.couplings_: np.ndarray | None = None
        self.ry_angles_: np.ndarray | None = None
        self.edges_: list[tuple[int, int]] = []
        self.backend_ = "born-inspired-fallback"
        self.targets_: MomentTargets | None = None

    def fit(self, spec: SystemSpec) -> None:
        self.spec_ = spec
        self.targets_ = targets_from_spec(spec)
        p = np.clip(self.targets_.marginals, 1e-6, 1 - 1e-6)
        self.bias_ = np.log(p / (1 - p))
        self.ry_angles_ = 2 * np.arcsin(np.sqrt(p))

        dep = np.maximum(spec.dependency_matrix(), 0.0)
        exposure = spec.exposure_matrix + spec.exposure_matrix.T
        if exposure.max() > 0:
            dep = dep + 0.25 * exposure / exposure.max()
        np.fill_diagonal(dep, 0.0)

        self.edges_ = [
            (i, j)
            for i in range(spec.n)
            for j in range(i + 1, spec.n)
            if dep[i, j] > 0.12
        ]
        couplings = np.zeros((spec.n, spec.n), dtype=float)
        for i, j in self.edges_:
            couplings[i, j] = couplings[j, i] = self.coupling_scale * dep[i, j]
        self.couplings_ = couplings

    def train(
        self,
        n_steps: int = 8,
        n_samples: int = 512,
        seed: int | None = None,
        lr_bias: float = 0.7,
        lr_coupling: float = 0.9,
        eta: float = 0.35,
    ) -> list[dict[str, float]]:
        """Lightweight moment matching over marginals and pairwise co-defaults."""
        require_fitted(self.spec_, self.name)
        require_fitted(self.bias_, self.name)
        require_fitted(self.couplings_, self.name)
        rng = np.random.default_rng(seed)
        require_fitted(self.targets_, self.name)
        target_p = self.targets_.marginals
        target_joint = self.targets_.pairwise_joint
        history: list[dict[str, float]] = []
        for step in range(n_steps):
            samples = self.sample(n_samples, seed=int(rng.integers(0, 2**32 - 1)))
            observed_p = samples.mean(axis=0)
            observed_joint = (samples.T @ samples) / max(n_samples, 1)
            marginal_error = target_p - observed_p
            joint_error = target_joint - observed_joint

            self.bias_ = self.bias_ + lr_bias * marginal_error
            for i, j in self.edges_:
                self.couplings_[i, j] += lr_coupling * eta * joint_error[i, j]
                self.couplings_[j, i] = self.couplings_[i, j]
            self.couplings_ = np.clip(self.couplings_, -1.5, 2.5)

            history.append(
                {
                    "step": float(step),
                    "marginal_rmse": float(np.sqrt(np.mean(marginal_error**2))),
                    "pairwise_rmse": float(np.sqrt(np.mean(joint_error**2))),
                }
            )
        return history

    def sample(self, n_samples: int, seed: int | None = None) -> np.ndarray:
        require_fitted(self.bias_, self.name)
        require_fitted(self.couplings_, self.name)
        rng = np.random.default_rng(seed)
        n = len(self.bias_)
        samples = np.zeros((n_samples, n), dtype=int)
        state = (rng.random(n) < self._sigmoid(self.bias_)).astype(int)

        for _ in range(self.burn_in):
            state = self._gibbs_sweep(state, rng)

        for row in range(n_samples):
            for _ in range(self.gibbs_sweeps):
                state = self._gibbs_sweep(state, rng)
            samples[row] = state
        return samples

    def circuit_description(self) -> dict[str, object]:
        require_fitted(self.spec_, self.name)
        return {
            "qubits": self.spec_.node_names,
            "encoding": "|0>=survives, |1>=initial default",
            "single_qubit_layers": "Ry rotations initialized from marginal default probabilities",
            "entangling_layers": "RZZ-style pair interactions on dependency/exposure graph edges",
            "edges": [(self.spec_.node_names[i], self.spec_.node_names[j]) for i, j in self.edges_],
            "backend": self.backend_,
        }

    def pqc_diagnostics(self) -> PQCDiagnostics:
        require_fitted(self.spec_, self.name)
        return PQCDiagnostics(
            backend=self.backend_,
            n_qubits=self.spec_.n,
            n_edges=len(self.edges_),
            layers=self.layers,
        )

    def _gibbs_sweep(self, state: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        order = rng.permutation(len(state))
        for i in order:
            field = self.bias_[i] + float(self.couplings_[i] @ state)
            prob = self._sigmoid(field)
            state[i] = int(rng.random() < prob)
        return state

    @staticmethod
    def _sigmoid(x: np.ndarray | float) -> np.ndarray | float:
        return 1.0 / (1.0 + np.exp(-np.clip(x, -40, 40)))
