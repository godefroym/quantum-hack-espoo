from __future__ import annotations

import numpy as np
from typing import Tuple

from systemic_risk.generators.quantum import ansatz as A
from systemic_risk.generators.quantum.statevector import StateVector, sample_bitstrings
from systemic_risk.generators.gaussian_copula import GaussianCopulaGenerator


class SimpleDiscriminator:
    """Small feedforward discriminator implemented in NumPy.

    Binary classifier returning probability P(real). Trained with SGD on
    cross-entropy.
    """

    def __init__(self, input_dim: int, hidden: int = 32, seed: int | None = None):
        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(scale=0.1, size=(input_dim, hidden))
        self.b1 = np.zeros(hidden)
        self.W2 = rng.normal(scale=0.1, size=(hidden, 1))
        self.b2 = np.zeros(1)

    def forward(self, X: np.ndarray) -> np.ndarray:
        h = np.tanh(X @ self.W1 + self.b1)
        logits = h @ self.W2 + self.b2
        probs = 1.0 / (1.0 + np.exp(-logits))
        return probs.ravel()

    def train_step(self, X_real: np.ndarray, X_fake: np.ndarray, lr: float = 1e-2) -> float:
        X = np.vstack([X_real, X_fake])
        y = np.concatenate([np.ones(X_real.shape[0]), np.zeros(X_fake.shape[0])])
        probs = self.forward(X)
        # cross-entropy loss
        eps = 1e-12
        loss = -np.mean(y * np.log(probs + eps) + (1 - y) * np.log(1 - probs + eps))

        # gradients via simple backprop
        h = np.tanh(X @ self.W1 + self.b1)
        dlogits = (probs - y)[:, None] / X.shape[0]
        dW2 = h.T @ dlogits
        db2 = dlogits.sum(axis=0)
        dh = dlogits @ self.W2.T
        drelu = (1.0 - h ** 2) * dh
        dW1 = X.T @ drelu
        db1 = drelu.sum(axis=0)

        # SGD step
        self.W2 -= lr * dW2
        self.b2 -= lr * db2
        self.W1 -= lr * dW1
        self.b1 -= lr * db1
        return float(loss)


class QGANTrainer:
    """Adversarial trainer for a parameterised Born machine using a NumPy discriminator.

    Notes:
    - This is a simple, small-scale qGAN suitable for statevector-backed training.
    - Generator parameters are the block `RY` and `CRY` angles from `ansatz._block_circuit`.
    - Generator gradients are estimated by forward finite differences on the exact
      statevector-based expectation of the discriminator loss.
    """

    def __init__(
        self,
        n_epochs: int = 10,
        batch_size: int = 256,
        disc_steps: int = 5,
        gen_steps: int = 1,
        lr_disc: float = 1e-2,
        lr_gen: float = 1e-2,
        fd_eps: float = 1e-3,
        seed: int | None = None,
    ) -> None:
        self.n_epochs = int(n_epochs)
        self.batch_size = int(batch_size)
        self.disc_steps = int(disc_steps)
        self.gen_steps = int(gen_steps)
        self.lr_disc = float(lr_disc)
        self.lr_gen = float(lr_gen)
        self.fd_eps = float(fd_eps)
        self.rng = np.random.default_rng(seed)

    def _build_circuit_params(self, circuit: A.EntangledCircuit) -> np.ndarray:
        return np.concatenate([circuit.ry, circuit.cry])

    def _unpack_params(self, circuit: A.EntangledCircuit, params: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        n = circuit.size
        ry = params[:n]
        cry = params[n : n + len(circuit.edges)]
        return ry, cry

    def _sample_from_params(self, circuit: A.EntangledCircuit, params: np.ndarray, n_samples: int) -> np.ndarray:
        ry, cry = self._unpack_params(circuit, params)
        sv = StateVector(circuit.size)
        for i, th in enumerate(ry):
            sv.ry(i, float(th))
        for e, (cidx, tidx) in enumerate(circuit.edges):
            sv.cry(int(cidx), int(tidx), float(cry[e]))
        probs = sv.probabilities()
        bits = sample_bitstrings(probs, circuit.size, n_samples, self.rng)
        # map to global indices: produce full-length vector with zeros then fill qubits
        out = np.zeros((n_samples, max(circuit.qubits) + 1), dtype=int)
        for col, q in enumerate(circuit.qubits):
            out[:, q] = bits[:, col]
        return out

    def fit(self, spec, *, n_real: int = 2000, block_qubits: list[int] | None = None):
        # construct real samples via Gaussian copula baseline (moment-matched)
        gauss = GaussianCopulaGenerator()
        gauss.fit(spec)
        real = gauss.sample(n_real, seed=self.rng.integers(1 << 30))

        # choose one analytic block circuit to train (small systems / demo)
        edges = A.dependency_edges(spec, threshold=0.02, within_clusters_only=False)
        blocks = A.partition_blocks(spec, edges, max_block=spec.n)
        # pick smallest block with size <= 12 for tractability
        chosen = None
        for b in blocks:
            if len(b) <= 12:
                chosen = b
                break
        if chosen is None:
            chosen = blocks[0]
        circuit = A._block_circuit(chosen, np.clip(spec.marginal_default_probs, 1e-6, 1 - 1e-6), A.target_covariance(spec), edges)

        # initial params
        theta = self._build_circuit_params(circuit)
        disc = SimpleDiscriminator(input_dim=spec.n, hidden=32, seed=int(self.rng.integers(1 << 30)))

        history = {"disc_loss": [], "gen_loss": []}
        for epoch in range(self.n_epochs):
            # discriminator steps
            for _ in range(self.disc_steps):
                idx = self.rng.integers(0, real.shape[0], size=self.batch_size)
                Xr = real[idx]
                Xf = self._sample_from_params(circuit, theta, self.batch_size)
                dl = disc.train_step(Xr, Xf, lr=self.lr_disc)
            history["disc_loss"].append(dl)

            # generator steps (finite-difference gradient estimate)
            for _ in range(self.gen_steps):
                # approximate loss = mean log(1 - D(x)) over samples
                Xf = self._sample_from_params(circuit, theta, self.batch_size)
                probs = disc.forward(Xf)
                eps = 1e-12
                gen_loss = float(np.mean(np.log(1.0 - probs + eps)))
                history["gen_loss"].append(gen_loss)

                # finite-difference gradient
                grad = np.zeros_like(theta)
                base_loss = gen_loss
                for i in range(theta.size):
                    theta_p = theta.copy()
                    theta_p[i] += self.fd_eps
                    Xp = self._sample_from_params(circuit, theta_p, self.batch_size)
                    pp = disc.forward(Xp)
                    loss_p = float(np.mean(np.log(1.0 - pp + eps)))
                    grad[i] = (loss_p - base_loss) / self.fd_eps

                # gradient ascent (we maximize log(1-D) to fool discriminator); use lr_gen
                theta = theta + self.lr_gen * grad

        # return trained circuit with final params packed
        ry_opt, cry_opt = self._unpack_params(circuit, theta)
        trained = A.EntangledCircuit(qubits=circuit.qubits, ry=ry_opt, edges=circuit.edges, cry=cry_opt, target_p=circuit.target_p, target_cov=circuit.target_cov)
        return {"circuit": trained, "history": history}
