from __future__ import annotations

import numpy as np
from typing import Tuple

from systemic_risk.generators.quantum import ansatz as A
from systemic_risk.generators.quantum.statevector import StateVector, sample_bitstrings
from systemic_risk.generators.gaussian_copula import GaussianCopulaGenerator


class SimpleDiscriminator:
    """Small feedforward discriminator implemented in NumPy.

    Binary classifier returning probability P(real). Trained with the Adam
    optimiser on cross-entropy.
    """

    def __init__(self, input_dim: int, hidden: int = 32, seed: int | None = None):
        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(scale=0.1, size=(input_dim, hidden))
        self.b1 = np.zeros(hidden)
        self.W2 = rng.normal(scale=0.1, size=(hidden, 1))
        self.b2 = np.zeros(1)
        # Adam moment buffers, keyed by parameter id
        self._m: dict[int, np.ndarray] = {}
        self._v: dict[int, np.ndarray] = {}
        self._t = 0

    def forward(self, X: np.ndarray) -> np.ndarray:
        h = np.tanh(X @ self.W1 + self.b1)
        logits = h @ self.W2 + self.b2
        probs = 1.0 / (1.0 + np.exp(-logits))
        return probs.ravel()

    def _adam_step(self, params: list[np.ndarray], grads: list[np.ndarray], lr: float,
                   b1: float = 0.9, b2: float = 0.999, eps: float = 1e-8) -> None:
        self._t += 1
        for p, g in zip(params, grads):
            key = id(p)
            m = self._m.get(key)
            v = self._v.get(key)
            if m is None:
                m = np.zeros_like(p)
                v = np.zeros_like(p)
            m = b1 * m + (1.0 - b1) * g
            v = b2 * v + (1.0 - b2) * (g * g)
            mhat = m / (1.0 - b1 ** self._t)
            vhat = v / (1.0 - b2 ** self._t)
            p -= lr * mhat / (np.sqrt(vhat) + eps)
            self._m[key] = m
            self._v[key] = v

    def train_step(self, X_real: np.ndarray, X_fake: np.ndarray, lr: float = 1e-2,
                   l2: float = 1e-4) -> float:
        X = np.vstack([X_real, X_fake])
        y = np.concatenate([np.ones(X_real.shape[0]), np.zeros(X_fake.shape[0])])
        probs = self.forward(X)
        # cross-entropy loss
        eps = 1e-12
        loss = -np.mean(y * np.log(probs + eps) + (1 - y) * np.log(1 - probs + eps))

        # gradients via simple backprop
        h = np.tanh(X @ self.W1 + self.b1)
        dlogits = (probs - y)[:, None] / X.shape[0]
        dW2 = h.T @ dlogits + l2 * self.W2
        db2 = dlogits.sum(axis=0)
        dh = dlogits @ self.W2.T
        drelu = (1.0 - h ** 2) * dh
        dW1 = X.T @ drelu + l2 * self.W1
        db1 = drelu.sum(axis=0)

        # Adam step (mutates weights in place)
        self._adam_step(
            [self.W1, self.b1, self.W2, self.b2],
            [dW1, db1, dW2, db2],
            lr,
        )
        return float(loss)


class QGANTrainer:
    """Adversarial trainer for a parameterised Born machine using a NumPy discriminator.

    Notes:
    - This is a small-scale qGAN suitable for statevector-backed training.
    - Generator parameters are the block ``RY`` and ``CRY`` angles from
      ``ansatz._block_circuit``.
    - The generator objective is evaluated as the **exact expectation** over the
      Born distribution (``sv.probabilities()``), so it carries no Monte-Carlo
      noise; its gradient is a deterministic **central finite difference** on
      that exact expectation. (On hardware the same gradient would be obtained by
      the parameter-shift rule from sampled expectations.) The generator uses the
      **non-saturating** objective ``max E_p[log D(fake)]`` and an Adam optimiser.
    - The discriminator is trained on sampled real/fake batches with Adam.
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

    def _clip_params(self, circuit: A.EntangledCircuit, params: np.ndarray) -> np.ndarray:
        """Keep RY in (0, pi) and CRY in a sane range so Adam cannot drift off-manifold."""
        n = circuit.size
        out = params.copy()
        out[:n] = np.clip(out[:n], 1e-6, np.pi - 1e-6)
        out[n:] = np.clip(out[n:], -2.5, 2.5)
        return out

    def _probabilities(self, circuit: A.EntangledCircuit, params: np.ndarray) -> np.ndarray:
        """Exact Born probabilities over the 2^size block basis (flat, C-order)."""
        ry, cry = self._unpack_params(circuit, params)
        sv = StateVector(circuit.size)
        for i, th in enumerate(ry):
            sv.ry(i, float(th))
        for e, (cidx, tidx) in enumerate(circuit.edges):
            sv.cry(int(cidx), int(tidx), float(cry[e]))
        return sv.probabilities()

    @staticmethod
    def _bit_table(size: int) -> np.ndarray:
        """(2^size, size) table of basis-state bits, matching ``sample_bitstrings``.

        Qubit ``i`` lives at bit position ``size - 1 - i`` of the flat index.
        """
        idx = np.arange(1 << size, dtype=np.uint64)
        pos = (size - 1 - np.arange(size)).astype(np.uint64)
        return ((idx[:, None] >> pos[None, :]) & np.uint64(1)).astype(int)

    def _sample_from_params(self, circuit: A.EntangledCircuit, params: np.ndarray, n_samples: int) -> np.ndarray:
        probs = self._probabilities(circuit, params)
        bits = sample_bitstrings(probs, circuit.size, n_samples, self.rng)
        out = np.zeros((n_samples, max(circuit.qubits) + 1), dtype=int)
        for col, q in enumerate(circuit.qubits):
            out[:, q] = bits[:, col]
        return out

    def _exact_moments(self, probs: np.ndarray, bit_table: np.ndarray):
        """Exact block marginals and pairwise-joint matrix from the probability vector."""
        marg = bit_table.T @ probs
        joint = (bit_table * probs[:, None]).T @ bit_table
        return marg, joint

    def _moment_error(self, circuit: A.EntangledCircuit, probs: np.ndarray, bit_table: np.ndarray) -> float:
        marg, joint = self._exact_moments(probs, bit_table)
        tp = circuit.target_p
        tj = circuit.target_cov + np.outer(tp, tp)
        iu = np.triu_indices(circuit.size, k=1)
        return float(np.sum((marg - tp) ** 2) + np.sum((joint[iu] - tj[iu]) ** 2))

    def fit(
        self,
        spec,
        *,
        n_real: int = 2000,
        block_qubits: list[int] | None = None,
        circuit: A.EntangledCircuit | None = None,
        real: np.ndarray | None = None,
    ):
        """Adversarially train one block circuit.

        ``circuit`` and ``real`` can be supplied to drive a hand-built (e.g.
        sparse, hardware-friendly) ansatz and custom target samples; otherwise a
        Gaussian-copula baseline and the strongest dependency block are used.
        """
        # real (target) samples -- Gaussian copula baseline unless supplied
        if real is None:
            gauss = GaussianCopulaGenerator()
            gauss.fit(spec)
            real = gauss.sample(n_real, seed=self.rng.integers(1 << 30))

        # generator circuit -- strongest tractable dependency block unless supplied
        if circuit is None:
            edges = A.dependency_edges(spec, threshold=0.02, within_clusters_only=False)
            blocks = A.partition_blocks(spec, edges, max_block=spec.n)
            chosen = None
            for b in blocks:
                if len(b) <= 12:
                    chosen = b
                    break
            if chosen is None:
                chosen = blocks[0]
            circuit = A._block_circuit(
                chosen,
                np.clip(spec.marginal_default_probs, 1e-6, 1 - 1e-6),
                A.target_covariance(spec),
                edges,
            )

        block = list(circuit.qubits)
        bit_table = self._bit_table(circuit.size)
        # design matrix: full-width (discriminator input) basis vectors, block columns filled
        full_width = real.shape[1]
        X_states = np.zeros((1 << circuit.size, full_width), dtype=float)
        for col, q in enumerate(block):
            if q < full_width:
                X_states[:, q] = bit_table[:, col]

        theta = self._clip_params(circuit, self._build_circuit_params(circuit))
        disc = SimpleDiscriminator(input_dim=full_width, hidden=32, seed=int(self.rng.integers(1 << 30)))

        # generator Adam state
        gm = np.zeros_like(theta)
        gv = np.zeros_like(theta)
        gt = 0
        b1, b2, adam_eps = 0.9, 0.999, 1e-8

        history = {"disc_loss": [], "gen_loss": [], "moment_err": []}
        best_theta = theta.copy()
        best_err = self._moment_error(circuit, self._probabilities(circuit, theta), bit_table)
        history["moment_err"].append(best_err)

        eps = 1e-12
        for epoch in range(self.n_epochs):
            # ---- discriminator steps (sampled batches) ----
            dl = 0.0
            for _ in range(self.disc_steps):
                idx = self.rng.integers(0, real.shape[0], size=self.batch_size)
                Xr = real[idx]
                Xf = self._sample_from_params(circuit, theta, self.batch_size)
                dl = disc.train_step(Xr, Xf, lr=self.lr_disc)
            history["disc_loss"].append(dl)

            # ---- generator steps (exact-expectation, central-difference grad) ----
            for _ in range(self.gen_steps):
                probs = self._probabilities(circuit, theta)
                d = disc.forward(X_states)
                g = np.log(d + eps)  # non-saturating: maximise E_p[log D]
                gen_obj = float(probs @ g)
                history["gen_loss"].append(gen_obj)

                grad = np.zeros_like(theta)
                for i in range(theta.size):
                    tp_ = theta.copy(); tp_[i] += self.fd_eps
                    tm_ = theta.copy(); tm_[i] -= self.fd_eps
                    pp = self._probabilities(circuit, self._clip_params(circuit, tp_))
                    pm = self._probabilities(circuit, self._clip_params(circuit, tm_))
                    grad[i] = ((pp - pm) @ g) / (2.0 * self.fd_eps)

                # Adam ascent (maximise gen_obj)
                gt += 1
                gm = b1 * gm + (1.0 - b1) * grad
                gv = b2 * gv + (1.0 - b2) * (grad * grad)
                mhat = gm / (1.0 - b1 ** gt)
                vhat = gv / (1.0 - b2 ** gt)
                theta = theta + self.lr_gen * mhat / (np.sqrt(vhat) + adam_eps)
                theta = self._clip_params(circuit, theta)

            err = self._moment_error(circuit, self._probabilities(circuit, theta), bit_table)
            history["moment_err"].append(err)
            if err < best_err:
                best_err = err
                best_theta = theta.copy()

        # return best circuit by exact moment error
        ry_opt, cry_opt = self._unpack_params(circuit, best_theta)
        trained = A.EntangledCircuit(
            qubits=circuit.qubits, ry=ry_opt, edges=circuit.edges, cry=cry_opt,
            target_p=circuit.target_p, target_cov=circuit.target_cov,
        )
        history["best_moment_err"] = best_err
        return {"circuit": trained, "history": history}
