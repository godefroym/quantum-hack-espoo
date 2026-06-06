import numpy as np

from scenario_generation.qgan_trainer import QGANTrainer


def test_qgan_trainer_smoke(monkeypatch):
    # create a minimal dummy spec
    class DummySpec:
        def __init__(self):
            self.n = 3
            self.marginal_default_probs = np.array([0.01, 0.02, 0.03])

    spec = DummySpec()

    # stub GaussianCopulaGenerator used inside trainer
    class DummyGauss:
        def fit(self, s):
            pass

        def sample(self, n, seed=None):
            rng = np.random.default_rng(seed)
            return rng.integers(0, 2, size=(n, spec.n))

    # stub ansatz utilities and StateVector behavior
    def fake_dependency_edges(spec, threshold, within_clusters_only=False):
        return []

    def fake_partition_blocks(spec, edges, max_block=0):
        return [[0, 1, 2]]

    class FakeCircuit:
        def __init__(self):
            self.size = 3
            self.qubits = [0, 1, 2]
            self.edges = []
            self.ry = np.zeros(self.size)
            self.cry = np.zeros(0)
            self.target_p = spec.marginal_default_probs
            self.target_cov = np.eye(self.size) * 1e-6

    def fake_block_circuit(chosen, clipped, cov, edges):
        return FakeCircuit()

    # replace imports inside module under test
    import scenario_generation.qgan_trainer as qmod

    monkeypatch.setattr(qmod, "GaussianCopulaGenerator", lambda: DummyGauss())
    monkeypatch.setattr(qmod.A, "dependency_edges", fake_dependency_edges)
    monkeypatch.setattr(qmod.A, "partition_blocks", lambda spec, edges, max_block: fake_partition_blocks(spec, edges, max_block))
    monkeypatch.setattr(qmod.A, "_block_circuit", lambda chosen, clipped, cov, edges: fake_block_circuit(chosen, clipped, cov, edges))
    # stub target covariance to avoid requiring spec.target_pairwise_joint_probs
    monkeypatch.setattr(qmod.A, "target_covariance", lambda s: np.eye(s.n) * 1e-6)

    # stub StateVector and sampling to return simple uniform probabilities
    class FakeSV:
        def __init__(self, size):
            self._size = size

        def ry(self, i, th):
            pass

        def cry(self, c, t, th):
            pass

        def probabilities(self):
            # uniform over 2^size
            probs = np.ones(1 << self._size, dtype=float)
            return probs / probs.sum()

    def fake_sample_bitstrings(probs, size, n_samples, rng):
        # sample indices then convert to bits
        idx = rng.choice(len(probs), size=n_samples, p=probs)
        bits = ((np.arange(1 << size)[:, None] & (1 << np.arange(size))) > 0).astype(int)
        return bits[idx][:, :size]

    monkeypatch.setattr(qmod, "StateVector", FakeSV)
    monkeypatch.setattr(qmod, "sample_bitstrings", fake_sample_bitstrings)

    trainer = QGANTrainer(n_epochs=1, batch_size=8, disc_steps=1, gen_steps=1, fd_eps=1e-3, seed=0)
    out = trainer.fit(spec, n_real=64)

    assert isinstance(out, dict)
    assert "circuit" in out and "history" in out
    assert len(out["history"]["disc_loss"]) >= 1
