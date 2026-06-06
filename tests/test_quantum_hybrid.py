from __future__ import annotations

import numpy as np

from systemic_risk.data_network.assemble import build_synthetic_system_spec

from scenario_generation.quantum_hybrid import HybridGAMGenerator


def test_hybrid_generator_basic():
    spec = build_synthetic_system_spec(n=8, seed=123)
    gen = HybridGAMGenerator(quantum_fraction=0.2, entangled_kwargs={"calibrate": False}, augmentation_strength=0.0)
    gen.fit(spec)
    samples = gen.sample(200, seed=1)
    assert samples.shape == (200, spec.n)
    assert np.all((samples == 0) | (samples == 1))
