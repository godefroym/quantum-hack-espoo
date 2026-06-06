from __future__ import annotations

import numpy as np

from systemic_risk.data_network.assemble import build_synthetic_system_spec
from scenario_generation.gam_generator import GAMGenerator
from scenario_generation.validation import validate_scenarios


def test_gam_generator_basic():
    spec = build_synthetic_system_spec(n=8, seed=123)
    gen = GAMGenerator()
    gen.fit(spec)
    samples1 = gen.sample(1000, seed=1)
    samples2 = gen.sample(1000, seed=1)
    # deterministic with same seed
    assert np.array_equal(samples1, samples2)
    # shape and schema
    assert samples1.shape == (1000, spec.n)
    assert validate_scenarios(samples1, spec)["schema_ok"]

    # marginals should be approximately close to targets (within 0.1)
    sampled_marginals = samples1.mean(axis=0)
    target = np.asarray(spec.marginal_default_probs)
    diff = np.abs(sampled_marginals - target)
    assert np.all(diff < 0.15)
