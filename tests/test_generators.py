from __future__ import annotations

import numpy as np

from systemic_risk.data import make_synthetic_system
from systemic_risk.generators import (
    BernoulliGenerator,
    EntangledPQCGenerator,
    GaussianCopulaGenerator,
    StudentTCopulaGenerator,
)


def test_generators_return_binary_samples() -> None:
    spec = make_synthetic_system(n=12, seed=8)
    generators = [
        BernoulliGenerator(),
        GaussianCopulaGenerator(),
        StudentTCopulaGenerator(),
        EntangledPQCGenerator(gibbs_sweeps=4, burn_in=5),
    ]

    for generator in generators:
        generator.fit(spec)
        samples = generator.sample(64, seed=123)
        assert samples.shape == (64, spec.n)
        assert np.all((samples == 0) | (samples == 1))


def test_bernoulli_is_seed_deterministic() -> None:
    spec = make_synthetic_system(n=12, seed=8)
    generator = BernoulliGenerator()
    generator.fit(spec)

    assert np.array_equal(generator.sample(20, seed=1), generator.sample(20, seed=1))
