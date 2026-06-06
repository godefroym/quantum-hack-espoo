from __future__ import annotations

import numpy as np

from systemic_risk.data import make_synthetic_system
from systemic_risk.generators import (
    BernoulliGenerator,
    EntangledPQCGenerator,
    GaussianCopulaGenerator,
    StudentTCopulaGenerator,
)
from systemic_risk.generators.moments import empirical_moments, targets_from_spec
from systemic_risk.spec import SystemSpec


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


def test_gaussian_copula_matches_binary_moments() -> None:
    p = np.array([0.20, 0.30, 0.25])
    corr = np.array(
        [
            [1.0, 0.16, 0.10],
            [0.16, 1.0, 0.08],
            [0.10, 0.08, 1.0],
        ]
    )
    spec = SystemSpec(
        node_names=["A", "B", "C"],
        node_types=["bank"] * 3,
        exposure_matrix=np.zeros((3, 3)),
        capital_buffers=np.ones(3),
        marginal_default_probs=p,
        target_pairwise_corr=corr,
        clusters=["test"] * 3,
        metadata={"correlation_space": "binary_default"},
    )
    generator = GaussianCopulaGenerator()
    generator.fit(spec)

    first = generator.sample(60_000, seed=321)
    second = generator.sample(60_000, seed=321)
    observed = empirical_moments(first)
    targets = targets_from_spec(spec)

    assert np.array_equal(first, second)
    assert np.max(np.abs(observed.marginals - targets.marginals)) < 0.01
    mask = targets.off_diagonal_mask
    assert np.max(
        np.abs(observed.pairwise_corr[mask] - targets.pairwise_corr[mask])
    ) < 0.025


def test_gaussian_copula_uses_latent_correlation_without_reinversion() -> None:
    p = np.array([0.12, 0.18, 0.25])
    latent_corr = np.array(
        [
            [1.0, 0.42, 0.20],
            [0.42, 1.0, 0.31],
            [0.20, 0.31, 1.0],
        ]
    )
    spec = SystemSpec(
        node_names=["A", "B", "C"],
        node_types=["bank"] * 3,
        exposure_matrix=np.zeros((3, 3)),
        capital_buffers=np.ones(3),
        marginal_default_probs=p,
        target_pairwise_corr=latent_corr,
        clusters=["test"] * 3,
        metadata={"correlation_space": "latent_gaussian"},
    )
    generator = GaussianCopulaGenerator()
    generator.fit(spec)
    targets = targets_from_spec(spec)
    observed = empirical_moments(generator.sample(80_000, seed=123))

    assert np.allclose(generator.corr_, latent_corr, atol=1e-8)
    assert targets.latent_gaussian_corr is not None
    assert not np.allclose(targets.pairwise_corr, latent_corr)
    mask = targets.off_diagonal_mask
    assert np.max(
        np.abs(observed.pairwise_joint[mask] - targets.pairwise_joint[mask])
    ) < 0.008


def test_b_and_c_share_identical_moment_targets() -> None:
    spec = make_synthetic_system(n=12, seed=8)
    gaussian = GaussianCopulaGenerator()
    entangled = EntangledPQCGenerator(gibbs_sweeps=2, burn_in=2)

    gaussian.fit(spec)
    entangled.fit(spec)

    assert np.allclose(gaussian.targets_.marginals, entangled.targets_.marginals)
    assert np.allclose(
        gaussian.targets_.pairwise_joint,
        entangled.targets_.pairwise_joint,
    )
