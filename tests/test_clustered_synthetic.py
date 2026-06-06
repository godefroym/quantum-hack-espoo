from __future__ import annotations

import numpy as np
import pytest

from systemic_risk.data import (
    ClusteredSystemConfig,
    cluster_block_means,
    make_clustered_system,
    planted_latent_correlation,
    reference_default_samples,
)
from systemic_risk.generators.base import sample_diagnostics
from systemic_risk.spec import CORRELATION_SPACE_LATENT_GAUSSIAN, SystemSpec


def _labels(spec: SystemSpec) -> np.ndarray:
    return np.asarray(spec.metadata["cluster_labels"], dtype=int)


def test_planted_labels_are_recorded_and_first_class() -> None:
    config = ClusteredSystemConfig(cluster_sizes=(6, 6, 4), cross_coupling=0.05)
    spec = make_clustered_system(config)

    # First-class, fully valid SystemSpec.
    assert isinstance(spec, SystemSpec)
    spec.validate()
    assert spec.n == 16
    assert spec.correlation_space == CORRELATION_SPACE_LATENT_GAUSSIAN

    # Ground-truth labels recorded on the object.
    assert spec.clusters is not None
    assert len(spec.clusters) == spec.n
    labels = _labels(spec)
    assert labels.tolist() == [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2]
    # The string cluster names agree with the integer labels.
    assert [f"cluster_{c}" for c in labels] == spec.clusters

    # Round-trips through the canonical serialisation.
    restored = SystemSpec.from_dict(spec.to_dict())
    assert restored.clusters == spec.clusters
    assert restored.metadata["cluster_labels"] == labels.tolist()


def test_within_cluster_latent_corr_exceeds_cross_cluster() -> None:
    config = ClusteredSystemConfig(
        cluster_sizes=(5, 5, 5), cross_coupling=0.08, within_coupling=0.4
    )
    spec = make_clustered_system(config)
    corr = spec.target_pairwise_corr
    labels = _labels(spec)

    within, cross = cluster_block_means(corr, labels)
    # Planted exactly: within = cross + within_coupling.
    assert np.isclose(within, 0.08 + 0.4)
    assert np.isclose(cross, 0.08)
    assert within > cross


def test_zero_dial_gives_block_diagonal_separable_clusters() -> None:
    config = ClusteredSystemConfig(cluster_sizes=(4, 4), cross_coupling=0.0)
    spec = make_clustered_system(config)
    corr = spec.target_pairwise_corr
    labels = _labels(spec)

    cross_mask = (labels[:, None] != labels[None, :])
    # Fully separable: cross-cluster latent correlation is exactly zero.
    assert np.allclose(corr[cross_mask], 0.0)


def test_planted_latent_correlation_is_psd() -> None:
    labels = np.array([0, 0, 1, 1, 2, 2])
    corr = planted_latent_correlation(labels, cross_coupling=0.1, within_coupling=0.6)
    eigvals = np.linalg.eigvalsh(corr)
    assert eigvals.min() > -1e-9


def test_dial_monotonically_increases_cross_cluster_co_movement() -> None:
    sizes = (8, 8)
    dials = [0.0, 0.05, 0.1, 0.2, 0.35]
    measured_cross = []
    measured_within = []
    for dial in dials:
        config = ClusteredSystemConfig(
            cluster_sizes=sizes,
            cross_coupling=dial,
            within_coupling=0.45,
            heterogeneous_marginals=False,
            marginal_default_prob=0.1,
            seed=0,
        )
        spec = make_clustered_system(config)
        samples = reference_default_samples(spec, n_samples=120_000, seed=7)
        corr = sample_diagnostics(samples).sampled_pairwise_corr
        within, cross = cluster_block_means(corr, _labels(spec))
        measured_cross.append(cross)
        measured_within.append(within)

    measured_cross = np.array(measured_cross)
    measured_within = np.array(measured_within)

    # Cross-cluster co-default correlation increases monotonically with the dial
    # (allowing a small sampling slack between consecutive steps).
    diffs = np.diff(measured_cross)
    assert np.all(diffs > -0.01)
    assert measured_cross[-1] > measured_cross[0] + 0.02

    # Within-cluster co-movement stays clearly above cross-cluster throughout.
    assert np.all(measured_within > measured_cross + 0.05)


def test_reference_sampling_reproduces_marginals_and_correlation() -> None:
    config = ClusteredSystemConfig(
        cluster_sizes=(6, 6),
        cross_coupling=0.06,
        within_coupling=0.4,
        heterogeneous_marginals=True,
        seed=3,
    )
    spec = make_clustered_system(config)
    samples = reference_default_samples(spec, n_samples=200_000, seed=11)
    diag = sample_diagnostics(samples)

    # Marginals reproduced to sampling tolerance.
    np.testing.assert_allclose(
        diag.sampled_marginals, spec.marginal_default_probs, atol=0.01
    )

    # Co-default (binary) correlation structure matches the thresholded latent model.
    target_joint = spec.target_pairwise_joint_probs()
    from systemic_risk.spec import joint_to_corr

    target_corr = joint_to_corr(target_joint, spec.marginal_default_probs)
    off = ~np.eye(spec.n, dtype=bool)
    np.testing.assert_allclose(
        diag.sampled_pairwise_corr[off], target_corr[off], atol=0.02
    )


def test_config_rejects_oversized_clusters() -> None:
    with pytest.raises(ValueError):
        ClusteredSystemConfig(cluster_sizes=(21,))


def test_config_rejects_loading_budget_overflow() -> None:
    with pytest.raises(ValueError):
        ClusteredSystemConfig(cluster_sizes=(4, 4), cross_coupling=0.6, within_coupling=0.5)
