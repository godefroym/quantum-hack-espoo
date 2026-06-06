"""Tests for budget-respecting cluster discovery.

The point of the suite: prove the clusterer (a) RECOVERS planted ground-truth
clusters at high partition agreement on well-separated instances, (b) degrades
gracefully as cross-cluster coupling rises, and (c) NEVER returns a cluster larger
than the hard qubit-budget cap, across normal and degenerate inputs.
"""

from __future__ import annotations

import numpy as np
import pytest

from systemic_risk.data import (
    ClusteredSystemConfig,
    make_clustered_system,
    reference_default_samples,
)
from systemic_risk.data_network.cluster import adjusted_rand_index
from systemic_risk.generators.quantum.ansatz import partition_blocks
from systemic_risk.generators.quantum.budget_clustering import (
    ClusterPartition,
    budget_clusters_from_dependency,
    dependency_for_clustering,
    discover_clusters,
    split_oversize_group,
)
from systemic_risk.spec import SystemSpec


# --------------------------------------------------------------------------- helpers


def _assert_valid_cover(partition: ClusterPartition, n: int, cap: int) -> None:
    """Every cluster fits the cap and the clusters are a disjoint cover of 0..n-1."""
    seen: list[int] = []
    for cluster in partition.clusters:
        assert 1 <= len(cluster) <= cap, f"cluster {cluster} violates cap {cap}"
        seen.extend(cluster)
    assert sorted(seen) == list(range(n)), "clusters must be a disjoint cover of all nodes"
    assert partition.labels.shape == (n,)
    # labels agree with the cluster lists
    for cid, cluster in enumerate(partition.clusters):
        for node in cluster:
            assert partition.labels[node] == cid


# --------------------------------------------------------------------------- recovery


def test_recovers_well_separated_planted_clusters() -> None:
    """Low cross-coupling -> near-perfect recovery of the planted partition."""
    config = ClusteredSystemConfig(
        cluster_sizes=(7, 7, 6),
        cross_coupling=0.02,
        within_coupling=0.5,
        seed=1,
    )
    spec = make_clustered_system(config)
    truth = np.asarray(spec.metadata["cluster_labels"], dtype=int)

    partition = discover_clusters(spec, max_cluster_size=8)
    _assert_valid_cover(partition, spec.n, cap=8)

    ari = adjusted_rand_index(truth, partition.labels)
    assert ari >= 0.95, f"expected near-perfect recovery, got ARI={ari:.3f}"


def test_recovery_degrades_gracefully_with_coupling() -> None:
    """ARI is high at low coupling and trends down (not up) as the dial rises."""
    sizes = (8, 8, 6)
    cap = 8
    aris: list[float] = []
    for cross in (0.01, 0.05, 0.15, 0.30):
        config = ClusteredSystemConfig(
            cluster_sizes=sizes,
            cross_coupling=cross,
            within_coupling=0.5,
            seed=3,
        )
        spec = make_clustered_system(config)
        truth = np.asarray(spec.metadata["cluster_labels"], dtype=int)
        partition = discover_clusters(spec, max_cluster_size=cap)
        _assert_valid_cover(partition, spec.n, cap=cap)
        aris.append(adjusted_rand_index(truth, partition.labels))

    # Well-separated end recovers nearly perfectly...
    assert aris[0] >= 0.95, f"low-coupling ARI too low: {aris}"
    # ...and increasing the cross-coupling never improves recovery (graceful decay).
    assert aris[-1] <= aris[0] + 1e-9, f"recovery should not improve with coupling: {aris}"


def test_recovery_holds_across_seeds() -> None:
    """Recovery is a property of the data, not a lucky seed."""
    for seed in range(5):
        config = ClusteredSystemConfig(
            cluster_sizes=(6, 6, 6),
            cross_coupling=0.02,
            within_coupling=0.5,
            seed=seed,
        )
        spec = make_clustered_system(config)
        truth = np.asarray(spec.metadata["cluster_labels"], dtype=int)
        partition = discover_clusters(spec, max_cluster_size=6)
        _assert_valid_cover(partition, spec.n, cap=6)
        ari = adjusted_rand_index(truth, partition.labels)
        assert ari >= 0.9, f"seed {seed}: ARI={ari:.3f}"


def test_recovery_from_measured_sample_correlations() -> None:
    """Cluster on the EMPIRICAL co-default correlation from finite samples.

    With a strong within-cluster excess the planted structure survives sampling noise and
    is recovered nearly perfectly; this exercises the realistic path where the dependency
    matrix is estimated, not handed to us exactly.
    """
    config = ClusteredSystemConfig(
        cluster_sizes=(8, 8, 6),
        cross_coupling=0.05,
        within_coupling=0.45,
        seed=3,
    )
    spec = make_clustered_system(config)
    truth = np.asarray(spec.metadata["cluster_labels"], dtype=int)

    samples = reference_default_samples(spec, 5000, seed=7).astype(float)
    measured = np.corrcoef(samples.T)
    measured = np.nan_to_num(measured, nan=0.0)
    np.fill_diagonal(measured, 0.0)

    partition = budget_clusters_from_dependency(np.abs(measured), max_cluster_size=8)
    _assert_valid_cover(partition, spec.n, cap=8)
    assert adjusted_rand_index(truth, partition.labels) >= 0.95


# --------------------------------------------------------------------------- size cap


def test_cap_forces_split_of_one_large_cohesive_cluster() -> None:
    """A single planted cluster larger than the cap must be split to fit the budget."""
    config = ClusteredSystemConfig(
        cluster_sizes=(18,),
        cross_coupling=0.02,
        within_coupling=0.5,
        seed=2,
    )
    spec = make_clustered_system(config)
    cap = 8
    partition = discover_clusters(spec, max_cluster_size=cap)
    _assert_valid_cover(partition, spec.n, cap=cap)
    assert partition.n_clusters >= 3  # 18 nodes cannot fit in < 3 clusters of <= 8


def test_cap_respected_for_many_small_planted_clusters() -> None:
    config = ClusteredSystemConfig(
        cluster_sizes=(5, 5, 5, 5),
        cross_coupling=0.03,
        within_coupling=0.5,
        seed=4,
    )
    spec = make_clustered_system(config)
    for cap in (3, 5, 7, 20):
        partition = discover_clusters(spec, max_cluster_size=cap)
        _assert_valid_cover(partition, spec.n, cap=cap)


# --------------------------------------------------------------------------- objective


def test_within_weight_exceeds_cut_on_well_separated_instance() -> None:
    config = ClusteredSystemConfig(
        cluster_sizes=(7, 7, 6),
        cross_coupling=0.02,
        within_coupling=0.5,
        seed=1,
    )
    spec = make_clustered_system(config)
    partition = discover_clusters(spec, max_cluster_size=8)
    assert partition.within_weight > partition.cut_weight
    assert 0.0 <= partition.cut_fraction < 0.5


# --------------------------------------------------------------------------- splitter


def test_splitter_cuts_weakest_link() -> None:
    """Two tight triangles joined by one thin edge must split along the thin edge."""
    # nodes 0,1,2 strongly tied; nodes 3,4,5 strongly tied; one weak 2-3 link.
    dep = np.zeros((6, 6))
    for a, b in [(0, 1), (0, 2), (1, 2)]:
        dep[a, b] = dep[b, a] = 0.9
    for a, b in [(3, 4), (3, 5), (4, 5)]:
        dep[a, b] = dep[b, a] = 0.9
    dep[2, 3] = dep[3, 2] = 0.05

    pieces = split_oversize_group(dep, list(range(6)), max_cluster_size=3)
    pieces = sorted([sorted(p) for p in pieces])
    assert pieces == [[0, 1, 2], [3, 4, 5]]


def test_splitter_beats_index_order_chop() -> None:
    """When the cohesive groups are interleaved by index, the index chop severs strong
    links but the weight-aware splitter keeps them together."""
    # cohesive set A = even indices, B = odd indices; an index-order chop of [0..5]
    # into [0,1,2]/[3,4,5] would cut strong A-A and B-B links.
    dep = np.zeros((6, 6))
    even = [0, 2, 4]
    odd = [1, 3, 5]
    for grp in (even, odd):
        for a in grp:
            for b in grp:
                if a < b:
                    dep[a, b] = dep[b, a] = 0.9
    # weak cross links
    dep[0, 1] = dep[1, 0] = 0.02

    pieces = split_oversize_group(dep, list(range(6)), max_cluster_size=3)
    pieces = sorted([sorted(p) for p in pieces])
    assert pieces == [[0, 2, 4], [1, 3, 5]]


def test_splitter_noop_when_within_cap() -> None:
    dep = np.ones((4, 4)) - np.eye(4)
    pieces = split_oversize_group(dep, [0, 1, 2, 3], max_cluster_size=8)
    assert pieces == [[0, 1, 2, 3]]


def test_splitter_no_edges_falls_back_to_chunks() -> None:
    dep = np.zeros((5, 5))
    pieces = split_oversize_group(dep, [0, 1, 2, 3, 4], max_cluster_size=2)
    for p in pieces:
        assert len(p) <= 2
    assert sorted(sum(pieces, [])) == [0, 1, 2, 3, 4]


# --------------------------------------------------------------------------- degenerate


def test_single_node() -> None:
    dep = np.zeros((1, 1))
    partition = budget_clusters_from_dependency(dep, max_cluster_size=5)
    assert partition.clusters == [[0]]
    _assert_valid_cover(partition, 1, cap=5)


def test_no_edges_within_cap_is_single_block() -> None:
    dep = np.zeros((4, 4))
    partition = budget_clusters_from_dependency(dep, max_cluster_size=8)
    _assert_valid_cover(partition, 4, cap=8)
    assert partition.n_clusters == 1


def test_no_edges_over_cap_splits_to_fit() -> None:
    dep = np.zeros((10, 10))
    partition = budget_clusters_from_dependency(dep, max_cluster_size=4)
    _assert_valid_cover(partition, 10, cap=4)


def test_empty_matrix() -> None:
    dep = np.zeros((0, 0))
    partition = budget_clusters_from_dependency(dep, max_cluster_size=5)
    assert partition.clusters == []
    assert partition.labels.shape == (0,)


def test_invalid_cap_raises() -> None:
    with pytest.raises(ValueError):
        budget_clusters_from_dependency(np.zeros((3, 3)), max_cluster_size=0)


# --------------------------------------------------------------------------- fallbacks


def test_exposure_fallback_when_no_correlation() -> None:
    """With no correlation, clustering must use the symmetrised exposure graph."""
    exposure = np.zeros((4, 4))
    exposure[0, 1] = exposure[1, 0] = 10.0
    exposure[2, 3] = exposure[3, 2] = 10.0
    spec = SystemSpec(
        node_names=["A", "B", "C", "D"],
        node_types=["bank"] * 4,
        exposure_matrix=exposure,
        capital_buffers=np.ones(4),
        marginal_default_probs=np.full(4, 0.1),
        target_pairwise_corr=None,
    )
    dep = dependency_for_clustering(spec)
    assert dep.max() > 0.0  # came from exposure
    partition = discover_clusters(spec, max_cluster_size=2)
    _assert_valid_cover(partition, 4, cap=2)
    pieces = sorted([sorted(c) for c in partition.clusters])
    assert pieces == [[0, 1], [2, 3]]


def test_signed_correlation_handled_by_magnitude() -> None:
    """A strong NEGATIVE correlation still binds two nodes into one cluster."""
    corr = np.eye(4)
    corr[0, 1] = corr[1, 0] = -0.9  # strong negative
    corr[2, 3] = corr[3, 2] = 0.9   # strong positive
    spec = SystemSpec(
        node_names=["A", "B", "C", "D"],
        node_types=["bank"] * 4,
        exposure_matrix=np.zeros((4, 4)),
        capital_buffers=np.ones(4),
        marginal_default_probs=np.full(4, 0.1),
        target_pairwise_corr=corr,
    )
    partition = discover_clusters(spec, max_cluster_size=2)
    pieces = sorted([sorted(c) for c in partition.clusters])
    assert pieces == [[0, 1], [2, 3]]


# --------------------------------------------------------------------------- ansatz integration


def test_partition_blocks_respects_cap_and_weight() -> None:
    """The ansatz partitioner now splits oversize components by weight, capped."""
    corr = np.eye(6)
    even, odd = [0, 2, 4], [1, 3, 5]
    for grp in (even, odd):
        for a in grp:
            for b in grp:
                if a < b:
                    corr[a, b] = corr[b, a] = 0.9
    corr[0, 1] = corr[1, 0] = 0.4  # one bridge so the whole thing is one component
    spec = SystemSpec(
        node_names=[f"n{i}" for i in range(6)],
        node_types=["bank"] * 6,
        exposure_matrix=np.zeros((6, 6)),
        capital_buffers=np.ones(6),
        marginal_default_probs=np.full(6, 0.1),
        target_pairwise_corr=corr,
    )
    # All within-cluster edges plus the bridge make a single connected component of 6.
    edges = [(a, b) for a in range(6) for b in range(a + 1, 6) if corr[a, b] > 0.0]
    blocks = partition_blocks(spec, edges, max_block=3)
    for block in blocks:
        assert len(block) <= 3
    assert sorted(sum(blocks, [])) == list(range(6))
    # The weak bridge (0-1) should be the cut, keeping the two cohesive triples together.
    blocks_sorted = sorted([sorted(b) for b in blocks])
    assert blocks_sorted == [[0, 2, 4], [1, 3, 5]]
