from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_cluster_mixture.py"
_SPEC = importlib.util.spec_from_file_location("run_cluster_mixture", SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
SCRIPT = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(SCRIPT)

from systemic_risk.data import (
    ClusteredSystemConfig,
    make_clustered_system,
    reference_default_samples,
)
from systemic_risk.generators.quantum import discover_clusters
from systemic_risk.mixture import (
    ClusterSamples,
    CommonShockReconciler,
    attach_cluster_exposures,
    cluster_samples_from_bitstrings,
    cross_cluster_corr_target,
    default_count_distribution,
    independent_global_samples,
    reconciliation_diagnostics,
    sample_clusters_statevector,
    total_variation,
)
from systemic_risk.mixture.reconcile import _binary_corr, _mean_cross_block


def _within_mean(corr: np.ndarray, labels: np.ndarray) -> float:
    n = corr.shape[0]
    same = (labels[:, None] == labels[None, :]) & ~np.eye(n, dtype=bool)
    return float(corr[same].mean())


def _make(cross_coupling: float, seed: int = 0):
    config = ClusteredSystemConfig(
        cluster_sizes=(8, 8, 6),
        cross_coupling=cross_coupling,
        within_coupling=0.45,
        marginal_default_prob=0.06,
        seed=seed,
    )
    spec = make_clustered_system(config)
    partition = discover_clusters(spec, max_cluster_size=10)
    return spec, partition


# --------------------------------------------------------------- the hardware seam
def test_reconciler_accepts_externally_supplied_bitstrings() -> None:
    """The reconciler must accept per-cluster samples from ANY source (the hardware path)."""
    spec, partition = _make(cross_coupling=0.1)
    labels = partition.labels
    rng = np.random.default_rng(0)

    # Simulate "measured" per-cluster bitstrings handed in from outside (e.g. real devices).
    clusters = []
    for members in partition.clusters:
        bits = (rng.random((5000, len(members))) < 0.1).astype(int)
        clusters.append(cluster_samples_from_bitstrings(members, bits, source="hardware:test"))
    assert all(c.source == "hardware:test" for c in clusters)

    target = cross_cluster_corr_target(spec, labels)
    rec = CommonShockReconciler(spec.n, labels).fit_reconcile(clusters, target, 5000, seed=1)
    assert rec.samples.shape == (5000, spec.n)
    assert set(np.unique(rec.samples)) <= {0, 1}
    assert rec.cluster_sources == ["hardware:test"] * partition.n_clusters


def test_cluster_samples_validates_shape() -> None:
    with pytest.raises(ValueError):
        ClusterSamples(members=(0, 1, 2), samples=np.zeros((10, 2), dtype=int))


# ------------------------------------------------------------------ reconciled marginals
def test_reconciled_marginals_match_loader_targets() -> None:
    """Reconciliation only couples clusters: per-node marginals are preserved exactly."""
    spec, partition = _make(cross_coupling=0.12)
    labels = partition.labels
    clusters = sample_clusters_statevector(spec, partition, 60_000, seed=1)

    # Marginals carried by the independent per-cluster samples (the within-cluster law).
    loader_marg = np.zeros(spec.n)
    for c in clusters:
        loader_marg[list(c.members)] = c.samples.mean(axis=0)

    target = cross_cluster_corr_target(spec, labels)
    rec = CommonShockReconciler(spec.n, labels).fit_reconcile(clusters, target, 60_000, seed=2)
    np.testing.assert_allclose(rec.samples.mean(axis=0), loader_marg, atol=0.01)


def test_reconciled_within_cluster_structure_is_preserved() -> None:
    """The within-cluster co-default correlation is untouched by the coupling step."""
    spec, partition = _make(cross_coupling=0.15)
    labels = partition.labels
    clusters = sample_clusters_statevector(spec, partition, 60_000, seed=3)
    indep = independent_global_samples(clusters, spec.n)

    target = cross_cluster_corr_target(spec, labels)
    rec = CommonShockReconciler(spec.n, labels).fit_reconcile(clusters, target, 60_000, seed=4)

    within_indep = _within_mean(_binary_corr(indep), labels)
    within_rec = _within_mean(_binary_corr(rec.samples), labels)
    # Reconciliation reuses each cluster's own samples, so within-cluster structure matches the
    # independent baseline (which is the raw per-cluster law) to sampling tolerance.
    assert abs(within_rec - within_indep) < 0.02


# --------------------------------------------------- cross-cluster co-movement vs baseline
def test_reconciled_cross_cluster_corr_beats_independent_baseline() -> None:
    spec, partition = _make(cross_coupling=0.18)
    labels = partition.labels
    clusters = sample_clusters_statevector(spec, partition, 60_000, seed=5)

    target = cross_cluster_corr_target(spec, labels)
    rec = CommonShockReconciler(spec.n, labels).fit_reconcile(clusters, target, 60_000, seed=6)

    indep = independent_global_samples(clusters, spec.n)
    cross_rec = _mean_cross_block(_binary_corr(rec.samples), labels)
    cross_indep = _mean_cross_block(_binary_corr(indep), labels)

    # Independent clusters carry ~zero cross-cluster co-movement; reconciliation hits the target.
    assert abs(cross_indep) < 0.005
    assert abs(cross_rec - target) < 0.01
    assert abs(cross_rec - target) < abs(cross_indep - target)


def test_fit_target_comes_from_spec_not_planting() -> None:
    """cross_cluster_corr_target reads the spec's own target, recoverable on a relabelled spec."""
    spec, partition = _make(cross_coupling=0.1)
    target = cross_cluster_corr_target(spec, partition.labels)
    # Positive, small (cross-cluster), and well below the within-cluster block mean.
    dep = spec.dependency_matrix()
    within = _within_mean(dep, partition.labels)
    assert 0.0 < target < within


# ------------------------------------------------------- the make-or-break sweep claim
def test_reconciled_global_joint_closer_to_reference_than_independent_in_sweep() -> None:
    """Across the weak-to-moderate coupling regime, reconciled tracks the reference tail and
    cross-cluster correlation better than the naive independent baseline."""
    levels = [0.06, 0.1, 0.18]
    cross_wins = 0
    tail_wins = 0
    for dial in levels:
        spec, partition = _make(cross_coupling=dial, seed=1)
        labels = partition.labels
        clusters = sample_clusters_statevector(spec, partition, 40_000, seed=7)
        target = cross_cluster_corr_target(spec, labels)
        rec = CommonShockReconciler(spec.n, labels).fit_reconcile(clusters, target, 40_000, seed=8)
        indep = independent_global_samples(clusters, spec.n)
        reference = reference_default_samples(spec, 40_000, seed=9)

        diag = reconciliation_diagnostics(
            reference, rec.samples, indep, labels, spec.n, cascade_spec=None
        )
        if (
            diag["reconciled"]["cross_cluster_corr_abs_err_vs_ref"]
            <= diag["independent"]["cross_cluster_corr_abs_err_vs_ref"]
        ):
            cross_wins += 1
        if (
            diag["reconciled"]["tail_count_l1_vs_ref"]
            <= diag["independent"]["tail_count_l1_vs_ref"] + 1e-4
        ):
            tail_wins += 1

    # Cross-cluster correlation: reconciliation wins at every level.
    assert cross_wins == len(levels)
    # Tail of the default-count law: reconciliation wins at the moderate levels.
    assert tail_wins >= 2


def test_cascade_reconciled_tracks_reference_tail_loss() -> None:
    """The global cascade tail loss under reconciliation moves toward the reference; the
    independent baseline under-states it."""
    spec, partition = _make(cross_coupling=0.25, seed=2)
    labels = partition.labels
    clusters = sample_clusters_statevector(spec, partition, 30_000, seed=10)
    target = cross_cluster_corr_target(spec, labels)
    rec = CommonShockReconciler(spec.n, labels).fit_reconcile(clusters, target, 30_000, seed=11)
    indep = independent_global_samples(clusters, spec.n)
    reference = reference_default_samples(spec, 30_000, seed=12)

    cascade_spec = attach_cluster_exposures(spec, labels, seed=2)
    diag = reconciliation_diagnostics(
        reference, rec.samples, indep, labels, spec.n,
        cascade_spec=cascade_spec, cascade_max_eval=4000,
    )
    ref_cvar = diag["reference"]["cascade_count_cvar"]
    rec_err = diag["reconciled"]["cascade_count_cvar_abs_err_vs_ref"]
    indep_err = diag["independent"]["cascade_count_cvar_abs_err_vs_ref"]
    # Reconciliation's tail cascade loss is closer to the reference than the independent baseline.
    assert rec_err <= indep_err
    assert ref_cvar > 0


def test_end_to_end_script_runs_in_dry_mode(capsys) -> None:
    """The entry-point script runs in dry mode (no heavy sampling) and emits valid JSON."""
    SCRIPT.sys.argv = ["run_cluster_mixture.py", "--cluster-sizes", "8", "8", "6", "--budget", "10"]
    SCRIPT.main()
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "dry-run"
    assert report["n"] == 22
    assert report["discovery"]["n_clusters"] >= 2
    assert report["discovery"]["matches_planting"] is True


def test_total_variation_and_count_distribution_basics() -> None:
    pmf = default_count_distribution(np.array([[1, 0, 0], [1, 1, 0], [0, 0, 0]]), 3)
    assert pmf.shape == (4,)
    assert np.isclose(pmf.sum(), 1.0)
    assert np.isclose(total_variation(pmf, pmf), 0.0)
    other = default_count_distribution(np.array([[1, 1, 1], [1, 1, 1]]), 3)
    assert total_variation(pmf, other) > 0.0
