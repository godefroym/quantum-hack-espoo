"""Classical+quantum MIXTURE pipeline: sample clusters independently, reconcile, cascade.

Runs the full roadmap end to end on a planted-cluster synthetic system:

  1. build (or accept) a clustered system with a tunable cross-cluster coupling dial;
  2. DISCOVER the clusters from the spec's dependency graph (budget-respecting, <= --budget
     qubits per cluster) -- the planted labels are only used for an after-the-fact sanity check;
  3. sample each cluster's correlated default distribution INDEPENDENTLY via its own statevector
     loader (the stand-in for one real device per cluster -- see the hardware hook below);
  4. RECONCILE the independent per-cluster samples into a global joint with a one-factor
     common-shock coupler fit to the spec's own cross-cluster co-default target;
  5. run the GLOBAL exposure cascade on the reconciled scenarios and report the tail risk;
  6. (sweep mode) compare reconciled vs the naive independent baseline against the ground-truth
     reference across coupling levels -- the make-or-break experiment.

Dry-run prints the plan (discovered partition, targets) without the heavy sampling/cascade;
``--run`` does the work and prints the JSON report; ``--sweep`` runs the coupling sweep.

REAL-HARDWARE SUBSTITUTION ("reconcile IBM machine results locally and solve")
-----------------------------------------------------------------------------
Step 3 is the only quantum step, and it is isolated behind one seam: a list of
``systemic_risk.mixture.ClusterSamples``. To swap in real per-cluster devices, run each
cluster's circuit on its own QPU, collect the measured 0/1 bitstrings, and build::

    from systemic_risk.mixture import cluster_samples_from_bitstrings, CommonShockReconciler
    clusters = [
        cluster_samples_from_bitstrings(members, measured_bits, source="hardware:ibm_boston")
        for members, measured_bits in per_device_results   # one entry per cluster/device
    ]
    target = cross_cluster_corr_target(spec, partition.labels)
    result = CommonShockReconciler(spec.n, partition.labels).fit_reconcile(clusters, target, N)

Nothing downstream changes: the reconciler and cascade are identical for simulated and measured
samples. Pass ``--hardware-samples a.npz b.npz ...`` (one ``.npz`` per cluster, each holding a
``samples`` 0/1 array in that cluster's member order) to take this path instead of simulating.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from systemic_risk.data import (
    ClusteredSystemConfig,
    cluster_block_means,
    make_clustered_system,
    reference_default_samples,
)
from systemic_risk.generators.quantum import discover_clusters
from systemic_risk.mixture import (
    CommonShockReconciler,
    attach_cluster_exposures,
    cluster_samples_from_bitstrings,
    cross_cluster_corr_target,
    independent_global_samples,
    reconciliation_diagnostics,
    sample_clusters_statevector,
)
from systemic_risk.mixture.reconcile import _binary_corr, _mean_cross_block


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cluster-sizes", type=int, nargs="+", default=[10, 10, 8],
                        help="Planted cluster sizes (each <= --budget).")
    parser.add_argument("--cross-coupling", type=float, default=0.1,
                        help="Cross-cluster dial beta_global for a single run.")
    parser.add_argument("--within-coupling", type=float, default=0.45)
    parser.add_argument("--marginal", type=float, default=0.05, help="Baseline marginal default prob.")
    parser.add_argument("--budget", type=int, default=12, help="Per-device qubit budget (max cluster size).")
    parser.add_argument("--n-samples", type=int, default=40_000, help="Samples per cluster / global.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run", action="store_true", help="Do the sampling + reconcile + cascade.")
    parser.add_argument("--sweep", action="store_true", help="Run the cross-coupling sweep experiment.")
    parser.add_argument("--sweep-levels", type=float, nargs="+",
                        default=[0.0, 0.03, 0.06, 0.1, 0.18, 0.3],
                        help="Cross-coupling dial values for the sweep.")
    parser.add_argument("--hardware-samples", type=Path, nargs="+", default=None,
                        help="One .npz per cluster (key 'samples') to reconcile instead of simulating.")
    parser.add_argument("--cascade-max-eval", type=int, default=4000,
                        help="Subsample size for the cascade tail-loss evaluation.")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "cluster_mixture")
    return parser.parse_args()


def build_spec(args: argparse.Namespace, cross_coupling: float):
    config = ClusteredSystemConfig(
        cluster_sizes=tuple(args.cluster_sizes),
        cross_coupling=cross_coupling,
        within_coupling=args.within_coupling,
        marginal_default_prob=args.marginal,
        seed=args.seed,
    )
    return make_clustered_system(config)


def discovery_block(spec, partition) -> dict:
    """Discovered-partition summary plus a sanity check against the planted labels."""
    planted = np.asarray(spec.metadata["cluster_labels"], dtype=int)
    dep = spec.dependency_matrix()
    within_disc, cross_disc = cluster_block_means(dep, partition.labels)
    within_plant, cross_plant = cluster_block_means(dep, planted)
    return {
        "discovered_sizes": partition.sizes,
        "n_clusters": partition.n_clusters,
        "cut_fraction": round(partition.cut_fraction, 4),
        "discovered_within_dep_mean": round(within_disc, 4),
        "discovered_cross_dep_mean": round(cross_disc, 4),
        "planted_within_dep_mean": round(within_plant, 4),
        "planted_cross_dep_mean": round(cross_plant, 4),
        "matches_planting": bool(_partition_matches(partition.labels, planted)),
    }


def _partition_matches(a: np.ndarray, b: np.ndarray) -> bool:
    """True if two label vectors induce the same partition (ignoring label names)."""
    a, b = np.asarray(a), np.asarray(b)
    return bool(np.all((a[:, None] == a[None, :]) == (b[:, None] == b[None, :])))


def load_hardware_clusters(paths, partition):
    """Build ClusterSamples from one .npz per cluster (the real-hardware path)."""
    if len(paths) != partition.n_clusters:
        raise ValueError(
            f"got {len(paths)} hardware sample files but {partition.n_clusters} discovered clusters"
        )
    clusters = []
    for members, path in zip(partition.clusters, paths):
        with np.load(path) as data:
            bits = np.asarray(data["samples"], dtype=int)
        clusters.append(
            cluster_samples_from_bitstrings(members, bits, source=f"hardware:{path.name}")
        )
    return clusters


def run_single(args, spec, partition, cross_coupling: float) -> dict:
    labels = partition.labels
    if args.hardware_samples is not None:
        clusters = load_hardware_clusters(args.hardware_samples, partition)
        n_samples = min(c.n_samples for c in clusters)
    else:
        clusters = sample_clusters_statevector(spec, partition, args.n_samples, seed=args.seed + 1)
        n_samples = args.n_samples

    target = cross_cluster_corr_target(spec, labels)
    reconciler = CommonShockReconciler(spec.n, labels)
    rec = reconciler.fit_reconcile(clusters, target, n_samples, seed=args.seed + 2)
    independent = independent_global_samples(clusters, spec.n)
    reference = reference_default_samples(spec, n_samples, seed=args.seed + 3)

    cascade_spec = attach_cluster_exposures(spec, labels, seed=args.seed)
    diag = reconciliation_diagnostics(
        reference, rec.samples, independent, labels, spec.n,
        cascade_spec=cascade_spec, cascade_max_eval=args.cascade_max_eval,
    )
    return {
        "cross_coupling": float(cross_coupling),
        "fitted_beta": round(rec.beta, 5),
        "target_cross_corr": round(rec.target_cross_corr, 5),
        "achieved_cross_corr": round(rec.achieved_cross_corr, 5),
        "cluster_sources": rec.cluster_sources,
        "n_samples": n_samples,
        "diagnostics": diag,
    }


def headline(level: dict) -> dict:
    """Squeeze one sweep level into a compact reconciled-vs-independent comparison row."""
    d = level["diagnostics"]
    return {
        "cross_coupling": level["cross_coupling"],
        "fitted_beta": level["fitted_beta"],
        "cross_corr_ref": round(d["reference"]["cross_cluster_corr"], 4),
        "cross_corr_reconciled": round(d["reconciled"]["cross_cluster_corr"], 4),
        "cross_corr_independent": round(d["independent"]["cross_cluster_corr"], 4),
        "cross_corr_err_reconciled": round(d["reconciled"]["cross_cluster_corr_abs_err_vs_ref"], 4),
        "cross_corr_err_independent": round(d["independent"]["cross_cluster_corr_abs_err_vs_ref"], 4),
        "tail_l1_reconciled": round(d["reconciled"]["tail_count_l1_vs_ref"], 5),
        "tail_l1_independent": round(d["independent"]["tail_count_l1_vs_ref"], 5),
        "jte_ref": round(d["reference"]["joint_tail_excess"], 5),
        "jte_reconciled": round(d["reconciled"]["joint_tail_excess"], 5),
        "jte_independent": round(d["independent"]["joint_tail_excess"], 5),
        "cascade_cvar_ref": round(d["reference"].get("cascade_count_cvar", float("nan")), 3),
        "cascade_cvar_reconciled": round(d["reconciled"].get("cascade_count_cvar", float("nan")), 3),
        "cascade_cvar_independent": round(d["independent"].get("cascade_count_cvar", float("nan")), 3),
        "cascade_cvar_err_reconciled": round(
            d["reconciled"].get("cascade_count_cvar_abs_err_vs_ref", float("nan")), 3),
        "cascade_cvar_err_independent": round(
            d["independent"].get("cascade_count_cvar_abs_err_vs_ref", float("nan")), 3),
        "reconciled_beats_independent_on_tail_l1": bool(
            d["reconciled"]["tail_count_l1_vs_ref"] <= d["independent"]["tail_count_l1_vs_ref"]
        ),
        "reconciled_beats_independent_on_cross_corr": bool(
            d["reconciled"]["cross_cluster_corr_abs_err_vs_ref"]
            <= d["independent"]["cross_cluster_corr_abs_err_vs_ref"]
        ),
    }


def main() -> None:
    args = parse_args()
    spec = build_spec(args, args.cross_coupling)
    partition = discover_clusters(spec, max_cluster_size=args.budget)

    plan = {
        "system": "planted-cluster synthetic",
        "n": spec.n,
        "planted_cluster_sizes": list(args.cluster_sizes),
        "budget_per_device": args.budget,
        "discovery": discovery_block(spec, partition),
        "n_samples": args.n_samples,
    }

    if not (args.run or args.sweep):
        print(json.dumps({
            "status": "dry-run",
            **plan,
            "next_command": "uv run python scripts/run_cluster_mixture.py --run   (or --sweep)",
        }, indent=2))
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.sweep:
        levels = []
        for dial in args.sweep_levels:
            spec_d = build_spec(args, dial)
            part_d = discover_clusters(spec_d, max_cluster_size=args.budget)
            levels.append(run_single(args, spec_d, part_d, dial))
        report = {
            "status": "sweep",
            "system": "planted-cluster synthetic",
            "n": spec.n,
            "budget_per_device": args.budget,
            "n_samples": args.n_samples,
            "headline": [headline(level) for level in levels],
            "levels": levels,
        }
        (args.output_dir / "sweep_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps({"status": "sweep", "headline": report["headline"]}, indent=2))
        return

    result = run_single(args, spec, partition, args.cross_coupling)
    report = {"status": "run", **plan, **result}
    (args.output_dir / "run_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
