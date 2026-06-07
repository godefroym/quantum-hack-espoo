"""SIMULATION preview of the 3-cluster mixture over the STRESSED real exposure network.

The no-hardware reference twin of ``scripts/run_real_cluster_mixture_hardware.py``. It applies
the 2008-style stress transform (``systemic_risk.data_network.apply_stress``) to the real
38-entity network, then runs the EXACT-STATEVECTOR cluster mixture on the EXISTING persisted
3-cluster partition (``outputs/data_network/real_network_partition.json``, sizes 14/14/10 --
reused, never re-derived):

  1. stress the baseline spec -> crisis marginals (mean PD ~15%), correlation UNCHANGED;
  2. confirm loadability: how many of the 38 stressed PDs clear the ~2.7% QPU noise floor;
  3. sample each cluster INDEPENDENTLY via its own exact statevector loader
     (``sample_clusters_statevector`` -- the simulated stand-in for one device per cluster);
  4. reconcile the per-cluster samples into the global joint with the common-shock reconciler
     fit to the stressed spec's own cross-cluster co-default target;
  5. run the GLOBAL cascade on the real exposure graph and report the tail risk
     (P(severe) / CVaR / mean cascade count) under stress.

This is the faithful reference a hardware run on the stressed spec SHOULD reproduce. No QPU is
touched -- compute only. Artifacts go to ``outputs/real_cluster_mixture_stress/``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from systemic_risk.data_network import QPU_NOISE_FLOOR, apply_stress, build_network_spec
from systemic_risk.generators.gaussian_copula import GaussianCopulaGenerator
from systemic_risk.generators.quantum.budget_clustering import (
    ClusterPartition,
    _within_cut_weight,
    dependency_for_clustering,
)
from systemic_risk.mixture import (
    CommonShockReconciler,
    cross_cluster_corr_target,
    independent_global_samples,
    reconciliation_diagnostics,
    sample_clusters_statevector,
)
from systemic_risk.mixture.reconcile import _binary_corr, _mean_cross_block
from systemic_risk.spec import SystemSpec

DEFAULT_PARTITION = ROOT / "outputs" / "data_network" / "real_network_partition.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--partition", type=Path, default=DEFAULT_PARTITION)
    parser.add_argument("--target-mean-pd", type=float, default=0.15,
                        help="Cross-entity mean stressed PD (2008 anchor; default 0.15).")
    parser.add_argument("--crisis-floor", type=float, default=QPU_NOISE_FLOOR,
                        help="Floor every stressed PD to this (default = noise floor 0.027) so "
                             "all entities clear the readout floor; re-solved to hold the mean.")
    parser.add_argument("--edge-threshold", type=float, default=0.02)
    parser.add_argument("--n-samples", type=int, default=200_000,
                        help="Samples per cluster / reference (exact statevector draws).")
    parser.add_argument("--calib-iters", type=int, default=30)
    parser.add_argument("--cascade-max-eval", type=int, default=4000)
    parser.add_argument("--severe-fraction", type=float, default=0.5,
                        help="P(severe) = P(post-cascade default count >= severe_fraction * n).")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "outputs" / "real_cluster_mixture_stress"
    )
    return parser.parse_args()


def load_partition(path: Path, spec: SystemSpec) -> tuple[ClusterPartition, dict]:
    """Rebuild the persisted partition as a ClusterPartition (clusters never re-derived)."""
    record = json.loads(path.read_text(encoding="utf-8"))
    clusters = [sorted(int(m) for m in c["member_indices"]) for c in record["clusters"]]
    labels = np.asarray(record["labels"], dtype=int)
    if labels.shape[0] != spec.n:
        raise ValueError(f"partition labels ({labels.shape[0]}) != spec.n ({spec.n})")
    dep = dependency_for_clustering(spec)
    within, cut = _within_cut_weight(dep, labels)
    partition = ClusterPartition(
        clusters=clusters,
        labels=labels,
        max_cluster_size=max(len(c) for c in clusters),
        within_weight=within,
        cut_weight=cut,
    )
    return partition, record


def cluster_targets(spec: SystemSpec, partition: ClusterPartition) -> list[dict]:
    """Per-cluster target marginals + within-cluster correlation the loader must reproduce."""
    p = np.asarray(spec.marginal_default_probs, dtype=float)
    dep = np.abs(spec.dependency_matrix())
    out = []
    for idx, members in enumerate(partition.clusters):
        pm = p[members]
        sub = dep[np.ix_(members, members)]
        iu = np.triu_indices(sub.shape[0], k=1)
        out.append({
            "cluster": idx,
            "size": len(members),
            "member_indices": list(members),
            "member_names": [spec.node_names[m] for m in members],
            "target_pd_min": round(float(pm.min()), 5),
            "target_pd_max": round(float(pm.max()), 5),
            "target_pd_mean": round(float(pm.mean()), 5),
            "n_above_noise_floor": int(np.sum(pm >= QPU_NOISE_FLOOR)),
            "target_within_abs_corr_mean": round(float(np.mean(sub[iu])), 4),
        })
    return out


def loadability_report(base: SystemSpec, stressed: SystemSpec, calib) -> dict:
    pb = np.asarray(base.marginal_default_probs, dtype=float)
    ps = np.asarray(stressed.marginal_default_probs, dtype=float)
    below = [i for i in range(stressed.n) if ps[i] < QPU_NOISE_FLOOR]
    return {
        "noise_floor": QPU_NOISE_FLOOR,
        "baseline_mean_pd": round(float(pb.mean()), 6),
        "baseline_n_above_floor": int(np.sum(pb >= QPU_NOISE_FLOOR)),
        "stressed_mean_pd": round(float(ps.mean()), 6),
        "stressed_min_pd": round(float(ps.min()), 6),
        "stressed_max_pd": round(float(ps.max()), 6),
        "stressed_n_above_floor": int(np.sum(ps >= QPU_NOISE_FLOOR)),
        "n_entities": stressed.n,
        "snr_min": round(float(ps.min() / QPU_NOISE_FLOOR), 3),
        "snr_mean": round(float(ps.mean() / QPU_NOISE_FLOOR), 3),
        "entities_below_floor": [
            {"index": i, "name": stressed.node_names[i], "pd": round(float(ps[i]), 6)}
            for i in below
        ],
        "all_clear_floor": len(below) == 0,
        "calibration": calib.to_dict(),
    }


def severe_probabilities(
    spec: SystemSpec, reference, reconciled, independent, severe_fraction: float, max_eval: int
) -> dict:
    """P(post-cascade default count >= severe_fraction * n) for each joint, on the real graph."""
    from systemic_risk.mixture.pipeline import cascade_loss_cvar

    threshold = int(np.ceil(severe_fraction * spec.n))

    def p_severe(samples) -> dict:
        _, counts = cascade_loss_cvar(samples, spec, alpha=0.95, max_eval=max_eval)
        return {
            "p_severe": round(float(np.mean(counts >= threshold)), 5),
            "mean_cascade_count": round(float(counts.mean()), 4),
        }

    return {
        "severe_threshold_count": threshold,
        "severe_fraction": severe_fraction,
        "reference": p_severe(reference),
        "reconciled": p_severe(reconciled),
        "independent": p_severe(independent),
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    base = build_network_spec().to_system_spec()
    stressed, calib = apply_stress(
        base, target_mean_pd=args.target_mean_pd, crisis_floor=args.crisis_floor
    )
    partition, partition_record = load_partition(args.partition, stressed)

    load_rep = loadability_report(base, stressed, calib)
    targets = cluster_targets(stressed, partition)

    # --- exact-statevector cluster mixture on the STRESSED spec ----------------------------
    labels = partition.labels
    clusters = sample_clusters_statevector(
        stressed, partition, args.n_samples, seed=args.seed + 1,
        edge_threshold=args.edge_threshold, calibration_iterations=args.calib_iters,
    )
    target_cross = cross_cluster_corr_target(stressed, labels)
    rec = CommonShockReconciler(stressed.n, labels).fit_reconcile(
        clusters, target_cross, args.n_samples, seed=args.seed + 2
    )
    independent = independent_global_samples(clusters, stressed.n)

    ref_gen = GaussianCopulaGenerator()
    ref_gen.fit(stressed)
    reference = ref_gen.sample(args.n_samples, seed=args.seed + 3)

    diag = reconciliation_diagnostics(
        reference, rec.samples, independent, labels, stressed.n,
        cascade_spec=stressed, cascade_max_eval=args.cascade_max_eval,
    )
    severe = severe_probabilities(
        stressed, reference, rec.samples, independent,
        args.severe_fraction, args.cascade_max_eval,
    )

    report = {
        "status": "simulation-preview",
        "system": "REAL 38-entity exposure network -- 2008 STRESS, exact statevector (no hardware)",
        "n": stressed.n,
        "partition_source": str(args.partition),
        "partition_content_hash": partition_record.get("content_hash"),
        "cluster_sizes": partition.sizes,
        "labels": labels.tolist(),
        "reference": "full-network Gaussian-copula joint fit to the STRESSED target correlation",
        "n_samples": args.n_samples,
        "loadability": load_rep,
        "cluster_targets": targets,
        "fitted_beta": round(rec.beta, 5),
        "target_cross_corr": round(rec.target_cross_corr, 5),
        "achieved_cross_corr": round(rec.achieved_cross_corr, 5),
        "cascade_severe": severe,
        "diagnostics": diag,
    }

    report_path = args.output_dir / "stress_preview_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    stressed.save_json(args.output_dir / "stressed_spec.json")
    np.savez_compressed(
        args.output_dir / "reconciled_global_stress.npz",
        reconciled=rec.samples, independent=independent,
        reference=reference, labels=labels,
    )

    print(json.dumps({
        "status": report["status"],
        "loadability": {
            k: load_rep[k] for k in (
                "stressed_mean_pd", "stressed_min_pd", "stressed_max_pd",
                "stressed_n_above_floor", "n_entities", "snr_min", "all_clear_floor",
            )
        },
        "cluster_targets": [
            {k: t[k] for k in (
                "cluster", "size", "target_pd_mean", "target_pd_min", "target_pd_max",
                "target_within_abs_corr_mean",
            )} for t in targets
        ],
        "fitted_beta": report["fitted_beta"],
        "target_cross_corr": report["target_cross_corr"],
        "achieved_cross_corr": report["achieved_cross_corr"],
        "cascade_severe": severe,
        "cascade_cvar": {
            "reference": round(diag["reference"]["cascade_count_cvar"], 3),
            "reconciled": round(diag["reconciled"]["cascade_count_cvar"], 3),
            "independent": round(diag["independent"]["cascade_count_cvar"], 3),
        },
        "mean_cascade_count": {
            "reference": round(diag["reference"]["mean_cascade_count"], 3),
            "reconciled": round(diag["reconciled"]["mean_cascade_count"], 3),
            "independent": round(diag["independent"]["mean_cascade_count"], 3),
        },
    }, indent=2))
    print(f"\nSaved report -> {report_path}")


if __name__ == "__main__":
    main()
