"""Run the 3-cluster mixture over the STRESSED real exposure network on IBM hardware.

This is the HARDWARE twin of ``scripts/run_real_cluster_mixture_stress.py`` (the compute-only
simulation preview) and the STRESSED analogue of ``scripts/run_real_cluster_mixture_hardware.py``
(the baseline, noise-dominated hardware run). It:

  1. builds the real 38-entity exposure-network spec,
  2. applies the 2008-style stress transform
     (``systemic_risk.data_network.apply_stress(spec, target_mean_pd=0.15,
     crisis_floor=0.027)``) -- a rank-preserving logit-space shift that lifts the cross-entity
     MEAN PD to ~15% (every entity above the ~2.7% QPU readout floor) while leaving the real
     correlation graph UNCHANGED,
  3. reuses the EXISTING persisted 3-cluster partition
     (``outputs/data_network/real_network_partition.json`` -- k=3, sizes 14/14/10; never
     re-derived),
  4. runs the SAME shape of experiment as the baseline hardware run: one logical QPU experiment
     per cluster, each cluster's fitted dense (RY + CRY) block circuit submitted to a backend,
     measured bitstrings reconciled locally into the global joint via the common-shock
     reconciler, and that joint fed through the cascade on the real exposure graph.

WHY STRESS, NOT FAITHFUL
------------------------
The baseline run (``run_real_cluster_mixture_hardware.py``) loaded the real ~0.2%-mean PDs
faithfully: they sit BELOW the ~2.7% noise floor, so the marginals came back as pure noise
(per-cluster marginal RMSE ~0.20-0.23, see ``outputs/real_cluster_mixture_hw/``). The stress
transform lifts every PD above the floor (SNR_mean ~5.6 in the sim preview), so this run is the
first that can plausibly recover the loaded marginals AND correlation on hardware. The faithful
reference it should reproduce is ``outputs/real_cluster_mixture_stress/stress_preview_report.json``.

HONESTY CAVEATS baked in:

  * The stress overlay is a HYPOTHETICAL uniform crisis (one logit shift for all entities),
    calibrated to historical aggregates (Moody's GFC default rates + FRED BAA-AAA spread ~3.85x),
    not an entity-by-entity 2008 re-rating. Recorded in ``stress_calibration`` and the spec's
    ``metadata['stress']``.
  * These are DENSE blocks (every within-cluster edge above ``--edge-threshold``), well past the
    documented coherence boundary, so decoherence still attacks the within-cluster correlation;
    how much survives is exactly what this run measures.

Always dry-run first (omit ``--submit``) to see per-cluster gate/qubit/entangler budget, the
shot-batching plan, and the stressed-PD inspection (proving every entity clears the floor). With
``--submit`` it spends metered QPU time: one logical experiment per cluster, shots auto-batched to
the backend per-job cap and aggregated. Artifacts go to ``outputs/real_cluster_mixture_stress_hw/``
(distinct from the baseline ``real_cluster_mixture_hw/`` and the sim preview
``real_cluster_mixture_stress/``).
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
from systemic_risk.generators.quantum import ansatz as A
from systemic_risk.generators.quantum.ibm_runtime import _split_shots, run_block
from systemic_risk.mixture import (
    CommonShockReconciler,
    cluster_samples_from_bitstrings,
    cross_cluster_corr_target,
    independent_global_samples,
    reconciliation_diagnostics,
)
from systemic_risk.mixture.cluster_loader import _statevector_block_moments
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
    parser.add_argument("--calib-iters", type=int, default=30)
    parser.add_argument("--backend", default="ibm_boston")
    parser.add_argument("--shots", type=int, default=200_000,
                        help="Total shots per cluster (auto-batched to backend cap).")
    parser.add_argument("--max-shots-per-job", type=int, default=None,
                        help="Override the backend per-job shot cap for batch planning "
                             "(dry-run only; live runs read the real cap from the backend).")
    parser.add_argument("--optimization-level", type=int, choices=range(4), default=3)
    parser.add_argument("--submit", action="store_true", help="Submit the metered IBM jobs.")
    parser.add_argument("--ref-samples", type=int, default=200_000,
                        help="Gaussian-copula reference sample count for diagnostics.")
    parser.add_argument("--cascade-max-eval", type=int, default=4000)
    parser.add_argument("--severe-fraction", type=float, default=0.5,
                        help="P(severe) = P(post-cascade default count >= severe_fraction * n).")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "outputs" / "real_cluster_mixture_stress_hw"
    )
    return parser.parse_args()


def load_partition(path: Path) -> tuple[list[list[int]], np.ndarray, dict]:
    """Load the persisted real-network partition: cluster member-index lists + global labels."""
    record = json.loads(path.read_text(encoding="utf-8"))
    clusters = [sorted(int(m) for m in c["member_indices"]) for c in record["clusters"]]
    labels = np.asarray(record["labels"], dtype=int)
    return clusters, labels, record


def build_block(spec: SystemSpec, members, *, threshold: float, calib_iters: int):
    """Fitted dense block circuit for one cluster (all within-cluster edges above threshold).

    Identical construction to ``run_real_cluster_mixture_hardware.build_block`` -- reused so the
    stressed run is the same circuit shape as the baseline one (only the marginals differ).
    """
    members = sorted(int(m) for m in members)
    p = np.clip(spec.marginal_default_probs, 1e-6, 1.0 - 1e-6)
    cov = A.target_covariance(spec)
    member_set = set(members)
    edges = [
        (i, j)
        for (i, j) in A.dependency_edges(spec, threshold=threshold, within_clusters_only=False)
        if i in member_set and j in member_set
    ]
    circuit = A._block_circuit(members, p, cov, edges)
    if circuit.size > 1 and circuit.edges:
        circuit = A.calibrate_block(circuit, _statevector_block_moments, iterations=calib_iters)
    return circuit


def within_abs_corr(samples: np.ndarray) -> float:
    """Mean |pairwise correlation| inside a cluster's samples (collapses to 0 when decohered)."""
    if samples.shape[1] < 2:
        return float("nan")
    corr = np.corrcoef(samples, rowvar=False)
    iu = np.triu_indices(corr.shape[0], k=1)
    return float(np.nanmean(np.abs(corr[iu])))


def target_within_abs_corr(spec: SystemSpec, members) -> float:
    dep = np.abs(spec.dependency_matrix())
    sub = dep[np.ix_(members, members)]
    iu = np.triu_indices(sub.shape[0], k=1)
    return float(np.mean(sub[iu]))


def shot_plan(shots: int, max_shots_per_job: int | None) -> dict:
    """Describe how ``shots`` would batch into per-job publications for the dry-run report."""
    if max_shots_per_job is None:
        return {
            "total_shots_per_cluster": shots,
            "max_shots_per_job": "read from backend at submit time",
            "note": "batches computed live from backend.configuration().max_shots",
        }
    batches = _split_shots(shots, max_shots_per_job)
    return {
        "total_shots_per_cluster": shots,
        "max_shots_per_job": max_shots_per_job,
        "n_jobs_per_cluster": len(batches),
        "shot_batches": list(batches),
    }


def loadability_report(base: SystemSpec, stressed: SystemSpec) -> dict:
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
        "all_clear_floor": len(below) == 0,
    }


def pd_inspection(stressed: SystemSpec, clusters) -> list[dict]:
    ps = np.asarray(stressed.marginal_default_probs, dtype=float)
    out = []
    for idx, members in enumerate(clusters):
        pm = ps[members]
        out.append({
            "cluster": idx,
            "members": members,
            "pd_min": round(float(pm.min()), 6),
            "pd_max": round(float(pm.max()), 6),
            "pd_mean": round(float(pm.mean()), 6),
            "n_above_noise_floor": int(np.sum(pm >= QPU_NOISE_FLOOR)),
            "all_above_noise_floor": bool(np.all(pm >= QPU_NOISE_FLOOR)),
        })
    return out


def main() -> None:
    args = parse_args()
    base_spec = build_network_spec().to_system_spec()
    spec, calib = apply_stress(
        base_spec, target_mean_pd=args.target_mean_pd, crisis_floor=args.crisis_floor
    )
    clusters, labels, partition_record = load_partition(args.partition)

    if labels.shape[0] != spec.n:
        raise ValueError(f"partition labels ({labels.shape[0]}) != spec.n ({spec.n})")

    blocks = [
        build_block(spec, members, threshold=args.edge_threshold, calib_iters=args.calib_iters)
        for members in clusters
    ]

    plan = {
        "system": "REAL 38-entity exposure network -- 2008 STRESS, DENSE blocks on hardware",
        "n": spec.n,
        "backend": args.backend,
        "partition_source": str(args.partition),
        "partition_content_hash": partition_record.get("content_hash"),
        "cluster_sizes": [b.size for b in blocks],
        "labels": labels.tolist(),
        "stress_calibration": calib.to_dict(),
        "loadability": loadability_report(base_spec, spec),
        "noise_floor_assumed": QPU_NOISE_FLOOR,
        "pd_inspection": pd_inspection(spec, clusters),
        "shot_plan": shot_plan(args.shots, args.max_shots_per_job),
        "per_cluster": [
            {
                "cluster": idx,
                "members": list(b.qubits),
                "qubits": b.size,
                "entanglers": len(b.edges),
                "approx_two_qubit_gates_pretranspile": 2 * len(b.edges),
                "entanglement_depth_pretranspile": b.entanglement_depth,
                "target_within_abs_corr": round(target_within_abs_corr(spec, list(b.qubits)), 4),
                "target_pd_mean": round(
                    float(np.asarray(spec.marginal_default_probs)[list(b.qubits)].mean()), 4
                ),
            }
            for idx, b in enumerate(blocks)
        ],
    }

    if not args.submit:
        print(json.dumps({
            "status": "dry-run",
            **plan,
            "note": ("STRESSED marginals (mean PD ~15%) all clear the ~2.7% noise floor, so "
                     "marginals should now be recoverable (unlike the baseline faithful run). "
                     "Dense blocks still past the coherence boundary -- correlation recovery is "
                     "what this measures. Add --submit to spend metered QPU."),
            "next_command": (
                "uv run --extra quantum python "
                "scripts/run_real_cluster_mixture_stress_hardware.py --submit"
            ),
        }, indent=2))
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    # Persist the resolved plan + stressed spec up front for resumability.
    (args.output_dir / "plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
    spec.save_json(args.output_dir / "stressed_spec.json")
    live_jobs: list[dict] = []
    live_path = args.output_dir / "job_ids_live.json"

    cluster_samples = []
    hw_reports = []
    p_stressed = np.asarray(spec.marginal_default_probs, dtype=float)
    for idx, block in enumerate(blocks):
        result = run_block(
            block,
            shots=args.shots,
            backend_name=args.backend,
            optimization_level=args.optimization_level,
        )
        members = list(block.qubits)
        # Persist job id immediately for resumability before doing any post-processing.
        live_jobs.append({
            "cluster": idx,
            "job_id": result.job_id,
            "backend": result.backend_name,
            "shots": result.shots,
            "shot_batches": list(result.shot_batches),
        })
        live_path.write_text(json.dumps({
            "run": "200k-shot STRESSED real-network 3-cluster mixture on hardware",
            "backend": args.backend,
            "shots_per_cluster": args.shots,
            "jobs": live_jobs,
            "note": "job ids persisted incrementally for resumability",
        }, indent=2), encoding="utf-8")

        npz_path = args.output_dir / f"cluster{idx}_{result.backend_name}_{result.job_id}.npz"
        np.savez_compressed(npz_path, samples=result.samples, members=np.asarray(members))
        cluster_samples.append(
            cluster_samples_from_bitstrings(
                members, result.samples, source=f"hardware:{result.backend_name}"
            )
        )
        observed = result.samples.mean(axis=0)
        target = p_stressed[members]
        hw_reports.append({
            "cluster": idx,
            "members": members,
            "backend": result.backend_name,
            "job_id": result.job_id,
            "shots": result.shots,
            "shot_batches": list(result.shot_batches),
            "n_jobs": len(result.shot_batches),
            "circuit_depth": result.circuit_depth,
            "two_qubit_gates": result.two_qubit_gates,
            "circuit_operations": result.circuit_operations,
            "marginal_rmse_vs_target": round(
                float(np.sqrt(np.mean((observed - target) ** 2))), 5
            ),
            "observed_marginals": [round(float(x), 5) for x in observed],
            "target_marginals": [round(float(x), 5) for x in target],
            "target_within_abs_corr": round(target_within_abs_corr(spec, members), 4),
            "observed_within_abs_corr": round(within_abs_corr(result.samples), 4),
            "samples_file": npz_path.name,
        })
        print(f"[cluster {idx}] job {result.job_id} on {result.backend_name}: "
              f"depth {result.circuit_depth}, 2q gates {result.two_qubit_gates}, "
              f"{len(result.shot_batches)} batch(es) -> {result.samples.shape[0]} shots, "
              f"marg RMSE {hw_reports[-1]['marginal_rmse_vs_target']}, "
              f"within-corr {hw_reports[-1]['observed_within_abs_corr']} "
              f"(target {hw_reports[-1]['target_within_abs_corr']})")

    # Reconcile the measured per-cluster samples into the global joint and solve the cascade
    # on the real network's OWN exposure graph (no synthetic overlay), under stress.
    n_samples = min(c.n_samples for c in cluster_samples)
    target_cross = cross_cluster_corr_target(spec, labels)
    rec = CommonShockReconciler(spec.n, labels).fit_reconcile(
        cluster_samples, target_cross, n_samples, seed=args.seed + 2
    )
    independent = independent_global_samples(cluster_samples, spec.n)

    ref_gen = GaussianCopulaGenerator()
    ref_gen.fit(spec)
    reference = ref_gen.sample(min(args.ref_samples, max(n_samples, 1000)), seed=args.seed + 3)

    diag = reconciliation_diagnostics(
        reference, rec.samples, independent, labels, spec.n,
        cascade_spec=spec, cascade_max_eval=args.cascade_max_eval,
    )

    # P(severe) over the stressed cascade, matching the sim-preview report structure.
    from systemic_risk.mixture.pipeline import cascade_loss_cvar
    threshold = int(np.ceil(args.severe_fraction * spec.n))

    def p_severe(samples) -> dict:
        _, counts = cascade_loss_cvar(samples, spec, alpha=0.95, max_eval=args.cascade_max_eval)
        return {
            "p_severe": round(float(np.mean(counts >= threshold)), 5),
            "mean_cascade_count": round(float(counts.mean()), 4),
        }

    cascade_severe = {
        "severe_threshold_count": threshold,
        "severe_fraction": args.severe_fraction,
        "reference": p_severe(reference),
        "reconciled": p_severe(rec.samples),
        "independent": p_severe(independent),
    }

    report = {
        "status": "run",
        **plan,
        "reference": "full-network Gaussian-copula joint fit to the STRESSED target correlation",
        "n_samples": n_samples,
        "hardware_per_cluster": hw_reports,
        "fitted_beta": round(rec.beta, 5),
        "target_cross_corr": round(rec.target_cross_corr, 5),
        "achieved_cross_corr": round(rec.achieved_cross_corr, 5),
        "n_reconciled_samples": n_samples,
        "cascade_severe": cascade_severe,
        "diagnostics": diag,
    }
    report_path = args.output_dir / "real_stress_hardware_run_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    np.savez_compressed(
        args.output_dir / "reconciled_global_stress.npz",
        reconciled=rec.samples,
        independent=independent,
        reference=reference,
        labels=labels,
    )
    print(json.dumps(report, indent=2))
    print(f"\nSaved report -> {report_path}")


if __name__ == "__main__":
    main()
