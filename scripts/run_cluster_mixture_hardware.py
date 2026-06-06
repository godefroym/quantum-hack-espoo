"""Run the 3-cluster classical+quantum MIXTURE with each cluster on REAL IBM hardware.

This is the hardware leg of ``scripts/run_cluster_mixture.py``: instead of the exact
statevector standing in for each per-cluster device, every discovered cluster's fitted
(RY + CRY) block circuit is submitted to a QPU as its own job, and the measured bitstrings
are reconciled locally into the global joint -- the literal "distribute clusters onto
separate machines, reconcile IBM results locally and solve."

DENSE BLOCKS, NO SPARSIFICATION. Each ~20-qubit cluster keeps every within-cluster entangler
above ``--edge-threshold`` (~190 CRY / ~380 two-qubit gates). This is past the documented
coherence boundary for today's devices, so the hardware samples are expected to be heavily
decohered -- the run quantifies that collapse (observed within-cluster correlation vs target)
and feeds whatever survives through the same reconciler the simulated path uses.

Always dry-run first (omit ``--submit``) to see the per-cluster gate budget. With ``--submit``
it spends metered QPU time: one job per cluster.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from systemic_risk.data import ClusteredSystemConfig, make_clustered_system, reference_default_samples
from systemic_risk.generators.quantum import discover_clusters
from systemic_risk.generators.quantum import ansatz as A
from systemic_risk.generators.quantum.ibm_runtime import run_block
from systemic_risk.mixture import (
    CommonShockReconciler,
    attach_cluster_exposures,
    cluster_samples_from_bitstrings,
    cross_cluster_corr_target,
    independent_global_samples,
    reconciliation_diagnostics,
)
from systemic_risk.mixture.cluster_loader import _statevector_block_moments


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cluster-sizes", type=int, nargs="+", default=[20, 20, 20])
    parser.add_argument("--cross-coupling", type=float, default=0.1)
    parser.add_argument("--within-coupling", type=float, default=0.45)
    parser.add_argument("--marginal", type=float, default=0.05)
    parser.add_argument("--budget", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--edge-threshold", type=float, default=0.02)
    parser.add_argument("--calib-iters", type=int, default=20)
    parser.add_argument("--backend", default="ibm_boston")
    parser.add_argument("--shots", type=int, default=4096)
    parser.add_argument("--optimization-level", type=int, choices=range(4), default=3)
    parser.add_argument("--submit", action="store_true", help="Submit the metered IBM jobs.")
    parser.add_argument("--cascade-max-eval", type=int, default=4000)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "cluster_mixture_hw")
    return parser.parse_args()


def build_block(spec, members, *, threshold, calib_iters):
    """Fitted dense block circuit for one cluster (all within-cluster edges above threshold)."""
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
    """Mean |pairwise correlation| inside a cluster's samples (collapses toward 0 when decohered)."""
    if samples.shape[1] < 2:
        return float("nan")
    corr = np.corrcoef(samples, rowvar=False)
    iu = np.triu_indices(corr.shape[0], k=1)
    return float(np.nanmean(np.abs(corr[iu])))


def target_within_abs_corr(spec, members) -> float:
    dep = np.abs(spec.dependency_matrix())
    sub = dep[np.ix_(members, members)]
    iu = np.triu_indices(sub.shape[0], k=1)
    return float(np.mean(sub[iu]))


def main() -> None:
    args = parse_args()
    spec = make_clustered_system(
        ClusteredSystemConfig(
            cluster_sizes=tuple(args.cluster_sizes),
            cross_coupling=args.cross_coupling,
            within_coupling=args.within_coupling,
            marginal_default_prob=args.marginal,
            seed=args.seed,
        )
    )
    partition = discover_clusters(spec, max_cluster_size=args.budget)

    blocks = [
        build_block(spec, members, threshold=args.edge_threshold, calib_iters=args.calib_iters)
        for members in partition.clusters
    ]
    plan = {
        "system": "planted-cluster synthetic, DENSE blocks on hardware",
        "n": spec.n,
        "backend": args.backend,
        "shots": args.shots,
        "discovered_sizes": partition.sizes,
        "cut_fraction": round(partition.cut_fraction, 4),
        "per_cluster": [
            {
                "members": list(b.qubits),
                "qubits": b.size,
                "entanglers": len(b.edges),
                "approx_two_qubit_gates_pretranspile": 2 * len(b.edges),
                "target_within_abs_corr": round(target_within_abs_corr(spec, list(b.qubits)), 4),
            }
            for b in blocks
        ],
    }

    if not args.submit:
        print(json.dumps({
            "status": "dry-run",
            **plan,
            "note": "dense blocks; expect heavy decoherence. add --submit to spend metered QPU.",
            "next_command": "uv run --extra quantum python scripts/run_cluster_mixture_hardware.py --submit",
        }, indent=2))
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)

    clusters = []
    hw_reports = []
    for idx, block in enumerate(blocks):
        result = run_block(
            block,
            shots=args.shots,
            backend_name=args.backend,
            optimization_level=args.optimization_level,
        )
        members = list(block.qubits)
        npz_path = args.output_dir / f"cluster{idx}_{result.backend_name}_{result.job_id}.npz"
        np.savez_compressed(npz_path, samples=result.samples, members=np.asarray(members))
        clusters.append(
            cluster_samples_from_bitstrings(members, result.samples, source=f"hardware:{result.backend_name}")
        )
        observed = result.samples.mean(axis=0)
        target = np.asarray(spec.marginal_default_probs)[members]
        hw_reports.append({
            "cluster": idx,
            "members": members,
            "backend": result.backend_name,
            "job_id": result.job_id,
            "circuit_depth": result.circuit_depth,
            "two_qubit_gates": result.two_qubit_gates,
            "marginal_rmse_vs_target": round(float(np.sqrt(np.mean((observed - target) ** 2))), 5),
            "target_within_abs_corr": round(target_within_abs_corr(spec, members), 4),
            "observed_within_abs_corr": round(within_abs_corr(result.samples), 4),
            "samples_file": npz_path.name,
        })

    # Reconcile the measured per-cluster samples into the global joint and solve the cascade.
    n_samples = min(c.n_samples for c in clusters)
    target = cross_cluster_corr_target(spec, partition.labels)
    rec = CommonShockReconciler(spec.n, partition.labels).fit_reconcile(
        clusters, target, n_samples, seed=args.seed + 2
    )
    independent = independent_global_samples(clusters, spec.n)
    reference = reference_default_samples(spec, n_samples, seed=args.seed + 3)
    cascade_spec = attach_cluster_exposures(spec, partition.labels, seed=args.seed)
    diag = reconciliation_diagnostics(
        reference, rec.samples, independent, partition.labels, spec.n,
        cascade_spec=cascade_spec, cascade_max_eval=args.cascade_max_eval,
    )

    report = {
        "status": "run",
        **plan,
        "hardware_per_cluster": hw_reports,
        "fitted_beta": round(rec.beta, 5),
        "target_cross_corr": round(rec.target_cross_corr, 5),
        "achieved_cross_corr": round(rec.achieved_cross_corr, 5),
        "n_samples": n_samples,
        "diagnostics": diag,
    }
    (args.output_dir / "hardware_run_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
