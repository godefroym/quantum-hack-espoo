"""Run the 3-cluster classical+quantum MIXTURE over the REAL exposure network on IBM hardware.

This is the REAL-NETWORK twin of ``scripts/run_cluster_mixture_hardware.py``. Instead of the
synthetic planted-cluster system (``make_clustered_system`` + ``discover_clusters``), it consumes:

  * the real 38-entity exposure network spec
    (``systemic_risk.data_network.build_network_spec().to_system_spec()``), and
  * the persisted real-network community partition
    (``outputs/data_network/real_network_partition.json`` -- k=3, sizes 14/14/10),

then runs the SAME shape of experiment: one QPU job per cluster, each cluster's fitted dense
(RY + CRY) block circuit submitted to a backend, the measured bitstrings reconciled locally into
the global joint via the common-shock reconciler, and that joint fed through the cascade. All the
block-building, reconciliation, and cascade machinery is reused from the repo unchanged.

HONESTY CAVEATS baked in (see memory ``real-network-not-hardware-loadable``):

  * Per-cluster marginal default probabilities are tiny (~1e-5..1.4e-2) -- at/below the ~2.7%
    QPU readout+decoherence noise floor. By DEFAULT this script loads them FAITHFULLY (no
    rescaling), so the marginals are expected to be unrecoverable; the run quantifies that wall.
    ``--inflate-marginals P`` optionally floors every per-cluster PD to ``P`` (e.g. 0.05, above
    the noise floor) purely to make the *correlation* structure observable -- this is recorded
    explicitly in the report under ``marginal_handling`` so it is never silent.
  * These are DENSE blocks (every within-cluster edge above ``--edge-threshold``), past the
    documented coherence boundary, so heavy decoherence is expected and is part of what is
    being measured (observed vs target within-cluster correlation).

The REFERENCE ground truth is the full-network Gaussian-copula joint fit to the real spec's own
target correlation (NOT a planted sampler -- the real spec carries no planting metadata). The
cascade runs on the real spec's OWN exposure graph (no synthetic overlay).

Always dry-run first (omit ``--submit``) to see per-cluster gate/qubit/entangler budget, the
shot-batching plan for the requested shot count, and the PD inspection. With ``--submit`` it
spends metered QPU time: one logical experiment per cluster, shots auto-batched to the backend's
per-job cap and aggregated.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from systemic_risk.data_network import build_network_spec
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
DEFAULT_NOISE_FLOOR = 0.027  # ~2.7% effective QPU readout+decoherence floor (see memory).


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--partition", type=Path, default=DEFAULT_PARTITION)
    parser.add_argument("--edge-threshold", type=float, default=0.02)
    parser.add_argument("--calib-iters", type=int, default=20)
    parser.add_argument("--backend", default="ibm_boston")
    parser.add_argument("--shots", type=int, default=1_000_000,
                        help="Total shots per cluster (auto-batched to backend cap).")
    parser.add_argument("--max-shots-per-job", type=int, default=None,
                        help="Override the backend's per-job shot cap for batch planning "
                             "(dry-run only; live runs read the real cap from the backend).")
    parser.add_argument("--optimization-level", type=int, choices=range(4), default=3)
    parser.add_argument(
        "--inflate-marginals", type=float, default=None,
        help="Floor every per-cluster PD to this value (e.g. 0.05) to lift it above the noise "
             "floor so correlation is observable. Recorded explicitly. Default: faithful (None).",
    )
    parser.add_argument("--submit", action="store_true", help="Submit the metered IBM jobs.")
    parser.add_argument("--ref-samples", type=int, default=200_000,
                        help="Gaussian-copula reference sample count for diagnostics.")
    parser.add_argument("--cascade-max-eval", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "outputs" / "real_cluster_mixture_hw"
    )
    return parser.parse_args()


def load_partition(path: Path) -> tuple[list[list[int]], np.ndarray, dict]:
    """Load the persisted real-network partition: cluster member-index lists + global labels."""
    record = json.loads(path.read_text(encoding="utf-8"))
    clusters = [sorted(int(m) for m in c["member_indices"]) for c in record["clusters"]]
    labels = np.asarray(record["labels"], dtype=int)
    return clusters, labels, record


def maybe_inflate(spec: SystemSpec, floor: float | None) -> tuple[SystemSpec, dict]:
    """Optionally floor marginal PDs above the noise floor; return (spec, handling-record)."""
    p = np.asarray(spec.marginal_default_probs, dtype=float)
    if floor is None:
        return spec, {"mode": "faithful", "rescaled": False,
                      "note": "real PDs loaded unchanged; expected below QPU noise floor"}
    inflated = np.maximum(p, float(floor))
    new_spec = SystemSpec(
        node_names=spec.node_names,
        node_types=spec.node_types,
        exposure_matrix=spec.exposure_matrix,
        capital_buffers=spec.capital_buffers,
        marginal_default_probs=inflated,
        target_pairwise_corr=spec.target_pairwise_corr,
        clusters=spec.clusters,
        metadata={**spec.metadata, "marginal_inflation_floor": float(floor)},
    )
    return new_spec, {
        "mode": "inflated", "rescaled": True, "floor": float(floor),
        "n_floored": int(np.sum(p < floor)),
        "note": ("per-cluster PDs floored to lift them above the ~2.7% noise floor so "
                 "WITHIN-cluster correlation is observable; marginals no longer match the "
                 "real network -- correlation comparison only"),
    }


def build_block(spec: SystemSpec, members, *, threshold: float, calib_iters: int):
    """Fitted dense block circuit for one cluster (all within-cluster edges above threshold).

    Identical construction to ``run_cluster_mixture_hardware.build_block`` -- reused verbatim so
    the real-network run is the same circuit shape as the synthetic one.
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


def main() -> None:
    args = parse_args()
    base_spec = build_network_spec().to_system_spec()
    spec, marginal_handling = maybe_inflate(base_spec, args.inflate_marginals)
    clusters, labels, partition_record = load_partition(args.partition)

    if labels.shape[0] != spec.n:
        raise ValueError(f"partition labels ({labels.shape[0]}) != spec.n ({spec.n})")

    blocks = [
        build_block(spec, members, threshold=args.edge_threshold, calib_iters=args.calib_iters)
        for members in clusters
    ]

    p_real = np.asarray(base_spec.marginal_default_probs, dtype=float)
    pd_inspection = []
    for idx, members in enumerate(clusters):
        pm = p_real[members]
        pd_inspection.append({
            "cluster": idx,
            "members": members,
            "pd_min": round(float(pm.min()), 6),
            "pd_max": round(float(pm.max()), 6),
            "pd_mean": round(float(pm.mean()), 6),
            "n_below_noise_floor": int(np.sum(pm < DEFAULT_NOISE_FLOOR)),
            "all_below_noise_floor": bool(np.all(pm < DEFAULT_NOISE_FLOOR)),
        })

    plan = {
        "system": "REAL 38-entity exposure network, DENSE blocks on hardware",
        "n": spec.n,
        "backend": args.backend,
        "partition_source": str(args.partition),
        "partition_content_hash": partition_record.get("content_hash"),
        "cluster_sizes": [b.size for b in blocks],
        "labels": labels.tolist(),
        "marginal_handling": marginal_handling,
        "noise_floor_assumed": DEFAULT_NOISE_FLOOR,
        "pd_inspection": pd_inspection,
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
            }
            for idx, b in enumerate(blocks)
        ],
    }

    if not args.submit:
        print(json.dumps({
            "status": "dry-run",
            **plan,
            "note": ("dense blocks past the coherence boundary; expect heavy decoherence. "
                     "Real PDs are below the noise floor unless --inflate-marginals is set. "
                     "Add --submit to spend metered QPU."),
            "next_command": (
                "uv run --extra quantum python scripts/run_real_cluster_mixture_hardware.py "
                "--submit"
            ),
        }, indent=2))
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)

    cluster_samples = []
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
        cluster_samples.append(
            cluster_samples_from_bitstrings(
                members, result.samples, source=f"hardware:{result.backend_name}"
            )
        )
        observed = result.samples.mean(axis=0)
        target = np.asarray(spec.marginal_default_probs)[members]
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
              f"{len(result.shot_batches)} batch(es) -> {result.samples.shape[0]} shots")

    # Reconcile the measured per-cluster samples into the global joint and solve the cascade
    # on the real network's OWN exposure graph (no synthetic overlay).
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

    report = {
        "status": "run",
        **plan,
        "reference": "full-network Gaussian-copula joint fit to real target correlation",
        "hardware_per_cluster": hw_reports,
        "fitted_beta": round(rec.beta, 5),
        "target_cross_corr": round(rec.target_cross_corr, 5),
        "achieved_cross_corr": round(rec.achieved_cross_corr, 5),
        "n_reconciled_samples": n_samples,
        "diagnostics": diag,
    }
    report_path = args.output_dir / "real_hardware_run_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    # Save the reconciled global joint for later analysis.
    np.savez_compressed(
        args.output_dir / "reconciled_global.npz",
        reconciled=rec.samples,
        independent=independent,
        reference=reference,
        labels=labels,
    )
    print(json.dumps(report, indent=2))
    print(f"\nSaved report -> {report_path}")


if __name__ == "__main__":
    main()
