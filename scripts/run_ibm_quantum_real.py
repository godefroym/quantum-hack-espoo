"""Run the REAL 28-bank G-SIB network on an IBM QPU, sparsified to the NISQ-feasible boundary.

Unlike the banded toy in ``run_ibm_quantum_large.py``, this loads the actual interbank
network (dense exposures, equity-derived correlation up to 0.88). A faithful all-strong-edge
circuit is ~365 entanglers / ~2600 two-qubit gates -- fully decohered on today's hardware. So
we keep only the strongest ``--max-degree`` entanglers per bank (~33 edges) and report, with no
spin, *which* correlations survive (the kept entangler pairs) and which collapse (the ~90% of
strong pairs we had to drop, plus device noise).

Validation is against the analytic moment targets; ``n = 28`` is past the exact-statevector
limit, so there is no exact ground truth -- the targets are the honest reference.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from systemic_risk.data_network.assemble import build_system_spec
from systemic_risk.generators.moments import empirical_moments, targets_from_spec
from systemic_risk.generators.quantum import ansatz as A
from systemic_risk.generators.quantum.ibm_runtime import DEFAULT_HARDWARE_SHOTS, run_block


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-degree", type=int, default=3, help="Max entanglers per bank.")
    parser.add_argument("--edge-threshold", type=float, default=0.02)
    parser.add_argument("--backend", default="ibm_boston")
    parser.add_argument("--shots", type=int, default=DEFAULT_HARDWARE_SHOTS)
    parser.add_argument("--optimization-level", type=int, choices=range(4), default=3)
    parser.add_argument("--submit", action="store_true", help="Submit the metered IBM job.")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "ibm_quantum")
    return parser.parse_args()


def rmse(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    if not mask.any():
        return float("nan")
    return float(np.sqrt(np.mean((a[mask] - b[mask]) ** 2)))


def main() -> None:
    args = parse_args()
    spec = build_system_spec()
    n = spec.n
    p = np.clip(spec.marginal_default_probs, 1e-6, 1.0 - 1e-6)
    cov = A.target_covariance(spec)

    # All strong pairs vs the sparsified subset we can actually run.
    strong = A.dependency_edges(spec, threshold=args.edge_threshold, within_clusters_only=False)
    kept = A.dependency_edges(
        spec, threshold=args.edge_threshold, within_clusters_only=False, max_degree=args.max_degree
    )
    # One circuit spanning all n banks; disconnected banks are pure-RY (uncalibrated analytic seed:
    # n=28 is past the exact-statevector limit, so the calibration loop is unavailable here).
    block = A._block_circuit(list(range(n)), p, cov, kept)

    targets = targets_from_spec(spec)
    off = ~np.eye(n, dtype=bool)
    kept_mask = np.zeros((n, n), dtype=bool)
    for i, j in kept:
        kept_mask[i, j] = kept_mask[j, i] = True
    dropped_mask = off & ~kept_mask

    summary = {
        "network": "real 28-bank G-SIB",
        "n_banks": n,
        "strong_pairs_total": len(strong),
        "entanglers_kept": len(kept),
        "pairs_dropped": int(dropped_mask.sum() // 2),
        "fraction_strong_dropped": round(1 - len(kept) / max(len(strong), 1), 3),
        "max_degree": args.max_degree,
    }

    if not args.submit:
        print(json.dumps({"status": "dry-run", **summary,
                          "next_command": "uv run --extra quantum python scripts/run_ibm_quantum_real.py --submit"},
                         indent=2))
        return

    result = run_block(block, shots=args.shots, backend_name=args.backend,
                       optimization_level=args.optimization_level)
    observed = empirical_moments(result.samples)

    report = {
        **summary,
        "backend": result.backend_name,
        "job_id": result.job_id,
        "shots": result.shots,
        "circuit_depth": result.circuit_depth,
        "two_qubit_gates": result.two_qubit_gates,
        "circuit_operations": result.circuit_operations,
        "marginal_rmse_vs_target": rmse(observed.marginals, targets.marginals, np.ones(n, bool)),
        # The decomposition that tells the honest story:
        "joint_rmse_kept_edges": rmse(observed.pairwise_joint, targets.pairwise_joint, kept_mask),
        "joint_rmse_dropped_pairs": rmse(observed.pairwise_joint, targets.pairwise_joint, dropped_mask),
        "joint_rmse_all_pairs": rmse(observed.pairwise_joint, targets.pairwise_joint, off),
        "target_marginals": targets.marginals.tolist(),
        "hardware_marginals": observed.marginals.tolist(),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"hardware_real28_degree{args.max_degree}_{result.backend_name}_{result.job_id}"
    np.savez_compressed(args.output_dir / f"{stem}_samples.npz", samples=result.samples)
    (args.output_dir / f"{stem}_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
