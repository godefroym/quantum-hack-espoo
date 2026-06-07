"""Export the 48-entity 2008-STRESS hardware run into the web demo's hardware.json.

The newer run (``outputs/real_cluster_mixture_stress_hw/``) is a 4-cluster *mixture*: each
community was run as its own circuit on IBM ``ibm_boston`` and the per-cluster shots were
reconciled into one 200k-shot, 48-entity global sample set. There is **no exact 48-qubit
simulator** (2^48 is infeasible), so the demo's third series ("ideal") is mapped to the run's
**full-network Gaussian-copula reference** instead — relabelled accordingly in the frontend.

It reuses the derivation helpers (`compute_posteriors`, `compute_graph`, `compute_tail`) from
``export_results_data.py`` so the JSON shape stays identical to what the results page reads.

Usage:
    uv run python scripts/export_stress_results_data.py
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # import sibling exporter helpers

from export_results_data import compute_graph, compute_posteriors, compute_tail  # noqa: E402

RUN_DIR = ROOT / "outputs" / "real_cluster_mixture_stress_hw"
OUT_DIR = ROOT / "frontend" / "public" / "results"

_REGION_NAME = {
    "US": "North America",
    "CA": "North America",
    "UK": "UK & Europe",
    "EU": "UK & Europe",
    "JP": "Japan",
    "LATAM": "Energy & LatAm",
    "LatAm": "Energy & LatAm",
}


def load_institutions() -> list[dict]:
    """Institution labels from the run's own qubit legend (authoritative, 48 entities)."""
    legend = json.loads((RUN_DIR / "qubit_legend.json").read_text())
    legend = sorted(legend, key=lambda r: int(r["qubit"]))  # ensure qubit order
    out = []
    for row in legend:
        region = row.get("region", "")
        out.append(
            {
                "ticker": row.get("ticker", f"Q{row['qubit'] + 1}"),
                "name": row.get("name", f"Institution {row['qubit'] + 1}"),
                "region": region,
                "regionName": _REGION_NAME.get(region, row.get("bloc", region or "Other")),
            }
        )
    return out


def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def _offdiag_rmse(a: np.ndarray, b: np.ndarray) -> float:
    mask = ~np.eye(a.shape[0], dtype=bool)
    return float(np.sqrt(np.mean((a[mask] - b[mask]) ** 2)))


def _aggregate_circuit(report: dict) -> dict:
    """Aggregate the 4 per-cluster circuits into single headline numbers."""
    per = report.get("per_cluster", [])
    hw = report.get("hardware_per_cluster", [])
    ops: dict[str, int] = {}
    for block in hw:
        for name, count in block.get("circuit_operations", {}).items():
            ops[name] = ops.get(name, 0) + int(count)
    return {
        "entanglers": int(sum(b.get("entanglers", 0) for b in per)),
        "entanglement_depth": int(
            max((b.get("entanglement_depth_pretranspile", 0) for b in per), default=0)
        ),
        "circuit_depth": int(max((b.get("circuit_depth", 0) for b in hw), default=0)),
        "two_qubit_gates": int(sum(b.get("two_qubit_gates", 0) for b in hw)),
        "circuit_operations": ops,
        "job_ids": [b.get("job_id", "") for b in hw],
    }


def main() -> None:
    from systemic_risk.spec import SystemSpec

    report = json.loads((RUN_DIR / "real_stress_hardware_run_report.json").read_text())
    spec = SystemSpec.from_dict(json.loads((RUN_DIR / "stressed_spec.json").read_text()))

    with np.load(RUN_DIR / "reconciled_global_stress.npz") as z:
        samples = z["reconciled"].astype(int)        # (200000, 48) hardware-reconciled
        reference = z["reference"].astype(int)        # full-network Gaussian-copula reference
        labels = z["labels"]
    # Column order must match the global node order (and the qubit legend).
    if not np.array_equal(labels, np.asarray(report["labels"])):
        raise SystemExit("reconciled column order does not match report labels")

    shots, n = samples.shape
    target_marg = np.asarray(spec.marginal_default_probs, dtype=float)
    # stressed_spec stores the latent correlation, not a joint; derive the analytic target
    # pairwise co-default probabilities through the canonical helper.
    target_joint = spec.target_pairwise_joint_probs()
    if target_marg.shape[0] != n:
        raise SystemExit(f"spec has {target_marg.shape[0]} entities, samples have {n}")

    hw_marg = samples.mean(axis=0)
    hw_joint = (samples.T @ samples) / shots
    ref_marg = reference.mean(axis=0)
    ref_joint = (reference.T @ reference) / shots

    counts = samples.sum(axis=1)
    count_hist = np.bincount(counts, minlength=n + 1).astype(int).tolist()
    corr = np.nan_to_num(np.corrcoef(samples.T))

    # top joint-default scenarios (bitstrings)
    rows = [tuple(int(x) for x in r) for r in samples]
    uniq: dict[tuple, int] = {}
    for r in rows:
        uniq[r] = uniq.get(r, 0) + 1
    top = sorted(uniq.items(), key=lambda kv: kv[1], reverse=True)[:16]
    top_patterns = [
        {"indices": [i for i, b in enumerate(pat) if b], "count": c, "freq": c / shots}
        for pat, c in top
    ]

    circuit = _aggregate_circuit(report)
    out = {
        # provenance / circuit
        "backend": report.get("backend", "ibm_boston"),
        "job_id": " + ".join(j for j in circuit["job_ids"] if j) or "4-cluster mixture",
        "job_ids": circuit["job_ids"],
        "shots": int(shots),
        "n_qubits": int(n),
        "institutions": load_institutions(),
        "max_degree": None,  # dense within-cluster blocks; no degree cap
        "entanglers": circuit["entanglers"],
        "entanglement_depth": circuit["entanglement_depth"],
        "circuit_depth": circuit["circuit_depth"],
        "two_qubit_gates": circuit["two_qubit_gates"],
        "circuit_operations": circuit["circuit_operations"],
        "exact_ground_truth": False,  # 2^48 cannot be simulated exactly
        # fidelity (note: "ideal" = full-network Gaussian-copula reference, not a simulator)
        "marginal_rmse_vs_target": round(_rmse(hw_marg, target_marg), 5),
        "pairwise_joint_rmse_vs_target": round(_offdiag_rmse(hw_joint, target_joint), 5),
        "marginal_rmse_vs_ideal": round(_rmse(hw_marg, ref_marg), 5),
        "pairwise_joint_rmse_vs_ideal": round(_offdiag_rmse(hw_joint, ref_joint), 5),
        # marginals (per institution)
        "target_marginals": [round(float(x), 5) for x in target_marg],
        "hardware_marginals": [round(float(x), 5) for x in hw_marg],
        "ideal_marginals": [round(float(x), 5) for x in ref_marg],
        # derived distributions
        "default_count_hist": count_hist,
        "mean_defaults": float(counts.mean()),
        "expected_defaults_target": float(np.sum(target_marg)),
        "pairwise_corr": [[round(float(x), 4) for x in row] for row in corr],
        "pairwise_joint": [[round(float(x), 5) for x in row] for row in hw_joint],
        "top_patterns": top_patterns,
        "n_unique_patterns": len(uniq),
        "posteriors": compute_posteriors(samples),
        "graph": compute_graph(samples),
        "tail": compute_tail(samples),
        # stress-run context (extra; ignored by the current UI but useful provenance)
        "stress": {
            "scenario": report.get("stress_calibration", {}).get("scenario", "2008"),
            "n_clusters": len(report.get("per_cluster", [])),
            "target_cross_corr": report.get("target_cross_corr"),
            "achieved_cross_corr": report.get("achieved_cross_corr"),
            "cascade_severe": report.get("cascade_severe"),
            "ideal_series_meaning": "full-network Gaussian-copula reference (no exact 48-qubit simulator)",
        },
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "hardware.json").write_text(json.dumps(out))
    print(f"wrote {OUT_DIR / 'hardware.json'}")
    print(
        f"  backend={out['backend']} n={n} shots={shots} clusters={out['stress']['n_clusters']} "
        f"unique_patterns={out['n_unique_patterns']} mean_defaults={out['mean_defaults']:.2f}"
    )
    print(
        f"  marginal RMSE vs target={out['marginal_rmse_vs_target']} "
        f"vs reference={out['marginal_rmse_vs_ideal']}"
    )


if __name__ == "__main__":
    main()
