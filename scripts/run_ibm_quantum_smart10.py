"""Real 10-bank subset on an IBM QPU with engineered transpilation -- the honest rare-event run.

Takes the 10 highest-PD banks from the real G-SIB network and loads them with every NISQ trick
that legitimately helps:

* **best physical qubits** -- a connected 10-qubit line on the device chosen to minimise readout
  + CZ error (the real readout floor is ~0.3%, not the ~2.7% seen at depth-119 / auto-layout);
* **zero-SWAP layout** -- banks are ordered so the most-correlated pairs are adjacent, entangled as
  a chain, and pinned to that physical line via ``initial_layout`` (router inserts no SWAPs);
* **error suppression** -- dynamical decoupling + measurement twirling;
* **exact validation** -- n=10 => the 2^10 statevector is the exact ground truth, and the chain is
  calibrated against it.

The point is NOT a pretty result: real PDs are ~0.3% (only 2 banks reach 1.4%), so even at high
shot counts a co-default pair occurs only a handful of times. The script quantifies this -- expected
events vs shot noise -- to show that direct *sampling* is statistically blind to rare-event risk
structure, which is the motivation for the QAE *calculation* surface.
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
from systemic_risk.generators.quantum.ibm_runtime import run_block
from systemic_risk.generators.quantum_born_machine import (
    _build_statevector,
    _statevector_block_moments,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", default="ibm_boston")
    parser.add_argument("--shots", type=int, default=32768)
    parser.add_argument("--optimization-level", type=int, choices=range(4), default=3)
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "ibm_quantum")
    return parser.parse_args()


def correlation_path(dep: np.ndarray) -> list[int]:
    """Greedy Hamiltonian-ish path: most-correlated banks placed adjacent (=> chain entanglers)."""
    n = dep.shape[0]
    start = int(dep.sum(axis=1).argmax())
    path = [start]
    used = {start}
    while len(path) < n:
        last = path[-1]
        cand = [(dep[last, k], k) for k in range(n) if k not in used]
        _, nxt = max(cand)
        path.append(nxt)
        used.add(nxt)
    return path


def best_line(backend, length: int) -> tuple[list[int], np.ndarray]:
    """Lowest-error connected qubit line on the device (readout + CZ error)."""
    tgt = backend.target
    nq = backend.num_qubits
    ro = np.array([tgt["measure"][(q,)].error for q in range(nq)])
    twoq = tgt["cz"] if "cz" in tgt else tgt["ecr"]
    edges = {p: pr.error for p, pr in twoq.items() if pr is not None and pr.error is not None}
    adj: dict[int, list[int]] = {}
    for (i, j) in edges:
        adj.setdefault(i, []).append(j)
    best, best_cost = None, 1e9
    for s in np.argsort(ro)[:40]:
        path, used, cost = [int(s)], {int(s)}, float(ro[s])
        for _ in range(length - 1):
            cur = path[-1]
            cand = [
                (edges.get((cur, n), edges.get((n, cur), 9.0)) + ro[n], n)
                for n in adj.get(cur, [])
                if n not in used
            ]
            if not cand:
                break
            c, nxt = min(cand)
            path.append(nxt)
            used.add(nxt)
            cost += c
        if len(path) == length and cost < best_cost:
            best, best_cost = path, cost
    return best, ro


def build_circuit_for(spec, banks: list[int]):
    """Order the chosen banks by correlation, build + calibrate the chain on local indices."""
    p_all = np.clip(spec.marginal_default_probs, 1e-6, 1.0 - 1e-6)
    cov_all = A.target_covariance(spec)
    dep = np.abs(spec.dependency_matrix())
    sub_dep = dep[np.ix_(banks, banks)]
    order = correlation_path(sub_dep)  # local order within the 10
    ordered = [banks[o] for o in order]
    p = p_all[ordered]
    cov = cov_all[np.ix_(ordered, ordered)]
    edges = [(k, k + 1) for k in range(len(ordered) - 1)]  # chain
    circ = A.EntangledCircuit(
        qubits=list(range(len(ordered))),
        ry=A.marginal_angles(p),
        edges=edges,
        cry=np.array([A.cry_angle(p[i], p[j], cov[i, j]) for i, j in edges]),
        target_p=p,
        target_cov=cov,
    )
    circ = A.calibrate_block(circ, _statevector_block_moments, iterations=60)
    return circ, ordered, edges


def main() -> None:
    args = parse_args()
    spec = build_system_spec()
    p_all = spec.marginal_default_probs
    banks = sorted(np.argsort(p_all)[-10:].tolist())  # 10 highest-PD banks
    circ, ordered, edges = build_circuit_for(spec, banks)
    n = len(ordered)

    # exact ground truth from the calibrated chain
    state = _build_statevector(circ.ry, circ.edges, circ.cry)
    ideal_marg, ideal_joint = state.marginals(), state.pairwise_joint()

    targets = targets_from_spec(spec)
    tgt_marg = targets.marginals[ordered]
    tgt_joint = targets.pairwise_joint[np.ix_(ordered, ordered)]

    from qiskit_ibm_runtime import QiskitRuntimeService
    import os
    svc = QiskitRuntimeService(
        token=os.environ.get("IBM_QUANTUM_TOKEN"),
        channel=os.environ.get("IBM_QUANTUM_CHANNEL", "ibm_quantum_platform"),
        instance=os.environ.get("IBM_QUANTUM_INSTANCE"),
    )
    backend = svc.backend(args.backend)
    line, ro = best_line(backend, n)

    # statistical-resolvability budget (independent of hardware noise)
    shot_sigma = np.sqrt(tgt_marg * (1 - tgt_marg) / args.shots)
    exp_defaults = args.shots * tgt_marg
    exp_codefaults = np.array([args.shots * tgt_joint[i, j] for i, j in edges])
    line_floor = float(np.mean(ro[line]))
    resolvable = [bool(tm > line_floor + 2 * ss) for tm, ss in zip(tgt_marg, shot_sigma)]
    stats = {
        "shots": args.shots,
        "line_readout_floor_pct": round(line_floor * 100, 3),
        "bank_pd_pct": [round(x * 100, 3) for x in tgt_marg.tolist()],
        "expected_defaults_per_bank": [round(x, 1) for x in exp_defaults.tolist()],
        "marginals_resolvable_above_floor": int(sum(resolvable)),
        "expected_codefaults_per_entangled_pair": [round(x, 1) for x in exp_codefaults.tolist()],
        "pairs_with_>=10_codefaults": int((exp_codefaults >= 10).sum()),
    }

    if not args.submit:
        print(json.dumps({"status": "dry-run", "n": n, "banks_global_idx": ordered,
                          "physical_line": line, "chain_edges": edges, **stats}, indent=2))
        return

    result = run_block(
        circ, shots=args.shots, backend_name=args.backend,
        optimization_level=args.optimization_level, initial_layout=line,
        dynamical_decoupling=True, measure_twirling=True,
    )
    observed = empirical_moments(result.samples)
    edge_mask = np.zeros((n, n), bool)
    for i, j in edges:
        edge_mask[i, j] = edge_mask[j, i] = True

    report = {
        "network": "real G-SIB, 10 highest-PD banks",
        "backend": result.backend_name,
        "job_id": result.job_id,
        "physical_line": line,
        "banks_global_idx": ordered,
        "two_qubit_gates": result.two_qubit_gates,
        "circuit_depth": result.circuit_depth,
        "swaps_added": result.two_qubit_gates - 2 * len(edges),
        **stats,
        "ideal_marginals_pct": [round(x * 100, 3) for x in ideal_marg.tolist()],
        "hardware_marginals_pct": [round(x * 100, 3) for x in observed.marginals.tolist()],
        "marginal_rmse_vs_ideal": float(np.sqrt(np.mean((observed.marginals - ideal_marg) ** 2))),
        "marginal_rmse_vs_target": float(np.sqrt(np.mean((observed.marginals - tgt_marg) ** 2))),
        "joint_rmse_chain_vs_ideal": float(
            np.sqrt(np.mean((observed.pairwise_joint[edge_mask] - ideal_joint[edge_mask]) ** 2))
        ),
        "hardware_codefaults_on_chain": [int(observed.pairwise_joint[i, j] * result.shots) for i, j in edges],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"hardware_smart10_{result.backend_name}_{result.job_id}"
    np.savez_compressed(args.output_dir / f"{stem}_samples.npz", samples=result.samples)
    (args.output_dir / f"{stem}_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
