"""Submit the trained qGAN circuit to a real IBM QPU and validate it.

The qGAN is trained in exact simulation (``scripts/train_qgan.py --mode hw``);
training on hardware is infeasible (each finite-difference gradient step is many
thousands of circuit evaluations). The legitimate hardware step is to take the
*final* trained circuit and run it on a QPU, comparing hardware-sampled moments
to the exact statevector.

The saved hw circuit is built to be loadable: inflated marginals (above the
~2-3% readout floor) on a max-degree-2 chain that pins to a low-error physical
line with zero routing SWAPs (dynamical decoupling + measurement twirling on).

Dry-run (no cloud job) by default; pass --submit to actually run.

    uv run --extra quantum python scripts/run_qgan_hardware.py            # dry-run
    uv run --extra quantum python scripts/run_qgan_hardware.py --submit   # real QPU
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from systemic_risk.generators.quantum import ansatz as A
from systemic_risk.generators.quantum.statevector import StateVector
from systemic_risk.generators.quantum.ibm_runtime import (
    DEFAULT_HARDWARE_SHOTS,
    best_qubit_line,
    run_block,
)


def load_circuit(path: Path) -> A.EntangledCircuit:
    d = np.load(path)
    return A.EntangledCircuit(
        qubits=d["qubits"].tolist(),
        ry=d["ry"],
        edges=[tuple(map(int, e)) for e in d["edges"]],
        cry=d["cry"],
        target_p=d["target_p"],
        target_cov=d["target_cov"],
    )


def ideal_moments(circuit: A.EntangledCircuit):
    sv = StateVector(circuit.size)
    for i, th in enumerate(circuit.ry):
        sv.ry(i, float(th))
    for e, (c, t) in enumerate(circuit.edges):
        sv.cry(int(c), int(t), float(circuit.cry[e]))
    return sv.marginals(), sv.pairwise_joint()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--circuit", type=Path, default=Path("outputs/qgan_hw_circuit.npz"))
    p.add_argument("--backend", default="ibm_boston")
    p.add_argument("--shots", type=int, default=DEFAULT_HARDWARE_SHOTS)
    p.add_argument("--optimization-level", type=int, choices=range(4), default=3)
    p.add_argument("--submit", action="store_true")
    p.add_argument("--output-dir", type=Path, default=Path("outputs/ibm_quantum"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    circuit = load_circuit(args.circuit)
    n = circuit.size
    edges = circuit.edges

    ideal_marg, ideal_joint = ideal_moments(circuit)
    tgt_marg = circuit.target_p
    tgt_joint = circuit.target_cov + np.outer(tgt_marg, tgt_marg)

    from qiskit_ibm_runtime import QiskitRuntimeService

    svc = QiskitRuntimeService(
        token=os.environ.get("IBM_QUANTUM_TOKEN"),
        channel=os.environ.get("IBM_QUANTUM_CHANNEL", "ibm_quantum_platform"),
        instance=os.environ.get("IBM_QUANTUM_INSTANCE"),
    )
    backend = svc.backend(args.backend)
    line, ro = best_qubit_line(backend, n)

    # statistical-resolvability budget
    shot_sigma = np.sqrt(tgt_marg * (1 - tgt_marg) / args.shots)
    line_floor = float(np.mean(ro[line]))
    resolvable = [bool(tm > line_floor + 2 * ss) for tm, ss in zip(tgt_marg, shot_sigma)]
    exp_codefaults = np.array([args.shots * ideal_joint[i, j] for i, j in edges])
    stats = {
        "shots": args.shots,
        "line_readout_floor_pct": round(line_floor * 100, 3),
        "ideal_marginals_pct": [round(x * 100, 2) for x in ideal_marg.tolist()],
        "marginals_resolvable_above_floor": f"{sum(resolvable)}/{n}",
        "expected_codefaults_per_chain_pair": [round(x, 1) for x in exp_codefaults.tolist()],
        "pairs_with_>=10_codefaults": int((exp_codefaults >= 10).sum()),
    }

    if not args.submit:
        print(json.dumps({
            "status": "dry-run (no cloud job)",
            "backend": args.backend,
            "n": n,
            "chain_edges_local": edges,
            "physical_line": line,
            "expected_two_qubit_gates": 2 * len(edges),
            **stats,
        }, indent=2))
        return

    result = run_block(
        circuit, shots=args.shots, backend_name=args.backend,
        optimization_level=args.optimization_level, initial_layout=line,
        dynamical_decoupling=True, measure_twirling=True,
    )
    samples = result.samples
    obs_marg = samples.mean(axis=0)
    obs_joint = (samples.T @ samples) / len(samples)

    edge_mask = np.zeros((n, n), bool)
    for i, j in edges:
        edge_mask[i, j] = edge_mask[j, i] = True

    report = {
        "source": "trained qGAN circuit (hw mode)",
        "backend": result.backend_name,
        "job_id": result.job_id,
        "physical_line": line,
        "two_qubit_gates": result.two_qubit_gates,
        "circuit_depth": result.circuit_depth,
        "swaps_added": result.two_qubit_gates - 2 * len(edges),
        **stats,
        "hardware_marginals_pct": [round(x * 100, 2) for x in obs_marg.tolist()],
        "marginal_rmse_vs_ideal": float(np.sqrt(np.mean((obs_marg - ideal_marg) ** 2))),
        "marginal_rmse_vs_target": float(np.sqrt(np.mean((obs_marg - tgt_marg) ** 2))),
        "joint_rmse_chain_vs_ideal": float(
            np.sqrt(np.mean((obs_joint[edge_mask] - ideal_joint[edge_mask]) ** 2))
        ),
        "hardware_codefaults_on_chain": [int(obs_joint[i, j] * result.shots) for i, j in edges],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"qgan_hardware_{result.backend_name}_{result.job_id}"
    np.savez_compressed(args.output_dir / f"{stem}_samples.npz", samples=samples)
    (args.output_dir / f"{stem}_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
