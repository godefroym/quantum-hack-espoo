"""Run a small fitted systemic-risk Born machine on an IBM Quantum QPU.

Authentication is read from a saved Qiskit Runtime account or the
``IBM_QUANTUM_TOKEN`` / ``IBM_QUANTUM_INSTANCE`` environment variables.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from systemic_risk.generators import EntangledBornMachineGenerator
from systemic_risk.generators.moments import empirical_moments
from systemic_risk.generators.quantum.ibm_runtime import run_block
from systemic_risk.spec import SystemSpec


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", help="IBM backend name; default selects the least busy QPU")
    parser.add_argument("--shots", type=int, default=4096)
    parser.add_argument("--optimization-level", type=int, choices=range(4), default=1)
    parser.add_argument(
        "--submit",
        action="store_true",
        help="Actually submit the paid/queued IBM Runtime job.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs" / "ibm_quantum",
    )
    return parser.parse_args()


def hardware_test_spec() -> SystemSpec:
    p = np.array([0.08, 0.12, 0.16, 0.20])
    corr = np.array(
        [
            [1.00, 0.16, 0.08, 0.05],
            [0.16, 1.00, 0.13, 0.07],
            [0.08, 0.13, 1.00, 0.11],
            [0.05, 0.07, 0.11, 1.00],
        ]
    )
    return SystemSpec(
        node_names=["Bank A", "Bank B", "Bank C", "Bank D"],
        node_types=["bank"] * 4,
        exposure_matrix=np.zeros((4, 4)),
        capital_buffers=np.ones(4),
        marginal_default_probs=p,
        target_pairwise_corr=corr,
        clusters=["hardware-test"] * 4,
        metadata={"correlation_space": "binary_default", "kind": "ibm-hardware-test"},
    )


def main() -> None:
    args = parse_args()
    spec = hardware_test_spec()
    generator = EntangledBornMachineGenerator(ansatz="entangled", backend="statevector")
    generator.fit(spec)
    if len(generator.blocks_) != 1:
        raise RuntimeError("the IBM smoke test requires one materialisable circuit block")
    block = generator.blocks_[0]
    ideal_marginals, ideal_joint = generator.exact_moments()
    if not args.submit:
        print(
            json.dumps(
                {
                    "status": "dry-run",
                    "n_qubits": block.size,
                    "entanglers": len(block.edges),
                    "shots": args.shots,
                    "backend": args.backend or "least_busy",
                    "next_command": (
                        "uv run --extra quantum python scripts/run_ibm_quantum_test.py --submit"
                    ),
                },
                indent=2,
            )
        )
        return

    result = run_block(
        block,
        shots=args.shots,
        backend_name=args.backend,
        optimization_level=args.optimization_level,
    )
    observed = empirical_moments(result.samples)
    mask = ~np.eye(spec.n, dtype=bool)
    marginal_rmse = float(np.sqrt(np.mean((observed.marginals - ideal_marginals) ** 2)))
    pairwise_joint_rmse = float(
        np.sqrt(np.mean((observed.pairwise_joint[mask] - ideal_joint[mask]) ** 2))
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output_dir / "hardware_samples.npz", samples=result.samples)
    report = {
        "backend": result.backend_name,
        "job_id": result.job_id,
        "shots": result.shots,
        "circuit_depth": result.circuit_depth,
        "two_qubit_gates": result.two_qubit_gates,
        "circuit_operations": result.circuit_operations,
        "marginal_rmse_vs_ideal": marginal_rmse,
        "pairwise_joint_rmse_vs_ideal": pairwise_joint_rmse,
        "ideal_marginals": ideal_marginals.tolist(),
        "hardware_marginals": observed.marginals.tolist(),
    }
    (args.output_dir / "hardware_report.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
