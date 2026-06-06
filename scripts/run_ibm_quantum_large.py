"""Run a larger fitted systemic-risk Born machine (default 20 qubits) on an IBM QPU.

Unlike ``run_ibm_quantum_test.py`` (capped at 4/6/8), this submits a single sparse
*entangled* block at arbitrary ``n``. The entanglement graph is kept local and sparse
(``--max-degree``) so the circuit routes onto Heron heavy-hex connectivity without SWAP
blow-up -- the only thing that matters for fidelity on a non-error-corrected device.

While ``n`` is small enough to form the ``2^n`` statevector, the hardware samples are
validated against the *exact* ideal distribution; beyond that the fall-back is the analytic
moment targets. Authentication: saved Qiskit Runtime account or ``IBM_QUANTUM_TOKEN`` /
``IBM_QUANTUM_INSTANCE`` environment variables.
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

# Largest n whose 2^n statevector is comfortably materialisable for calibration + exact
# validation. Beyond this we fit from the analytic angle seed and validate against targets only.
EXACT_LIMIT = 24


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qubits", type=int, default=20)
    parser.add_argument("--max-degree", type=int, default=2, help="Max entanglers per qubit.")
    parser.add_argument("--backend", default="ibm_boston", help="IBM backend name.")
    parser.add_argument("--shots", type=int, default=4096)
    parser.add_argument("--optimization-level", type=int, choices=range(4), default=3)
    parser.add_argument("--submit", action="store_true", help="Submit the metered IBM job.")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "ibm_quantum")
    return parser.parse_args()


def banded_spec(n: int) -> SystemSpec:
    """Heterogeneous marginals with a banded (near-neighbor) co-default correlation."""
    p = np.linspace(0.05, 0.25, n)
    index = np.arange(n)
    distance = np.abs(index[:, None] - index[None, :])
    corr = 0.04 + 0.14 * np.exp(-distance / 2.0)
    np.fill_diagonal(corr, 1.0)
    return SystemSpec(
        node_names=[f"Bank {i + 1}" for i in range(n)],
        node_types=["bank"] * n,
        exposure_matrix=np.zeros((n, n)),
        capital_buffers=np.ones(n),
        marginal_default_probs=p,
        target_pairwise_corr=corr,
        clusters=["hardware-test"] * n,
        metadata={"correlation_space": "binary_default", "kind": "ibm-hardware-large"},
    )


def main() -> None:
    args = parse_args()
    spec = banded_spec(args.qubits)
    can_exact = args.qubits <= EXACT_LIMIT
    # Calibration and exact validation both form the 2^n statevector, so only enable them while
    # n is small enough. Past EXACT_LIMIT the analytic angle seed is faithful for a sparse,
    # near-local entangler graph and we validate against the analytic targets instead.
    generator = EntangledBornMachineGenerator(
        ansatz="entangled",
        backend="qiskit",
        max_degree=args.max_degree,
        max_block_qubits=max(args.qubits, 22),
        calibrate=can_exact,
    )
    generator.fit(spec)
    if len(generator.blocks_) != 1 or generator.blocks_[0].size != args.qubits:
        raise RuntimeError(
            f"expected one block spanning all {args.qubits} qubits; "
            f"got {len(generator.blocks_)} block(s)"
        )
    block = generator.blocks_[0]

    targets = generator.targets_
    mask = ~np.eye(spec.n, dtype=bool)
    ideal_marginals, ideal_joint = generator.exact_moments() if can_exact else (None, None)

    if not args.submit:
        seed = ideal_marginals if can_exact else targets.marginals
        print(
            json.dumps(
                {
                    "status": "dry-run",
                    "n_qubits": block.size,
                    "entanglers": len(block.edges),
                    "max_degree": args.max_degree,
                    "shots": args.shots,
                    "backend": args.backend,
                    "exact_ground_truth": can_exact,
                    "loader_marginal_rmse_vs_target": float(
                        np.sqrt(np.mean((seed - targets.marginals) ** 2))
                    ),
                    "next_command": (
                        "uv run --extra quantum python scripts/run_ibm_quantum_large.py "
                        f"--qubits {args.qubits} --max-degree {args.max_degree} --submit"
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
    report = {
        "backend": result.backend_name,
        "job_id": result.job_id,
        "shots": result.shots,
        "n_qubits": args.qubits,
        "max_degree": args.max_degree,
        "entanglers": len(block.edges),
        "circuit_depth": result.circuit_depth,
        "two_qubit_gates": result.two_qubit_gates,
        "circuit_operations": result.circuit_operations,
        "exact_ground_truth": can_exact,
        "marginal_rmse_vs_target": float(
            np.sqrt(np.mean((observed.marginals - targets.marginals) ** 2))
        ),
        "pairwise_joint_rmse_vs_target": float(
            np.sqrt(np.mean((observed.pairwise_joint[mask] - targets.pairwise_joint[mask]) ** 2))
        ),
        "target_marginals": targets.marginals.tolist(),
        "hardware_marginals": observed.marginals.tolist(),
    }
    if can_exact:
        report["marginal_rmse_vs_ideal"] = float(
            np.sqrt(np.mean((observed.marginals - ideal_marginals) ** 2))
        )
        report["pairwise_joint_rmse_vs_ideal"] = float(
            np.sqrt(np.mean((observed.pairwise_joint[mask] - ideal_joint[mask]) ** 2))
        )
        report["ideal_marginals"] = ideal_marginals.tolist()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"hardware_{args.qubits}q_degree{args.max_degree}_{result.backend_name}_{result.job_id}"
    np.savez_compressed(args.output_dir / f"{stem}_samples.npz", samples=result.samples)
    (args.output_dir / f"{stem}_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
