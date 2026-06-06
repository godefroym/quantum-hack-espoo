"""Compare Gaussian, ideal-QCBM, and IBM-QCBM scenarios on 28 or 38 institutions.

The empirical bank identities, rating PD ordering, equity-return dependence, reconstructed
exposures, and capital buffers come from the project's real-data pipeline. Because the raw
one-year PDs sit below current QPU readout noise, a transparent log-odds stress shift raises the
mean PD to 5% while preserving bank ranking. All three branches then use the same stressed spec.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import expit, logit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from systemic_risk.data_network.assemble import build_system_spec
from systemic_risk.generators import GaussianCopulaGenerator
from systemic_risk.generators.moments import (
    empirical_moments,
    moment_errors,
    targets_from_spec,
)
from systemic_risk.generators.quantum import ansatz as A
from systemic_risk.generators.quantum import mps_backend
from systemic_risk.generators.quantum.ibm_runtime import (
    DEFAULT_HARDWARE_SHOTS,
    best_qubit_line,
    dependency_aware_layout,
    mitigate_readout_moments,
    run_block,
    run_readout_calibration,
)
from systemic_risk.generators.quantum.qiskit_backend import build_circuit
from systemic_risk.spec import SystemSpec, joint_to_corr


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shots", type=int, default=DEFAULT_HARDWARE_SHOTS)
    parser.add_argument(
        "--reference-shots",
        type=int,
        default=None,
        help="Optional separate shot count for Gaussian/noiseless references.",
    )
    parser.add_argument("--stress-mean-pd", type=float, default=0.05)
    parser.add_argument("--backend", default="ibm_boston")
    parser.add_argument("--max-depth", type=int, default=50)
    parser.add_argument(
        "--entanglement-layout",
        choices=("chain", "topology"),
        default="chain",
        help="Use the original chain or a backend-aware graph expanded up to --max-depth.",
    )
    parser.add_argument("--calibration-iterations", type=int, default=5)
    parser.add_argument("--calibration-shots", type=int, default=5_000)
    parser.add_argument("--optimization-level", type=int, choices=range(4), default=3)
    parser.add_argument("--twirling-randomizations", type=int, default=32)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--submit", action="store_true")
    parser.add_argument(
        "--scope",
        choices=("banks", "all"),
        default="banks",
        help="'banks' uses 28 banks; 'all' adds the 10 corporates for 38 qubits.",
    )
    parser.add_argument("--include-corporates", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
    )
    return parser.parse_args()


def stressed_real_bank_spec(target_mean_pd: float = 0.05) -> SystemSpec:
    """Return all 28 banks, excluding corporates, with a transparent PD stress shift."""
    return stressed_real_institution_spec(target_mean_pd, include_corporates=False)


def stressed_real_institution_spec(
    target_mean_pd: float = 0.05,
    *,
    include_corporates: bool = True,
) -> SystemSpec:
    """Return the selected real entities with a common, ranking-preserving PD stress shift."""
    if not 0 < target_mean_pd < 1:
        raise ValueError("target_mean_pd must lie in (0, 1)")

    full = build_system_spec()
    selected_indices = np.arange(full.n) if include_corporates else np.array(
        [index for index, node_type in enumerate(full.node_types) if node_type != "corporate"]
    )
    expected = 38 if include_corporates else 28
    if len(selected_indices) != expected:
        raise RuntimeError(f"expected {expected} real institutions, found {len(selected_indices)}")

    base_pd = np.clip(full.marginal_default_probs[selected_indices], 1e-8, 1.0 - 1e-8)
    shift = brentq(
        lambda value: float(expit(logit(base_pd) + value).mean() - target_mean_pd),
        -30.0,
        30.0,
    )
    stressed_pd = expit(logit(base_pd) + shift)
    grid = np.ix_(selected_indices, selected_indices)
    subset = (
        "38 real institutions: 28 banks and 10 corporates"
        if include_corporates
        else "28 financial institutions; corporates excluded"
    )
    spec = SystemSpec(
        node_names=[full.node_names[i] for i in selected_indices],
        node_types=[full.node_types[i] for i in selected_indices],
        exposure_matrix=full.exposure_matrix[grid].copy(),
        capital_buffers=full.capital_buffers[selected_indices].copy(),
        marginal_default_probs=stressed_pd,
        target_pairwise_corr=full.target_pairwise_corr[grid].copy(),
        clusters=None
        if full.clusters is None
        else [full.clusters[i] for i in selected_indices],
        metadata={
            **full.metadata,
            "subset": subset,
            "base_marginal_default_probs": base_pd.tolist(),
            "stress_method": "common log-odds shift",
            "stress_log_odds_shift": float(shift),
            "stress_mean_pd": float(target_mean_pd),
        },
    )
    return _order_by_dependency_path(spec)


def _order_by_dependency_path(spec: SystemSpec) -> SystemSpec:
    dependency = np.abs(spec.dependency_matrix())
    start = int(dependency.sum(axis=1).argmax())
    order = [start]
    unused = set(range(spec.n)) - {start}
    while unused:
        current = order[-1]
        next_node = max(unused, key=lambda candidate: (dependency[current, candidate], -candidate))
        order.append(next_node)
        unused.remove(next_node)

    index = np.array(order)
    grid = np.ix_(index, index)
    return SystemSpec(
        node_names=[spec.node_names[i] for i in index],
        node_types=[spec.node_types[i] for i in index],
        exposure_matrix=spec.exposure_matrix[grid].copy(),
        capital_buffers=spec.capital_buffers[index].copy(),
        marginal_default_probs=spec.marginal_default_probs[index].copy(),
        target_pairwise_corr=spec.target_pairwise_corr[grid].copy(),
        clusters=None if spec.clusters is None else [spec.clusters[i] for i in index],
        metadata={**spec.metadata, "dependency_path_order": index.tolist()},
    )


def fitted_chain(spec: SystemSpec) -> A.EntangledCircuit:
    """Build and MPS-calibrate a two-layer chain spanning every institution."""
    p = np.clip(spec.marginal_default_probs, 1e-6, 1.0 - 1e-6)
    edges = [(index, index + 1) for index in range(spec.n - 1)]
    edges = [edge for layer in A.schedule_entanglement_edges(edges) for edge in layer]
    block = A._block_circuit(list(range(spec.n)), p, A.target_covariance(spec), edges)
    return A.calibrate_block(block, mps_backend.block_moments, iterations=30)


def _sample_calibrated_block(
    spec: SystemSpec,
    edges: list[tuple[int, int]],
    *,
    iterations: int,
    shots: int,
    seed: int,
) -> A.EntangledCircuit:
    """Calibrate a cyclic sparse graph from deterministic MPS sample moments."""
    p = np.clip(spec.marginal_default_probs, 1e-6, 1.0 - 1e-6)
    block = A._block_circuit(
        list(range(spec.n)),
        p,
        A.target_covariance(spec),
        edges,
    )
    call_index = 0

    def sampled_moments(
        ry: np.ndarray,
        block_edges: list[tuple[int, int]],
        cry: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        nonlocal call_index
        samples = mps_backend.sample_block(
            ry,
            block_edges,
            cry,
            shots,
            seed=seed + call_index,
        ).astype(float)
        call_index += 1
        return samples.mean(axis=0), (samples.T @ samples) / len(samples)

    return A.calibrate_block(
        block,
        sampled_moments,
        iterations=iterations,
        tol=0.0,
    )


def fitted_topology_graph(
    spec: SystemSpec,
    backend,
    *,
    max_depth: int,
    optimization_level: int,
    calibration_iterations: int,
    calibration_shots: int,
    seed: int,
) -> tuple[A.EntangledCircuit, list[int], object, dict[str, object]]:
    """Fit the strongest backend-compatible graph whose transpiled depth stays bounded."""
    from qiskit.transpiler import generate_preset_pass_manager

    dependency = np.abs(spec.dependency_matrix())
    initial_layout, native_edges, readout = dependency_aware_layout(
        backend,
        dependency,
        seed=seed,
    )
    native = {tuple(sorted(edge)) for edge in native_edges}
    candidates = sorted(
        (
            (float(dependency[i, j]), i, j)
            for i in range(spec.n)
            for j in range(i + 1, spec.n)
            if (i, j) not in native
        ),
        reverse=True,
    )
    selected = list(native_edges)
    preview = None
    routed_edges = 0

    for _, source, target in candidates:
        trial = selected + [(source, target)]
        trial.sort(key=lambda edge: dependency[edge], reverse=True)
        scheduled = [
            edge
            for layer in A.schedule_entanglement_edges(trial)
            for edge in layer
        ]
        seed_block = A._block_circuit(
            list(range(spec.n)),
            np.clip(spec.marginal_default_probs, 1e-6, 1.0 - 1e-6),
            A.target_covariance(spec),
            scheduled,
        )
        candidate_preview = generate_preset_pass_manager(
            optimization_level=optimization_level,
            backend=backend,
            initial_layout=initial_layout,
            seed_transpiler=seed,
        ).run(build_circuit(seed_block.ry, seed_block.edges, seed_block.cry, measure=True))
        if candidate_preview.depth() <= max_depth:
            selected.append((source, target))
            preview = candidate_preview
            routed_edges += 1

    selected.sort(key=lambda edge: dependency[edge], reverse=True)
    scheduled = [
        edge
        for layer in A.schedule_entanglement_edges(selected)
        for edge in layer
    ]
    block = _sample_calibrated_block(
        spec,
        scheduled,
        iterations=calibration_iterations,
        shots=calibration_shots,
        seed=seed + 10_000,
    )
    preview = generate_preset_pass_manager(
        optimization_level=optimization_level,
        backend=backend,
        initial_layout=initial_layout,
        seed_transpiler=seed,
    ).run(build_circuit(block.ry, block.edges, block.cry, measure=True))
    metadata = {
        "native_entanglers": len(native_edges),
        "routed_entanglers": routed_edges,
        "dependency_score": float(sum(dependency[edge] for edge in block.edges)),
        "mean_layout_readout_error": float(readout[initial_layout].mean()),
        "calibration_method": "sampled MPS moments",
        "calibration_iterations": calibration_iterations,
        "calibration_shots_per_iteration": calibration_shots,
        "transpiled_depth": int(preview.depth()),
        "transpiled_two_qubit_gates": int(
            sum(len(instruction.qubits) == 2 for instruction in preview.data)
        ),
    }
    return block, initial_layout, preview, metadata


def vectorized_cascade(samples: np.ndarray, spec: SystemSpec) -> tuple[np.ndarray, np.ndarray]:
    """Return final failure counts and cascade depths for the deterministic exposure engine."""
    state = np.asarray(samples, dtype=bool).copy()
    frontier = state.copy()
    cumulative_losses = np.zeros(state.shape, dtype=float)
    depths = np.zeros(len(state), dtype=int)
    active = np.any(frontier, axis=1) & ~np.all(state, axis=1)
    for round_index in range(1, spec.n + 1):
        if not np.any(active):
            break
        cumulative_losses[active] += (
            frontier[active].astype(float) @ spec.exposure_matrix.T
        )
        new_failures = (
            ~state[active]
            & (cumulative_losses[active] > spec.capital_buffers[None, :])
        )
        active_indices = np.flatnonzero(active)
        changed = np.any(new_failures, axis=1)
        if np.any(changed):
            changed_indices = active_indices[changed]
            state[changed_indices] |= new_failures[changed]
            frontier[changed_indices] = new_failures[changed]
            depths[changed_indices] = round_index
        finished_indices = active_indices[~changed]
        frontier[finished_indices] = False
        active = np.any(frontier, axis=1) & ~np.all(state, axis=1)
    return state.sum(axis=1), depths


def evaluate(label: str, samples: np.ndarray, spec: SystemSpec) -> dict[str, float | str]:
    targets = targets_from_spec(spec)
    errors = moment_errors(samples, targets)
    initial_counts = samples.sum(axis=1)
    final_counts, depths = vectorized_cascade(samples, spec)
    severe_threshold = int(np.ceil(spec.n / 2))
    return {
        "generator": label,
        "n_samples": float(len(samples)),
        "marginal_rmse": errors.marginal_rmse,
        "pairwise_joint_rmse": errors.pairwise_joint_rmse,
        "pairwise_corr_rmse": errors.pairwise_corr_rmse,
        "mean_initial_defaults": float(initial_counts.mean()),
        "mean_final_failures": float(final_counts.mean()),
        "mean_cascade_amplification": float((final_counts - initial_counts).mean()),
        "mean_cascade_depth": float(depths.mean()),
        "max_final_failures": float(final_counts.max()),
        "p_severe": float(np.mean(final_counts >= severe_threshold)),
        "tail_mean_1pct": _tail_mean(final_counts, 0.01),
        "tail_mean_5pct": _tail_mean(final_counts, 0.05),
        "scenario_diversity": float(np.unique(samples, axis=0).shape[0] / len(samples)),
    }


def _tail_mean(values: np.ndarray, fraction: float) -> float:
    count = max(1, int(np.ceil(fraction * len(values))))
    return float(np.sort(values)[-count:].mean())


def _plot(
    path: Path,
    samples_by_label: dict[str, np.ndarray],
    spec: SystemSpec,
    *,
    entangled_edges: list[tuple[int, int]],
    mitigated_moments: tuple[np.ndarray, np.ndarray] | None = None,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    targets = targets_from_spec(spec)
    labels = list(samples_by_label)
    colors = ["#3366A3", "#B84A3A", "#2D8C68"]
    final_counts = {
        label: vectorized_cascade(samples, spec)[0]
        for label, samples in samples_by_label.items()
    }

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    ax = axes[0, 0]
    for label, color in zip(labels, colors):
        marginals = samples_by_label[label].mean(axis=0)
        ax.scatter(targets.marginals, marginals, s=32, alpha=0.8, label=label, color=color)
    if mitigated_moments is not None:
        corrected_marginals, _ = mitigated_moments
        ax.scatter(
            targets.marginals,
            corrected_marginals,
            s=42,
            marker="x",
            linewidth=1.5,
            label="IBM readout-corrected moments",
            color="#7A5195",
        )
    limit = max(
        float(targets.marginals.max()),
        *(float(samples.mean(axis=0).max()) for samples in samples_by_label.values()),
    )
    ax.plot([0, limit], [0, limit], "k--", linewidth=1)
    ax.set(xlabel="Target P(default)", ylabel="Observed P(default)", title="Marginal calibration")
    ax.legend()

    kept = np.zeros((spec.n, spec.n), dtype=bool)
    for source, target in entangled_edges:
        kept[source, target] = kept[target, source] = True
    ax = axes[0, 1]
    target_joint = targets.pairwise_joint[kept]
    for label, color in zip(labels, colors):
        samples = samples_by_label[label]
        joint = (samples.T @ samples) / len(samples)
        ax.scatter(target_joint, joint[kept], s=28, alpha=0.75, label=label, color=color)
    if mitigated_moments is not None:
        _, corrected_joint = mitigated_moments
        ax.scatter(
            target_joint,
            corrected_joint[kept],
            s=42,
            marker="x",
            linewidth=1.5,
            label="IBM readout-corrected moments",
            color="#7A5195",
        )
    joint_limit = max(float(target_joint.max()), ax.get_xlim()[1], ax.get_ylim()[1])
    ax.plot([0, joint_limit], [0, joint_limit], "k--", linewidth=1)
    ax.set(
        xlabel="Target P(co-default)",
        ylabel="Observed P(co-default)",
        title="Directly encoded entanglement edges",
    )

    ax = axes[1, 0]
    bins = np.arange(-0.5, spec.n + 1.5)
    for label, color in zip(labels, colors):
        ax.hist(
            final_counts[label],
            bins=bins,
            density=True,
            histtype="step",
            linewidth=2,
            label=label,
            color=color,
        )
    ax.set(xlabel="Final failed institutions", ylabel="Probability", title="Contagion outcomes")
    ax.legend()

    ax = axes[1, 1]
    thresholds = np.arange(spec.n + 1)
    for label, color in zip(labels, colors):
        tail = [np.mean(final_counts[label] >= threshold) for threshold in thresholds]
        ax.step(thresholds, tail, where="post", linewidth=2, label=label, color=color)
    ax.set_yscale("log")
    ax.set_ylim(1e-5, 1.05)
    ax.set(xlabel="Failure threshold", ylabel="P(final failures >= threshold)", title="Cascade tail")
    ax.legend()

    n_corporates = sum(node_type == "corporate" for node_type in spec.node_types)
    scope = (
        f"{spec.n} real institutions ({n_corporates} corporates)"
        if n_corporates
        else f"{spec.n} real banks"
    )
    fig.suptitle(f"{scope}: Gaussian copula vs ideal and IBM quantum circuits")
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.shots <= 0:
        raise ValueError("--shots must be positive")
    if args.reference_shots is not None and args.reference_shots <= 0:
        raise ValueError("--reference-shots must be positive")
    if args.max_depth <= 0:
        raise ValueError("--max-depth must be positive")
    if args.calibration_iterations < 0:
        raise ValueError("--calibration-iterations must be nonnegative")
    if args.calibration_shots <= 0:
        raise ValueError("--calibration-shots must be positive")

    include_corporates = args.scope == "all" or args.include_corporates
    output_dir = args.output_dir or (
        ROOT
        / "outputs"
        / (
            "real_institutions_quantum_comparison"
            if include_corporates
            else "real_banks_quantum_comparison"
        )
    )
    spec = stressed_real_institution_spec(
        args.stress_mean_pd,
        include_corporates=include_corporates,
    )
    reference_shots = args.reference_shots or args.shots
    service = None
    backend = None
    physical_layout: list[int] | None = None
    preview = None
    layout_metadata: dict[str, object] = {
        "strategy": args.entanglement_layout,
    }
    if args.entanglement_layout == "topology":
        from qiskit_ibm_runtime import QiskitRuntimeService

        service_kwargs: dict[str, str] = {}
        if os.environ.get("IBM_QUANTUM_TOKEN"):
            service_kwargs["token"] = os.environ["IBM_QUANTUM_TOKEN"]
            service_kwargs["channel"] = os.environ.get(
                "IBM_QUANTUM_CHANNEL", "ibm_quantum_platform"
            )
        if os.environ.get("IBM_QUANTUM_INSTANCE"):
            service_kwargs["instance"] = os.environ["IBM_QUANTUM_INSTANCE"]
        service = QiskitRuntimeService(**service_kwargs)
        backend = service.backend(args.backend)
        block, physical_layout, preview, fitted_metadata = fitted_topology_graph(
            spec,
            backend,
            max_depth=args.max_depth,
            optimization_level=args.optimization_level,
            calibration_iterations=args.calibration_iterations,
            calibration_shots=args.calibration_shots,
            seed=args.seed,
        )
        layout_metadata.update(fitted_metadata)
    else:
        block = fitted_chain(spec)
    gaussian = GaussianCopulaGenerator()
    gaussian.fit(spec)

    samples_by_label = {
        "Gaussian copula": gaussian.sample(reference_shots, seed=args.seed),
        "Quantum circuit - noiseless reference": mps_backend.sample_block(
            block.ry,
            block.edges,
            block.cry,
            reference_shots,
            seed=args.seed + 1,
        ),
    }
    hardware_metadata: dict[str, object] = {}
    mitigated_moments: tuple[np.ndarray, np.ndarray] | None = None

    if args.submit:
        from qiskit.transpiler import generate_preset_pass_manager
        from qiskit_ibm_runtime import QiskitRuntimeService

        if service is None:
            service_kwargs: dict[str, str] = {}
            if os.environ.get("IBM_QUANTUM_TOKEN"):
                service_kwargs["token"] = os.environ["IBM_QUANTUM_TOKEN"]
                service_kwargs["channel"] = os.environ.get(
                    "IBM_QUANTUM_CHANNEL", "ibm_quantum_platform"
                )
            if os.environ.get("IBM_QUANTUM_INSTANCE"):
                service_kwargs["instance"] = os.environ["IBM_QUANTUM_INSTANCE"]
            service = QiskitRuntimeService(**service_kwargs)
        if backend is None:
            backend = service.backend(args.backend)
        if physical_layout is None:
            physical_layout, readout = best_qubit_line(backend, spec.n)
        else:
            readout = np.array(
                [
                    backend.target["measure"][(q,)].error
                    for q in range(backend.num_qubits)
                ]
            )
        if preview is None:
            preview = generate_preset_pass_manager(
                optimization_level=args.optimization_level,
                backend=backend,
                initial_layout=physical_layout,
                seed_transpiler=args.seed,
            ).run(build_circuit(block.ry, block.edges, block.cry, measure=True))
        preview_depth = int(preview.depth())
        if preview_depth > args.max_depth:
            raise RuntimeError(
                f"transpiled depth {preview_depth} exceeds --max-depth {args.max_depth}"
            )

        hardware = run_block(
            block,
            shots=args.shots,
            backend_name=args.backend,
            optimization_level=args.optimization_level,
            initial_layout=physical_layout,
            dynamical_decoupling=True,
            measure_twirling=True,
            gate_twirling=True,
            twirling_randomizations=args.twirling_randomizations,
            seed_transpiler=args.seed,
            service=service,
        )
        calibration = run_readout_calibration(
            spec.n,
            shots=min(args.shots, DEFAULT_HARDWARE_SHOTS),
            backend_name=args.backend,
            optimization_level=args.optimization_level,
            initial_layout=physical_layout,
            measure_twirling=True,
            seed_transpiler=args.seed,
            service=service,
        )
        mitigated_moments = mitigate_readout_moments(hardware.samples, calibration)
        samples_by_label["Quantum circuit - IBM hardware"] = hardware.samples
        hardware_metadata = {
            "backend": hardware.backend_name,
            "job_id": hardware.job_id,
            "shot_batches": list(hardware.shot_batches),
            "circuit_depth": hardware.circuit_depth,
            "two_qubit_gates": hardware.two_qubit_gates,
            "circuit_operations": hardware.circuit_operations,
            "physical_layout": physical_layout,
            "mean_layout_readout_error": float(readout[physical_layout].mean()),
            "gate_twirling": True,
            "measure_twirling": True,
            "twirling_randomizations": args.twirling_randomizations,
            "readout_calibration_job_id": calibration.job_id,
            "readout_calibration_shots_per_state": calibration.shots,
            "mean_p_meas_1_given_0": float(calibration.p_meas_1_given_0.mean()),
            "mean_p_meas_0_given_1": float(calibration.p_meas_0_given_1.mean()),
        }

    rows = [evaluate(label, samples, spec) for label, samples in samples_by_label.items()]
    frame = pd.DataFrame(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / "comparison.csv", index=False)
    np.savez_compressed(
        output_dir / "samples.npz",
        **{label.lower().replace(" ", "_"): samples for label, samples in samples_by_label.items()},
    )
    figure = output_dir / "comparison.png"
    _plot(
        figure,
        samples_by_label,
        spec,
        entangled_edges=block.edges,
        mitigated_moments=mitigated_moments,
    )

    if args.entanglement_layout == "topology":
        sampled_reference = empirical_moments(
            samples_by_label["Quantum circuit - noiseless reference"]
        )
        ideal_marginals = sampled_reference.marginals
        ideal_joint = sampled_reference.pairwise_joint
        ideal_moment_source = "sampled noiseless MPS reference"
    else:
        ideal_marginals, ideal_joint = mps_backend.block_moments(
            block.ry, block.edges, block.cry
        )
        ideal_moment_source = "exact MPS expectations"
    targets = targets_from_spec(spec)
    kept = np.zeros((spec.n, spec.n), dtype=bool)
    for source, target in block.edges:
        kept[source, target] = kept[target, source] = True
    report = {
        "network": spec.metadata["subset"],
        "scope": "all" if include_corporates else "banks",
        "reference_shots_per_generator": reference_shots,
        "hardware_shots": args.shots if args.submit else None,
        "stress_mean_pd": args.stress_mean_pd,
        "institution_order": spec.node_names,
        "institution_types": spec.node_types,
        "n_qubits": spec.n,
        "entanglers": len(block.edges),
        "possible_pairs": spec.n * (spec.n - 1) // 2,
        "directly_entangled_pair_fraction": (
            len(block.edges) / (spec.n * (spec.n - 1) // 2)
        ),
        "entanglement_depth": block.entanglement_depth,
        "entanglement_layout": layout_metadata,
        "ideal_marginal_rmse": float(
            np.sqrt(np.mean((ideal_marginals - targets.marginals) ** 2))
        ),
        "ideal_moment_source": ideal_moment_source,
        "ideal_joint_rmse_kept_edges": float(
            np.sqrt(np.mean((ideal_joint[kept] - targets.pairwise_joint[kept]) ** 2))
        ),
        "results": rows,
        "hardware": hardware_metadata,
    }
    if mitigated_moments is not None:
        corrected_marginals, corrected_joint = mitigated_moments
        off_diagonal = ~np.eye(spec.n, dtype=bool)
        dropped = off_diagonal & ~kept
        corrected_corr = joint_to_corr(corrected_joint, corrected_marginals)
        report["readout_mitigated_moments"] = {
            "marginal_rmse": float(
                np.sqrt(np.mean((corrected_marginals - targets.marginals) ** 2))
            ),
            "pairwise_joint_rmse": float(
                np.sqrt(
                    np.mean(
                        (
                            corrected_joint[off_diagonal]
                            - targets.pairwise_joint[off_diagonal]
                        )
                        ** 2
                    )
                )
            ),
            "pairwise_joint_rmse_kept_edges": float(
                np.sqrt(
                    np.mean(
                        (
                            corrected_joint[kept]
                            - targets.pairwise_joint[kept]
                        )
                        ** 2
                    )
                )
            ),
            "pairwise_corr_rmse": float(
                np.sqrt(
                    np.mean(
                        (
                            corrected_corr[off_diagonal]
                            - targets.pairwise_corr[off_diagonal]
                        )
                        ** 2
                    )
                )
            ),
            "pairwise_corr_rmse_kept_edges": float(
                np.sqrt(
                    np.mean(
                        (
                            corrected_corr[kept]
                            - targets.pairwise_corr[kept]
                        )
                        ** 2
                    )
                )
            ),
            "pairwise_corr_rmse_dropped_edges": float(
                np.sqrt(
                    np.mean(
                        (
                            corrected_corr[dropped]
                            - targets.pairwise_corr[dropped]
                        )
                        ** 2
                    )
                )
            ),
            "marginals": corrected_marginals.tolist(),
            "pairwise_joint": corrected_joint.tolist(),
            "scope": (
                "Independent readout mitigation for first/pairwise moments only; "
                "cascade metrics remain based on raw hardware shots."
            ),
        }
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    print(frame.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    printable = {key: value for key, value in report.items() if key != "results"}
    if "readout_mitigated_moments" in printable:
        printable["readout_mitigated_moments"] = {
            key: value
            for key, value in printable["readout_mitigated_moments"].items()
            if key not in {"marginals", "pairwise_joint"}
        }
    print(json.dumps(printable, indent=2))
    print(f"Figure: {figure}")


if __name__ == "__main__":
    main()
