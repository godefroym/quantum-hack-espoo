"""Explicit IBM Quantum Runtime execution for fitted Born-machine circuit blocks.

This module is intentionally separate from the generator's normal ``sample`` method:
submitting a cloud job must always be an explicit action. Credentials are loaded from a
saved Qiskit Runtime account or from ``IBM_QUANTUM_TOKEN`` and
``IBM_QUANTUM_INSTANCE`` environment variables.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import os
from typing import Any

import numpy as np

from systemic_risk.generators.quantum.ansatz import EntangledCircuit


DEFAULT_HARDWARE_SHOTS = 100_000


def _split_shots(shots: int, max_shots: int) -> tuple[int, ...]:
    """Split a requested sample count into backend-compatible Runtime publications."""
    if shots <= 0 or max_shots <= 0:
        raise ValueError("shots and max_shots must be positive")
    full_batches, remainder = divmod(shots, max_shots)
    batches = [max_shots] * full_batches
    if remainder:
        batches.append(remainder)
    return tuple(batches)


def best_qubit_line(backend: Any, length: int) -> tuple[list[int], np.ndarray]:
    """Return a low-error connected physical-qubit line for a chain circuit."""
    if length <= 0 or length > backend.num_qubits:
        raise ValueError("length must lie between 1 and backend.num_qubits")

    target = backend.target
    readout = np.array([target["measure"][(q,)].error for q in range(backend.num_qubits)])
    two_qubit_gate = next(
        (name for name in ("cz", "ecr", "cx") if name in target),
        None,
    )
    if two_qubit_gate is None:
        raise RuntimeError("backend exposes no supported two-qubit gate")
    two_qubit = target[two_qubit_gate]
    edge_error = {
        pair: properties.error
        for pair, properties in two_qubit.items()
        if properties is not None and properties.error is not None
    }
    adjacency: dict[int, list[int]] = {}
    for source, target_qubit in edge_error:
        adjacency.setdefault(source, []).append(target_qubit)

    best: list[int] | None = None
    best_cost = float("inf")
    for start in np.argsort(readout)[: min(60, backend.num_qubits)]:
        path = [int(start)]
        used = {int(start)}
        cost = float(readout[start])
        for _ in range(length - 1):
            current = path[-1]
            candidates = [
                (
                    edge_error.get(
                        (current, neighbor),
                        edge_error.get((neighbor, current), float("inf")),
                    )
                    + readout[neighbor],
                    neighbor,
                )
                for neighbor in adjacency.get(current, [])
                if neighbor not in used
            ]
            if not candidates:
                break
            edge_cost, next_qubit = min(candidates)
            path.append(next_qubit)
            used.add(next_qubit)
            cost += float(edge_cost)
        if len(path) == length and cost < best_cost:
            best = path
            best_cost = cost

    if best is None:
        raise RuntimeError(f"no connected physical-qubit line of length {length} found")
    return best, readout


def dependency_aware_layout(
    backend: Any,
    dependency: np.ndarray,
    *,
    seed: int = 2026,
    annealing_steps: int = 100_000,
) -> tuple[list[int], list[tuple[int, int]], np.ndarray]:
    """Map institutions to a compact native subgraph, maximizing encoded dependency.

    Returns ``(initial_layout, logical_edges, readout_errors)``. Every returned logical
    edge is native under ``initial_layout``, so the corresponding CRY gates need no SWAPs.
    """
    dependency = np.asarray(dependency, dtype=float)
    if dependency.ndim != 2 or dependency.shape[0] != dependency.shape[1]:
        raise ValueError("dependency must be a square matrix")
    if annealing_steps < 0:
        raise ValueError("annealing_steps must be nonnegative")
    length = len(dependency)
    if length <= 0 or length > backend.num_qubits:
        raise ValueError("dependency size must lie between 1 and backend.num_qubits")

    target = backend.target
    readout = np.array([target["measure"][(q,)].error for q in range(backend.num_qubits)])
    two_qubit_gate = next(
        (name for name in ("cz", "ecr", "cx") if name in target),
        None,
    )
    if two_qubit_gate is None:
        raise RuntimeError("backend exposes no supported two-qubit gate")
    two_qubit = target[two_qubit_gate]
    edge_error = {
        tuple(sorted(pair)): properties.error
        for pair, properties in two_qubit.items()
        if properties is not None and properties.error is not None
    }
    physical_edges = sorted(edge_error)
    adjacency: dict[int, set[int]] = {
        qubit: set() for qubit in range(backend.num_qubits)
    }
    for source, target_qubit in physical_edges:
        adjacency[source].add(target_qubit)
        adjacency[target_qubit].add(source)

    def grow(seed_qubit: int) -> set[int] | None:
        selected = {seed_qubit}
        while len(selected) < length:
            frontier = set().union(*(adjacency[q] for q in selected)) - selected
            if not frontier:
                return None
            next_qubit = max(
                frontier,
                key=lambda q: (
                    sum(neighbor in selected for neighbor in adjacency[q]),
                    -readout[q],
                    len(adjacency[q]),
                    -q,
                ),
            )
            selected.add(next_qubit)
        return selected

    best: tuple[tuple[float, float, float], set[int], list[tuple[int, int]]] | None = None
    for seed_qubit in range(backend.num_qubits):
        selected = grow(seed_qubit)
        if selected is None:
            continue
        induced = [
            edge
            for edge in physical_edges
            if edge[0] in selected and edge[1] in selected
        ]
        score = (
            float(len(induced)),
            -float(np.mean([edge_error[edge] for edge in induced])),
            -float(readout[list(selected)].mean()),
        )
        if best is None or score > best[0]:
            best = (score, selected, induced)
    if best is None:
        raise RuntimeError(f"no connected physical subgraph of size {length} found")

    _, selected, induced = best
    physical = sorted(selected)
    slot = {qubit: index for index, qubit in enumerate(physical)}
    slot_edges = [(slot[source], slot[target_qubit]) for source, target_qubit in induced]
    physical_degree = np.array(
        [sum(index in edge for edge in slot_edges) for index in range(length)]
    )
    logical_strength = np.abs(dependency).sum(axis=1)
    assignment = np.empty(length, dtype=int)
    assignment[np.argsort(-physical_degree)] = np.argsort(-logical_strength)

    def objective(values: np.ndarray) -> float:
        return float(sum(dependency[values[i], values[j]] for i, j in slot_edges))

    rng = np.random.default_rng(seed)
    current = objective(assignment)
    best_assignment = assignment.copy()
    best_objective = current
    for step in range(annealing_steps):
        first, second = rng.choice(length, size=2, replace=False)
        assignment[first], assignment[second] = assignment[second], assignment[first]
        candidate = objective(assignment)
        temperature = 0.02 * (1.0 - step / max(annealing_steps, 1)) + 1e-5
        if candidate >= current or rng.random() < math.exp((candidate - current) / temperature):
            current = candidate
        else:
            assignment[first], assignment[second] = assignment[second], assignment[first]
        if current > best_objective:
            best_objective = current
            best_assignment = assignment.copy()

    initial_layout = [0] * length
    for physical_slot, logical_qubit in enumerate(best_assignment):
        initial_layout[int(logical_qubit)] = physical[physical_slot]
    logical_edges = [
        (int(best_assignment[first]), int(best_assignment[second]))
        for first, second in slot_edges
    ]
    return initial_layout, logical_edges, readout


@dataclass(frozen=True)
class IBMHardwareResult:
    backend_name: str
    job_id: str
    shots: int
    shot_batches: tuple[int, ...]
    samples: np.ndarray
    circuit_depth: int
    circuit_operations: dict[str, int]
    two_qubit_gates: int


@dataclass(frozen=True)
class ReadoutCalibration:
    """Independent per-qubit assignment errors measured on one physical layout."""

    backend_name: str
    job_id: str
    shots: int
    p_meas_1_given_0: np.ndarray
    p_meas_0_given_1: np.ndarray


def run_block(
    block: EntangledCircuit,
    *,
    shots: int = DEFAULT_HARDWARE_SHOTS,
    backend_name: str | None = None,
    optimization_level: int = 1,
    initial_layout: list[int] | None = None,
    dynamical_decoupling: bool = False,
    measure_twirling: bool = False,
    gate_twirling: bool = False,
    twirling_randomizations: int | None = None,
    seed_transpiler: int | None = None,
    service: Any | None = None,
) -> IBMHardwareResult:
    """Transpile and execute one fitted circuit block with IBM Runtime ``SamplerV2``.

    ``initial_layout`` pins logical qubit ``k`` to physical qubit ``initial_layout[k]`` -- use it
    to place an already-ordered entangler chain onto a hand-picked, low-error line so the router
    inserts no SWAPs. ``dynamical_decoupling`` and ``measure_twirling``/``gate_twirling`` enable
    cheap error suppression (idle-qubit DD, readout/gate Pauli twirling).
    """
    if shots <= 0:
        raise ValueError("shots must be positive")
    if optimization_level not in {0, 1, 2, 3}:
        raise ValueError("optimization_level must be one of 0, 1, 2, 3")

    from qiskit.transpiler import generate_preset_pass_manager
    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2

    from systemic_risk.generators.quantum.qiskit_backend import build_circuit

    if service is None:
        token = os.environ.get("IBM_QUANTUM_TOKEN")
        instance = os.environ.get("IBM_QUANTUM_INSTANCE")
        kwargs: dict[str, str] = {}
        if token:
            kwargs["token"] = token
            kwargs["channel"] = os.environ.get(
                "IBM_QUANTUM_CHANNEL",
                "ibm_quantum_platform",
            )
        if instance:
            kwargs["instance"] = instance
        service = QiskitRuntimeService(**kwargs)

    backend = (
        service.backend(backend_name)
        if backend_name
        else service.least_busy(
            operational=True,
            simulator=False,
            min_num_qubits=block.size,
        )
    )
    circuit = build_circuit(block.ry, block.edges, block.cry, measure=True)
    pass_manager = generate_preset_pass_manager(
        optimization_level=optimization_level,
        backend=backend,
        initial_layout=initial_layout,
        seed_transpiler=seed_transpiler,
    )
    isa_circuit = pass_manager.run(circuit)

    sampler = SamplerV2(mode=backend)
    if dynamical_decoupling:
        sampler.options.dynamical_decoupling.enable = True
        sampler.options.dynamical_decoupling.sequence_type = "XY4"
    if measure_twirling or gate_twirling:
        sampler.options.twirling.enable_measure = measure_twirling
        sampler.options.twirling.enable_gates = gate_twirling
        if twirling_randomizations is not None:
            if twirling_randomizations <= 0:
                raise ValueError("twirling_randomizations must be positive")
            sampler.options.twirling.num_randomizations = twirling_randomizations
    max_shots = int(getattr(backend.configuration(), "max_shots", shots) or shots)
    shot_batches = _split_shots(shots, max_shots)
    pubs = [(isa_circuit, None, batch_shots) for batch_shots in shot_batches]
    job = sampler.run(pubs)
    bitstrings = [
        bitstring
        for pub_result in job.result()
        for bitstring in pub_result.data.meas.get_bitstrings()
    ]
    samples = _bitstrings_to_samples(bitstrings, block.size)

    operations = {str(name): int(count) for name, count in isa_circuit.count_ops().items()}
    two_qubit_gates = sum(
        1 for instruction in isa_circuit.data if len(instruction.qubits) == 2
    )
    return IBMHardwareResult(
        backend_name=_backend_name(backend),
        job_id=str(job.job_id()),
        shots=shots,
        shot_batches=shot_batches,
        samples=samples,
        circuit_depth=int(isa_circuit.depth()),
        circuit_operations=operations,
        two_qubit_gates=two_qubit_gates,
    )


def run_readout_calibration(
    n_qubits: int,
    *,
    shots: int = DEFAULT_HARDWARE_SHOTS,
    backend_name: str | None = None,
    optimization_level: int = 1,
    initial_layout: list[int] | None = None,
    measure_twirling: bool = True,
    seed_transpiler: int | None = None,
    service: Any | None = None,
) -> ReadoutCalibration:
    """Measure simultaneous ``|0...0>``/``|1...1>`` assignment errors on a layout."""
    if n_qubits <= 0:
        raise ValueError("n_qubits must be positive")
    if shots <= 0:
        raise ValueError("shots must be positive")

    from qiskit import QuantumCircuit
    from qiskit.transpiler import generate_preset_pass_manager
    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2

    if service is None:
        token = os.environ.get("IBM_QUANTUM_TOKEN")
        instance = os.environ.get("IBM_QUANTUM_INSTANCE")
        kwargs: dict[str, str] = {}
        if token:
            kwargs["token"] = token
            kwargs["channel"] = os.environ.get(
                "IBM_QUANTUM_CHANNEL",
                "ibm_quantum_platform",
            )
        if instance:
            kwargs["instance"] = instance
        service = QiskitRuntimeService(**kwargs)

    backend = (
        service.backend(backend_name)
        if backend_name
        else service.least_busy(
            operational=True,
            simulator=False,
            min_num_qubits=n_qubits,
        )
    )
    zero = QuantumCircuit(n_qubits)
    zero.measure_all()
    one = QuantumCircuit(n_qubits)
    one.x(range(n_qubits))
    one.measure_all()
    pass_manager = generate_preset_pass_manager(
        optimization_level=optimization_level,
        backend=backend,
        initial_layout=initial_layout,
        seed_transpiler=seed_transpiler,
    )
    circuits = [pass_manager.run(zero), pass_manager.run(one)]

    sampler = SamplerV2(mode=backend)
    if measure_twirling:
        sampler.options.twirling.enable_measure = True
    job = sampler.run(circuits, shots=shots)
    results = job.result()
    measured_zero = _bitstrings_to_samples(
        results[0].data.meas.get_bitstrings(),
        n_qubits,
    )
    measured_one = _bitstrings_to_samples(
        results[1].data.meas.get_bitstrings(),
        n_qubits,
    )
    return ReadoutCalibration(
        backend_name=_backend_name(backend),
        job_id=str(job.job_id()),
        shots=shots,
        p_meas_1_given_0=measured_zero.mean(axis=0),
        p_meas_0_given_1=1.0 - measured_one.mean(axis=0),
    )


def mitigate_readout_moments(
    samples: np.ndarray,
    calibration: ReadoutCalibration,
) -> tuple[np.ndarray, np.ndarray]:
    """Correct first and second moments under an independent assignment-error model."""
    observed = np.asarray(samples, dtype=float)
    if observed.ndim != 2:
        raise ValueError("samples must be a two-dimensional array")
    if observed.shape[1] != len(calibration.p_meas_1_given_0):
        raise ValueError("sample width must match readout calibration")

    observed_marginals = observed.mean(axis=0)
    observed_joint = (observed.T @ observed) / len(observed)
    false_positive = calibration.p_meas_1_given_0
    visibility = 1.0 - false_positive - calibration.p_meas_0_given_1
    if np.any(visibility <= 0.05):
        raise RuntimeError("readout calibration is too ill-conditioned to invert")

    marginals = np.clip(
        (observed_marginals - false_positive) / visibility,
        0.0,
        1.0,
    )
    pairwise = np.diag(marginals)
    for i in range(observed.shape[1]):
        for j in range(i + 1, observed.shape[1]):
            numerator = (
                observed_joint[i, j]
                - false_positive[i] * false_positive[j]
                - false_positive[i] * visibility[j] * marginals[j]
                - false_positive[j] * visibility[i] * marginals[i]
            )
            corrected = numerator / (visibility[i] * visibility[j])
            lower = max(0.0, marginals[i] + marginals[j] - 1.0)
            upper = min(marginals[i], marginals[j])
            pairwise[i, j] = pairwise[j, i] = np.clip(corrected, lower, upper)
    return marginals, pairwise


def _bitstrings_to_samples(bitstrings: list[str], n_qubits: int) -> np.ndarray:
    """Convert Qiskit's ``c[n-1]...c[0]`` strings into columns ``q[0]...q[n-1]``."""
    samples = np.empty((len(bitstrings), n_qubits), dtype=int)
    for row, bitstring in enumerate(bitstrings):
        compact = bitstring.replace(" ", "")
        if len(compact) != n_qubits or set(compact) - {"0", "1"}:
            raise ValueError(f"unexpected measurement bitstring {bitstring!r}")
        samples[row] = [int(bit) for bit in reversed(compact)]
    return samples


def _backend_name(backend: Any) -> str:
    name = getattr(backend, "name", None)
    return str(name() if callable(name) else name)
