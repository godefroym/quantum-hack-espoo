"""Explicit IBM Quantum Runtime execution for fitted Born-machine circuit blocks.

This module is intentionally separate from the generator's normal ``sample`` method:
submitting a cloud job must always be an explicit action. Credentials are loaded from a
saved Qiskit Runtime account or from ``IBM_QUANTUM_TOKEN`` and
``IBM_QUANTUM_INSTANCE`` environment variables.
"""

from __future__ import annotations

from dataclasses import dataclass
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
    two_qubit = target["cz"] if "cz" in target else target["ecr"]
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
