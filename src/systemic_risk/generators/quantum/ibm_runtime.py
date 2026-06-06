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


@dataclass(frozen=True)
class IBMHardwareResult:
    backend_name: str
    job_id: str
    shots: int
    samples: np.ndarray
    circuit_depth: int
    circuit_operations: dict[str, int]
    two_qubit_gates: int


def run_block(
    block: EntangledCircuit,
    *,
    shots: int = 4096,
    backend_name: str | None = None,
    optimization_level: int = 1,
    initial_layout: list[int] | None = None,
    dynamical_decoupling: bool = False,
    measure_twirling: bool = False,
    gate_twirling: bool = False,
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
    job = sampler.run([isa_circuit], shots=shots)
    pub_result = job.result()[0]
    bitstrings = pub_result.data.meas.get_bitstrings()
    samples = _bitstrings_to_samples(bitstrings, block.size)

    operations = {str(name): int(count) for name, count in isa_circuit.count_ops().items()}
    two_qubit_gates = sum(
        1 for instruction in isa_circuit.data if len(instruction.qubits) == 2
    )
    return IBMHardwareResult(
        backend_name=_backend_name(backend),
        job_id=str(job.job_id()),
        shots=shots,
        samples=samples,
        circuit_depth=int(isa_circuit.depth()),
        circuit_operations=operations,
        two_qubit_gates=two_qubit_gates,
    )


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
