from __future__ import annotations

import numpy as np
import pytest

from systemic_risk.generators.quantum.ibm_runtime import (
    DEFAULT_HARDWARE_SHOTS,
    _bitstrings_to_samples,
)
from systemic_risk.generators.quantum.qiskit_backend import build_circuit
from systemic_risk.spec import SystemSpec


def test_default_hardware_shot_budget_is_one_hundred_thousand() -> None:
    assert DEFAULT_HARDWARE_SHOTS == 100_000


def test_ibm_bitstrings_are_returned_in_institution_column_order() -> None:
    samples = _bitstrings_to_samples(["0011", "1000"], n_qubits=4)
    assert np.array_equal(
        samples,
        np.array(
            [
                [1, 1, 0, 0],
                [0, 0, 0, 1],
            ]
        ),
    )


def test_qiskit_hardware_circuit_contains_measurements() -> None:
    pytest.importorskip("qiskit")
    circuit = build_circuit(
        np.array([0.1, 0.2]),
        [(0, 1)],
        np.array([0.3]),
        measure=True,
    )
    assert circuit.num_qubits == 2
    assert circuit.count_ops()["measure"] == 2


def test_ibm_runtime_executes_against_small_fake_backend() -> None:
    pytest.importorskip("qiskit_ibm_runtime")
    from qiskit_ibm_runtime.fake_provider import FakeOslo

    from systemic_risk.generators import EntangledBornMachineGenerator
    from systemic_risk.generators.quantum.ibm_runtime import run_block

    class FakeService:
        def least_busy(self, **kwargs):
            return FakeOslo()

    corr = np.full((4, 4), 0.1)
    np.fill_diagonal(corr, 1.0)
    spec = SystemSpec(
        node_names=["A", "B", "C", "D"],
        node_types=["bank"] * 4,
        exposure_matrix=np.zeros((4, 4)),
        capital_buffers=np.ones(4),
        marginal_default_probs=np.array([0.08, 0.12, 0.16, 0.20]),
        target_pairwise_corr=corr,
        clusters=["test"] * 4,
    )
    generator = EntangledBornMachineGenerator()
    generator.fit(spec)
    result = run_block(generator.blocks_[0], shots=32, service=FakeService())

    assert result.backend_name == "fake_oslo"
    assert result.samples.shape == (32, 4)
    assert result.circuit_depth > 0
    assert result.two_qubit_gates > 0
