from __future__ import annotations

import numpy as np
import pytest

from systemic_risk.generators.quantum.ibm_runtime import (
    DEFAULT_HARDWARE_SHOTS,
    ReadoutCalibration,
    _bitstrings_to_samples,
    _split_shots,
    dependency_aware_layout,
    mitigate_readout_moments,
)
from systemic_risk.generators.quantum.qiskit_backend import build_circuit
from systemic_risk.spec import SystemSpec


def test_default_hardware_shot_budget_is_one_hundred_thousand() -> None:
    assert DEFAULT_HARDWARE_SHOTS == 100_000


def test_large_hardware_run_is_split_at_backend_limit() -> None:
    assert _split_shots(1_000_000, 100_000) == (100_000,) * 10
    assert _split_shots(250_001, 100_000) == (100_000, 100_000, 50_001)


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


def test_readout_mitigation_recovers_known_moments() -> None:
    rng = np.random.default_rng(11)
    latent = (rng.random((200_000, 2)) < np.array([0.15, 0.35])).astype(int)
    latent[:, 1] |= latent[:, 0]
    p10 = np.array([0.04, 0.06])
    p01 = np.array([0.02, 0.03])
    observed = latent.copy()
    false_positive = (latent == 0) & (rng.random(latent.shape) < p01)
    false_negative = (latent == 1) & (rng.random(latent.shape) < p10)
    observed[false_positive] = 1
    observed[false_negative] = 0
    calibration = ReadoutCalibration(
        backend_name="fake",
        job_id="cal",
        shots=len(latent),
        p_meas_1_given_0=p01,
        p_meas_0_given_1=p10,
    )

    marginals, pairwise = mitigate_readout_moments(observed, calibration)

    expected_marginals = latent.mean(axis=0)
    expected_joint = (latent.T @ latent) / len(latent)
    assert np.allclose(marginals, expected_marginals, atol=0.003)
    assert abs(pairwise[0, 1] - expected_joint[0, 1]) < 0.003


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


def test_dependency_aware_layout_uses_native_backend_edges() -> None:
    pytest.importorskip("qiskit_ibm_runtime")
    from qiskit_ibm_runtime.fake_provider import FakeOslo

    backend = FakeOslo()
    dependency = np.full((4, 4), 0.2)
    np.fill_diagonal(dependency, 0.0)
    layout, logical_edges, _ = dependency_aware_layout(
        backend,
        dependency,
        annealing_steps=100,
    )
    gate = next(name for name in ("cz", "ecr", "cx") if name in backend.target)
    native = {tuple(sorted(pair)) for pair in backend.target[gate]}

    assert len(layout) == 4
    assert logical_edges
    assert all(
        tuple(sorted((layout[source], layout[target]))) in native
        for source, target in logical_edges
    )
