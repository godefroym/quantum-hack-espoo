import importlib.util
from pathlib import Path

import numpy as np
import pytest

from systemic_risk.generators import EntangledBornMachineGenerator

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_ibm_quantum_test.py"
SCRIPT_SPEC = importlib.util.spec_from_file_location("run_ibm_quantum_test", SCRIPT_PATH)
assert SCRIPT_SPEC is not None and SCRIPT_SPEC.loader is not None
SCRIPT_MODULE = importlib.util.module_from_spec(SCRIPT_SPEC)
SCRIPT_SPEC.loader.exec_module(SCRIPT_MODULE)
hardware_test_spec = SCRIPT_MODULE.hardware_test_spec


@pytest.mark.parametrize("n_qubits", [4, 6, 8])
def test_hardware_test_spec_has_requested_size(n_qubits: int) -> None:
    spec = hardware_test_spec(n_qubits)

    assert spec.n == n_qubits
    assert np.allclose(spec.marginal_default_probs, np.linspace(0.08, 0.20, n_qubits))
    assert spec.target_pairwise_corr.shape == (n_qubits, n_qubits)
    assert np.allclose(np.diag(spec.target_pairwise_corr), 1.0)


def test_hardware_test_spec_rejects_unsupported_size() -> None:
    with pytest.raises(ValueError, match="4, 6, or 8"):
        hardware_test_spec(10)


def test_degree_limited_eight_qubit_block_respects_cap() -> None:
    generator = EntangledBornMachineGenerator(
        ansatz="entangled",
        backend="statevector",
        max_degree=2,
    )
    generator.fit(hardware_test_spec(8))

    assert len(generator.blocks_) == 1
    degrees = np.zeros(8, dtype=int)
    for source, target in generator.blocks_[0].edges:
        degrees[source] += 1
        degrees[target] += 1
    assert degrees.max() <= 2
