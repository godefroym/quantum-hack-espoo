from __future__ import annotations

from systemic_risk.data_network.assemble import build_synthetic_system_spec
from scenario_generation.qcbm_trainer import QCBMTrainer


def test_qcbm_trainer_runs():
    spec = build_synthetic_system_spec(n=6, seed=2)
    trainer = QCBMTrainer(maxiter=5)
    results = trainer.fit(spec)
    assert isinstance(results, dict)
    # at least one block result
    assert len(results) >= 1
