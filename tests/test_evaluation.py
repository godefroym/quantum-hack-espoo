from __future__ import annotations

from systemic_risk.data import make_synthetic_system
from systemic_risk.evaluation import EvaluationHarness
from systemic_risk.generators import BernoulliGenerator, EntangledPQCGenerator


def test_evaluation_harness_returns_comparison_rows() -> None:
    spec = make_synthetic_system(n=12, seed=5)
    harness = EvaluationHarness(spec, n_samples=80, seed=10)
    results = harness.run(
        [
            BernoulliGenerator(),
            EntangledPQCGenerator(gibbs_sweeps=3, burn_in=5),
        ]
    )
    frame = harness.to_frame(results)

    assert len(results) == 2
    assert {
        "generator",
        "mean_cascade_size",
        "mean_cascade_depth",
        "systemic_collapse_frequency",
        "p_severe_cascade",
    }.issubset(frame.columns)
