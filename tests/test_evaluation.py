from __future__ import annotations

import numpy as np

from systemic_risk.data import make_huang_2008_style_system, make_synthetic_system
from systemic_risk.evaluation import (
    EvaluationHarness,
    ExposureCascadeChannel,
    HuangFireSaleChannel,
)
from systemic_risk.generators import BernoulliGenerator, EntangledPQCGenerator
from systemic_risk.simulator import CascadeResult, HuangCascadeResult


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
    assert all(isinstance(r, CascadeResult) for r in results[0].cascade_results)
    assert {
        "generator",
        "mean_cascade_size",
        "mean_cascade_depth",
        "systemic_collapse_frequency",
        "p_severe_cascade",
    }.issubset(frame.columns)


def test_bare_spec_uses_exposure_cascade_channel() -> None:
    spec = make_synthetic_system(n=12, seed=5)
    harness = EvaluationHarness(spec, n_samples=40, seed=10)

    assert isinstance(harness.channel, ExposureCascadeChannel)
    assert harness.spec is spec


def test_harness_evaluates_huang_fire_sale_channel_end_to_end() -> None:
    bank_asset_spec = make_huang_2008_style_system(n_banks=12, seed=7)
    channel = HuangFireSaleChannel(
        bank_asset_spec,
        asset_price_shocks={
            "construction_and_land_development": 0.90,
            "nonfarm_nonresidential": 0.95,
        },
        alpha=0.08,
        eta=0.0,
        seed=11,
    )
    harness = EvaluationHarness(channel, n_samples=120, seed=3)
    results = harness.run([BernoulliGenerator()])

    assert harness.spec.n == bank_asset_spec.n_banks
    assert len(results) == 1
    cascades = results[0].cascade_results
    assert len(cascades) == 120
    assert all(isinstance(r, HuangCascadeResult) for r in cascades)
    # The same tail metrics are produced as for the exposure cascade.
    assert {
        "mean_cascade_size",
        "p_severe_cascade",
        "cascade_count_cvar_95",
        "aggregate_tail_dependence",
    }.issubset(results[0].metrics)


def test_explicit_exposure_channel_matches_bare_spec_path() -> None:
    spec = make_synthetic_system(n=12, seed=5)
    via_spec = EvaluationHarness(spec, n_samples=60, seed=10).run([BernoulliGenerator()])
    via_channel = EvaluationHarness(
        ExposureCascadeChannel(spec), n_samples=60, seed=10
    ).run([BernoulliGenerator()])

    assert np.array_equal(via_spec[0].samples, via_channel[0].samples)
    assert via_spec[0].metrics == via_channel[0].metrics


def test_evaluation_harness_can_skip_expensive_joint_structure() -> None:
    spec = make_synthetic_system(n=12, seed=5)
    harness = EvaluationHarness(
        spec,
        n_samples=80,
        seed=10,
        include_joint_structure=False,
    )
    result = harness.run([BernoulliGenerator()])[0]

    assert "p_severe_cascade" in result.metrics
    assert "excess_coskewness_rms" not in result.metrics
