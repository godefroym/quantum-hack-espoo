from __future__ import annotations

import numpy as np
import pytest

from systemic_risk.data_network import (
    QPU_NOISE_FLOOR,
    apply_stress,
    build_network_spec,
    stressed_marginals,
)
from systemic_risk.data_network.stress import GFC_SPREAD_MULTIPLIER


@pytest.fixture(scope="module")
def base_spec():
    return build_network_spec(prefer_snapshot=True).to_system_spec()


def test_stressed_marginals_hit_target_mean():
    base = np.array([1e-5, 0.0006, 0.0008, 0.0031, 0.0139])
    p, calib = stressed_marginals(base, target_mean_pd=0.15, crisis_floor=0.0)
    assert p.mean() == pytest.approx(0.15, abs=1e-6)
    assert calib.stressed_mean == pytest.approx(0.15, abs=1e-4)


def test_stress_is_rank_preserving():
    base = np.array([1e-5, 0.0006, 0.0008, 0.0031, 0.0139])
    p, _ = stressed_marginals(base, target_mean_pd=0.15, crisis_floor=0.0)
    # logit shift is monotone, so order is preserved exactly
    assert np.array_equal(np.argsort(base), np.argsort(p))
    assert np.all(np.diff(p[np.argsort(base)]) >= 0)


def test_crisis_floor_lifts_everything_above_noise_floor():
    base = np.array([1e-5, 0.0006, 0.0008, 0.0031, 0.0139])
    p, calib = stressed_marginals(base, target_mean_pd=0.15, crisis_floor=QPU_NOISE_FLOOR)
    assert p.min() >= QPU_NOISE_FLOOR - 1e-12
    assert calib.n_below_noise_floor == 0
    # floor must not break the target mean
    assert p.mean() == pytest.approx(0.15, abs=1e-6)


def test_apply_stress_keeps_correlation_and_drops_stale_joint(base_spec):
    stressed, calib = apply_stress(base_spec)
    assert stressed.n == base_spec.n
    # correlation graph is byte-identical (only marginals lifted)
    assert np.allclose(stressed.target_pairwise_corr, base_spec.target_pairwise_corr)
    # the joint is re-derived from (correlation, marginals), not frozen
    assert stressed.target_joint_probs is None
    # marginals rose and clear the floor
    p = np.asarray(stressed.marginal_default_probs)
    assert p.mean() == pytest.approx(0.15, abs=1e-3)
    assert np.all(p >= QPU_NOISE_FLOOR - 1e-9)
    assert calib.n_above_noise_floor == stressed.n
    stressed.validate()


def test_joint_re_derives_coherently_with_higher_marginals(base_spec):
    """Co-default probabilities rise consistently with the lifted marginals at fixed coupling."""
    stressed, _ = apply_stress(base_spec)
    jb = base_spec.target_pairwise_joint_probs()
    js = stressed.target_pairwise_joint_probs()
    iu = np.triu_indices(base_spec.n, k=1)
    # off-diagonal co-default mass strictly increases under stress
    assert js[iu].mean() > jb[iu].mean()
    # diagonal equals the stressed marginals (joint stays internally consistent)
    assert np.allclose(np.diag(js), stressed.marginal_default_probs)


def test_target_mean_is_parameterizable(base_spec):
    stressed, _ = apply_stress(base_spec, target_mean_pd=0.10, crisis_floor=0.0)
    assert float(np.mean(stressed.marginal_default_probs)) == pytest.approx(0.10, abs=1e-3)


def test_calibration_metadata_recorded(base_spec):
    stressed, calib = apply_stress(base_spec)
    meta = stressed.metadata["stress"]
    assert meta["target_mean_pd"] == 0.15
    assert meta["correlation_unchanged"] is True
    assert meta["ordering_preserved"] is True
    assert calib.gfc_spread_multiplier == GFC_SPREAD_MULTIPLIER


def test_invalid_arguments_raise():
    base = np.array([0.001, 0.01])
    with pytest.raises(ValueError):
        stressed_marginals(base, target_mean_pd=1.5)
    with pytest.raises(ValueError):
        stressed_marginals(base, target_mean_pd=0.15, crisis_floor=0.2)
