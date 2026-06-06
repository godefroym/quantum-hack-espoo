from __future__ import annotations

import numpy as np

from systemic_risk.edge_metrics import EdgeMetricConfig, compute_edge_metrics


def test_components_zero_on_absent_edges() -> None:
    W = np.array([[0.0, 1.0], [0.0, 0.0]])
    em = compute_edge_metrics(W, ["bank", "corporate"])
    assert em.effective[0, 1] > 0.0
    # No edge anywhere else -> every component zero there.
    for m in (em.effective, em.lgd, em.maturity_stress, em.wrong_way, em.substitutability):
        assert m[1, 0] == 0.0 and m[0, 0] == 0.0 and m[1, 1] == 0.0


def test_lgd_reduces_loss_below_notional_when_no_amplifiers() -> None:
    # Long-maturity, uncorrelated, single counterparty: only LGD and (mild) concentration act.
    W = np.array([[0.0, 1.0], [1.0, 0.0]])
    cfg = EdgeMetricConfig(rollover_beta=0.0, wrong_way_gamma=0.0, concentration_delta=0.0)
    em = compute_edge_metrics(W, ["bank", "bank"], config=cfg)
    assert np.all(em.recovery[W > 0] > 0) and np.all(em.recovery[W > 0] < 1)
    # effective == notional * lgd  (lgd < 1) -> strictly smaller.
    assert em.effective[0, 1] < W[0, 1]


def test_directionality_makes_effective_asymmetric() -> None:
    # Symmetric notional but different debtor types -> asymmetric effective loss.
    W = np.array([[0.0, 1.0], [1.0, 0.0]])
    em = compute_edge_metrics(W, ["CCP", "corporate"])
    assert not np.isclose(em.effective[0, 1], em.effective[1, 0])


def test_short_maturity_raises_rollover_stress() -> None:
    W = np.array([[0.0, 1.0], [1.0, 0.0]])
    em = compute_edge_metrics(W, ["CCP", "corporate"])  # debtor 0=CCP short, 1=corporate long
    # Edge whose debtor is the short-tenor CCP carries more rollover stress.
    assert em.maturity_stress[1, 0] > em.maturity_stress[0, 1]


def test_wrong_way_increases_with_correlation() -> None:
    W = np.array([[0.0, 1.0], [1.0, 0.0]])
    lo = compute_edge_metrics(W, ["bank", "bank"],
                              correlation=np.array([[1.0, 0.0], [0.0, 1.0]]))
    hi = compute_edge_metrics(W, ["bank", "bank"],
                              correlation=np.array([[1.0, 0.9], [0.9, 1.0]]))
    assert hi.effective[0, 1] > lo.effective[0, 1]


def test_collateralized_debtor_recovers_more() -> None:
    # A claim on a CCP (highly collateralized) loses less than a claim on a fund.
    W = np.array([[0.0, 1.0, 1.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    em = compute_edge_metrics(W, ["bank", "CCP", "fund"])
    assert em.recovery[0, 1] > em.recovery[0, 2]   # CCP recovery > fund recovery


def test_substitutability_amplifies_non_substitutable_provider() -> None:
    W = np.array([[0.0, 1.0, 1.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    em = compute_edge_metrics(W, ["bank", "CCP", "bank"])
    assert em.substitutability[0, 1] > em.substitutability[0, 2]  # CCP non-substitutable
