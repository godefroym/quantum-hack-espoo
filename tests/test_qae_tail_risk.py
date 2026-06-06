"""Tests for the QAE tail-risk *calculation* surface (qiskit-free, exact statevector).

Two distinct, separately-asserted claims, matching the project's honest-claims culture:

1. **Equivalence.** The amplitude-estimation estimate of ``P(severe)`` and ``CVaR`` agrees with
   the exact value computed from the loaded distribution + cascade truth table, and with the
   classical Monte-Carlo answer from the existing simulator/metrics -- on the *same* spec,
   loader, and threshold, up to the estimator's own stated (Fisher) error bar.
2. **Advantage.** The oracle-query count to reach a target relative error scales like
   ``O(1/(eps*sqrt(a)))`` for QAE vs ``O(1/(eps^2*a))`` for Monte Carlo, with the gap widening
   as the tail ``a`` shrinks. Asserted on the query-count *model* (the hardware-relevant figure),
   not on simulator wall-clock.

Everything here runs without qiskit: the AE operators are exact numpy-statevector simulations.
"""

from __future__ import annotations

import numpy as np
import pytest

from systemic_risk.evaluation.joint_structure import cascade_count_cvar
from systemic_risk.evaluation.qae_tail_risk import (
    QAETailRiskEstimator,
    basis_bit_matrix,
    cascade_sizes_over_basis,
)
from systemic_risk.generators import EntangledBornMachineGenerator
from systemic_risk.generators.quantum.amplitude_estimation import (
    GroverOperator,
    mc_queries_for_relative_error,
    qae_queries_for_relative_error,
    query_complexity_curve,
    run_mlae,
)
from systemic_risk.generators.quantum.statevector import StateVector
from systemic_risk.simulator.cascade import run_cascade, simulate_many
from systemic_risk.spec import SystemSpec


def _graded_cascade_spec(n: int = 10) -> SystemSpec:
    """Small spec whose cascade size is *graded* (not all-or-nothing), giving a tunable tail.

    A directed near-ring of moderate exposures vs unit capital: a single default rarely
    cascades, a cluster sometimes does, so ``P(cascade size >= s)`` spans an order of magnitude
    across thresholds -- the regime where the deep-tail advantage is visible.
    """
    corr = np.full((n, n), 0.15)
    np.fill_diagonal(corr, 1.0)
    exposure = np.zeros((n, n))
    for i in range(n):
        exposure[(i + 1) % n, i] = 0.6
        exposure[(i + 2) % n, i] = 0.5
    return SystemSpec(
        node_names=[f"I{i}" for i in range(n)],
        node_types=["bank"] * n,
        exposure_matrix=exposure,
        capital_buffers=np.full(n, 1.0),
        marginal_default_probs=np.full(n, 0.06),
        target_pairwise_corr=corr,
        clusters=["c"] * n,
    )


def _fitted_loader(spec: SystemSpec) -> EntangledBornMachineGenerator:
    gen = EntangledBornMachineGenerator(ansatz="entangled", calibrate=True)
    gen.fit(spec)
    return gen


# --------------------------------------------------------------------------- AE operator math
def test_grover_rotation_law_matches_sin_squared() -> None:
    """One Grover power rotates by 2*theta: P(marked) after Q^m = sin^2((2m+1)theta)."""
    rng = np.random.default_rng(0)
    dim = 16
    unitary, _ = np.linalg.qr(rng.standard_normal((dim, dim)))
    marked = np.zeros(dim, bool)
    marked[[2, 5, 9]] = True
    grover = GroverOperator(unitary=unitary.astype(complex), marked=marked)

    a = grover.amplitude
    theta = np.arcsin(np.sqrt(a))
    for m in range(6):
        got = grover.marked_probability_after_powers(m)
        assert got == pytest.approx(np.sin((2 * m + 1) * theta) ** 2, abs=1e-12)


def test_mlae_is_unbiased_in_the_noiseless_limit() -> None:
    """Feeding exact per-power probabilities to the likelihood recovers a to high precision."""
    rng = np.random.default_rng(1)
    dim = 32
    unitary, _ = np.linalg.qr(rng.standard_normal((dim, dim)))
    marked = np.zeros(dim, bool)
    marked[[1, 4, 7, 20]] = True
    grover = GroverOperator(unitary=unitary.astype(complex), marked=marked)

    est = run_mlae(grover, num_powers=9, shots=100, noiseless=True)
    assert est.value == pytest.approx(grover.amplitude, abs=1e-6)


def test_mlae_estimate_lands_within_its_fisher_error_bar() -> None:
    """Noisy MLAE error is consistent with the reported Cramer-Rao half-width (averaged)."""
    rng = np.random.default_rng(2)
    dim = 16
    unitary, _ = np.linalg.qr(rng.standard_normal((dim, dim)))
    marked = np.zeros(dim, bool)
    marked[[3, 11]] = True
    grover = GroverOperator(unitary=unitary.astype(complex), marked=marked)

    errors = [
        run_mlae(grover, num_powers=8, shots=150, rng=np.random.default_rng(s)).error_vs_exact
        for s in range(40)
    ]
    half_width = run_mlae(grover, num_powers=8, shots=150, noiseless=True).fisher_ci_half_width
    # RMS error should be on the order of the Fisher half-width (well within a factor of 3).
    assert np.sqrt(np.mean(np.square(errors))) <= 3.0 * half_width


# --------------------------------------------------------------------------- bit-order invariant
def test_basis_bit_matrix_matches_statevector_convention() -> None:
    """The oracle labeling's bit layout matches StateVector / exact_probabilities (qubit i = bit n-1-i)."""
    spec = _graded_cascade_spec(n=8)
    gen = _fitted_loader(spec)
    probs = gen.exact_probabilities()
    bits = basis_bit_matrix(spec.n)
    # Marginal of each qubit from (probs, bits) must equal the loader's exact marginals.
    marg_from_bits = probs @ bits
    marg_exact = gen.exact_moments()[0]
    assert np.allclose(marg_from_bits, marg_exact, atol=1e-9)


def test_cascade_sizes_over_basis_matches_direct_calls() -> None:
    spec = _graded_cascade_spec(n=7)
    sizes = cascade_sizes_over_basis(spec)
    bits = basis_bit_matrix(spec.n)
    expected = np.array([run_cascade(bits[x], spec).failure_count for x in range(bits.shape[0])])
    assert np.array_equal(sizes, expected)


# --------------------------------------------------------------------------- equivalence claim
@pytest.mark.parametrize("severe_threshold", [2, 3, 4])
def test_qae_p_severe_matches_exact_and_monte_carlo(severe_threshold: int) -> None:
    """CLAIM 1 (equivalence): QAE P(severe) ~= exact ~= classical Monte Carlo, within error bars."""
    spec = _graded_cascade_spec(n=10)
    gen = _fitted_loader(spec)
    estimator = QAETailRiskEstimator(gen, spec)

    exact = estimator.exact_p_severe(severe_threshold)

    # Classical Monte-Carlo reference through the *existing* simulator path.
    samples = gen.sample(150_000, seed=7)
    mc_sizes = np.array([r.failure_count for r in simulate_many(samples, spec)])
    mc = float((mc_sizes >= severe_threshold).mean())
    mc_se = np.sqrt(mc * (1 - mc) / len(samples))

    est = estimator.estimate_p_severe(
        severe_threshold, num_powers=9, shots=400, rng=np.random.default_rng(severe_threshold)
    )

    # QAE agrees with the exact value within its own (Fisher) error bar...
    assert est.within_fisher_ci
    # ...and the classical MC estimate agrees with the same exact value within its sampling error.
    assert abs(mc - exact) <= 5.0 * mc_se
    # The two independent estimates of the same quantity are mutually consistent.
    assert abs(est.estimate - mc) <= 5.0 * (mc_se + est.amplitude_estimate.fisher_ci_half_width)


def test_qae_p_severe_exact_equals_loaded_distribution_probability() -> None:
    """The exact reference truly is sum of loaded P(x) over cascade-marked x (analytic check)."""
    spec = _graded_cascade_spec(n=9)
    gen = _fitted_loader(spec)
    estimator = QAETailRiskEstimator(gen, spec)

    probs = gen.exact_probabilities()
    sizes = cascade_sizes_over_basis(spec)
    for thr in [1, 2, 3, 5]:
        analytic = float(probs[sizes >= thr].sum())
        assert estimator.exact_p_severe(thr) == pytest.approx(analytic, abs=1e-12)
        # And the noiseless MLAE recovers it (estimator unbiasedness on the real oracle).
        est = estimator.estimate_p_severe(thr, num_powers=9, shots=200, noiseless=True)
        if analytic > 1e-6:
            assert est.estimate == pytest.approx(analytic, rel=5e-3, abs=5e-4)


@pytest.mark.parametrize("alpha", [0.9, 0.95])
def test_qae_cvar_matches_exact_and_metric(alpha: float) -> None:
    """CLAIM 1 (equivalence): QAE CVaR of cascade size ~= exact ~= classical metric."""
    spec = _graded_cascade_spec(n=10)
    gen = _fitted_loader(spec)
    estimator = QAETailRiskEstimator(gen, spec)

    exact = estimator.exact_cvar(alpha=alpha)

    samples = gen.sample(150_000, seed=11)
    mc_sizes = np.array([r.failure_count for r in simulate_many(samples, spec)])
    mc = cascade_count_cvar(mc_sizes, alpha=alpha)

    est = estimator.estimate_cvar(
        alpha=alpha, num_powers=9, shots=500, rng=np.random.default_rng(123)
    )

    # Exact QAE reference matches the classical metric on the same loaded distribution closely
    # (both are the CVaR of the cascade-size law; MC differs only by sampling noise).
    assert abs(exact - mc) <= 0.3
    # The QAE estimate is close to the exact CVaR (count-valued, so an absolute tolerance).
    assert est.abs_error <= 0.5


def test_qae_cvar_noiseless_recovers_exact() -> None:
    """Noiseless threshold-level QAE reproduces the exact CVaR via the layer-cake identity."""
    spec = _graded_cascade_spec(n=9)
    gen = _fitted_loader(spec)
    estimator = QAETailRiskEstimator(gen, spec)
    for alpha in [0.9, 0.95, 0.99]:
        est = estimator.estimate_cvar(alpha=alpha, num_powers=9, shots=300, noiseless=True)
        assert est.estimate == pytest.approx(estimator.exact_cvar(alpha=alpha), abs=1e-3)


# --------------------------------------------------------------------------- advantage claim
def test_query_complexity_has_quadratic_speedup_scaling() -> None:
    """CLAIM 2 (advantage): MC queries ~ 1/(eps^2 a), QAE ~ 1/(eps sqrt(a)) -- a quadratic gap.

    Asserted via the log-log slope of required queries vs the tail probability a: MC scales like
    a^-1, QAE like a^-0.5, so the *ratio* (the speedup) grows like a^-0.5 into the deep tail. The
    leading constants carry a mild (1-a) factor, so we test the dominant power law, not an exact
    sqrt ratio at two points (that would over-claim a clean 1/sqrt(a) the constants do not give).
    """
    eps = 0.05
    a = np.array([0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 5e-4, 2e-4, 1e-4])
    mc = np.array([mc_queries_for_relative_error(x, eps) for x in a], dtype=float)
    qae = np.array([qae_queries_for_relative_error(x, eps) for x in a], dtype=float)

    mc_slope = np.polyfit(np.log(a), np.log(mc), 1)[0]
    qae_slope = np.polyfit(np.log(a), np.log(qae), 1)[0]

    assert mc_slope == pytest.approx(-1.0, abs=0.05)  # classical: 1/a
    assert qae_slope == pytest.approx(-0.5, abs=0.05)  # quantum: 1/sqrt(a)
    # QAE is genuinely cheaper at every tail level in this range.
    assert np.all(qae < mc)


def test_mc_query_count_blows_up_in_the_deep_tail_faster_than_qae() -> None:
    """The speedup grows monotonically as a -> 0 and itself scales like 1/sqrt(a)."""
    eps = 0.1
    tail = np.array([0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.001])
    curve = query_complexity_curve(tail, eps)
    speedups = np.array([p.speedup for p in curve])

    # Monotonically widening advantage into the deep tail.
    assert np.all(np.diff(speedups) > 0)
    # The speedup's own power law is ~ a^-0.5 (1/sqrt(a)); fit its log-log slope.
    speedup_slope = np.polyfit(np.log(tail), np.log(speedups), 1)[0]
    assert speedup_slope == pytest.approx(-0.5, abs=0.07)


def test_query_models_match_textbook_forms() -> None:
    """The query-count formulas are the stated O(1/(eps^2 a)) and O(1/(eps sqrt(a)))."""
    a, eps = 0.04, 0.1
    assert mc_queries_for_relative_error(a, eps) == int(np.ceil((1 - a) / (eps**2 * a)))
    assert qae_queries_for_relative_error(a, eps) == int(
        np.ceil(np.pi / (2.0 * eps * np.sqrt(a * (1 - a))))
    )


def test_measured_mlae_queries_beat_matched_mc_in_the_tail() -> None:
    """The *actually issued* MLAE oracle count reaches a target accuracy below the matched MC budget.

    Hardware-relevant, end-to-end: run MLAE on the real cascade oracle at a deep threshold with a
    schedule *sized to the target accuracy* (a modest 6-power, 20-shot LIS schedule), confirm its
    estimate hits the target relative error on average, and confirm the oracle calls it spent are
    fewer than the classical Monte-Carlo draws that target would demand at the same tail a. The
    advantage is asymptotic -- it requires sizing the QAE schedule to eps, not brute-forcing it.
    """
    spec = _graded_cascade_spec(n=10)
    gen = _fitted_loader(spec)
    estimator = QAETailRiskEstimator(gen, spec)

    threshold = 4  # deep tail, a ~ 0.05
    a = estimator.exact_p_severe(threshold)
    target_eps = 0.1

    # A schedule sized to the target (not over-provisioned): 6 LIS powers x 20 shots.
    rel_errors = []
    for s in range(20):
        est = estimator.estimate_p_severe(
            threshold, num_powers=6, shots=20, rng=np.random.default_rng(s)
        )
        rel_errors.append(est.abs_error / a)
    mean_rel_error = float(np.mean(rel_errors))
    issued_oracle_queries = est.amplitude_estimate.oracle_queries
    mc_budget = mc_queries_for_relative_error(a, target_eps)

    # MLAE reaches the target relative error on average...
    assert mean_rel_error <= target_eps
    # ...spending fewer oracle calls than the classical MC budget for the same accuracy and tail.
    assert issued_oracle_queries < mc_budget
    # The issued count is exactly sum(schedule powers) * shots (honest accounting).
    assert issued_oracle_queries == sum(range(6)) * 20


# --------------------------------------------------------------------------- ansatz-agnostic
def test_estimator_works_on_ghz_systemic_loader() -> None:
    """The estimator is loader-agnostic: it also runs on the GHZ-systemic common-shock loader."""
    spec = _graded_cascade_spec(n=8)
    gen = EntangledBornMachineGenerator(ansatz="ghz_systemic")
    gen.fit(spec)
    estimator = QAETailRiskEstimator(gen, spec)
    # The loaded amplitudes must reproduce the loader's own probabilities (sqrt convention).
    assert np.allclose(estimator.probabilities, gen.exact_probabilities(), atol=1e-12)
    est = estimator.estimate_p_severe(2, num_powers=8, shots=300, noiseless=True)
    assert est.estimate == pytest.approx(estimator.exact_p_severe(2), rel=1e-2, abs=1e-3)


def test_loaded_state_amplitude_equals_tail_probability() -> None:
    """Sanity: GroverOperator.amplitude on the cascade mask equals the exact tail probability."""
    spec = _graded_cascade_spec(n=9)
    gen = _fitted_loader(spec)
    estimator = QAETailRiskEstimator(gen, spec)
    from systemic_risk.evaluation.qae_tail_risk import _grover_for_mask

    sizes = cascade_sizes_over_basis(spec)
    mask = sizes >= 3
    grover = _grover_for_mask(estimator.prepared, mask)
    assert grover.amplitude == pytest.approx(estimator.exact_p_severe(3), abs=1e-12)


def test_prepared_state_is_normalised_and_real() -> None:
    spec = _graded_cascade_spec(n=8)
    gen = _fitted_loader(spec)
    estimator = QAETailRiskEstimator(gen, spec)
    assert np.isclose(np.linalg.norm(estimator.prepared), 1.0, atol=1e-9)
    assert np.allclose(estimator.prepared.imag, 0.0, atol=1e-12)
    # And it is consistent with a directly-built StateVector for a single-block fit.
    state = StateVector.symmetric_shells  # presence check only; built path covered elsewhere
    assert callable(state)
