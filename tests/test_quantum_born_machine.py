"""Tests for the entangled, quantum-native Born-machine scenario generator.

Covers the generator contract, the analytic angle-setting facts it rests on, the exact
statevector readout, the ``n = 54`` mean-field-oracle validation, and the honest
higher-order / tail-dependence claim. The discrimination statistics are computed locally
so this file does not depend on the evaluation package; the Gaussian foil is a Monte-Carlo
copula fit to each generator's *own* realized marginals and correlation, which is the only
honest second-order reference (an under-correlated foil would conflate a missing-correlation
artifact with genuine higher-order structure).
"""

from __future__ import annotations

import numpy as np
import pytest

from systemic_risk.data import make_synthetic_system
from systemic_risk.generators import EntangledBornMachineGenerator, EntangledPQCGenerator
from systemic_risk.generators.base import sample_diagnostics
from systemic_risk.generators.gaussian_copula import GaussianCopulaGenerator
from systemic_risk.generators.quantum import ansatz as A
from systemic_risk.generators.quantum.statevector import StateVector, sample_bitstrings
from systemic_risk.models.ising import LossDistribution
from systemic_risk.models.mean_field_oracle import (
    MeanFieldIsingOracle,
    total_variation_distance,
)
from systemic_risk.spec import SystemSpec


def _homogeneous_spec(n: int, p: float, rho: float) -> SystemSpec:
    corr = np.full((n, n), rho)
    np.fill_diagonal(corr, 1.0)
    return SystemSpec(
        node_names=[f"I{i}" for i in range(n)],
        node_types=["bank"] * n,
        exposure_matrix=np.zeros((n, n)),
        capital_buffers=np.ones(n),
        marginal_default_probs=np.full(n, p),
        target_pairwise_corr=corr,
        clusters=["c"] * n,
    )


def _coskewness_rms(samples: np.ndarray) -> float:
    x = samples.astype(float)
    centered = x - x.mean(axis=0)
    scale = np.sqrt((centered**2).mean(axis=0))
    scale = np.where(scale > 0, scale, 1.0)
    n = x.shape[1]
    values = [
        (centered[:, i] * centered[:, j] * centered[:, k]).mean()
        / (scale[i] * scale[j] * scale[k])
        for i in range(n)
        for j in range(i + 1, n)
        for k in range(j + 1, n)
    ]
    return float(np.sqrt(np.mean(np.square(values)))) if values else 0.0


def _matched_gaussian_foil(samples: np.ndarray, seed: int) -> np.ndarray:
    """Gaussian copula calibrated to ``samples``' OWN realized marginals + correlation.

    This is the honest second-order reference: by construction it matches the first two
    moments of the sample, so any leftover co-skewness difference is genuine higher-order
    structure rather than a correlation mismatch.
    """
    n = samples.shape[1]
    corr = sample_diagnostics(samples).sampled_pairwise_corr.copy()
    np.fill_diagonal(corr, 1.0)
    spec = SystemSpec(
        node_names=[f"I{i}" for i in range(n)],
        node_types=["bank"] * n,
        exposure_matrix=np.zeros((n, n)),
        capital_buffers=np.ones(n),
        marginal_default_probs=samples.mean(axis=0),
        target_pairwise_corr=corr,
        clusters=["c"] * n,
    )
    gaussian = GaussianCopulaGenerator()
    gaussian.fit(spec)
    return gaussian.sample(samples.shape[0], seed=seed)


def _excess_coskewness(samples: np.ndarray, foil_seed: int) -> float:
    """Co-skewness rms of ``samples`` minus that of a moment-matched Gaussian copula.

    Near zero for any second-order (elliptical) model; positive only for a genuinely
    non-elliptical joint whose third-order co-default cannot be pinned by marginals +
    correlation.
    """
    return _coskewness_rms(samples) - _coskewness_rms(_matched_gaussian_foil(samples, foil_seed))


# --------------------------------------------------------------- analytic angle facts
def test_marginal_angle_reproduces_probability_exactly() -> None:
    p = np.array([0.01, 0.1, 0.5, 0.9])
    state = StateVector(len(p))
    for qubit, theta in enumerate(A.marginal_angles(p)):
        state.ry(qubit, float(theta))
    assert np.allclose(state.marginals(), p, atol=1e-12)


def test_cry_covariance_matches_closed_form() -> None:
    p_i, p_j, target_cov = 0.2, 0.2, 0.03
    alpha = A.cry_angle(p_i, p_j, target_cov)
    state = StateVector(2)
    state.ry(0, float(A.marginal_angles(np.array([p_i]))[0]))
    state.ry(1, float(A.marginal_angles(np.array([p_j]))[0]))
    state.cry(0, 1, alpha)
    joint = state.pairwise_joint()
    marginals = state.marginals()
    cov = joint[0, 1] - marginals[0] * marginals[1]
    assert abs(cov - target_cov) < 1e-12


def test_entanglement_edges_are_scheduled_into_parallel_layers() -> None:
    ring = [(i, i + 1) for i in range(19)] + [(0, 19)]
    layers = A.schedule_entanglement_edges(ring)

    assert len(layers) == 2
    assert sorted(edge for layer in layers for edge in layer) == sorted(ring)
    for layer in layers:
        touched = [qubit for edge in layer for qubit in edge]
        assert len(touched) == len(set(touched))


def test_twenty_qubit_banded_circuit_has_two_entanglement_layers() -> None:
    n = 20
    index = np.arange(n)
    distance = np.abs(index[:, None] - index[None, :])
    corr = 0.04 + 0.14 * np.exp(-distance / 2.0)
    np.fill_diagonal(corr, 1.0)
    spec = SystemSpec(
        node_names=[f"Bank {i + 1}" for i in range(n)],
        node_types=["bank"] * n,
        exposure_matrix=np.zeros((n, n)),
        capital_buffers=np.ones(n),
        marginal_default_probs=np.linspace(0.05, 0.25, n),
        target_pairwise_corr=corr,
        clusters=["hardware-test"] * n,
    )
    generator = EntangledBornMachineGenerator(
        ansatz="entangled",
        max_degree=2,
        max_block_qubits=22,
        calibrate=False,
    )

    generator.fit(spec)

    assert len(generator.blocks_) == 1
    assert len(generator.blocks_[0].edges) == 20
    assert generator.blocks_[0].entanglement_depth == 2
    pytest.importorskip("qiskit")
    from systemic_risk.generators.quantum.qiskit_backend import build_circuit

    block = generator.blocks_[0]
    circuit = build_circuit(block.ry, block.edges, block.cry, measure=True)
    assert circuit.depth() == 4


def test_z_diagonal_phase_is_inert_in_measurement_basis() -> None:
    """A Z-diagonal gate (e.g. RZZ) cannot move Z-basis marginals/covariance -- only phase.

    This is the physics the old placeholder got wrong; the generator's correlations come from
    amplitude mixing instead.
    """
    state = StateVector(2)
    state.ry(0, 0.9)
    state.ry(1, 1.3)
    before = state.probabilities().copy()
    z = np.array([1.0, -1.0])
    state.amplitudes = state.amplitudes * np.exp(1j * 0.7 * np.outer(z, z))
    assert np.allclose(before, state.probabilities(), atol=1e-15)


def test_sample_bitstrings_bit_order_matches_marginals() -> None:
    p = [0.1, 0.3, 0.6, 0.85]
    state = StateVector(len(p))
    for qubit, theta in enumerate(A.marginal_angles(np.array(p))):
        state.ry(qubit, float(theta))
    rng = np.random.default_rng(0)
    samples = sample_bitstrings(state.probabilities(), len(p), 200_000, rng)
    assert np.allclose(samples.mean(axis=0), state.marginals(), atol=0.01)


# ------------------------------------------------------------------- generator contract
def test_samples_are_binary_and_correct_shape() -> None:
    spec = make_synthetic_system(n=12, seed=8)
    generator = EntangledBornMachineGenerator(ansatz="entangled")
    generator.fit(spec)
    samples = generator.sample(64, seed=1)
    assert samples.shape == (64, spec.n)
    assert np.all((samples == 0) | (samples == 1))


def test_sampling_is_seed_deterministic() -> None:
    spec = make_synthetic_system(n=12, seed=8)
    generator = EntangledBornMachineGenerator(ansatz="entangled")
    generator.fit(spec)
    assert np.array_equal(generator.sample(50, seed=3), generator.sample(50, seed=3))


def test_ghz_ansatz_is_binary_and_deterministic() -> None:
    spec = make_synthetic_system(n=12, seed=8)
    generator = EntangledBornMachineGenerator(ansatz="ghz_systemic")
    generator.fit(spec)
    samples = generator.sample(64, seed=2)
    assert samples.shape == (64, spec.n)
    assert np.all((samples == 0) | (samples == 1))
    assert np.array_equal(generator.sample(40, seed=2), generator.sample(40, seed=2))


def test_legacy_alias_accepts_old_keywords() -> None:
    spec = make_synthetic_system(n=12, seed=8)
    generator = EntangledPQCGenerator(layers=2, gibbs_sweeps=4, burn_in=5)
    generator.fit(spec)
    assert generator.sample(16, seed=0).shape == (16, spec.n)


# --------------------------------------------------- drop-in accuracy vs the spec targets
def test_exact_marginals_and_correlations_match_target() -> None:
    spec = make_synthetic_system(n=12, seed=8)
    p = spec.marginal_default_probs
    corr = spec.target_pairwise_corr.copy()
    np.fill_diagonal(corr, 0.0)
    iu = np.triu_indices(spec.n, k=1)

    generator = EntangledBornMachineGenerator(ansatz="entangled")
    generator.fit(spec)
    marginals, joint = generator.exact_moments()
    cov = joint - np.outer(marginals, marginals)
    denom = np.sqrt(np.outer(marginals * (1 - marginals), marginals * (1 - marginals)))
    rho = np.where(denom > 0, cov / denom, 0.0)

    # Tight match by analytic angles + a light calibration loop -- as good as (here better
    # than) the strongest classical baseline at hitting the spec's first two moments.
    assert np.max(np.abs(marginals - p)) < 1e-3
    assert np.mean(np.abs(rho - corr)[iu]) < 5e-3
    assert np.max(np.abs(rho - corr)[iu]) < 2e-2


def test_exact_probabilities_are_a_normalised_distribution() -> None:
    spec = make_synthetic_system(n=12, seed=8)
    generator = EntangledBornMachineGenerator(ansatz="entangled")
    generator.fit(spec)
    probs = generator.exact_probabilities()
    assert probs.shape == (2**spec.n,)
    assert abs(probs.sum() - 1.0) < 1e-12
    assert np.all(probs >= -1e-15)


def test_sampled_moments_track_exact_moments() -> None:
    spec = make_synthetic_system(n=12, seed=8)
    generator = EntangledBornMachineGenerator(ansatz="entangled")
    generator.fit(spec)
    exact_marginals, _ = generator.exact_moments()
    samples = generator.sample(60_000, seed=7)
    assert np.max(np.abs(samples.mean(axis=0) - exact_marginals)) < 0.02


# ---------------------------------------------------------- n = 54 oracle validation
@pytest.mark.parametrize("p,rho", [(0.05, 0.1), (0.1, 0.3), (0.03, 0.2)])
def test_homogeneous_matches_mean_field_oracle_at_54(p: float, rho: float) -> None:
    n = 54
    spec = _homogeneous_spec(n, p, rho)
    oracle = MeanFieldIsingOracle.from_targets(n, p, rho)

    generator = EntangledBornMachineGenerator(ansatz="entangled")
    generator.fit(spec)
    pmf = generator.loss_count_pmf()
    loss = LossDistribution(pmf=pmf, exact=True)

    counts = np.arange(n + 1)
    e_k = float(np.dot(counts, pmf))
    e_kkm1 = float(np.dot(counts * (counts - 1), pmf))
    marginal = e_k / n
    co_default = e_kkm1 / (n * (n - 1))
    default_corr = (co_default - marginal**2) / (marginal * (1 - marginal))

    # Exact at any n: the homogeneous symmetric loader reproduces the oracle's loss-count law.
    assert total_variation_distance(loss, oracle.loss_distribution()) < 1e-9
    assert abs(marginal - oracle.marginal_default_prob()) < 1e-6
    assert abs(default_corr - oracle.default_correlation()) < 1e-6


def test_symmetric_loader_closed_form_matches_exact_statevector() -> None:
    """At small n the closed-form loss-count law equals the exact statevector's."""
    n = 8
    loader = A.SymmetricIsingLoader.from_targets(n, target_marginal=0.2, target_default_corr=0.3)
    state = StateVector.symmetric_shells(loader.shell_amplitudes())
    counts = np.array([bin(i).count("1") for i in range(2**n)])
    state_pmf = np.bincount(counts, weights=state.probabilities(), minlength=n + 1)
    assert np.allclose(state_pmf, loader.loss_count_pmf(), atol=1e-12)


def test_ghz_blend_closed_form_matches_exact_statevector() -> None:
    n = 8
    blend = A.GHZBlend.from_targets(n, target_marginal=0.15, target_default_corr=0.25)
    state = StateVector.product_blend(n, blend.weight, blend.benign, blend.systemic)
    counts = np.array([bin(i).count("1") for i in range(2**n)])
    state_pmf = np.bincount(counts, weights=state.probabilities(), minlength=n + 1)
    assert np.allclose(state_pmf, blend.loss_count_pmf(), atol=1e-12)


# ------------------------------------------- higher-order structure vs a moment-matched foil
def test_symmetric_loader_carries_coskewness_beyond_a_moment_matched_gaussian() -> None:
    """The central scientific claim, stated honestly.

    On a homogeneous credit spec the symmetric (entangled) loader reproduces the exchangeable
    mean-field Ising law, whose third-order co-default is fixed by the spec and is genuinely
    *non-elliptical*. Against a Gaussian copula matched to the loader's own realized marginals
    and correlation, its excess co-skewness is large and stable, while the same statistic
    applied to a pure Gaussian sample vanishes -- proof the statistic isolates beyond-second-
    order structure rather than re-measuring correlation.
    """
    n = 8
    spec = _homogeneous_spec(n, p=0.05, rho=0.4)
    n_samples = 200_000

    symmetric = EntangledBornMachineGenerator(ansatz="entangled")  # homogeneous -> symmetric loader
    symmetric.fit(spec)
    gaussian = GaussianCopulaGenerator()
    gaussian.fit(spec)

    symmetric_excess = _excess_coskewness(symmetric.sample(n_samples, seed=1), foil_seed=99)
    gaussian_excess = _excess_coskewness(gaussian.sample(n_samples, seed=1), foil_seed=99)

    assert symmetric_excess > 0.2
    assert abs(gaussian_excess) < 0.05
    assert symmetric_excess > 4.0 * abs(gaussian_excess)


def test_ghz_blend_concentrates_systemic_co_default_mass() -> None:
    """The GHZ ansatz's defining feature is its coherent all-default (systemic) mode.

    Its closed-form loss-count law puts amplitude on the all-default string many orders of
    magnitude above the independence baseline -- the rare "everyone fails together" event the
    state is built to carry. (Unlike the symmetric loader, the GHZ blend's higher-order
    structure is *not* pinned by the spec: it is dialled by ``benign_fraction``, so we assert
    only the systemic-mode property it always satisfies, not a generic beyond-Gaussian claim.)
    """
    n = 8
    spec = _homogeneous_spec(n, p=0.05, rho=0.4)
    ghz = EntangledBornMachineGenerator(ansatz="ghz_systemic")
    ghz.fit(spec)

    all_default = float(ghz.loss_count_pmf()[n])
    independent_all_default = 0.05**n
    assert all_default > 1e4 * independent_all_default


# ------------------------------------------------------------ real-data spec consumption
def test_consumes_real_and_synthetic_specs_end_to_end() -> None:
    from systemic_risk.data_network import build_synthetic_system_spec, build_system_spec

    for spec in (build_system_spec(), build_synthetic_system_spec(54)):
        generator = EntangledBornMachineGenerator(ansatz="entangled")
        generator.fit(spec)
        samples = generator.sample(256, seed=4)
        assert samples.shape == (256, spec.n)
        assert np.all((samples == 0) | (samples == 1))
        # Marginals stay near target even fitting across cluster blocks at n = 54.
        assert abs(samples.mean() - spec.marginal_default_probs.mean()) < 0.01


# --------------------------------------------------------------- backend interchangeability
def test_qiskit_backend_matches_numpy_statevector() -> None:
    pytest.importorskip("qiskit")
    spec = make_synthetic_system(n=12, seed=8)
    numpy_gen = EntangledBornMachineGenerator(ansatz="entangled", backend="statevector")
    qiskit_gen = EntangledBornMachineGenerator(ansatz="entangled", backend="qiskit")
    numpy_gen.fit(spec)
    qiskit_gen.fit(spec)

    np_marginals, np_joint = numpy_gen.exact_moments()
    qk_marginals, qk_joint = qiskit_gen.exact_moments()
    assert np.allclose(np_marginals, qk_marginals, atol=1e-12)
    assert np.allclose(np_joint, qk_joint, atol=1e-12)
