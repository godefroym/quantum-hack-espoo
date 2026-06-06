from __future__ import annotations

import numpy as np
import pytest

from systemic_risk.data_network import (
    build_network_spec,
    build_synthetic_system_spec,
    build_system_spec,
)
from systemic_risk.data_network.cluster import (
    adjusted_rand_index,
    cluster_with_stability,
    detect_communities,
)
from systemic_risk.data_network.clean import reconcile, whole_letter
from systemic_risk.data_network.reconstruct import max_entropy, min_density, reconstruct
from systemic_risk.data_network.sources.roster import load_roster
from systemic_risk.data_network.spec import GENERATOR, SIMULATOR, NetworkSpec
from systemic_risk.data_network.validate import validate_spec
from systemic_risk.generators.gaussian_copula import GaussianCopulaGenerator
from systemic_risk.simulator.cascade import run_cascade
from systemic_risk.spec import SystemSpec


# --- a module-scoped real spec so the (snapshot-backed) build runs once ------ #
@pytest.fixture(scope="module")
def network_spec() -> NetworkSpec:
    return build_network_spec(prefer_snapshot=True)


# --- roster + cleaning ------------------------------------------------------- #
def test_roster_loads_and_is_unique() -> None:
    rows = load_roster()
    assert len(rows) >= 20
    assert len({r.bank_id for r in rows}) == len(rows)
    assert len({r.ticker for r in rows}) == len(rows)


@pytest.mark.parametrize(
    "raw,expected",
    [("A-", "A"), ("BBB+", "BBB"), ("AA-", "AA"), ("BB-", "BB"), ("A+", "A"), ("CCC+", "CCC")],
)
def test_whole_letter_buckets(raw: str, expected: str) -> None:
    assert whole_letter(raw) == expected


def test_reconcile_orders_and_buckets() -> None:
    nodes = reconcile(load_roster())
    assert all(n.rating_bucket in {"AAA", "AA", "A", "BBB", "BB", "B", "CCC"} for n in nodes)


# --- reconstruction honours the marginals ------------------------------------ #
def test_max_entropy_matches_marginals() -> None:
    assets = np.array([3.0, 2.0, 5.0, 1.0])
    liabilities = np.array([2.0, 4.0, 1.0, 4.0])
    W = max_entropy(assets, liabilities)
    assert np.allclose(W.sum(axis=1), assets, atol=1e-6)
    assert np.allclose(W.sum(axis=0), liabilities, atol=1e-6)
    assert np.allclose(np.diag(W), 0.0)


def test_min_density_is_sparser_than_max_entropy() -> None:
    assets = np.array([3.0, 2.0, 5.0, 1.0])
    liabilities = np.array([2.0, 4.0, 1.0, 4.0])
    dense = max_entropy(assets, liabilities)
    sparse = min_density(assets, liabilities)
    assert np.allclose(sparse.sum(axis=1), assets, atol=1e-6)
    assert (sparse > 0).sum() < (dense > 0).sum()


def test_reconstruct_applies_single_counterparty_cap() -> None:
    assets = np.array([10.0, 1.0, 1.0])
    liabilities = np.array([4.0, 4.0, 4.0])
    cap = np.array([1.0, 1.0, 1.0])
    layer = reconstruct("max_entropy", assets, liabilities, single_counterparty_cap=cap)
    assert np.all(layer.edge_matrix <= cap[:, None] + 1e-9)


# --- clustering -------------------------------------------------------------- #
def test_adjusted_rand_index_identity_and_disjoint() -> None:
    a = np.array([0, 0, 1, 1])
    assert adjusted_rand_index(a, a) == pytest.approx(1.0)
    assert adjusted_rand_index(a, np.array([0, 1, 0, 1])) < 0.5


def test_detect_communities_on_block_structure() -> None:
    # Two clean blocks -> two communities.
    corr = np.array(
        [
            [1.0, 0.9, 0.1, 0.1],
            [0.9, 1.0, 0.1, 0.1],
            [0.1, 0.1, 1.0, 0.9],
            [0.1, 0.1, 0.9, 1.0],
        ]
    )
    labels, _ = detect_communities(corr, threshold=0.5)
    assert labels[0] == labels[1]
    assert labels[2] == labels[3]
    assert labels[0] != labels[2]


# --- the layered spec round-trips and assembles down ------------------------- #
def test_network_spec_json_roundtrip(network_spec: NetworkSpec) -> None:
    reloaded = NetworkSpec.from_json(network_spec.to_json())
    assert reloaded.empirical.node_ids == network_spec.empirical.node_ids
    assert np.array_equal(
        reloaded.reconstructed.edge_matrix, network_spec.reconstructed.edge_matrix
    )
    assert np.array_equal(
        reloaded.empirical.correlation_matrix, network_spec.empirical.correlation_matrix
    )
    assert reloaded.clusters == network_spec.clusters
    assert reloaded.compute_content_hash() == network_spec.compute_content_hash()


def test_to_system_spec_validates_and_roundtrips(tmp_path, network_spec: NetworkSpec) -> None:
    system = network_spec.to_system_spec()
    assert isinstance(system, SystemSpec)
    path = tmp_path / "system.json"
    system.save_json(path)
    loaded = SystemSpec.load_json(path)
    assert loaded.node_names == system.node_names
    assert np.allclose(loaded.exposure_matrix, system.exposure_matrix)
    assert np.allclose(np.diag(loaded.target_pairwise_corr), 1.0)
    assert loaded.correlation_space == "latent_gaussian"


def test_consumer_views_enforce_visibility(network_spec: NetworkSpec) -> None:
    gen_view = network_spec.view_for(GENERATOR)
    sim_view = network_spec.view_for(SIMULATOR)
    assert "marginal_default_prob" in gen_view.keys()
    assert "exposure_matrix" not in gen_view.keys()   # generators don't see edges
    assert "exposure_matrix" in sim_view.keys()
    assert "marginal_default_prob" not in sim_view.keys()


# --- B/C/D actually consume the assembled spec ------------------------------- #
def test_generator_and_cascade_consume_real_spec(network_spec: NetworkSpec) -> None:
    system = network_spec.to_system_spec()
    gen = GaussianCopulaGenerator()
    gen.fit(system)
    assert np.allclose(gen.corr_, system.target_pairwise_corr, atol=1e-8)
    samples = gen.sample(128, seed=1)
    assert samples.shape == (128, system.n)
    assert np.all((samples == 0) | (samples == 1))

    worst = np.argmax(samples.sum(axis=1))
    result = run_cascade(samples[worst], system)
    assert result.final_defaults.shape == (system.n,)
    assert result.failure_count >= int(samples[worst].sum())


# --- the full A end-to-end test ---------------------------------------------- #
def test_end_to_end_validation_passes(network_spec: NetworkSpec) -> None:
    report = validate_spec(network_spec, n_perturb=6)
    assert report.roundtrip_ok
    assert report.clusters_stable
    assert report.contract_ok
    assert report.ok


def test_build_is_deterministic() -> None:
    a = build_system_spec(prefer_snapshot=True)
    b = build_system_spec(prefer_snapshot=True)
    assert a.metadata["content_hash"] == b.metadata["content_hash"]
    assert np.allclose(a.exposure_matrix, b.exposure_matrix)


def test_bcd_entrypoints_return_consumable_system_specs() -> None:
    # Both entrypoints must return a flat SystemSpec that a generator fits and the cascade
    # runs on — i.e. they are drop-in for the synthetic make_* helpers B/C/D already use.
    for spec in (build_system_spec(prefer_snapshot=True), build_synthetic_system_spec(n=30)):
        assert isinstance(spec, SystemSpec)
        gen = GaussianCopulaGenerator()
        gen.fit(spec)
        samples = gen.sample(32, seed=0)
        assert samples.shape == (32, spec.n)
        result = run_cascade(samples[int(np.argmax(samples.sum(axis=1)))], spec)
        assert result.final_defaults.shape == (spec.n,)


def test_synthetic_entrypoint_is_deterministic() -> None:
    a = build_synthetic_system_spec(n=40, seed=3)
    b = build_synthetic_system_spec(n=40, seed=3)
    assert a.metadata["content_hash"] == b.metadata["content_hash"]
