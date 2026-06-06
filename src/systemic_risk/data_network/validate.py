"""End-to-end validation: round-trip, cluster stability, and B/C/D contract conformance.

This is the A end-to-end test in code form:

    load raw -> emit a valid spec -> confirm it round-trips losslessly,
    its communities are stable, and it feeds B/C/D (generators + cascade) without loss.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from systemic_risk.data_network.cluster import cluster_stability
from systemic_risk.data_network.spec import GENERATOR, SIMULATOR, NetworkSpec


@dataclass
class ValidationReport:
    roundtrip_ok: bool
    clusters_stable: bool
    cluster_mean_ari: float
    contract_ok: bool
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.roundtrip_ok and self.clusters_stable and self.contract_ok


def check_roundtrip(spec: NetworkSpec) -> bool:
    """JSON round-trip must reproduce every array and field exactly."""
    reloaded = NetworkSpec.from_json(spec.to_json())
    emp_a, emp_b = spec.empirical, reloaded.empirical
    if emp_a.node_ids != emp_b.node_ids:
        return False
    arrays_equal = (
        np.array_equal(emp_a.correlation_matrix, emp_b.correlation_matrix)
        and np.array_equal(spec.reconstructed.edge_matrix, reloaded.reconstructed.edge_matrix)
        and np.allclose(emp_a.marginals, emp_b.marginals)
        and np.allclose(emp_a.capital_buffers, emp_b.capital_buffers)
        and np.allclose(emp_a.interbank_assets, emp_b.interbank_assets)
        and np.allclose(emp_a.interbank_liabilities, emp_b.interbank_liabilities)
    )
    meta_equal = (
        spec.clusters == reloaded.clusters
        and spec.reconstructed.method == reloaded.reconstructed.method
        and spec.feature_schema.to_dict() == reloaded.feature_schema.to_dict()
        and spec.provenance.content_hash == reloaded.provenance.content_hash
        and reloaded.compute_content_hash() == spec.compute_content_hash()
    )
    return bool(arrays_equal and meta_equal)


def check_clusters_stable(
    spec: NetworkSpec, n_perturb: int = 8, noise: float = 0.05, stable_ari: float = 0.6
) -> tuple[bool, float]:
    """Communities must survive small correlation perturbations (mean ARI >= threshold)."""
    report = cluster_stability(
        spec.empirical.correlation_matrix,
        np.asarray(spec.clusters, dtype=int),
        n_perturb=n_perturb,
        noise=noise,
        stable_ari=stable_ari,
    )
    return report.stable, report.mean_ari


def check_contract(spec: NetworkSpec) -> tuple[bool, dict[str, Any]]:
    """The flat ``SystemSpec`` must validate and feed the generators + cascade (B/C/D)."""
    # Lazy imports: keep validate importable without pulling the whole B/C/D stack.
    from systemic_risk.generators.gaussian_copula import GaussianCopulaGenerator
    from systemic_risk.simulator.cascade import run_cascade

    detail: dict[str, Any] = {}
    system = spec.to_system_spec()  # raises if the flat spec is invalid
    detail["n"] = system.n

    # B/C: a generator must fit the spec and emit well-formed binary samples.
    gen = GaussianCopulaGenerator()
    gen.fit(system)
    samples = gen.sample(64, seed=0)
    gen_ok = (
        samples.shape == (64, system.n)
        and np.all((samples == 0) | (samples == 1))
    )
    detail["generator_samples_ok"] = bool(gen_ok)

    # D: the cascade must run on a seed scenario and return a consistent result.
    seed_scenario = np.zeros(system.n, dtype=int)
    seed_scenario[int(np.argmax(system.marginal_default_probs))] = 1
    result = run_cascade(seed_scenario, system)
    cascade_ok = result.final_defaults.shape == (system.n,) and result.failure_count >= 1
    detail["cascade_ok"] = bool(cascade_ok)

    # The consumer views must expose exactly the contracted fields.
    gen_view = spec.view_for(GENERATOR)
    sim_view = spec.view_for(SIMULATOR)
    views_ok = (
        "marginal_default_prob" in gen_view.keys()
        and "exposure_matrix" not in gen_view.keys()
        and "exposure_matrix" in sim_view.keys()
        and "marginal_default_prob" not in sim_view.keys()
    )
    detail["consumer_views_ok"] = bool(views_ok)

    return bool(gen_ok and cascade_ok and views_ok), detail


def validate_spec(
    spec: NetworkSpec,
    n_perturb: int = 8,
    noise: float = 0.05,
    stable_ari: float = 0.6,
) -> ValidationReport:
    """Run all three checks and return a structured report."""
    roundtrip_ok = check_roundtrip(spec)
    clusters_stable, mean_ari = check_clusters_stable(
        spec, n_perturb=n_perturb, noise=noise, stable_ari=stable_ari
    )
    contract_ok, detail = check_contract(spec)
    return ValidationReport(
        roundtrip_ok=roundtrip_ok,
        clusters_stable=clusters_stable,
        cluster_mean_ari=mean_ari,
        contract_ok=contract_ok,
        details=detail,
    )
