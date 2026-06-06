"""Assemble the layers into a validated ``NetworkSpec`` (and the flat ``SystemSpec``).

This is the end-to-end real pipeline:

    roster -> clean -> estimate (marginals, correlation, totals, buffers)
           -> reconstruct (bilateral exposures) -> cluster -> NetworkSpec

The returned ``NetworkSpec`` is frozen, carries provenance with a content hash, and emits
the flat ``SystemSpec`` that parts B/C/D consume via ``.to_system_spec()``.

Entrypoints for B/C/D
---------------------
B (copula baseline), C (entangled generator) and D (cascade simulator) all consume the flat
:class:`systemic_risk.spec.SystemSpec`. The two one-call entrypoints below return exactly
that — drop-in for the synthetic ``make_synthetic_system`` / ``make_scalable_system`` they
already use::

    from systemic_risk.data_network import build_system_spec, build_synthetic_system_spec

    spec = build_system_spec()                 # the REAL 28-bank network
    spec = build_synthetic_system_spec(n=54)   # calibrated-synthetic, scales to 54 qubits

    # ... then exactly as today:
    gen = GaussianCopulaGenerator(); gen.fit(spec); samples = gen.sample(2000)
    cascades = simulate_many(samples, spec)

Both are deterministic (the real build is snapshot-backed; the synthetic build is seeded), so
re-running gives an identical spec. To avoid rebuilding every process, build once and cache to
disk with ``scripts/build_system_spec.py`` then ``SystemSpec.load_json(...)``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from systemic_risk.data_network import clean as clean_mod
from systemic_risk.data_network import estimate as estimate_mod
from systemic_risk.data_network import reconstruct as reconstruct_mod
from systemic_risk.data_network.cluster import cluster_with_stability
from systemic_risk.data_network.sources.equity_returns import load_or_fetch_correlation
from systemic_risk.data_network.sources.roster import load_roster
from systemic_risk.data_network.spec import (
    EmpiricalLayer,
    FeatureSchema,
    NetworkSpec,
    Provenance,
)
from systemic_risk.spec import SystemSpec


def build_network_spec(
    roster_path: str | Path | None = None,
    *,
    interbank_share: float = 0.20,
    buffer_ratio: float = 0.06,
    single_counterparty_cap_frac: float = 0.25,
    reconstruction_method: str = "max_entropy",
    cluster_threshold: float = 0.0,
    prefer_snapshot: bool = True,
    write_snapshot: bool = False,
    stable_ari: float = 0.6,
    cluster_seed: int = 0,
) -> NetworkSpec:
    """Run the full real pipeline and return a content-hashed ``NetworkSpec``."""
    rows = load_roster(roster_path)
    nodes = clean_mod.reconcile(rows)
    node_ids = tuple(node.node_id for node in nodes)

    # --- empirical layer --------------------------------------------------- #
    marginals, pd_source = estimate_mod.marginals_from_ratings(nodes)
    ec = load_or_fetch_correlation(
        [node.ticker for node in nodes],
        prefer_snapshot=prefer_snapshot,
        write_snapshot=write_snapshot,
    )
    corr = estimate_mod.correlation_from_equity(nodes, ec)
    assets, liabilities = estimate_mod.interbank_totals(nodes, interbank_share)
    buffers = estimate_mod.capital_buffers(nodes, buffer_ratio)

    node_attributes = {
        node.node_id: {
            "name": node.name,
            "ticker": node.ticker,
            "node_type": node.node_type,
            "business_type": node.business_type,
            "region": node.region,
            "country": node.country,
            "rating": node.sp_rating,
            "rating_bucket": node.rating_bucket,
        }
        for node in nodes
    }
    node_totals = {node.node_id: float(node.total_assets_usd_bn) for node in nodes}

    empirical = EmpiricalLayer(
        node_ids=node_ids,
        marginals=tuple(float(x) for x in marginals),
        correlation_matrix=corr,
        capital_buffers=tuple(float(x) for x in buffers),
        interbank_assets=tuple(float(x) for x in assets),
        interbank_liabilities=tuple(float(x) for x in liabilities),
        node_totals=node_totals,
        node_attributes=node_attributes,
    )

    # --- reconstructed layer ---------------------------------------------- #
    cap = single_counterparty_cap_frac * buffers
    reconstructed = reconstruct_mod.reconstruct(
        reconstruction_method,
        assets,
        liabilities,
        single_counterparty_cap=cap,
        record={"single_counterparty_cap_frac": single_counterparty_cap_frac},
    )

    # --- communities ------------------------------------------------------- #
    report = cluster_with_stability(
        corr, threshold=cluster_threshold, seed=cluster_seed, stable_ari=stable_ari
    )
    clusters = tuple(int(c) for c in report.labels)

    provenance = Provenance(
        source=(
            "Real anchor: G-SIB / large-bank roster "
            "(data/external/banks/gsib_roster.csv). "
            f"Marginals: {pd_source}. Correlation: {ec.source}."
        ),
        fit_params={
            "interbank_share": interbank_share,
            "buffer_ratio": buffer_ratio,
            "single_counterparty_cap_frac": single_counterparty_cap_frac,
            "reconstruction_method": reconstruction_method,
            "cluster_threshold": cluster_threshold,
            "n_nodes": len(node_ids),
            "n_communities": report.n_communities,
            "modularity": round(report.modularity, 4),
            "cluster_mean_ari": round(report.mean_ari, 4),
            "cluster_min_ari": round(report.min_ari, 4),
            "cluster_stable": bool(report.stable),
            "equity_window": f"{ec.start}..{ec.end}",
            "equity_n_obs": ec.n_obs,
        },
        notes=(
            "Bilateral exposures are reconstructed (real bilateral data is confidential); "
            "marginals and equity correlation are empirical."
        ),
    )

    spec = NetworkSpec(
        empirical=empirical,
        reconstructed=reconstructed,
        clusters=clusters,
        feature_schema=FeatureSchema.default(),
        provenance=provenance,
    )
    return spec.with_content_hash()


def build_system_spec(roster_path: str | Path | None = None, **kwargs) -> SystemSpec:
    """B/C/D entrypoint — build the REAL bank network and return the flat ``SystemSpec``.

    One call returns a validated ``SystemSpec`` (node names/types, exposure matrix, capital
    buffers, marginal default probs, target pairwise correlation, community labels, and a
    provenance/feature-schema-bearing ``metadata`` dict). ``**kwargs`` forwards to
    :func:`build_network_spec` (e.g. ``reconstruction_method="min_density"``).
    """
    return build_network_spec(roster_path, **kwargs).to_system_spec()


def build_synthetic_system_spec(n: int = 54, seed: int = 11) -> SystemSpec:
    """B/C/D entrypoint — calibrated-synthetic network as a flat ``SystemSpec``.

    Use this to scale beyond the fixed real roster (up to the ``n = 54`` quantum-hardware
    target) while keeping the same layered provenance. Deterministic given ``(n, seed)``.
    """
    # Imported lazily: the synthetic source imports systemic_risk.data, and importing it at
    # module load would create an import cycle through the data package.
    from systemic_risk.data_network.sources.synthetic import synthetic_network_spec

    return synthetic_network_spec(n=n, seed=seed).to_system_spec()
