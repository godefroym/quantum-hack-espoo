"""Calibrated-synthetic source — scale beyond the real roster to the 54-qubit target.

The real anchor roster has a fixed size (~28 banks). For scaling experiments and the
quantum-hardware target (n up to 54), wrap the existing calibrated ``make_scalable_system``
and lift its flat ``SystemSpec`` into the same layered ``NetworkSpec`` shape, so downstream
code can treat the real and synthetic specs identically. Communities are detected on the
synthetic dependency target with the same detector used for the real network.
"""

from __future__ import annotations

import numpy as np

from systemic_risk.data import make_scalable_system
from systemic_risk.data_network.cluster import cluster_with_stability
from systemic_risk.data_network.spec import (
    EmpiricalLayer,
    FeatureSchema,
    NetworkSpec,
    Provenance,
    ReconstructedLayer,
)


def synthetic_network_spec(n: int = 54, seed: int = 11) -> NetworkSpec:
    """Lift ``make_scalable_system(n, seed)`` into a layered ``NetworkSpec``."""
    spec = make_scalable_system(n=n, seed=seed)
    W = np.asarray(spec.exposure_matrix, dtype=float)
    corr = (
        np.asarray(spec.target_pairwise_corr, dtype=float)
        if spec.target_pairwise_corr is not None
        else np.eye(n)
    )

    node_ids = tuple(name.replace(" ", "_") for name in spec.node_names)
    interbank_assets = W.sum(axis=1)
    interbank_liabilities = W.sum(axis=0)

    node_attributes = {
        nid: {
            "name": spec.node_names[i],
            "ticker": node_ids[i],
            "node_type": spec.node_types[i],
            "business_type": spec.node_types[i],
            "region": (spec.clusters[i] if spec.clusters else ""),
            "country": "synthetic",
            "rating": (spec.metadata.get("ratings", [""] * n)[i]
                       if isinstance(spec.metadata.get("ratings"), list) else ""),
            "rating_bucket": "",
        }
        for i, nid in enumerate(node_ids)
    }

    empirical = EmpiricalLayer(
        node_ids=node_ids,
        marginals=tuple(float(x) for x in spec.marginal_default_probs),
        correlation_matrix=corr,
        capital_buffers=tuple(float(x) for x in spec.capital_buffers),
        interbank_assets=tuple(float(x) for x in interbank_assets),
        interbank_liabilities=tuple(float(x) for x in interbank_liabilities),
        node_totals={nid: float(interbank_assets[i]) for i, nid in enumerate(node_ids)},
        node_attributes=node_attributes,
    )
    reconstructed = ReconstructedLayer(
        edge_matrix=W,
        method=str(spec.metadata.get("topology", "synthetic-gravity")),
        method_params={"seed": float(seed), "n": float(n)},
    )

    report = cluster_with_stability(corr, threshold=float(np.median(corr[corr < 1.0])))
    clusters = tuple(int(c) for c in report.labels)

    provenance = Provenance(
        source=f"Calibrated-synthetic make_scalable_system(n={n}, seed={seed})",
        fit_params={
            "n": n,
            "seed": seed,
            "n_communities": report.n_communities,
            "cluster_mean_ari": round(report.mean_ari, 4),
            **{k: v for k, v in spec.metadata.items()
               if isinstance(v, (int, float, str, bool))},
        },
        notes="Synthetic scaling source (scale-free core-periphery gravity network).",
    )
    return NetworkSpec(
        empirical=empirical,
        reconstructed=reconstructed,
        clusters=clusters,
        feature_schema=FeatureSchema.default(),
        provenance=provenance,
    ).with_content_hash()
