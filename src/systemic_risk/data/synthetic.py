from __future__ import annotations

import numpy as np

from systemic_risk.spec import SystemSpec


_TEMPLATE = [
    ("Global Bank A", "bank", "banks"),
    ("Global Bank B", "bank", "banks"),
    ("Regional Bank", "bank", "banks"),
    ("Mortgage Lender", "bank", "banks"),
    ("Life Insurer", "insurer", "insurers"),
    ("Reinsurer", "insurer", "insurers"),
    ("Pension Fund", "fund", "funds"),
    ("Credit Fund", "fund", "funds"),
    ("Real Estate Fund", "fund", "funds"),
    ("Industrial Corporate", "corporate", "corporates"),
    ("Energy Corporate", "corporate", "corporates"),
    ("Retail Corporate", "corporate", "corporates"),
    ("Sovereign A", "sovereign", "sovereign_ccp"),
    ("Sovereign B", "sovereign", "sovereign_ccp"),
    ("Central Counterparty", "CCP", "sovereign_ccp"),
    ("Payments Utility", "CCP", "sovereign_ccp"),
    ("Asset Manager", "fund", "funds"),
    ("Trade Finance Bank", "bank", "banks"),
    ("Health Insurer", "insurer", "insurers"),
    ("Transport Corporate", "corporate", "corporates"),
]


_BASE_DEFAULT_PROBS = {
    "bank": 0.025,
    "insurer": 0.018,
    "fund": 0.035,
    "corporate": 0.045,
    "sovereign": 0.012,
    "CCP": 0.006,
}


def make_synthetic_system(n: int = 16, seed: int = 7) -> SystemSpec:
    """Create a deterministic synthetic financial network with community structure."""
    if not 12 <= n <= len(_TEMPLATE):
        raise ValueError(f"n must be between 12 and {len(_TEMPLATE)}")

    rng = np.random.default_rng(seed)
    selected = _TEMPLATE[:n]
    node_names = [item[0] for item in selected]
    node_types = [item[1] for item in selected]
    clusters = [item[2] for item in selected]

    W = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            same_cluster = clusters[i] == clusters[j]
            system_node = "sovereign_ccp" in (clusters[i], clusters[j])
            edge_prob = 0.62 if same_cluster else 0.22
            if system_node:
                edge_prob += 0.08
            if rng.random() < edge_prob:
                base = 0.12 if same_cluster else 0.055
                if node_types[j] in {"sovereign", "CCP"}:
                    base *= 1.4
                if node_types[i] == "bank":
                    base *= 1.2
                W[i, j] = rng.lognormal(mean=np.log(base), sigma=0.42)

    incoming_exposure = W.sum(axis=1)
    capital_buffers = 0.28 * incoming_exposure + rng.uniform(0.08, 0.16, size=n)

    p = np.array([_BASE_DEFAULT_PROBS[node_type] for node_type in node_types], dtype=float)
    p = np.clip(p * rng.uniform(0.82, 1.24, size=n), 0.003, 0.12)

    corr = np.eye(n, dtype=float)
    exposure_strength = W + W.T
    max_strength = exposure_strength.max() if exposure_strength.max() > 0 else 1.0
    for i in range(n):
        for j in range(i + 1, n):
            value = 0.06
            if clusters[i] == clusters[j]:
                value += 0.28
            value += 0.18 * exposure_strength[i, j] / max_strength
            if {"sovereign", "CCP"} & {node_types[i], node_types[j]}:
                value += 0.04
            corr[i, j] = corr[j, i] = min(value, 0.65)

    return SystemSpec(
        node_names=node_names,
        node_types=node_types,
        exposure_matrix=W,
        capital_buffers=capital_buffers,
        marginal_default_probs=p,
        target_pairwise_corr=corr,
        clusters=clusters,
        metadata={
            "name": "Synthetic systemic stress network",
            "seed": seed,
            "description": "Directed weighted exposures with visible financial communities.",
        },
    )
