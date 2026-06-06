from __future__ import annotations

from pathlib import Path

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


# ---------------------------------------------------------------------------
# Scalable (n up to 54) system generator
# ---------------------------------------------------------------------------
#
# Calibration anchors (see research/README.md sections 1-3 and
# research/sections/03_statistical_mechanics_ising.md):
#  - per-rating one-year PDs from S&P / Moody's annual default studies;
#  - spec-grade aggregate PD ~3.8-4.2%;
#  - sparse, scale-free, core-periphery interbank topology (gamma ~ 2-3, ~20% core),
#    NOT Erdos-Renyi (Bardoscia et al. 2021; section 01);
#  - capital buffers ~4-8% of assets; single-counterparty exposure <= 25% of Tier-1;
#    interbank-asset share ~20% of total assets (Gai-Kapadia / Basel; section 01).

# Per-institution-type rating mix: ordered list of (rating, weight). Weights are
# unnormalised relative frequencies within a type. Banks/insurers/CCPs sit at the
# investment-grade end; funds and corporates carry more speculative-grade mass.
_TYPE_RATING_MIX: dict[str, list[tuple[str, float]]] = {
    "bank": [("A", 0.30), ("BBB", 0.50), ("BB", 0.18), ("B", 0.02)],
    "insurer": [("AA", 0.20), ("A", 0.45), ("BBB", 0.32), ("BB", 0.03)],
    "fund": [("BBB", 0.34), ("BB", 0.42), ("B", 0.21), ("CCC", 0.03)],
    "corporate": [("BBB", 0.26), ("BB", 0.36), ("B", 0.32), ("CCC", 0.06)],
    "sovereign": [("AA", 0.45), ("A", 0.40), ("BBB", 0.15)],
    "CCP": [("AA", 0.55), ("A", 0.45)],
}

# Literature-default one-year default probabilities by rating (S&P/Moody's studies;
# research/README.md section 3). Mid-range values within the cited bands. These are the
# fallback if the real Moody's PD CSV is absent.
_RATING_PD_DEFAULT: dict[str, float] = {
    "AAA": 0.0001,
    "AA": 0.0004,
    "A": 0.0008,
    "BBB": 0.0025,
    "BB": 0.0140,
    "B": 0.0550,
    "CCC": 0.2200,
}

# Real Moody's whole-letter ratings (Exhibit 17) -> our S&P-style rating keys.
_MOODYS_TO_RATING: dict[str, str] = {
    "Aaa": "AAA",
    "Aa": "AA",
    "A": "A",
    "Baa": "BBB",
    "Ba": "BB",
    "B": "B",
    "Caa-C": "CCC",
}

# Optional real PD source populated by the data agent (research/README.md section 5).
_RATING_PD_CSV = (
    Path(__file__).resolve().parents[3]
    / "data"
    / "external"
    / "ratings"
    / "moodys_pd_by_rating.csv"
)


def _load_rating_pd() -> tuple[dict[str, float], str]:
    """Return ``(rating -> 1y PD, source)``, preferring the real Moody's CSV.

    Reads the whole-letter (Exhibit 17) rows of ``moodys_pd_by_rating.csv`` if present and
    maps them onto our rating keys; otherwise falls back to the literature defaults. The
    speculative-grade aggregate of the Moody's whole-letter table is ~3.8%, matching the
    ~3.8-4.2% calibration band in ``research/README.md`` section 3.
    """
    table = dict(_RATING_PD_DEFAULT)
    source = "literature defaults (S&P/Moody's annual default studies, research/README.md s.3)"
    if not _RATING_PD_CSV.exists():
        return table, source
    try:
        rows: dict[str, float] = {}
        text = _RATING_PD_CSV.read_text(encoding="utf-8").splitlines()
        for line in text[1:]:
            if not line.strip():
                continue
            # Split into rating, PD, and the (quoted, comma-bearing) source field.
            parts = line.split(",", 2)
            if len(parts) < 3:
                continue
            rating, pd_str, src = parts[0].strip(), parts[1], parts[2]
            # Use only the whole-letter table (Exhibit 17) so all PDs are on one consistent
            # scale; the alphanumeric table (Exhibit 19) lacks whole-letter Ba/B rows.
            if "Exhibit 17" not in src:
                continue
            if rating in _MOODYS_TO_RATING and _MOODYS_TO_RATING[rating] not in rows:
                rows[_MOODYS_TO_RATING[rating]] = float(pd_str)
        if rows:
            # Floor near-zero high-grade PDs so logit fields stay finite downstream.
            table.update({k: max(v, 1e-5) for k, v in rows.items()})
            source = "Moody's Corporate Default & Recovery Rates 1920-2004, Exhibit 17 (whole-letter, Year-1)"
    except (OSError, ValueError):
        # Any parse problem -> silent fall back to defaults (soft dependency).
        return dict(_RATING_PD_DEFAULT), source
    return table, source


_RATING_PD, _RATING_PD_SOURCE = _load_rating_pd()

# Type weights used to build the institution mix programmatically for any n.
# Banks dominate, then funds/corporates, with a thin sovereign/CCP backbone.
_TYPE_WEIGHTS: list[tuple[str, str, float]] = [
    ("bank", "banks", 0.30),
    ("fund", "funds", 0.22),
    ("corporate", "corporates", 0.22),
    ("insurer", "insurers", 0.14),
    ("sovereign", "sovereign_ccp", 0.07),
    ("CCP", "sovereign_ccp", 0.05),
]

# Relative balance-sheet scale by type (drives node "fitness" -> degree & size).
_TYPE_SIZE: dict[str, float] = {
    "bank": 1.0,
    "insurer": 0.7,
    "fund": 0.45,
    "corporate": 0.4,
    "sovereign": 1.6,
    "CCP": 1.3,
}

_TYPE_LABEL: dict[str, str] = {
    "bank": "Bank",
    "insurer": "Insurer",
    "fund": "Fund",
    "corporate": "Corporate",
    "sovereign": "Sovereign",
    "CCP": "CCP",
}

# Descriptive, type-specific institution names so the synthetic network reads like a real
# roster ("Global Bank A", "Life Insurer", "Central Counterparty") instead of "Bank 01".
# Drawn in order per type; once a pool is exhausted a "(2)", "(3)" suffix is appended.
_TYPE_NAME_POOL: dict[str, list[str]] = {
    "bank": [
        "Global Bank A", "Global Bank B", "Continental Bank", "Regional Bank",
        "Community Bank", "Mortgage Lender", "Trade Finance Bank", "Investment Bank",
        "Merchant Bank", "Universal Bank", "Savings Bank", "Commercial Bank",
        "Private Bank", "Custody Bank", "Cooperative Bank", "Digital Bank",
    ],
    "insurer": [
        "Life Insurer", "Reinsurer", "Health Insurer", "Property Insurer",
        "Casualty Insurer", "Credit Insurer", "Title Insurer", "Specialty Insurer",
    ],
    "fund": [
        "Pension Fund", "Credit Fund", "Real Estate Fund", "Money Market Fund",
        "Hedge Fund", "Sovereign Wealth Fund", "Mutual Fund", "Asset Manager",
        "Private Equity Fund", "Infrastructure Fund", "Bond Fund", "Equity Fund",
    ],
    "corporate": [
        "Industrial Corporate", "Energy Corporate", "Retail Corporate", "Transport Corporate",
        "Telecom Corporate", "Utility Corporate", "Healthcare Corporate", "Materials Corporate",
        "Consumer Goods Corporate", "Technology Corporate", "Construction Corporate",
        "Aerospace Corporate",
    ],
    "sovereign": [
        "Sovereign A", "Sovereign B", "Sovereign C", "Sovereign D", "Sovereign E",
    ],
    "CCP": [
        "Central Counterparty", "Payments Utility", "Securities Depository", "Clearing House",
    ],
}


def _descriptive_names(node_types: list[str]) -> list[str]:
    """Assign a descriptive, unique name to each node from its type's name pool.

    Names are handed out in pool order per type; if a type has more nodes than pool entries,
    later ones get a ``" (2)"`` / ``" (3)"`` suffix so every name stays unique. Falls back to
    ``"<Type> NN"`` for any type without a pool.
    """
    counters: dict[str, int] = {}
    names: list[str] = []
    for node_type in node_types:
        idx = counters.get(node_type, 0)
        counters[node_type] = idx + 1
        pool = _TYPE_NAME_POOL.get(node_type)
        if not pool:
            names.append(f"{_TYPE_LABEL.get(node_type, node_type.title())} {idx + 1:02d}")
            continue
        base = pool[idx % len(pool)]
        cycle = idx // len(pool)
        names.append(base if cycle == 0 else f"{base} ({cycle + 1})")
    return names


def _allocate_types(n: int) -> list[tuple[str, str]]:
    """Deterministically allocate ``n`` institutions across types by target weights.

    Returns a list of ``(node_type, cluster)`` of length ``n``. Uses largest-remainder
    rounding on the type weights, guaranteeing at least one sovereign and one CCP (the
    systemic backbone) whenever ``n`` is large enough to afford them.
    """
    weights = np.array([w for _, _, w in _TYPE_WEIGHTS], dtype=float)
    weights = weights / weights.sum()
    raw = weights * n
    counts = np.floor(raw).astype(int)
    remainder = n - counts.sum()
    # Distribute the remaining slots to the largest fractional parts.
    order = np.argsort(-(raw - counts))
    for k in range(remainder):
        counts[order[k % len(counts)]] += 1

    # Ensure a sovereign and a CCP exist for n that can afford the backbone.
    type_index = {t: i for i, (t, _, _) in enumerate(_TYPE_WEIGHTS)}
    if n >= 8:
        for backbone in ("sovereign", "CCP"):
            bi = type_index[backbone]
            if counts[bi] == 0:
                donor = int(np.argmax(counts))
                counts[donor] -= 1
                counts[bi] += 1

    allocation: list[tuple[str, str]] = []
    for (node_type, cluster, _), c in zip(_TYPE_WEIGHTS, counts):
        allocation.extend([(node_type, cluster)] * int(c))
    return allocation[:n]


def _assign_ratings(
    node_types: list[str], rng: np.random.Generator
) -> tuple[list[str], np.ndarray]:
    """Draw a credit rating per node from its type's mix; return (ratings, PDs)."""
    ratings: list[str] = []
    for node_type in node_types:
        mix = _TYPE_RATING_MIX[node_type]
        labels = [r for r, _ in mix]
        probs = np.array([w for _, w in mix], dtype=float)
        probs = probs / probs.sum()
        ratings.append(str(rng.choice(labels, p=probs)))
    pd = np.array([_RATING_PD[r] for r in ratings], dtype=float)
    return ratings, pd


def _scale_free_fitness(
    node_types: list[str], rng: np.random.Generator, gamma: float
) -> np.ndarray:
    """Per-node fitness with a power-law (scale-free) tail, scaled by type size.

    Fitness drives both the link probability (hubs connect more) and the exposure
    weights in the gravity model, producing a core-periphery network rather than a
    homogeneous Erdos-Renyi graph. The power-law exponent ``gamma`` controls tail
    heaviness (empirical interbank ``gamma ~ 2-3``).
    """
    n = len(node_types)
    # Pareto-distributed base fitness: P(f > x) ~ x^-(gamma-1) for the degree tail.
    u = rng.random(n)
    base = (1.0 - u) ** (-1.0 / max(gamma - 1.0, 0.5))
    base = np.clip(base, 1.0, 50.0)  # cap the heaviest hub for numerical sanity
    size = np.array([_TYPE_SIZE[t] for t in node_types], dtype=float)
    return base * size


def make_scalable_system(n: int = 54, seed: int = 11) -> SystemSpec:
    """Create a deterministic scale-free, core-periphery systemic-risk specification.

    Scales to ``n = 54`` (the quantum-hardware target), unlike :func:`make_synthetic_system`
    (capped at 20). Everything is deterministic given ``seed``. The construction follows the
    calibration anchors in ``research/README.md`` (sections 1-3):

    - **Institution mix.** ``n`` nodes allocated across banks / funds / corporates / insurers
      / sovereigns / a CCP by fixed target weights (largest-remainder rounding), with a
      sovereign + CCP backbone guaranteed for ``n >= 8``.
    - **Marginals ``p_i``.** Each node draws a credit rating from its type's rating mix and
      takes the representative one-year PD for that rating (BBB ~0.25%, BB ~1.4%, B ~5.5%,
      CCC ~22%); the speculative-grade aggregate lands in the cited ~3.8-4.2% band.
    - **Topology.** Sparse, **scale-free / core-periphery** (power-law node fitness, exponent
      ``gamma in [2, 3]``, ~20% core) reconstructed with the density-corrected gravity model
      ``p_ij = z chi_i chi_j / (1 + z chi_i chi_j)`` (Cimini et al. 2015) -- *not*
      Erdos-Renyi. Exposures point from each creditor to its debtors.
    - **Balance sheets.** Interbank assets ~20% of total assets distributed as exposures;
      capital buffers ~4-8% of assets; every single-counterparty exposure capped at <=25% of
      the creditor's Tier-1 (the buffer).
    - **Targets.** A ``target_pairwise_corr`` consistent with the exposure graph and cluster
      structure is attached for the copula baselines and diagnostics; couplings for the Ising
      generator are derived from the exposure matrix.
    """
    if not 2 <= n <= 54:
        raise ValueError("n must be between 2 and 54")

    rng = np.random.default_rng(seed)

    allocation = _allocate_types(n)
    node_types = [a[0] for a in allocation]
    clusters = [a[1] for a in allocation]

    # Descriptive, type-specific names ("Global Bank A", "Life Insurer", ...) so the
    # network reads like a named roster rather than anonymous "Bank 01" placeholders.
    node_names = _descriptive_names(node_types)

    ratings, p = _assign_ratings(node_types, rng)

    # --- Topology: scale-free fitness -> density-corrected gravity exposure graph ---
    gamma = float(rng.uniform(2.1, 2.8))
    fitness = _scale_free_fitness(node_types, rng, gamma)

    # Target density: sparse but connected. ~20% of nodes form the high-fitness core.
    # Mean degree grows mildly with n (empirical interbank mean degree ~8-23).
    target_density = float(np.clip(6.0 / max(n - 1, 1), 0.04, 0.35))

    p_link = _gravity_link_probability(fitness, target_density)
    # Directed edges: creditor i lends to debtor j. Draw the support symmetrically
    # (an exposure relationship exists), then assign directed weights both ways.
    iu = np.triu_indices(n, k=1)
    edge_mask = np.zeros((n, n), dtype=bool)
    draws = rng.random(len(iu[0]))
    present = draws < p_link[iu]
    edge_mask[iu[0][present], iu[1][present]] = True
    edge_mask |= edge_mask.T

    # Total assets per node scale with fitness; interbank assets are ~20% of that.
    total_assets = 1.0 + 4.0 * (fitness / fitness.mean())
    interbank_share = np.clip(rng.normal(0.20, 0.03, size=n), 0.10, 0.30)
    interbank_assets = interbank_share * total_assets

    # Gravity weights on present edges: W[i, j] = exposure of creditor i to debtor j,
    # i.e. the loss to i if j defaults (the cascade convention). Distribute each
    # creditor's interbank-asset budget across its debtors proportionally to debtor size.
    W = np.zeros((n, n), dtype=float)
    debtor_pull = fitness.copy()
    for i in range(n):
        debtors = np.nonzero(edge_mask[i])[0]
        if debtors.size == 0:
            continue
        weights = debtor_pull[debtors]
        weights = weights / weights.sum()
        W[i, debtors] = interbank_assets[i] * weights

    # --- Capital buffers (Tier-1): 4-8% of total assets ---
    buffer_ratio = np.clip(rng.normal(0.06, 0.012, size=n), 0.04, 0.08)
    # Systemic backbone (sovereign / CCP) is better capitalised.
    backbone = np.array([t in ("sovereign", "CCP") for t in node_types])
    buffer_ratio = np.where(backbone, np.clip(buffer_ratio + 0.02, 0.04, 0.10), buffer_ratio)
    capital_buffers = buffer_ratio * total_assets

    # --- Single-counterparty cap: each exposure W[i, j] <= 25% of creditor i's Tier-1 ---
    cap = 0.25 * capital_buffers
    for i in range(n):
        row = W[i]
        over = row > cap[i]
        if np.any(over):
            row[over] = cap[i]
        W[i] = row

    # --- Target pairwise correlation consistent with exposure + cluster structure ---
    corr = _exposure_correlation(W, clusters, node_types)

    spec_grade = p[np.isin(ratings, ["BB", "B", "CCC", "CC", "C"])]
    metadata = {
        "name": "Scalable systemic stress network",
        "seed": int(seed),
        "n": int(n),
        "topology": "scale-free / core-periphery (density-corrected gravity model)",
        "gamma": round(gamma, 3),
        "target_density": round(target_density, 4),
        "edge_count": int(edge_mask[iu].sum()),
        "mean_degree": round(float(edge_mask.sum(axis=1).mean()), 3),
        "ratings": ratings,
        "spec_grade_aggregate_pd": round(float(spec_grade.mean()), 5) if spec_grade.size else None,
        "aggregate_pd": round(float(p.mean()), 5),
        "interbank_asset_share_mean": round(float(interbank_share.mean()), 4),
        "capital_buffer_ratio_mean": round(float(buffer_ratio.mean()), 4),
        "single_counterparty_cap_frac_tier1": 0.25,
        "type_counts": {t: int(node_types.count(t)) for t in sorted(set(node_types))},
        "pd_source": _RATING_PD_SOURCE,
        "calibration_sources": [
            _RATING_PD_SOURCE,
            "Cimini et al. 2015 (density-corrected gravity reconstruction)",
            "Bardoscia et al. 2021 (scale-free core-periphery interbank topology)",
            "Gai-Kapadia 2010 / Basel III (capital buffers, single-counterparty cap)",
        ],
        "description": (
            "Programmatic institution mix on a sparse scale-free core-periphery exposure "
            "graph; per-rating marginals, 4-8% Tier-1 buffers, <=25%-of-Tier-1 "
            "single-counterparty cap, ~20% interbank-asset share."
        ),
    }

    return SystemSpec(
        node_names=node_names,
        node_types=node_types,
        exposure_matrix=W,
        capital_buffers=capital_buffers,
        marginal_default_probs=p,
        target_pairwise_corr=corr,
        clusters=clusters,
        metadata=metadata,
    )


def _gravity_link_probability(fitness: np.ndarray, target_density: float) -> np.ndarray:
    """Density-corrected gravity link probabilities ``p_ij = z f_i f_j / (1 + z f_i f_j)``.

    ``z`` is the single global parameter fixed so the expected number of links matches
    ``target_density * n (n - 1) / 2`` (Cimini et al. 2015). Returns a symmetric matrix
    with zero diagonal.
    """
    fitness = np.asarray(fitness, dtype=float)
    n = len(fitness)
    if n < 2:
        return np.zeros((n, n))
    f = fitness / (fitness.mean() + 1e-12)
    target_links = float(np.clip(target_density, 0.0, 1.0)) * n * (n - 1) / 2.0
    outer = np.outer(f, f)
    iu = np.triu_indices(n, k=1)
    pairs = outer[iu]

    def expected_links(z: float) -> float:
        t = z * pairs
        return float(np.sum(t / (1.0 + t)))

    lo, hi = 1e-9, 1e9
    for _ in range(200):
        mid = np.sqrt(lo * hi)
        if expected_links(mid) < target_links:
            lo = mid
        else:
            hi = mid
    z = np.sqrt(lo * hi)

    t = z * outer
    p_link = t / (1.0 + t)
    np.fill_diagonal(p_link, 0.0)
    return (p_link + p_link.T) / 2.0


def _exposure_correlation(
    W: np.ndarray, clusters: list[str], node_types: list[str]
) -> np.ndarray:
    """Build a target Bernoulli correlation matrix from exposure + cluster structure.

    Default (event) correlations are small (research section 2: ``rho_D ~ 0.001-0.03``,
    within-cluster > cross-cluster). We set a small within/cross-cluster base plus a bump
    proportional to mutual exposure strength, then symmetrise. The values stay in a
    realistic low-correlation band; the cascade simulator, not this matrix, is the arbiter
    of joint tail probabilities.
    """
    n = W.shape[0]
    strength = W + W.T
    max_strength = strength.max() if strength.max() > 0 else 1.0
    corr = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            same_cluster = clusters[i] == clusters[j]
            value = 0.015 if same_cluster else 0.004
            value += 0.02 * strength[i, j] / max_strength
            if {"sovereign", "CCP"} & {node_types[i], node_types[j]}:
                value += 0.004
            corr[i, j] = corr[j, i] = float(min(value, 0.08))
    return corr

