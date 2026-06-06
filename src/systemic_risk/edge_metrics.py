"""Risk-adjusted, directed edge weights for the exposure network.

A raw exposure ``notional[i, j]`` (creditor ``i``'s gross claim on debtor ``j``) overstates
how much loss actually propagates in a crisis. Five channels modulate it — each a per-edge,
*directed* multiplier so ``A``'s loss when ``B`` fails need not equal the reverse:

1. **Loss-given-default (recovery / collateralization / seniority)** — only the unrecovered
   fraction ``(1 - recovery)`` is lost. Secured/senior claims on well-collateralized debtors
   (CCPs, sovereigns) recover far more than unsecured claims on speculative-grade corporates.
2. **Maturity / rollover risk** — short-tenor funding must be rolled over precisely when it is
   hardest, so it transmits stress more violently. A short maturity raises the weight.
3. **Wrong-way risk / conditionality** — an exposure to a counterparty that fails *together*
   with the creditor (high co-movement) is worth less than its notional suggests; it activates
   when it hurts most. High positive correlation raises the weight.
4. **Concentration / substitutability** — a claim on a non-substitutable provider (a CCP, a
   sovereign, a sole payment utility), or one that is a large share of the creditor's book,
   is harder to replace and transmits more.
5. **Directionality** — all of the above are computed per ordered pair ``(i, j)``, so the
   effective matrix is genuinely asymmetric (unlike a symmetrised "mutual exposure").

``effective[i, j] = notional[i, j] * lgd[i,j] * maturity_stress[i,j]
                                  * wrong_way[i,j] * substitutability[i,j]``

The result is what the cascade simulator should propagate (loss to ``i`` if ``j`` defaults).
Everything is **deterministic** given the inputs — no RNG — so it is reproducible and needs
no extra state to round-trip. The notional matrix is kept alongside for transparency.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np

# --- defaults grounded in rating-agency recovery studies + Basel maturity buckets ---------
# Senior-unsecured recovery is ~40%; secured/collateralized links recover much more. CCP and
# sovereign exposures are the most collateralized (margin / sovereign seniority).
_DEFAULT_RECOVERY_BY_TYPE: dict[str, float] = {
    "bank": 0.45,
    "insurer": 0.40,
    "fund": 0.30,
    "corporate": 0.40,
    "sovereign": 0.55,
    "CCP": 0.70,
}

# Extra recovery haircut by the *debtor's* rating bucket (worse credit -> lower recovery).
_RECOVERY_RATING_ADJ: dict[str, float] = {
    "AAA": 0.10, "AA": 0.07, "A": 0.04, "BBB": 0.0,
    "BB": -0.05, "B": -0.10, "CCC": -0.18,
}

# Representative tenor (years) of a claim, keyed by the debtor's type. Interbank / financial
# funding is short (rollover-fragile); corporate loans and sovereign bonds are longer.
_DEFAULT_MATURITY_YEARS_BY_TYPE: dict[str, float] = {
    "bank": 0.5,
    "insurer": 1.5,
    "fund": 0.5,
    "corporate": 4.0,
    "sovereign": 3.0,
    "CCP": 0.1,
}

# Non-substitutable providers transmit more (you cannot quickly replace a CCP or a sovereign).
_DEFAULT_SUBSTITUTABILITY_BY_TYPE: dict[str, float] = {
    "bank": 1.0,
    "insurer": 1.0,
    "fund": 1.0,
    "corporate": 1.0,
    "sovereign": 1.2,
    "CCP": 1.4,
}


@dataclass(frozen=True)
class EdgeMetricConfig:
    """Tunable coefficients for the five edge channels."""

    recovery_by_type: Mapping[str, float] = field(
        default_factory=lambda: dict(_DEFAULT_RECOVERY_BY_TYPE))
    recovery_rating_adj: Mapping[str, float] = field(
        default_factory=lambda: dict(_RECOVERY_RATING_ADJ))
    maturity_years_by_type: Mapping[str, float] = field(
        default_factory=lambda: dict(_DEFAULT_MATURITY_YEARS_BY_TYPE))
    substitutability_by_type: Mapping[str, float] = field(
        default_factory=lambda: dict(_DEFAULT_SUBSTITUTABILITY_BY_TYPE))
    rollover_beta: float = 0.6      # strength of the short-maturity penalty
    rollover_tau: float = 1.0       # years; maturities << tau get the full penalty
    wrong_way_gamma: float = 0.5    # strength of the correlation-conditional penalty
    concentration_delta: float = 0.5  # strength of the single-name concentration penalty
    default_recovery: float = 0.40
    default_substitutability: float = 1.0
    default_maturity_years: float = 1.0

    def to_summary(self) -> dict[str, float]:
        """Scalar-only summary for provenance / metadata (no per-type dicts)."""
        return {
            "rollover_beta": self.rollover_beta,
            "rollover_tau": self.rollover_tau,
            "wrong_way_gamma": self.wrong_way_gamma,
            "concentration_delta": self.concentration_delta,
        }


@dataclass(frozen=True)
class EdgeMetrics:
    """Per-edge directed component matrices and the combined effective-loss matrix."""

    notional: np.ndarray
    recovery: np.ndarray
    lgd: np.ndarray
    maturity_years: np.ndarray
    maturity_stress: np.ndarray
    wrong_way: np.ndarray
    substitutability: np.ndarray
    effective: np.ndarray

    def components_dict(self) -> dict[str, list]:
        """JSON-serializable component matrices (for metadata / visualization)."""
        return {
            "notional": self.notional.tolist(),
            "recovery": self.recovery.tolist(),
            "lgd": self.lgd.tolist(),
            "maturity_years": self.maturity_years.tolist(),
            "maturity_stress": self.maturity_stress.tolist(),
            "wrong_way": self.wrong_way.tolist(),
            "substitutability": self.substitutability.tolist(),
        }


def compute_edge_metrics(
    notional: np.ndarray,
    node_types: list[str],
    *,
    ratings: list[str] | None = None,
    correlation: np.ndarray | None = None,
    config: EdgeMetricConfig | None = None,
) -> EdgeMetrics:
    """Combine the five channels into a directed effective-loss matrix.

    ``notional[i, j]`` is creditor ``i``'s gross claim on debtor ``j``. Per-edge values key
    off the *debtor* ``j`` for recovery/maturity/substitutability (a claim's risk is driven by
    who you are exposed to) and off the ordered pair for wrong-way (co-movement of ``i`` and
    ``j``) and concentration (``j``'s share of ``i``'s book) — keeping the result asymmetric.
    """
    cfg = config or EdgeMetricConfig()
    W = np.asarray(notional, dtype=float)
    n = W.shape[0]
    if W.shape != (n, n):
        raise ValueError("notional must be square")
    if len(node_types) != n:
        raise ValueError("node_types must have one entry per node")

    # --- per-debtor (column) base vectors ---
    rec_base = np.array(
        [cfg.recovery_by_type.get(t, cfg.default_recovery) for t in node_types], dtype=float)
    if ratings is not None:
        if len(ratings) != n:
            raise ValueError("ratings must have one entry per node")
        rec_adj = np.array(
            [cfg.recovery_rating_adj.get(_bucket(r), 0.0) for r in ratings], dtype=float)
        rec_base = rec_base + rec_adj
    rec_base = np.clip(rec_base, 0.05, 0.95)

    mat_years_col = np.array(
        [cfg.maturity_years_by_type.get(t, cfg.default_maturity_years) for t in node_types],
        dtype=float)
    sub_col = np.array(
        [cfg.substitutability_by_type.get(t, cfg.default_substitutability) for t in node_types],
        dtype=float)

    # Broadcast debtor (column) vectors across creditors (rows).
    recovery = np.tile(rec_base, (n, 1))
    lgd = 1.0 - recovery
    maturity_years = np.tile(mat_years_col, (n, 1))
    maturity_stress = 1.0 + cfg.rollover_beta * np.exp(-maturity_years / max(cfg.rollover_tau, 1e-9))

    # Wrong-way: only positive co-movement amplifies; needs a correlation matrix.
    if correlation is not None:
        corr = np.asarray(correlation, dtype=float)
        wrong_way = 1.0 + cfg.wrong_way_gamma * np.clip(corr, 0.0, 1.0)
    else:
        wrong_way = np.ones((n, n))

    # Concentration: debtor j's share of creditor i's total book (row-normalized notional).
    row_sums = W.sum(axis=1, keepdims=True)
    share = np.divide(W, row_sums, out=np.zeros_like(W), where=row_sums > 0)
    substitutability = np.tile(sub_col, (n, 1)) * (1.0 + cfg.concentration_delta * share)

    effective = W * lgd * maturity_stress * wrong_way * substitutability
    np.fill_diagonal(effective, 0.0)

    # Zero-out every component on absent edges so the matrices read cleanly.
    mask = W > 0
    for m in (recovery, lgd, maturity_years, maturity_stress, wrong_way, substitutability):
        m *= mask
    return EdgeMetrics(
        notional=W,
        recovery=recovery,
        lgd=lgd,
        maturity_years=maturity_years,
        maturity_stress=maturity_stress,
        wrong_way=wrong_way,
        substitutability=substitutability,
        effective=effective,
    )


def _bucket(rating: str) -> str:
    """Collapse an S&P-style rating to its whole-letter bucket (``A-`` -> ``A``)."""
    r = (rating or "").strip().upper().rstrip("+-")
    r = "".join(ch for ch in r if ch.isalpha())
    for key in ("AAA", "AA", "A", "BBB", "BB", "B"):
        if r.startswith(key):
            return key
    if r:
        return "CCC"
    return "BBB"
