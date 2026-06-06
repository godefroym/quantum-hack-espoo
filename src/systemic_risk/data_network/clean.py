"""Normalization and ID reconciliation.

Turns the raw roster rows into clean, reconciled, deterministically-ordered node records:
canonical node ids, normalized ratings (S&P notation -> whole-letter PD key), and a single
ordered table the downstream estimate/reconstruct/cluster steps all index by position.
"""

from __future__ import annotations

from dataclasses import dataclass

from systemic_risk.data_network.sources.roster import RosterRow

# S&P / Moody's whole-letter buckets we price PDs at (matches the Moody's Exhibit-17 keys).
_WHOLE_LETTERS = ("AAA", "AA", "A", "BBB", "BB", "B", "CCC")


def normalize_sp_rating(raw: str) -> str:
    """Canonicalize an S&P rating string (upper-case, strip stray spaces)."""
    return raw.strip().upper().replace(" ", "")


def whole_letter(rating: str) -> str:
    """Collapse a notched S&P rating to its whole-letter PD bucket.

    ``A-`` -> ``A``, ``BBB+`` -> ``BBB``, ``AA-`` -> ``AA``, ``BB-`` -> ``BB``. Anything at
    or below CCC maps to ``CCC``; unknown strings fall back to ``BBB`` (mid investment grade).
    """
    r = normalize_sp_rating(rating).rstrip("+-")
    # Drop numeric notches if present (e.g. Moody's-style "A1").
    r = "".join(ch for ch in r if ch.isalpha())
    if r in _WHOLE_LETTERS:
        return r
    if r.startswith("CC") or r.startswith("C") or r.startswith("D"):
        return "CCC"
    if r.startswith("AAA"):
        return "AAA"
    if r.startswith("AA"):
        return "AA"
    if r.startswith("BBB"):
        return "BBB"
    if r.startswith("BB"):
        return "BB"
    if r.startswith("B"):
        return "B"
    if r.startswith("A"):
        return "A"
    return "BBB"


# SystemSpec node classes the roster may carry.
_VALID_NODE_TYPES = {"bank", "insurer", "fund", "corporate", "sovereign", "CCP"}


@dataclass(frozen=True)
class CleanNode:
    """A reconciled institution record, indexed by position in the spec."""

    node_id: str
    name: str
    ticker: str
    country: str
    region: str
    business_type: str
    node_type: str          # the SystemSpec class: "bank", "corporate", ...
    sp_rating: str
    rating_bucket: str      # whole-letter PD key
    total_assets_usd_bn: float


def reconcile(rows: tuple[RosterRow, ...]) -> tuple[CleanNode, ...]:
    """Reconcile and normalize roster rows into an ordered tuple of clean nodes."""
    nodes: list[CleanNode] = []
    seen: set[str] = set()
    for row in rows:
        node_id = row.bank_id.strip().upper()
        if node_id in seen:
            raise ValueError(f"duplicate node id after reconciliation: {node_id}")
        seen.add(node_id)
        sp = normalize_sp_rating(row.sp_rating)
        node_type = row.node_type if row.node_type in _VALID_NODE_TYPES else "bank"
        nodes.append(
            CleanNode(
                node_id=node_id,
                name=row.name,
                ticker=row.ticker.strip().upper(),
                country=row.country,
                region=row.region,
                business_type=row.business_type,
                node_type=node_type,
                sp_rating=sp,
                rating_bucket=whole_letter(sp),
                total_assets_usd_bn=row.total_assets_usd_bn,
            )
        )
    return tuple(nodes)
