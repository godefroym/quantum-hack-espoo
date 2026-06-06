"""13F institutional holdings -> portfolio-overlap (common-asset / fire-sale) network.

This is the *common-asset contagion* channel that the counterparty graph cannot see: two
institutions are linked when they hold the same securities, so a forced sale by one marks down
the assets of the other (Gualdi, Cimini, Primicerio, Di Clemente & Challet 2016, "Statistically
validated network of portfolio overlaps and systemic risk", arXiv:1603.05914 — note Cimini is
the same author as the gravity reconstruction in ``reconstruct.py``). It is the real-data
counterpart of the fire-sale layer deliberately left out of ``systemic_risk.edge_metrics``.

Pipeline:

    holdings.csv (CIK, CUSIP, value, shares, quarter)
        -> HoldingsPanel (long form, normalized columns)
        -> HoldingsMatrix  H[institution, asset]  (dollar value)
        -> overlap measures:
             * cosine_overlap            — weighted portfolio similarity (symmetric)
             * validated_overlap_network — hypergeometric statistically-validated links
             * liquidity_weighted_overlap — fire-sale co-holding, optionally illiquidity-scaled
             * directed_fire_sale_matrix — asymmetric loss-to-j-when-i-deleverages

Source data: the EDGAR-Parsing project (https://elsaifym.github.io/EDGAR-Parsing/) publishes
``holdings.csv`` (validated 13F positions, 1999-2020) and ``crspq.csv`` (CRSP prices/volume for
asset illiquidity). Download those into ``data/external/holdings_13f/`` (see the module README).
Nothing here requires the files at import time; a :func:`synthetic_holdings_panel` lets the
whole module run and be tested offline.

NB: the column names below are *guessed* from the standard 13F information-table schema and the
EDGAR-Parsing outputs; :class:`ColumnSpec.detect` resolves them case-insensitively against a
list of candidates, and you can always pass an explicit :class:`ColumnSpec`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_HOLDINGS_DIR = (
    Path(__file__).resolve().parents[4] / "data" / "external" / "holdings_13f"
)
DEFAULT_HOLDINGS_CSV = DEFAULT_HOLDINGS_DIR / "holdings.csv"
DEFAULT_CRSP_CSV = DEFAULT_HOLDINGS_DIR / "crspq.csv"


# --------------------------------------------------------------------------- #
# Column resolution (robust to the real file's naming)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ColumnSpec:
    """Resolved column names in a holdings table."""

    institution: str       # filer id (CIK)
    asset: str             # security id (CUSIP)
    value: str             # position market value
    shares: str | None = None
    quarter: str | None = None   # reporting period

    # Candidate names tried (case-insensitive) when auto-detecting.
    _INSTITUTION = ("cik", "manager_cik", "filer_cik", "managercik", "filer")
    _ASSET = ("cusip", "cusip8", "cusip9", "cusip6")
    _VALUE = ("value", "position_value", "dollar_value", "mktval", "marketvalue", "val")
    _SHARES = ("shares", "shares_held", "sshprnamt", "ssh_prnamt", "shrs", "quantity")
    _QUARTER = ("quarter", "period", "report_period", "reportperiod", "qtr", "date",
                "period_of_report", "filing_period")

    @classmethod
    def detect(cls, columns: list[str]) -> "ColumnSpec":
        lower = {c.lower().strip(): c for c in columns}

        def pick(cands: tuple[str, ...], required: bool, what: str) -> str | None:
            for cand in cands:
                if cand in lower:
                    return lower[cand]
            if required:
                raise KeyError(
                    f"could not find a {what} column among {list(columns)}; "
                    f"tried {cands}. Pass an explicit ColumnSpec."
                )
            return None

        return cls(
            institution=pick(cls._INSTITUTION, True, "institution/CIK"),  # type: ignore[arg-type]
            asset=pick(cls._ASSET, True, "asset/CUSIP"),                  # type: ignore[arg-type]
            value=pick(cls._VALUE, True, "position-value"),              # type: ignore[arg-type]
            shares=pick(cls._SHARES, False, "shares"),
            quarter=pick(cls._QUARTER, False, "quarter"),
        )


# --------------------------------------------------------------------------- #
# Panel + matrix
# --------------------------------------------------------------------------- #
@dataclass
class HoldingsPanel:
    """Long-form holdings: one row per (institution, asset, quarter) position."""

    df: pd.DataFrame   # normalized columns: institution, asset, value, [shares], [quarter]
    spec: ColumnSpec

    @property
    def quarters(self) -> list[str]:
        if "quarter" not in self.df.columns:
            return []
        return sorted(self.df["quarter"].dropna().astype(str).unique().tolist())

    def for_quarter(self, quarter: str | None) -> pd.DataFrame:
        if quarter is None or "quarter" not in self.df.columns:
            return self.df
        return self.df[self.df["quarter"].astype(str) == str(quarter)]


@dataclass(frozen=True)
class HoldingsMatrix:
    """Dense institution x asset dollar-holdings matrix for one quarter."""

    H: np.ndarray                       # (n_institutions, n_assets), nonnegative
    institutions: tuple[str, ...]
    assets: tuple[str, ...]
    quarter: str | None = None

    @property
    def n_institutions(self) -> int:
        return self.H.shape[0]

    @property
    def n_assets(self) -> int:
        return self.H.shape[1]

    def weights(self) -> np.ndarray:
        """Row-normalized portfolio weights (each institution sums to 1)."""
        totals = self.H.sum(axis=1, keepdims=True)
        return np.divide(self.H, totals, out=np.zeros_like(self.H), where=totals > 0)

    def binary(self) -> np.ndarray:
        """Binary holdings incidence (1 if the institution holds the asset)."""
        return (self.H > 0).astype(int)


def load_holdings(
    path: str | Path = DEFAULT_HOLDINGS_CSV,
    spec: ColumnSpec | None = None,
    nrows: int | None = None,
) -> HoldingsPanel:
    """Load and normalize a 13F holdings CSV into a :class:`HoldingsPanel`.

    Resolves the institution/asset/value/shares/quarter columns (auto-detected unless ``spec``
    is given), coerces value/shares to numbers, drops non-positive or unkeyed positions, and
    aggregates duplicate (institution, asset, quarter) rows by summing value/shares.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"13F holdings file not found: {path}. Download holdings.csv from "
            "https://elsaifym.github.io/EDGAR-Parsing/ into data/external/holdings_13f/."
        )
    raw = pd.read_csv(path, nrows=nrows, dtype=str, low_memory=False)
    spec = spec or ColumnSpec.detect(list(raw.columns))
    return _normalize(raw, spec)


def panel_from_frame(df: pd.DataFrame, spec: ColumnSpec | None = None) -> HoldingsPanel:
    """Build a panel from an already-loaded frame (handy for tests / custom ingestion)."""
    spec = spec or ColumnSpec.detect(list(df.columns))
    return _normalize(df.astype({spec.value: float}, errors="ignore"), spec)


def _normalize(raw: pd.DataFrame, spec: ColumnSpec) -> HoldingsPanel:
    cols = {spec.institution: "institution", spec.asset: "asset", spec.value: "value"}
    if spec.shares:
        cols[spec.shares] = "shares"
    if spec.quarter:
        cols[spec.quarter] = "quarter"
    df = raw[list(cols)].rename(columns=cols).copy()

    df["institution"] = df["institution"].astype(str).str.strip()
    df["asset"] = df["asset"].astype(str).str.strip().str.upper()
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    if "shares" in df.columns:
        df["shares"] = pd.to_numeric(df["shares"], errors="coerce")
    if "quarter" in df.columns:
        df["quarter"] = df["quarter"].astype(str).str.strip()

    df = df[(df["value"] > 0) & df["institution"].ne("") & df["asset"].ne("")]
    group_keys = ["institution", "asset"] + (["quarter"] if "quarter" in df.columns else [])
    agg = {"value": "sum"}
    if "shares" in df.columns:
        agg["shares"] = "sum"
    df = df.groupby(group_keys, as_index=False).agg(agg)
    return HoldingsPanel(df=df.reset_index(drop=True), spec=spec)


def holdings_matrix(
    panel: HoldingsPanel,
    quarter: str | None = None,
    *,
    min_positions: int = 1,
    min_assets_held: int = 1,
    top_institutions: int | None = None,
    value_col: str = "value",
) -> HoldingsMatrix:
    """Pivot a panel into a dense institution x asset matrix for one quarter.

    Filters keep the problem tractable on the full 13F universe (thousands x thousands):

    - ``min_assets_held`` — drop tiny portfolios (institutions holding fewer than this many
      distinct assets) — overlap is meaningless for 1-2 line portfolios.
    - ``min_positions`` — drop assets held by fewer than this many institutions (a co-holding
      needs at least 2 holders to matter).
    - ``top_institutions`` — keep only the largest-AUM institutions (by summed value).
    """
    df = panel.for_quarter(quarter)
    if df.empty:
        raise ValueError(f"no holdings for quarter={quarter!r}")

    if top_institutions is not None:
        aum = df.groupby("institution")[value_col].sum().nlargest(top_institutions)
        df = df[df["institution"].isin(aum.index)]

    pivot = df.pivot_table(index="institution", columns="asset",
                           values=value_col, aggfunc="sum", fill_value=0.0)

    if min_assets_held > 1:
        pivot = pivot[(pivot > 0).sum(axis=1) >= min_assets_held]
    if min_positions > 1:
        pivot = pivot.loc[:, (pivot > 0).sum(axis=0) >= min_positions]
    # Drop any institutions/assets left fully empty after column/row trimming.
    pivot = pivot.loc[(pivot > 0).any(axis=1), (pivot > 0).any(axis=0)]
    if pivot.shape[0] < 2 or pivot.shape[1] < 1:
        raise ValueError("too few institutions/assets after filtering for an overlap network")

    return HoldingsMatrix(
        H=pivot.to_numpy(dtype=float),
        institutions=tuple(str(i) for i in pivot.index),
        assets=tuple(str(a) for a in pivot.columns),
        quarter=quarter,
    )


# --------------------------------------------------------------------------- #
# Overlap measures
# --------------------------------------------------------------------------- #
def cosine_overlap(matrix: HoldingsMatrix) -> np.ndarray:
    """Weighted portfolio-similarity overlap: cosine of row-normalized weight vectors.

    ``O[i, j] = (w_i . w_j) / (||w_i|| ||w_j||)`` in [0, 1] (holdings are nonnegative).
    Symmetric, unit diagonal-free (diagonal set to 0).
    """
    w = matrix.weights()
    norms = np.linalg.norm(w, axis=1, keepdims=True)
    wn = np.divide(w, norms, out=np.zeros_like(w), where=norms > 0)
    overlap = wn @ wn.T
    np.fill_diagonal(overlap, 0.0)
    return np.clip(overlap, 0.0, 1.0)


def validated_overlap_network(
    matrix: HoldingsMatrix,
    alpha: float = 0.01,
    bonferroni: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Statistically validated overlap network (Gualdi et al. 2016; Tumminello et al. 2011).

    For each pair, the number of commonly held assets ``c_ij`` is compared to the
    hypergeometric null that preserves each institution's diversification (it holds ``n_i`` of
    ``N`` assets at random): ``P(X >= c_ij) = hypergeom.sf(c_ij - 1, N, n_j, n_i)``. A link is
    *validated* (over-expressed overlap) when that p-value is below ``alpha``, Bonferroni-
    corrected over all institution pairs by default.

    Returns ``(adjacency, pvalues)`` — ``adjacency[i, j] = 1`` for validated links (symmetric,
    zero diagonal); ``pvalues`` carries the raw upper-tail p-values.
    """
    from scipy.stats import hypergeom

    B = matrix.binary()
    n = B.sum(axis=1)                       # assets held per institution
    N = B.shape[1]                          # universe size (assets this quarter)
    common = B @ B.T                        # c_ij = shared assets
    m = B.shape[0]

    ni = n[:, None] * np.ones((1, m), dtype=int)
    nj = n[None, :] * np.ones((m, 1), dtype=int)
    # P(X >= c) = sf(c-1); population N, successes n_j, draws n_i.
    pvalues = hypergeom.sf(common - 1, N, nj, ni)
    pvalues = np.clip(pvalues, 0.0, 1.0)
    np.fill_diagonal(pvalues, 1.0)

    n_pairs = m * (m - 1) / 2.0
    threshold = alpha / n_pairs if (bonferroni and n_pairs > 0) else alpha
    adjacency = ((pvalues < threshold) & (common > 0)).astype(int)
    adjacency = np.triu(adjacency, 1)
    adjacency = adjacency + adjacency.T
    return adjacency, pvalues


def liquidity_weighted_overlap(
    matrix: HoldingsMatrix,
    illiquidity: np.ndarray | None = None,
) -> np.ndarray:
    """Fire-sale co-holding matrix: ``LWO[i, j] = sum_a illiquidity_a * H[i,a] * H[j,a]``.

    With ``illiquidity`` per asset (e.g. an Amihud-style price-impact coefficient from CRSP),
    this approximates the mark-to-market loss inflicted on ``j`` when ``i`` liquidates common
    holdings (Greenwood, Landier & Thesmar 2015). If ``illiquidity`` is None, all assets are
    weighted equally and ``LWO`` is the plain dollar co-holding matrix. Symmetric, zero
    diagonal. Scale is arbitrary; normalize before comparing across quarters.
    """
    H = matrix.H
    if illiquidity is None:
        ill = np.ones(H.shape[1])
    else:
        ill = np.asarray(illiquidity, dtype=float)
        if ill.shape != (H.shape[1],):
            raise ValueError("illiquidity must have one entry per asset (column)")
    weighted = H * ill[None, :]
    lwo = weighted @ H.T
    np.fill_diagonal(lwo, 0.0)
    return lwo


def directed_fire_sale_matrix(
    matrix: HoldingsMatrix,
    illiquidity: np.ndarray | None = None,
    deleverage: np.ndarray | None = None,
) -> np.ndarray:
    """Directed fire-sale loss: ``F[i, j]`` = loss to ``j`` when ``i`` deleverages.

    ``F[i, j] = deleverage_i * sum_a illiquidity_a * (H[i,a] / sum_b H[i,b]) * H[j,a]`` —
    institution ``i`` sells a fraction ``deleverage_i`` of its book pro-rata across its
    holdings; the price impact on each common asset (``illiquidity_a`` times the dollar sold)
    marks down ``j``'s holding of it. Asymmetric by construction (``F[i,j] != F[j,i]``), which
    is the directionality a symmetrized "mutual overlap" loses. ``deleverage`` defaults to 1
    for every institution (use leverage targets for a Greenwood-Landier-Thesmar calibration).
    """
    H = matrix.H
    m, k = H.shape
    ill = np.ones(k) if illiquidity is None else np.asarray(illiquidity, dtype=float)
    if ill.shape != (k,):
        raise ValueError("illiquidity must have one entry per asset (column)")
    delv = np.ones(m) if deleverage is None else np.asarray(deleverage, dtype=float)
    if delv.shape != (m,):
        raise ValueError("deleverage must have one entry per institution (row)")

    weights = matrix.weights()                  # seller sells pro-rata across its book
    sold = (delv[:, None] * weights) * ill[None, :]   # dollar-impact sold per asset, per seller
    F = sold @ H.T                              # marked-down loss to each holder j
    np.fill_diagonal(F, 0.0)
    return F


# --------------------------------------------------------------------------- #
# Asset illiquidity from CRSP (optional)
# --------------------------------------------------------------------------- #
def load_illiquidity(
    matrix: HoldingsMatrix,
    crsp_path: str | Path = DEFAULT_CRSP_CSV,
    *,
    cusip_col_candidates: tuple[str, ...] = ("cusip", "cusip8", "cusip9"),
    impact_col_candidates: tuple[str, ...] = ("amihud", "illiquidity", "price_impact"),
    mktcap_col_candidates: tuple[str, ...] = ("mktcap", "market_cap", "me", "marketcap"),
    volume_col_candidates: tuple[str, ...] = ("vol", "volume", "dollar_volume", "dvol"),
    default: float = 1.0,
) -> np.ndarray:
    """Per-asset illiquidity aligned to ``matrix.assets``, from a CRSP-style quarterly file.

    Prefers an explicit price-impact/Amihud column; else uses ``1 / market_cap`` or
    ``1 / dollar_volume`` as a price-impact proxy (bigger, more-traded assets absorb sales
    with less impact). Assets missing from CRSP get ``default``. Returns a vector aligned to
    ``matrix.assets``. This is the only place CRSP is needed; overlap/SVN work without it.
    """
    crsp_path = Path(crsp_path)
    if not crsp_path.exists():
        raise FileNotFoundError(
            f"CRSP file not found: {crsp_path}. Download crspq.csv from EDGAR-Parsing, or call "
            "the overlap functions without illiquidity for an equal-weighted fire-sale matrix."
        )
    df = pd.read_csv(crsp_path, dtype=str, low_memory=False)
    lower = {c.lower().strip(): c for c in df.columns}

    def find(cands: tuple[str, ...]) -> str | None:
        for c in cands:
            if c in lower:
                return lower[c]
        return None

    cusip_col = find(cusip_col_candidates)
    if cusip_col is None:
        raise KeyError(f"no CUSIP column in CRSP file among {list(df.columns)}")
    df[cusip_col] = df[cusip_col].astype(str).str.strip().str.upper()

    impact_col = find(impact_col_candidates)
    if impact_col is not None:
        series = pd.to_numeric(df[impact_col], errors="coerce")
    else:
        mktcap_col = find(mktcap_col_candidates) or find(volume_col_candidates)
        if mktcap_col is None:
            raise KeyError(
                "CRSP file has no amihud/illiquidity, market-cap, or volume column to derive "
                f"price impact from; columns were {list(df.columns)}"
            )
        denom = pd.to_numeric(df[mktcap_col], errors="coerce")
        series = 1.0 / denom.where(denom > 0)

    table = (
        pd.DataFrame({"cusip": df[cusip_col], "ill": series})
        .dropna()
        .groupby("cusip")["ill"].mean()
    )
    return np.array([float(table.get(a, default)) for a in matrix.assets], dtype=float)


# --------------------------------------------------------------------------- #
# Synthetic panel (offline demo / tests)
# --------------------------------------------------------------------------- #
def synthetic_holdings_panel(
    n_institutions: int = 30,
    n_assets: int = 80,
    n_blocks: int = 3,
    seed: int = 0,
    quarter: str = "2008Q3",
) -> HoldingsPanel:
    """A small, block-structured synthetic 13F panel for offline demos and tests.

    Institutions are split into ``n_blocks`` style groups; institutions in the same block hold
    overlapping assets (so the validated-overlap network recovers the blocks), plus a diffuse
    market-wide core every institution holds a little of. Deterministic given ``seed``.
    """
    rng = np.random.default_rng(seed)
    block_of = rng.integers(0, n_blocks, size=n_institutions)
    assets_per_block = n_assets // (n_blocks + 1)
    core = np.arange(assets_per_block)  # market-wide assets everyone holds a little of

    rows: list[dict] = []
    for i in range(n_institutions):
        b = int(block_of[i])
        block_assets = np.arange(assets_per_block * (b + 1), assets_per_block * (b + 2))
        # Hold most of the block's assets (strong, statistically-validatable overlap with
        # same-block peers) plus a thin slice of the market-wide core (diffuse noise).
        n_block = max(3, int(0.8 * len(block_assets)))
        held = set(rng.choice(block_assets, size=n_block, replace=False))
        held |= set(rng.choice(core, size=max(1, len(core) // 5), replace=False))
        for a in held:
            value = float(rng.lognormal(mean=12.0, sigma=1.0))  # ~ position in $000s
            rows.append({
                "cik": f"CIK{i:04d}",
                "cusip": f"CUSIP{a:05d}",
                "value": value,
                "sshprnamt": float(value / rng.uniform(5, 50)),
                "quarter": quarter,
            })
    df = pd.DataFrame(rows)
    return panel_from_frame(df)
