"""The real anchor loader — a curated roster of real systemically important banks.

Why a curated roster rather than a raw EBA / FFIEC bulk download: the supervisory bulk
files (EBA Transparency Exercise, FR Y-15) are gated behind browser bot-challenges and
bespoke multi-sheet schemas (see ``data/external/CATALOG.md``). The roster committed at
``data/external/banks/gsib_roster.csv`` is a small, fully reproducible real anchor: each
row is a real, publicly listed bank with its public S&P long-term issuer rating and an
approximate total-assets figure from its FY2023 report. From it we derive:

- **marginals ``p_i``** — rating -> 1-year PD via the committed Moody's table
  (``estimate.marginals_from_ratings``);
- **node totals** — interbank asset / liability sums used as the *constraints* for the
  bilateral-exposure reconstruction (real bilateral data is confidential);
- **equity tickers** — the keys for the real equity-return correlation matrix.

The total-assets figures are approximate public values (rounded to the nearest ~$10bn)
used only as a *relative scale* anchor for reconstruction; they are not precise
accounting figures and are not used as marginals.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

# data/external/banks/gsib_roster.csv, four levels up from this file.
DEFAULT_ROSTER_CSV = (
    Path(__file__).resolve().parents[4]
    / "data"
    / "external"
    / "banks"
    / "gsib_roster.csv"
)


@dataclass(frozen=True)
class RosterRow:
    """One real institution in the anchor roster (a bank or a non-financial corporate)."""

    bank_id: str
    name: str
    ticker: str
    country: str
    region: str
    node_type: str          # SystemSpec class: "bank" | "corporate" | ...
    business_type: str
    sp_rating: str
    total_assets_usd_bn: float
    source: str


def load_roster(path: str | Path | None = None) -> tuple[RosterRow, ...]:
    """Load and lightly validate the real bank roster CSV.

    Returns rows in file order (deterministic). Raises ``FileNotFoundError`` if the
    committed roster is missing and ``ValueError`` on malformed / duplicate rows.
    """
    csv_path = Path(path) if path is not None else DEFAULT_ROSTER_CSV
    if not csv_path.exists():
        raise FileNotFoundError(f"roster CSV not found: {csv_path}")

    rows: list[RosterRow] = []
    seen_ids: set[str] = set()
    seen_tickers: set[str] = set()
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "bank_id",
            "name",
            "ticker",
            "country",
            "region",
            "business_type",
            "sp_rating",
            "total_assets_usd_bn",
            "source",
        }
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"roster CSV missing columns: {sorted(missing)}")
        for line in reader:
            bank_id = line["bank_id"].strip()
            ticker = line["ticker"].strip().upper()
            if not bank_id or not ticker:
                raise ValueError("roster rows must have non-empty bank_id and ticker")
            if bank_id in seen_ids:
                raise ValueError(f"duplicate bank_id in roster: {bank_id}")
            if ticker in seen_tickers:
                raise ValueError(f"duplicate ticker in roster: {ticker}")
            seen_ids.add(bank_id)
            seen_tickers.add(ticker)
            try:
                assets = float(line["total_assets_usd_bn"])
            except ValueError as exc:  # pragma: no cover - defensive
                raise ValueError(
                    f"bad total_assets_usd_bn for {bank_id}: {line['total_assets_usd_bn']!r}"
                ) from exc
            if assets <= 0:
                raise ValueError(f"total_assets_usd_bn must be positive for {bank_id}")
            rows.append(
                RosterRow(
                    bank_id=bank_id,
                    name=line["name"].strip(),
                    ticker=ticker,
                    country=line["country"].strip(),
                    region=line["region"].strip(),
                    # node_type is optional for backward compatibility with older rosters.
                    node_type=(line.get("node_type") or "bank").strip() or "bank",
                    business_type=line["business_type"].strip(),
                    sp_rating=line["sp_rating"].strip(),
                    total_assets_usd_bn=assets,
                    source=line["source"].strip(),
                )
            )
    if not rows:
        raise ValueError("roster CSV contains no rows")
    return tuple(rows)
