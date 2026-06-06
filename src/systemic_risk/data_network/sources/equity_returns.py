"""Real equity-return co-movement -> empirical correlation matrix.

The correlation matrix is the genuinely-real network signal in the empirical layer: it
is estimated from daily equity log-returns of the roster banks (banks that co-move under
common shocks cluster together). It doubles as the latent asset-return correlation that
the Gaussian/Student-t copula baselines threshold into correlated defaults (the Merton /
single-factor ASRF reading), so it feeds B/C/D directly.

Data path:

1. Prefer the committed snapshot ``data/external/banks/equity_corr.csv`` (+ ``.meta.json``)
   so the whole pipeline runs offline and reproducibly.
2. Otherwise fetch daily adjusted closes from the Yahoo chart API (stdlib ``urllib`` only,
   no extra dependency) over a pinned window, compute log-return correlations, and — if
   asked — write the snapshot back.

Yahoo is rate-limited without a browser User-Agent; the keyless endpoint is best-effort
and is *not* a hard dependency of the test suite (the committed snapshot is).
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import time
import urllib.error
import urllib.request

import numpy as np

_BANKS_DIR = Path(__file__).resolve().parents[4] / "data" / "external" / "banks"
DEFAULT_CORR_CSV = _BANKS_DIR / "equity_corr.csv"
DEFAULT_META_JSON = _BANKS_DIR / "equity_corr.meta.json"

# Pinned estimation window for the committed snapshot (reproducible).
DEFAULT_START = "2021-06-01"
DEFAULT_END = "2024-06-01"

_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)
_CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"


@dataclass(frozen=True)
class EquityCorrelation:
    """An estimated equity-return correlation matrix with its provenance."""

    tickers: tuple[str, ...]
    matrix: np.ndarray  # (n, n) symmetric, unit diagonal
    start: str
    end: str
    n_obs: int
    source: str

    def reordered(self, tickers: list[str]) -> np.ndarray:
        """Return the correlation submatrix in the requested ticker order."""
        index = {t: i for i, t in enumerate(self.tickers)}
        missing = [t for t in tickers if t not in index]
        if missing:
            raise KeyError(f"correlation matrix missing tickers: {missing}")
        order = [index[t] for t in tickers]
        return self.matrix[np.ix_(order, order)]


def _unix(date: str) -> int:
    return int(time.mktime(time.strptime(date, "%Y-%m-%d")))


def _fetch_adjclose(ticker: str, start: str, end: str, timeout: float = 20.0) -> dict[int, float]:
    """Fetch {unix_day: adjusted_close} for one ticker from the Yahoo chart API."""
    url = (
        _CHART_URL.format(ticker=ticker)
        + f"?period1={_unix(start)}&period2={_unix(end)}&interval=1d"
    )
    request = urllib.request.Request(url, headers={"User-Agent": _UA})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))
    result = payload["chart"]["result"]
    if not result:
        raise ValueError(f"empty chart result for {ticker}")
    block = result[0]
    timestamps = block.get("timestamp") or []
    adj = block["indicators"].get("adjclose")
    closes = adj[0]["adjclose"] if adj else block["indicators"]["quote"][0]["close"]
    series: dict[int, float] = {}
    for ts, px in zip(timestamps, closes):
        if px is not None:
            series[int(ts) // 86400] = float(px)  # bucket by day index
    return series


def fetch_correlation(
    tickers: list[str],
    start: str = DEFAULT_START,
    end: str = DEFAULT_END,
    pause: float = 0.3,
) -> EquityCorrelation:
    """Fetch prices, align on common trading days, and correlate daily log-returns.

    Tickers that fail to fetch or have too little history are dropped (with the survivors
    kept), so a transient single-ticker failure does not sink the run.
    """
    prices: dict[str, dict[int, float]] = {}
    for ticker in tickers:
        try:
            series = _fetch_adjclose(ticker, start, end)
        except (urllib.error.URLError, KeyError, ValueError, TimeoutError):
            continue
        if len(series) > 30:
            prices[ticker] = series
        time.sleep(pause)

    kept = [t for t in tickers if t in prices]
    if len(kept) < 2:
        raise RuntimeError("equity fetch returned fewer than 2 usable tickers")

    # Align on days present for every kept ticker.
    common_days = set.intersection(*(set(prices[t]) for t in kept))
    days = sorted(common_days)
    if len(days) < 30:
        raise RuntimeError(f"only {len(days)} common trading days across tickers")

    price_matrix = np.array([[prices[t][d] for d in days] for t in kept], dtype=float)
    log_returns = np.diff(np.log(price_matrix), axis=1)
    corr = np.corrcoef(log_returns)
    corr = _clean_correlation(corr)
    return EquityCorrelation(
        tickers=tuple(kept),
        matrix=corr,
        start=start,
        end=end,
        n_obs=log_returns.shape[1],
        source=f"Yahoo Finance daily adjusted close, log-return correlation ({start}..{end})",
    )


def _clean_correlation(corr: np.ndarray) -> np.ndarray:
    corr = np.asarray(corr, dtype=float)
    corr = (corr + corr.T) / 2.0
    corr = np.clip(corr, -1.0, 1.0)
    np.fill_diagonal(corr, 1.0)
    return corr


def save_snapshot(
    ec: EquityCorrelation,
    corr_csv: str | Path = DEFAULT_CORR_CSV,
    meta_json: str | Path = DEFAULT_META_JSON,
) -> None:
    """Write the correlation matrix (CSV, ticker-labelled) and a metadata sidecar."""
    corr_csv = Path(corr_csv)
    corr_csv.parent.mkdir(parents=True, exist_ok=True)
    header = "," + ",".join(ec.tickers)
    lines = [header]
    for i, ticker in enumerate(ec.tickers):
        cells = ",".join(f"{v:.6f}" for v in ec.matrix[i])
        lines.append(f"{ticker},{cells}")
    corr_csv.write_text("\n".join(lines) + "\n", encoding="utf-8")
    Path(meta_json).write_text(
        json.dumps(
            {
                "tickers": list(ec.tickers),
                "start": ec.start,
                "end": ec.end,
                "n_obs": ec.n_obs,
                "source": ec.source,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def load_snapshot(
    corr_csv: str | Path = DEFAULT_CORR_CSV,
    meta_json: str | Path = DEFAULT_META_JSON,
) -> EquityCorrelation:
    """Load a committed correlation snapshot."""
    corr_csv = Path(corr_csv)
    rows = corr_csv.read_text(encoding="utf-8").strip().splitlines()
    header = rows[0].split(",")[1:]
    tickers = [h.strip().upper() for h in header]
    matrix = np.array([[float(x) for x in r.split(",")[1:]] for r in rows[1:]], dtype=float)
    if matrix.shape != (len(tickers), len(tickers)):
        raise ValueError("correlation snapshot is not square / header-consistent")
    meta_path = Path(meta_json)
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    return EquityCorrelation(
        tickers=tuple(tickers),
        matrix=_clean_correlation(matrix),
        start=str(meta.get("start", DEFAULT_START)),
        end=str(meta.get("end", DEFAULT_END)),
        n_obs=int(meta.get("n_obs", matrix.shape[0])),
        source=str(meta.get("source", "committed equity-correlation snapshot")),
    )


def load_or_fetch_correlation(
    tickers: list[str],
    *,
    prefer_snapshot: bool = True,
    write_snapshot: bool = False,
    start: str = DEFAULT_START,
    end: str = DEFAULT_END,
) -> EquityCorrelation:
    """Return the equity correlation, preferring the committed snapshot.

    If ``prefer_snapshot`` and the snapshot exists and covers all requested tickers, use
    it. Otherwise fetch live; if ``write_snapshot`` is set, persist the fetched result.
    """
    if prefer_snapshot and DEFAULT_CORR_CSV.exists():
        snapshot = load_snapshot()
        if set(tickers).issubset(set(snapshot.tickers)):
            return snapshot
    ec = fetch_correlation(tickers, start=start, end=end)
    if write_snapshot:
        save_snapshot(ec)
    return ec
