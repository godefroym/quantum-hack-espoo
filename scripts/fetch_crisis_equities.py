"""Fetch daily equity prices for the 2008-crisis institution roster.

Reads ``data/external/crisis_2008/affected_institutions.csv`` and downloads daily
adjusted closes for each listed ``yahoo_ticker`` over the crisis window from the
Yahoo chart API (stdlib urllib only, browser User-Agent, same approach as
``systemic_risk.data_network.sources.equity_returns``). Each series is written to
``data/external/crisis_2008/prices/<id>.csv`` and a ``fetch_log.json`` records
what succeeded.

Some entries have no usable Yahoo series (e.g. thinly-covered exchanges); those
are skipped and logged, not treated as errors. The keyless endpoint is
best-effort, so re-run if rate-limited.

Usage:
    uv run python scripts/fetch_crisis_equities.py
    uv run python scripts/fetch_crisis_equities.py --start 2007-01-01 --end 2010-12-31
"""

from __future__ import annotations

import argparse
import csv
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIR = ROOT / "data" / "external" / "crisis_2008"
ROSTER = DIR / "affected_institutions.csv"
PRICES = DIR / "prices"

_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)
_CHART = "https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"


def _unix(date: str) -> int:
    return int(time.mktime(time.strptime(date, "%Y-%m-%d")))


def fetch_series(ticker: str, start: str, end: str, timeout: float = 25.0) -> list[tuple[str, float]]:
    """Return [(YYYY-MM-DD, adjusted_close)] for one ticker, or [] on no data."""
    url = (
        _CHART.format(ticker=urllib.request.quote(ticker))
        + f"?period1={_unix(start)}&period2={_unix(end)}&interval=1d"
    )
    req = urllib.request.Request(url, headers={"User-Agent": _UA})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    result = (payload.get("chart") or {}).get("result")
    if not result:
        return []
    res = result[0]
    stamps = res.get("timestamp") or []
    indicators = res.get("indicators") or {}
    adj = (indicators.get("adjclose") or [{}])[0].get("adjclose")
    closes = (indicators.get("quote") or [{}])[0].get("close")
    series_vals = adj if adj else closes
    if not stamps or not series_vals:
        return []
    out: list[tuple[str, float]] = []
    for ts, v in zip(stamps, series_vals):
        if v is None:
            continue
        day = time.strftime("%Y-%m-%d", time.gmtime(ts))
        out.append((day, float(v)))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2007-01-01")
    ap.add_argument("--end", default="2010-12-31")
    args = ap.parse_args()

    PRICES.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(ROSTER.open(encoding="utf-8")))
    log: list[dict] = []

    for r in rows:
        rid = r["id"]
        yt = r["yahoo_ticker"].strip()
        status: dict = {"id": rid, "yahoo_ticker": yt}
        try:
            series = fetch_series(yt, args.start, args.end)
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
            status.update(ok=False, error=f"{type(e).__name__}: {e}", n_obs=0)
            print(f"  {rid:7s} {yt:10s} FAILED {type(e).__name__}", flush=True)
            log.append(status)
            time.sleep(0.7)
            continue

        if not series:
            status.update(ok=False, error="no data on Yahoo", n_obs=0)
            print(f"  {rid:7s} {yt:10s} no data", flush=True)
        else:
            out_path = PRICES / f"{rid}.csv"
            with out_path.open("w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["date", "adjclose"])
                w.writerows(series)
            status.update(ok=True, n_obs=len(series), first=series[0][0], last=series[-1][0])
            print(f"  {rid:7s} {yt:10s} {len(series)} obs  {series[0][0]}..{series[-1][0]}", flush=True)
        log.append(status)
        time.sleep(0.7)  # be polite to the keyless endpoint

    ok = sum(1 for s in log if s.get("ok"))
    (DIR / "fetch_log.json").write_text(
        json.dumps(
            {"start": args.start, "end": args.end, "n_ok": ok, "n_total": len(log), "results": log},
            indent=2,
        )
    )
    print(f"\nfetched {ok}/{len(log)} series into {PRICES}")


if __name__ == "__main__":
    main()
