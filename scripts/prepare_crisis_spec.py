"""Prepare the 2008-crisis institutions as a runnable spec for a future quantum run.

Takes the fetched daily prices in ``data/external/crisis_2008/prices/`` and the
roster, computes the real equity-return correlation (the dependency structure the
entangled generator / copula baselines encode), and assembles a ``SystemSpec``
ready to load into the generator pipeline. Marginals are provisional placeholders
(documented) so the spec validates and can be sampled; replace them with
rating-derived PDs when an actual run is configured.

Outputs (in data/external/crisis_2008/):
* ``equity_corr.csv`` (+ ``equity_corr.meta.json``) — correlation matrix.
* ``quantum_spec.json`` / ``.npz`` — the SystemSpec (marginals + correlation).

Usage:
    uv run python scripts/prepare_crisis_spec.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from systemic_risk.spec import SystemSpec

ROOT = Path(__file__).resolve().parents[1]
DIR = ROOT / "data" / "external" / "crisis_2008"
ROSTER = DIR / "affected_institutions.csv"
PRICES = DIR / "prices"

# provisional marginal default probabilities (1y), pending rating-derived PDs
PROVISIONAL_PD = {"bank": 0.010, "insurer": 0.012}


def load_prices(rid: str) -> dict[str, float]:
    path = PRICES / f"{rid}.csv"
    if not path.exists():
        return {}
    out: dict[str, float] = {}
    with path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                out[row["date"]] = float(row["adjclose"])
            except (KeyError, ValueError):
                continue
    return out


def main() -> None:
    roster = list(csv.DictReader(ROSTER.open(encoding="utf-8")))
    # keep only institutions that have a fetched price series
    rows = [r for r in roster if (PRICES / f"{r['id']}.csv").exists()]
    ids = [r["id"] for r in rows]
    series = {r["id"]: load_prices(r["id"]) for r in rows}

    # align on the common trading days across all series
    common = set.intersection(*(set(series[i].keys()) for i in ids))
    dates = sorted(common)
    if len(dates) < 50:
        raise SystemExit(f"only {len(dates)} common trading days; too few")

    # log-returns -> correlation
    prices = np.array([[series[i][d] for d in dates] for i in ids])  # (n, T)
    logret = np.diff(np.log(prices), axis=1)  # (n, T-1)
    corr = np.corrcoef(logret)
    corr = (corr + corr.T) / 2
    np.fill_diagonal(corr, 1.0)
    corr = np.clip(corr, -1.0, 1.0)

    n = len(ids)
    types = [r["type"] for r in rows]
    marginals = np.array([PROVISIONAL_PD.get(t, 0.01) for t in types])

    spec = SystemSpec(
        node_names=ids,
        node_types=types,
        exposure_matrix=np.zeros((n, n)),
        capital_buffers=np.ones(n),
        marginal_default_probs=marginals,
        target_pairwise_corr=corr,
        clusters=[r["region"] for r in rows],
        metadata={
            "source": "data/external/crisis_2008 (equity correlation 2007-2010)",
            "marginals": "PROVISIONAL placeholders, not rating-derived",
            "window_days": len(dates),
            "window": [dates[0], dates[-1]],
        },
    )
    spec.save_json(DIR / "quantum_spec.json")
    spec.save_npz(DIR / "quantum_spec.npz")

    # also write the correlation matrix in the banks/equity_corr.csv style
    with (DIR / "equity_corr.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([""] + ids)
        for i, rid in enumerate(ids):
            w.writerow([rid] + [f"{corr[i, j]:.6f}" for j in range(n)])
    (DIR / "equity_corr.meta.json").write_text(
        json.dumps(
            {
                "tickers": ids,
                "n": n,
                "n_obs": len(dates),
                "start": dates[0],
                "end": dates[-1],
                "source": "Yahoo daily adjusted closes; log-return Pearson correlation",
            },
            indent=2,
        )
    )

    print(f"prepared spec for {n} institutions over {len(dates)} common days "
          f"({dates[0]}..{dates[-1]})")
    print(f"  wrote {DIR / 'quantum_spec.json'}")
    print(f"  wrote {DIR / 'equity_corr.csv'}")
    # quick sanity: strongest off-diagonal correlations
    iu = np.triu_indices(n, 1)
    order = np.argsort(corr[iu])[::-1][:5]
    print("  top correlations:")
    for k in order:
        a, b = iu[0][k], iu[1][k]
        print(f"    {ids[a]:6s} {ids[b]:6s} {corr[a, b]:.3f}")


if __name__ == "__main__":
    main()
