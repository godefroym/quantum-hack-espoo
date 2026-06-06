# `holdings_13f/` — 13F institutional holdings (common-asset / fire-sale channel)

Source for the **portfolio-overlap network**: two institutions are linked when they hold the
same securities, so a forced sale by one marks down the assets of the other (Gualdi, Cimini,
Primicerio, Di Clemente & Challet 2016, *Statistically validated network of portfolio overlaps
and systemic risk*, arXiv:1603.05914). This is the real-data counterpart of the fire-sale
layer left out of `src/systemic_risk/edge_metrics.py`.

Parsed by `src/systemic_risk/data_network/sources/holdings_13f.py` (runs offline against a
synthetic panel until the files below are present).

## What to download

From the **EDGAR-Parsing** project (https://elsaifym.github.io/EDGAR-Parsing/ →
[Dropbox](https://www.dropbox.com/sh/27mxydmiume3t0e/AADyZkVZjwZe5Id9n7FbGTlea?dl=0)),
validated SEC Form 13F holdings, 1999–2020. Place files here:

| File | Need | Why |
|---|---|---|
| `holdings.csv` | **essential** | validated positions (CIK, CUSIP, value, shares) → the institution × asset matrix |
| `biographical.csv` | yes | manager name ↔ CIK → node identities / roster |
| `crspq.csv` | recommended | CRSP prices/volume/market-cap → asset **illiquidity** for the fire-sale weighting |
| `holdings_raw.csv`, `cusip_universe/`, `master_files/`, `raw_filings/`, `raw_tables/`, `processed_tables/`, `errors.csv` | skip | raw parsing intermediates; only use `processed_tables/` (keyed by CIK) if `holdings.csv` lacks a CIK+quarter key |

**Tips:** don't pull all of 1999–2020 — a couple of quarters (e.g. **2008Q3** crisis + a recent
one) tell the before/during-crisis story and stay small. Confirm `holdings.csv` carries a
**CIK** and **quarter/period** column; if the names differ, pass an explicit `ColumnSpec` (the
loader otherwise auto-detects common spellings).

This directory is git-ignored for the large CSVs; only this README is committed.

## Quick use

```python
from systemic_risk.data_network.sources.holdings_13f import (
    load_holdings, holdings_matrix, validated_overlap_network,
    liquidity_weighted_overlap, load_illiquidity, directed_fire_sale_matrix,
)

panel = load_holdings("data/external/holdings_13f/holdings.csv")
mat = holdings_matrix(panel, quarter="2008Q3", min_assets_held=5,
                      min_positions=2, top_institutions=200)

svn, pvals = validated_overlap_network(mat)          # statistically-validated overlap links
ill = load_illiquidity(mat)                           # per-asset price impact from crspq.csv
fire = directed_fire_sale_matrix(mat, illiquidity=ill)  # directed loss-to-j-when-i-sells
```
