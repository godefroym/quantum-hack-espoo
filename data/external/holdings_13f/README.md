# `holdings_13f/` — 13F portfolio-overlap (common-asset / fire-sale) slices

Source for the **portfolio-overlap network**: two institutions are linked when they hold the
same securities, so a forced sale by one marks down the assets of the other (Gualdi, Cimini,
Primicerio, Di Clemente & Challet 2016, *Statistically validated network of portfolio overlaps
and systemic risk*, arXiv:1603.05914). The real-data counterpart of the fire-sale layer in
`src/systemic_risk/edge_metrics.py`.

Parsed by `src/systemic_risk/data_network/sources/holdings_13f.py` (runs offline against a
synthetic panel until the files below are present).

## Layout

- **Raw bulk downloads** live in **`data/external/13F/`** (the large source files):
  - `holdings.csv` (~5 GB) — validated 13F positions, columns
    `cik, rdate, fdate, form, permno, shares, value, accession`. **Asset id = CRSP `permno`**
    (not CUSIP); **quarter = `rdate`** (quarter-end snapshot, e.g. `2008-09-30`).
  - `crspq.csv` (~43 MB) — `permno, ncusip, yearqtr, prc, shrout, split`; gives market cap
    (`|prc| * shrout`) per quarter → asset illiquidity.
  - `biographical/` — manager name ↔ CIK.
- **This directory (`holdings_13f/`)** holds the small **carved slices** the code writes, e.g.
  `holdings_slice_20080930.csv` (one quarter, top-N filers; a few MB). Git-ignored.

## Why slicing

`holdings.csv` is too large to load. `sample_holdings_csv` streams it once in row chunks,
keeps a single quarter (`rdate`) and the top-N filers by AUM, and writes a compact slice here.
Recommended quarters (line up with our committed stress-index peaks): **2007-06-30** (pre),
**2008-09-30** (Lehman), **2008-12-31** (peak), **2010-12-31** (recovery).

## Build it

```bash
# stream 2008-Q3 -> slice -> overlap network (validated + cosine + fire-sale) + plot
uv run python scripts/build_13f_overlap.py --rdate 2008-09-30 --top 250
```

Outputs → `outputs/data_network/overlap_13f/`: `overlap_13f.npz` (all matrices + institution
ids), `summary.json`, `overlap_network.png`.

```python
from systemic_risk.data_network.sources.holdings_13f import (
    load_or_sample_holdings, holdings_matrix, validated_overlap_network,
    crsp_illiquidity, rdate_to_yearqtr, directed_fire_sale_matrix,
)
panel = load_or_sample_holdings(("2008-09-30",), top_institutions=250)   # streams once, caches
mat = holdings_matrix(panel, quarter="2008-09-30", min_positions=3, min_assets_held=5)
svn, _ = validated_overlap_network(mat)
ill = crsp_illiquidity(mat, yearqtr=rdate_to_yearqtr("2008-09-30"))
fire = directed_fire_sale_matrix(mat, illiquidity=ill)
```
