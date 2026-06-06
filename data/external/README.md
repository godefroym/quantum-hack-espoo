# `data/external/` — Real-world calibration & validation data

Free, public datasets that calibrate and validate the systemic-stress generator.
Each dataset maps to one model parameter: `p_i` (marginal default propensity),
`J_ij` (pairwise coupling/exposure), **events** (ground-truth tail), or
**macro-state** (stress regime). See **[`CATALOG.md`](CATALOG.md)** for the full
per-dataset provenance, license, source URL, and parameter mapping.

## Layout

```
data/external/
├── CATALOG.md         # authoritative: one entry per dataset + model-parameter mapping
├── fetch.sh           # reproduce every download (run: bash data/external/fetch.sh)
├── README.md          # this file
├── fdic/              # FDIC failed-bank events  -> events
│   ├── failures.csv             (2000-2024, 587 rows)
│   └── failures_1980_2024.csv   (1980-2024, 3548 rows; S&L + GFC clusters)
├── fred/              # FRED credit/stress series -> macro-state (+ p_i proxies)
│   ├── AAA.csv  BAA.csv          (Moody's Aaa/Baa yields; BAA-AAA = credit spread)
│   ├── BAA10Y.csv                (Baa - 10Y Treasury spread, daily)
│   ├── BAMLH0A0HYM2.csv          (ICE BofA HY OAS; keyless = 2023+ ONLY, see note)
│   ├── KCFSI.csv  STLFSI4.csv    (KC Fed / St. Louis Fed stress indices)
│   ├── NFCI.csv   ANFCI.csv      (Chicago Fed financial-conditions indices)
│   └── VIXCLS.csv               (CBOE VIX, daily)
├── ecb/
│   └── ciss_euro_area.csv        (ECB CISS systemic-stress index, daily since 1980)
├── ratings/           # rating-agency PDs -> p_i
│   ├── moodys_pd_by_rating.csv   (1-yr PD by rating; Moody's Exhibits 17 & 19)
│   └── Moodys_Default_1920-2004.pdf
├── vlab/              # NYU V-Lab SRISK -> p_i  (stub: documented-only, see CATALOG.md)
├── ffiec/             # FR Y-15 / FR Y-9C -> J_ij (documented-only)
└── bis/               # BIS consolidated banking stats -> J_ij (documented-only)
```

## Coverage guarantee (the stress episodes we care about)

The generator must reproduce **correlated tail events**, so the calibration data must
contain the 2008 GFC and the COVID-2020 shock:

- **STLFSI4** peaks at **9.67 on 2008-10-10** (Lehman) and spikes in 2020.
- **KCFSI** peaks at **5.82 on 2008-11-01**; **NFCI/ANFCI** carry full weekly 2008 + 2020 history.
- **BAA-AAA credit spread** (from `AAA.csv`/`BAA.csv`, full monthly history since 1919)
  blows out in 2008 and 2020 — the stand-in for the high-yield OAS series whose keyless
  history starts only in 2023.
- **ECB CISS** covers **1980→present** daily (every European crisis incl. 2008/2020/2011 sovereign).
- **FDIC `failures_1980_2024.csv`** holds two real co-failure clusters: the S&L crisis
  (1988–1992, hundreds/yr) and the GFC (2009–2011) — the empirical right-tail to validate against.

## Refresh

```bash
bash data/external/fetch.sh
```

Idempotent (re-fetches & overwrites). Requirements: `bash`, `curl`, `python3`
(+ optional `pdftotext` to re-extract the Moody's table).

### Known fetch caveats (all encoded in `fetch.sh`)

- **FRED HTTP 504s.** The keyless `fredgraph.csv` endpoint intermittently times out
  (HTTP 504) on full-history single requests and under concurrent load. The script
  fetches the heavy **daily** series (`BAA10Y`, `VIXCLS`) in multi-year **slices** and
  concatenates, and retries every request with backoff. If a `fred/*.csv` is missing or
  empty after a run, simply re-run `fetch.sh` (the edge recovers).
- **`BAMLH0A0HYM2` keyless = 2023+ only.** Full 1996+ history needs a free FRED API key
  (`export FRED_API_KEY=...`, see the bottom of `fetch.sh`). The BAA-AAA spread covers
  2008/2020 in the meantime.
- **`OFRFSI` not on FRED keyless** (404). Use the OFR portal CSV or the FRED API; see CATALOG.md.
- **Documented-only sources** (V-Lab SRISK, FFIEC FR Y-15/Y-9C, BIS consolidated stats):
  blocked by client-side rendering / bot challenge / file size. `fetch.sh` prints the exact
  verified URLs + methods; full details and the field→`J_ij` mapping are in `CATALOG.md`.

## Licensing (summary)

US-government sources (FDIC, FRED Fed-compiled series, OFR, FFIEC) are **public domain**.
FRED-redistributed third-party series (ICE BofA, CBOE) carry the **provider's** terms
(free, non-commercial). **ECB** and **BIS** are free with attribution. The **Moody's** study
is used from a free academic PDF mirror for research. Per-dataset licenses are in `CATALOG.md`.
Everything here is free/public; no proprietary or confidential supervisory data is included
(real bilateral interbank matrices are confidential — hence the gravity-model `J_ij`
reconstruction; see `research/sections/04_systemic_risk_measures.md` §4).
