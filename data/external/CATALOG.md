# External Data Catalog

Provenance and model-parameter mapping for every external dataset under `data/external/`.

**Model parameters** (see `research/sections/04_systemic_risk_measures.md`, Part D):
- `p_i` — per-institution marginal default / distress propensity (Ising field `h_i = logit(p_i)`).
- `J_ij` — pairwise coupling / bilateral exposure between institutions.
- `events` — realised default labels (ground-truth tail events to validate the generator).
- `macro-state` — system-level stress regime used to condition / scale `J_ij` in stressed periods.

**Status legend:** ✅ downloaded · 📝 documented-only (verified URL, not committed) · ⚠️ partial.

All sources are free / public. Licenses recorded per row. URLs verified during this session
(2026-06-06) unless marked **[UNVERIFIED]**.

---

## Downloaded datasets

### `fdic/failures.csv` — FDIC failed-bank list (2000–2024) ✅
- **Contents:** Real US bank failures & assisted transactions. Columns: `CERT, CITYST, COST, FAILDATE, ID, NAME, QBFASSET` (total assets, $000), `QBFDEP` (total deposits, $000), `RESTYPE` (resolution type).
- **Granularity:** one row per failed/assisted institution (event-level). 587 rows.
- **Coverage:** 2000-01-14 → 2024. Includes the 2008–2010 GFC failure wave.
- **Format:** CSV. **License:** US Government work — public domain.
- **Source:** FDIC BankFind Suite — Failures API. `https://api.fdic.gov/banks/failures`
- **Maps to:** **events** (ground-truth correlated-cascade tail; failure clustering by year is the co-default signal).
- *Note:* preserved as originally pulled by the earlier agent; do not overwrite.

### `fdic/failures_1980_2024.csv` — FDIC failed-bank list (1980–2024, extended) ✅
- **Contents:** Same schema as above plus a `FAILYR` column. 3,548 rows (2,960 `FAILURE` + 589 `ASSISTANCE`).
- **Coverage:** 1980 → 2024. Adds the **S&L-crisis cluster** (1988=470, 1989=534, 1990=382, 1991=271, 1992=181 events/yr) on top of the GFC cluster (2009=148, 2010=157, 2011=92).
- **Format:** CSV. **License:** US Government work — public domain.
- **Source:** FDIC BankFind Suite — Failures API, `filters=FAILYR:[1980 TO 2024]`.
  `https://api.fdic.gov/banks/failures?fields=NAME,CERT,CITYST,FAILDATE,QBFDEP,QBFASSET,COST,RESTYPE,FAILYR&filters=FAILYR:[1980%20TO%202024]&limit=10000&format=csv`
- **Maps to:** **events** (preferred long-history file: two distinct correlated-failure clusters for tail validation).

### `fred/AAA.csv` — Moody's Seasoned Aaa Corporate Bond Yield ✅
- **Contents:** `observation_date, AAA` (percent, monthly). 1,289 rows, **1919-01-01 → 2026-05-01**.
- **Format:** CSV. **License:** FRED redistribution of a Fed-compiled series (public; Moody's underlying).
- **Source:** `https://fred.stlouisfed.org/graph/fredgraph.csv?id=AAA&cosd=1990-01-01`
- **Maps to:** **macro-state** (credit-cycle level); with `BAA` forms the **BAA-AAA credit spread**, a market-implied stress / PD proxy → also informs `p_i` scaling.

### `fred/BAA.csv` — Moody's Seasoned Baa Corporate Bond Yield ✅
- **Contents:** `observation_date, BAA` (percent, monthly). 1,289 rows, **1919-01-01 → 2026-05-01**.
- **Source:** `https://fred.stlouisfed.org/graph/fredgraph.csv?id=BAA&cosd=1990-01-01`
- **Maps to:** **macro-state** / `p_i` proxy. **BAA-AAA spread peaks at the 2008 credit crisis** (and 1932), confirming GFC coverage even though the keyless HY-OAS series (below) starts only in 2023.
- **License:** as AAA.

### `fred/BAMLH0A0HYM2.csv` — ICE BofA US High Yield Index Option-Adjusted Spread ⚠️
- **Contents:** `observation_date, BAMLH0A0HYM2` (percent, daily). 795 rows, **2023-06-06 → 2026-06-04**.
- **LIMITATION:** the keyless FRED graph endpoint serves **only 2023-06-06 onward** for this licensed ICE BofA series (confirmed: both the CSV endpoint and the FRED "table data" HTML page return the same 796-date 2023–2026 window regardless of `cosd`). **It therefore MISSES 2008 and COVID-2020.**
  - **Full 1996+ history** requires the authenticated FRED API with a key:
    `https://api.stlouisfed.org/fred/series/observations?series_id=BAMLH0A0HYM2&api_key=YOUR_KEY&file_type=csv` (free key: https://fredaccount.stlouisfed.org/apikey).
  - **Workaround already in place:** the **BAA-AAA spread** (from `AAA.csv`/`BAA.csv`, full history) covers the 2008/2020 credit-stress episodes for `macro-state` purposes.
- **Format:** CSV. **License:** FRED/ICE BofA (free for non-commercial; ICE provider terms).
- **Maps to:** **macro-state** (high-yield credit-spread stress; usable 2023+ only via keyless).

### `fred/KCFSI.csv` — Kansas City Fed Financial Stress Index ✅
- **Contents:** `observation_date, KCFSI` (monthly z-score-style; >0 = above-average stress). 436 rows, **1990-02-01 → 2026-05-01**.
- **Coverage check:** **peaks at 5.82 on 2008-11-01** (GFC) and spikes in 2020Q1–Q2 (COVID). ✅
- **Source:** `https://fred.stlouisfed.org/graph/fredgraph.csv?id=KCFSI&cosd=1990-01-01`
- **Maps to:** **macro-state** (crisis-period label for regime-conditional `J_ij`).
- **License:** Federal Reserve Bank of Kansas City via FRED — free.

### `fred/STLFSI4.csv` — St. Louis Fed Financial Stress Index v4 ✅
- **Contents:** `observation_date, STLFSI4` (weekly; mean 0). 1,692 rows, **1993-12-31 → 2026-05-29**.
- **Coverage check:** **peaks at 9.67 on 2008-10-10** (Lehman) and spikes in 2020. ✅
- **Source:** `https://fred.stlouisfed.org/graph/fredgraph.csv?id=STLFSI4&cosd=1990-01-01` (series natural start 1993).
- **Maps to:** **macro-state**.
- **License:** Federal Reserve Bank of St. Louis via FRED — free.

### `fred/NFCI.csv` — Chicago Fed National Financial Conditions Index ✅
- **Contents:** `observation_date, NFCI` (weekly; mean 0, SD 1 since 1973). 1,900 rows, **1990-01-05 → 2026-05-29**.
- **Coverage check:** 52 weekly obs in 2008 + 8 in 2020Q1–Q2. ✅
- **Source:** `https://fred.stlouisfed.org/graph/fredgraph.csv?id=NFCI&cosd=1990-01-01`
- **Maps to:** **macro-state** (broad financial-conditions regime).
- **License:** Federal Reserve Bank of Chicago via FRED — free.

### `fred/ANFCI.csv` — Chicago Fed Adjusted NFCI (cycle-adjusted) ✅
- **Contents:** `observation_date, ANFCI` (weekly; NFCI with macro cycle removed). 1,900 rows, **1990-01-05 → 2026-05-29**.
- **Source:** `https://fred.stlouisfed.org/graph/fredgraph.csv?id=ANFCI&cosd=1990-01-01`
- **Maps to:** **macro-state** (financial stress orthogonal to the business cycle).
- **License:** Federal Reserve Bank of Chicago via FRED — free.

### `fred/BAA10Y.csv` — Moody's Baa Corporate minus 10Y Treasury spread ✅
- **Contents:** `observation_date, BAA10Y` (percent, daily). **5,066 rows, 2007-01-02 → 2026-06-04** (contiguous; covers 2008 GFC + COVID-2020). The keyless endpoint 504-times-out on a full-history single request, so it is pulled in multi-year **windows** and concatenated (de-duplicated by date) — see `fetch.sh`.
- **Source (per window):** `https://fred.stlouisfed.org/graph/fredgraph.csv?id=BAA10Y&cosd=<start>&coed=<end>`
- **Maps to:** **macro-state** + `p_i` proxy (daily corporate credit-risk premium; the daily analogue of BAA-AAA).
- **License:** Fed-compiled (Moody's underlying) via FRED — free.
- *If the file is absent/empty, re-run `fetch.sh` (FRED edge intermittently returns HTTP 504; windowed fetch + retries recovers it). Pre-2007 history available by adding earlier windows.*

### `fred/VIXCLS.csv` — CBOE Volatility Index (VIX), close ✅
- **Contents:** `observation_date, VIXCLS` (daily). **5,065 rows, 2007-01-03 → 2026-06-04** (contiguous; covers 2008 GFC + COVID-2020). Pulled in multi-year **windows** and concatenated (full-history single request 504-times-out) — see `fetch.sh`.
- **Source (per window):** `https://fred.stlouisfed.org/graph/fredgraph.csv?id=VIXCLS&cosd=<start>&coed=<end>`
- **Maps to:** **macro-state** (market fear / volatility regime — a CoVaR/ΔCoVaR state variable).
- **License:** CBOE via FRED — free for non-commercial.
- *If absent/empty, re-run `fetch.sh` (transient FRED 504s; windowed fetch + retries recovers it). Pre-2007 history available by adding earlier windows.*

### `ecb/ciss_euro_area.csv` — ECB Composite Indicator of Systemic Stress (CISS) ✅
- **Contents:** `observation_date, CISS` (daily, euro area; pure number 0–1, rises when stress is correlated across market segments). 12,119 rows, **1980-01-03 → 2026-06-04**. (Trimmed from the full SDMX response to date+value.)
- **Format:** CSV (from SDMX). **License:** European Central Bank — free, attribution expected.
- **Source:** ECB Data Portal SDMX:
  `https://data-api.ecb.europa.eu/service/data/CISS/D.U2.Z0Z.4F.EC.SS_CIN.IDX?format=csvdata`
- **Maps to:** **macro-state** (EU systemic-stress regime; its portfolio-theory aggregation is conceptually the "co-failure" weighting we put into `J_ij`).

### `ratings/moodys_pd_by_rating.csv` — Moody's one-year default rates by rating ✅
- **Contents:** `rating, one_year_pd` (decimal fraction; 0.0456 = 4.56%), `source`. 30 rows = two cohorts:
  Exhibit 19 (alphanumeric Aaa…Caa-C, 1983–2004) and Exhibit 17 (whole-letter Aaa…Caa-C, 1920–2004),
  each with IG / SG / All-Rated aggregates. Year-1 column of the cumulative-default-rate tables.
- **Format:** CSV. **License:** free academic PDF mirror (Boston University); Moody's study, cited for research.
- **Source:** Moody's *Corporate Default & Recovery Rates 1920–2004* —
  `https://www.bu.edu/econ/files/2015/01/Moodys_Default_1920-2004.pdf` (also committed as
  `ratings/Moodys_Default_1920-2004.pdf`, Exhibits 17 & 19).
- **Maps to:** **`p_i`** (the primary marginal default-probability table; `h_i = logit(p_i)`).

### `ratings/Moodys_Default_1920-2004.pdf` — source PDF ✅
- The full study (1.9 MB) retained for provenance/audit of the extracted table above. **License:** free PDF mirror.

---

## Documented-only (verified URL/method; not committed) 📝

### `vlab/srisk.csv` — NYU Stern V-Lab SRISK rankings 📝 (stub written, no rows)
- **Wanted:** firm, SRISK%, LRMES, leverage (LVG) — firm-level, weekly.
- **Why not fetched:** `https://vlab.stern.nyu.edu/srisk` is a **client-rendered Next.js app**. Column
  labels ("SRISK ($m)", "LRMES", "LVG") are confirmed in the page i18n bundle, but the firm rows are
  loaded by a runtime client-side data call — **not** present in the static HTML or the `__NEXT_DATA__`
  blob (which holds only geo/country metadata). `/_next/data/<buildId>/srisk.json` returns 404 (build
  id rotates); per-portfolio analysis pages (e.g. `…/analysis/RISK.USFIN-MR.MES`) likewise embed no
  numeric rows. Extracting requires executing the page JS (headless browser) to capture the backing
  XHR — not possible from a fetch-only environment. We do not fabricate rows.
- **How to obtain:** (1) manual copy/export of the rendered table; (2) Playwright/Puppeteer to load the
  page and capture the table's network response (inspect DevTools → Network for the exact backend URL +
  params, which are not in the static bundle); (3) docs: `https://vlab.stern.nyu.edu/docs/srisk`.
- **License:** free, open-access (academic, NYU Stern). Attribution expected.
- **Maps to:** **`p_i`** (SRISK% / LRMES = ready-made distress-propensity per firm).
- *Full method recorded inline in `vlab/srisk.csv`.*

### FFIEC FR Y-15 (Systemic Risk Report) + FR Y-9C 📝 (`ffiec/` empty)
- **Wanted:** GSIB interconnectedness / size / complexity line items (intra-financial assets &
  liabilities, secured/unsecured wholesale funding) → balance-sheet edges for `J_ij`; FR Y-9C
  consolidated financials → bottom-up `p_i` (leverage / capital → distance-to-default).
- **Why not fetched:** `https://www.ffiec.gov/npw/FinancialReport/FinancialDataDownload` returns
  **HTTP 403** to non-browser clients (Sec-CH-UA bot challenge). Needs a real browser session /
  manual download.
- **How to obtain:** open the Financial Data Download UI in a browser and download the FR Y-9C and
  FR Y-15 **caret(^)-delimited TXT** bulk files (FR Y-9C back to 2000; FR Y-15 annual snapshots).
  FR Y-15 reports index: `https://www.ffiec.gov/npw/FinancialReport/FRY15Reports`.
- **Format:** caret-delimited TXT (zipped). **License:** US Government work — public domain.
- **Field → `J_ij` mapping:** FR Y-15 **Schedule B (Interconnectedness)** — *Intra-financial system
  assets* (Item 1) and *Intra-financial system liabilities* (Item 2), plus securities/wholesale-funding
  items — give each reporter's gross exposure to other financial institutions; combined with size
  (Schedule A total exposures) these calibrate the per-node weights of the gravity-model reconstruction
  `J_ij ∝ w_ij` (Cimini 2015). FR Y-9C leverage/capital fields → `p_i`.

### BIS Consolidated Banking Statistics (+ Quarterly Review Table 9B) 📝 (`bis/` empty)
- **Wanted:** cross-border claims by reporting country × counterparty country/sector → sector/country-level
  exposure edges; EU sovereign cross-holdings (Table 9B, as used by Elliott–Golub–Jackson).
- **Status:** **bulk file verified reachable** this session (HTTP 200, **~84.7 MB zip**,
  last-modified 2026-06-03) but **not committed** (size; and the targeted SDMX slice needs the exact
  13-dimension key, which we did not want to guess). The SDMX API itself responds correctly
  (returns a well-formed "No results for query" for a wrong key).
- **How to obtain:**
  - Bulk flat CSV: `https://data.bis.org/static/bulk/WS_CBS_PUB_csv_flat.zip`
  - SDMX REST (targeted slice): `https://stats.bis.org/api/v2/data/dataflow/BIS/WS_CBS_PUB/1.0/<KEY>?startPeriod=2007`
    with `Accept: application/vnd.sdmx.data+csv` (resolve `<KEY>` via the dataflow's DSD at
    `https://stats.bis.org/api/v2/structure/datastructure/BIS/...`).
  - EU sovereign cross-holdings: **BIS Quarterly Review, Table 9B** (PDF/statistical annex),
    `https://www.bis.org/statistics/secstats.htm` / Quarterly Review landing `https://www.bis.org/publ/quarterly.htm`.
- **Format:** zipped flat CSV / SDMX-ML / SDMX-CSV; Table 9B is PDF + annex CSV. **License:** free, BIS terms (attribution; no redistribution of bulk as-is).
- **Field → `J_ij` mapping:** in the CBS flat file, the rows keyed by *reporting country* (`L_REP_CTY`)
  × *counterparty country* (`L_CP_COUNTRY`) × *counterparty sector* (`L_CP_SECTOR` = banks) with
  measure *consolidated claims, ultimate-risk basis* are directed country→country bank-exposure weights
  → the country/sector-aggregated `J_ij` edges.

### OFR Financial Stress Index (OFRFSI) 📝 (not on FRED keyless)
- **Wanted:** daily market-based global financial-stress index (33 variables) → macro-state.
- **Why not fetched:** the FRED series id `OFRFSI` returns a **404 "Error – St. Louis Fed"** page on the
  keyless graph endpoint (no longer served there, or requires the authenticated API).
- **How to obtain:**
  - OFR direct (primary, public domain): "Download all data" CSV from
    `https://www.financialresearch.gov/financial-stress-index/` (indicator list at
    `…/financial-stress-index/files/indicators/`).
  - Or authenticated FRED API: `https://api.stlouisfed.org/fred/series/observations?series_id=OFRFSI&api_key=YOUR_KEY&file_type=csv` (verify id still exists). **[UNVERIFIED that OFRFSI still resolves on FRED]**
- **Format:** CSV. **License:** US Treasury / Office of Financial Research — public domain.
- **Maps to:** **macro-state**.

---

## Summary: dataset → model parameter

| Dataset | Status | Model parameter |
|---|---|---|
| `fdic/failures.csv` (2000–2024) | ✅ | **events** |
| `fdic/failures_1980_2024.csv` | ✅ | **events** (S&L + GFC clusters) |
| `ratings/moodys_pd_by_rating.csv` | ✅ | **`p_i`** |
| `fred/AAA.csv`, `fred/BAA.csv` (+ BAA-AAA spread) | ✅ | **macro-state** / `p_i` proxy |
| `fred/BAA10Y.csv` | ✅ | **macro-state** / `p_i` proxy |
| `fred/BAMLH0A0HYM2.csv` | ⚠️ 2023+ only (keyless) | **macro-state** |
| `fred/KCFSI.csv`, `STLFSI4.csv`, `NFCI.csv`, `ANFCI.csv` | ✅ | **macro-state** |
| `fred/VIXCLS.csv` | ✅ | **macro-state** |
| `ecb/ciss_euro_area.csv` | ✅ | **macro-state** |
| NYU V-Lab SRISK | 📝 | **`p_i`** |
| FFIEC FR Y-15 / FR Y-9C | 📝 | **`J_ij`** (+ `p_i` from Y-9C) |
| BIS consolidated banking stats / Table 9B | 📝 | **`J_ij`** |
| OFR FSI (OFRFSI) | 📝 | **macro-state** |
