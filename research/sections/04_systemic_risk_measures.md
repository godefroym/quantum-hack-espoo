# 04 — Empirical Systemic-Risk Measures and Their Public Data Infrastructure

**Scope.** Survey of the empirical systemic-risk literature (market-based contribution
measures, connectedness/network measures, and financial-stress/conditions indices) with a
deliberate emphasis on **where the data comes from** and **what is publicly downloadable for
free**. This sub-area is unusually rich in free data: central-bank research departments, the OFR,
the FDIC, and the FFIEC all publish machine-readable series and APIs, and NYU V-Lab publishes
firm-level systemic-risk rankings updated weekly.

**Why this matters for our project.** Our generator is a Boltzmann/Ising "plausibility model"
over binary default configurations, parameterised by marginal default probabilities `p_i` and
pairwise couplings `J_ij` (see `scenario_generation.md`). The datasets below feed three distinct
needs:

1. **Calibrating `p_i`** (per-institution distress/default propensity): SRISK%, MES/LRMES,
   Δ-CoVaR, PD proxies from CDS/Call-Report leverage, FR Y-15 systemic-footprint indicators.
2. **Calibrating `J_ij`** (pairwise co-distress / exposure couplings): Diebold-Yilmaz and
   Billio-et-al. connectedness networks, equity-return covariance, FR Y-15 interconnectedness
   line items, BIS/ECB cross-border exposures.
3. **Validating the tail / supplying real co-failure events**: FDIC failed-bank list (actual
   default events with dates, assets, deposits, resolution cost) — the single best source of
   *ground-truth* default labels — plus stress-index spikes as crisis-period markers.

> Verification note: All URLs below were checked via web search and (where feasible) WebFetch in
> June 2026. Items I could not directly confirm are tagged **[UNVERIFIED]**. No citations, DOIs,
> dataset names, or URLs were fabricated.

---

## Part A — The measures (papers, what they measure, data, provenance)

### A1. SES / MES — Acharya, Pedersen, Philippon & Richardson (2017), "Measuring Systemic Risk"
- **Citation.** Acharya, V. V., Pedersen, L. H., Philippon, T., & Richardson, M. (2017).
  *Measuring Systemic Risk.* The Review of Financial Studies, 30(1), 2–47. DOI: 10.1093/rfs/hhw088.
  (NBER working-paper precursor: w16223 / the SES framework; the original "Measuring Systemic
  Risk" circulated from ~2010.)
- **What it measures.** Each firm's **Systemic Expected Shortfall (SES)** — its propensity to be
  undercapitalised when the whole system is undercapitalised. SES rises with leverage and with
  **Marginal Expected Shortfall (MES)**, the firm's average equity loss on the worst market days.
- **Data inputs.** Daily/period equity returns of financial firms; a market index; balance-sheet
  leverage (book debt + market equity). MES is estimated purely from **public equity-return data**.
- **Source / provider.** Returns from CRSP/Datastream/Yahoo/Bloomberg-class feeds; leverage from
  financial statements. The *measure itself* is reproduced and published free by NYU V-Lab.
- **Public vs proprietary.** Method = public. Raw returns = proprietary feed OR free (Yahoo/Stooq).
  Computed MES/SRISK = **FREE via V-Lab**.
- **Link.** https://academic.oup.com/rfs/article-abstract/30/1/2/2682977 (NBER:
  https://www.nber.org/papers/w16223)

### A2. SRISK — Brownlees & Engle (2017); Acharya, Engle & Richardson (2012)
- **Citations.**
  - Brownlees, C., & Engle, R. F. (2017). *SRISK: A Conditional Capital Shortfall Measure of
    Systemic Risk.* The Review of Financial Studies, 30(1), 48–79. DOI: 10.1093/rfs/hhw060.
  - Acharya, V., Engle, R., & Richardson, M. (2012). *Capital Shortfall: A New Approach to Ranking
    and Regulating Systemic Risks.* American Economic Review, 102(3), 59–64. DOI: 10.1257/aer.102.3.59.
- **What it measures.** **SRISK** = the expected capital shortfall of a firm conditional on a
  severe market decline (a prolonged ~40% market drop). A function of size, leverage, and
  **LRMES** (Long-Run MES). Aggregate SRISK is an early-warning signal for the real economy.
- **Data inputs.** Equity returns + a market index (to estimate LRMES via a GARCH-DCC model);
  market cap; book value of debt (book liabilities); a prudential capital ratio (e.g., 8%).
- **Source / provider.** Computed and published **firm-by-firm, weekly, globally** by NYU Stern
  **V-Lab**. Underlying returns/balance-sheet from market feeds.
- **Public vs proprietary.** **FREE** — V-Lab publishes SRISK (US$ bn), SRISK%, LRMES, leverage,
  and rankings for global and US financials.
- **Link.** Paper: https://academic.oup.com/rfs/article/30/1/48/2669965 — Live data:
  https://vlab.stern.nyu.edu/srisk — Docs: https://vlab.stern.nyu.edu/docs/srisk

### A3. CoVaR / ΔCoVaR — Adrian & Brunnermeier (2016)
- **Citation.** Adrian, T., & Brunnermeier, M. K. (2016). *CoVaR.* American Economic Review,
  106(7), 1705–1741. DOI: 10.1257/aer.20120555.
- **What it measures.** **CoVaR** = the Value-at-Risk of the *financial system* conditional on a
  particular institution being in distress. **ΔCoVaR** = the system VaR when the institution is in
  distress *minus* the system VaR when it is at its median state — i.e. that institution's
  marginal contribution to systemic risk. Forward-looking version predicts crisis-period CoVaR.
- **Data inputs.** Quantile (tail) regressions of system returns on institution returns, plus
  state variables: VIX, liquidity spreads (3M repo–T-bill), the yield-curve slope, credit spreads,
  and equity/real-estate returns. Firm characteristics (leverage, size, maturity mismatch).
- **Source / provider.** Equity-return panels + market state variables. The state variables are
  almost entirely **available free on FRED** (VIX, Treasury yields, credit spreads).
- **Public vs proprietary.** Method = public; inputs largely **FREE on FRED**; firm returns from a
  feed (free Yahoo/Stooq workable).
- **Link.** https://www.aeaweb.org/articles?id=10.1257/aer.20120555 (author copy:
  https://markus.scholar.princeton.edu/publications/covar)

### A4. Granger-causality / connectedness networks — Billio, Getmansky, Lo & Pelizzon (2012)
- **Citation.** Billio, M., Getmansky, M., Lo, A. W., & Pelizzon, L. (2012). *Econometric Measures
  of Connectedness and Systemic Risk in the Finance and Insurance Sectors.* Journal of Financial
  Economics, 104(3), 535–559. DOI: 10.1016/j.jfineco.2011.12.010.
- **What it measures.** Builds **principal-component** and **pairwise Granger-causality networks**
  among monthly returns of banks, broker/dealers, insurers, and hedge funds. Network density,
  degree, and directionality proxy systemic connectedness; banks transmit shocks asymmetrically.
- **Data inputs.** Monthly equity returns of the four sectors (hedge-fund index returns proprietary;
  banks/brokers/insurers from public equity data).
- **Source / provider.** CRSP/Datastream-style returns; hedge-fund returns from databases (TASS,
  proprietary). **Bank/broker/insurer returns reproducible from free equity data.** This directly
  inspires our pairwise `J_ij` coupling structure.
- **Public vs proprietary.** Method = public; hedge-fund inputs proprietary; the
  bank/insurer network is reproducible with **free** equity returns.
- **Link.** https://www.sciencedirect.com/science/article/abs/pii/S0304405X11002868 (open NBER
  precursor: https://www.nber.org/papers/w16223 is *different*; this paper's open MIT copy:
  https://dspace.mit.edu/handle/1721.1/110542)

### A5. Connectedness via variance decompositions — Diebold & Yilmaz (2014)
- **Citation.** Diebold, F. X., & Yılmaz, K. (2014). *On the Network Topology of Variance
  Decompositions: Measuring the Connectedness of Financial Firms.* Journal of Econometrics, 182(1),
  119–134. DOI: 10.1016/j.jeconom.2014.04.012. (NBER w17490.)
- **What it measures.** Defines total/directional/pairwise **connectedness** from the
  forecast-error variance decomposition of a VAR on firm return volatilities. The variance-share
  matrix *is* a weighted directed network — a direct, estimable analogue of our `J_ij` matrix.
- **Data inputs.** Daily return *volatilities* (range-based) of major US financial institutions.
- **Source / provider.** Public equity/intraday data. The authors maintain a project site with
  data and methodology.
- **Public vs proprietary.** Method = public; inputs **FREE** (equity prices). Reproducible.
- **Link.** Paper: https://ideas.repec.org/a/eee/econom/v182y2014i1p119-134.html — Project site:
  https://financialconnectedness.org/ — PDF:
  https://www.sas.upenn.edu/~fdiebold/papers2/DDLYpaper.pdf

---

## Part B — Financial-stress / financial-conditions indices (all FREE)

These are ready-made aggregate "stress" time series. Useful to us as **crisis-period labels** (to
identify when correlated defaults cluster, i.e. to calibrate the *strength* of `J_ij` in stressed
regimes) and as **macro state variables** for conditioning the generator.

### B1. OFR Financial Stress Index (OFR FSI)
- **What.** Daily, **market-based** snapshot of global financial stress from **33 variables** in
  5 categories (credit, equity valuation, funding, safe assets, volatility). Decomposed by 3 regions
  (US, other advanced economies, emerging markets). Mean-zero (0 = average stress).
- **Data inputs.** 33 market series (yield spreads, valuations, rates, vols) sourced by OFR from
  Refinitiv Datastream.
- **Provider / license.** US Treasury **Office of Financial Research** — US Government work,
  **public domain / free**. "Download all data" button on the page; history + daily updates
  (current to ~2 business days prior).
- **Link.** https://www.financialresearch.gov/financial-stress-index/ — Indicator list:
  https://www.financialresearch.gov/financial-stress-index/files/indicators/ — Working paper
  (methodology, OFR WP 17-04):
  https://www.financialresearch.gov/working-papers/files/OFRwp-17-04_The-OFR-Financial-Stress-Index.pdf
  - Also mirrored on FRED: series **OFRFSI** (https://fred.stlouisfed.org/series/OFRFSI).

### B2. Chicago Fed National Financial Conditions Index (NFCI / ANFCI)
- **What.** **Weekly** weighted average of **~105** indicators across money, debt, equity markets
  and traditional + "shadow" banking. Mean 0, SD 1 (since 1973). Subindexes: **risk, credit,
  leverage**. **ANFCI** = adjusted for the macro cycle (inflation/activity removed).
- **Provider / license.** Federal Reserve Bank of Chicago — **free**. Published on Chicago Fed +
  FRED.
- **Link.** https://www.chicagofed.org/research/data/nfci/about — Current data:
  https://www.chicagofed.org/research/data/nfci/current-data — FRED series **NFCI**
  (https://fred.stlouisfed.org/series/NFCI), **ANFCI** (https://fred.stlouisfed.org/series/ANFCI),
  plus subindexes NFCIRISK / NFCICREDIT / NFCILEVERAGE.

### B3. St. Louis Fed Financial Stress Index (STLFSI4)
- **What.** **Weekly** index from **18 series** (7 interest rates, 6 yield spreads, 5 other). Mean
  0. Current version **STLFSI4** (uses forward-looking SOFR). Older STLFSI/STLFSI2/STLFSI3
  discontinued.
- **Provider / license.** Federal Reserve Bank of St. Louis — **free** on FRED.
- **Link.** https://fred.stlouisfed.org/series/STLFSI4 — FRED blog (v4 methodology):
  https://fredblog.stlouisfed.org/2022/11/the-st-louis-feds-financial-stress-index-version-4/

### B4. Kansas City Fed Financial Stress Index (KCFSI)
- **What.** **Monthly** index from **11** financial-market variables; positive = above-average
  stress. (Note: an underlying bank-stock subindex was switched to the S&P US BMI Banks Index in
  Sept 2021, with back-filled predicted values pre-2004.)
- **Provider / license.** Federal Reserve Bank of Kansas City — **free** on FRED + KC Fed.
- **Link.** https://fred.stlouisfed.org/series/KCFSI —
  https://www.kansascityfed.org/data-and-trends/kansas-city-financial-stress-index/

### B5. Cleveland Fed Financial Stress Index (CFSI) — **DISCONTINUED**
- **What.** Daily coincident index across 6 market types (credit, equity, FX, funding, real estate,
  securitization), as z-scores. **Discontinued** in 2016 after calculation errors were found; the
  historical series remains on FRED for backtesting only.
- **Provider / license.** Federal Reserve Bank of Cleveland — **free** (archival).
- **Link.** https://fred.stlouisfed.org/series/CFSI — release:
  https://fred.stlouisfed.org/release?rid=302

### B6. ECB CISS — Composite Indicator of Systemic Stress (and SovCISS)
- **Citation (method).** Holló, D., Kremer, M., & Lo Duca, M. (2012). *CISS — A Composite Indicator
  of Systemic Stress in the Financial System.* ECB Working Paper No. 1426.
- **What.** **Daily/weekly** indicator aggregating **15** raw stress measures into **5
  market-segment subindices** (money, bond, equity, financial intermediaries, FX), aggregated using
  **portfolio theory** so that the headline rises when stress is correlated *across* segments —
  conceptually close to our "co-failure" emphasis. Computed for the euro area (and a few countries).
  **SovCISS** is the sovereign-stress analogue.
- **Provider / license.** European Central Bank — **free** via the ECB Data Portal / SDW (SDMX API,
  CSV).
- **Link.** Data: https://data.ecb.europa.eu/data/datasets/CISS/data-information — Series e.g.
  `CISS.D.U2.Z0Z.4F.EC.SS_CIN.IDX` — WP:
  https://www.ecb.europa.eu/pub/pdf/scpwps/ecbwp1426.pdf

---

## Part C — FREE / PUBLIC DATA PORTALS (the important table)

> This is the operational core. Each row is a portal we can actually pull from. "What it gives us"
> maps to `p_i` (distress/default), `J_ij` (couplings/exposures), or **events** (real failures).

| # | Portal | Contents | Granularity | Format / API | License | Landing URL |
|---|--------|----------|-------------|--------------|---------|-------------|
| 1 | **NYU Stern V-Lab — Systemic Risk** | SRISK (US$ bn), **SRISK%**, **LRMES**, leverage, MES; rankings for global + US financials | Firm-level, **weekly**; history ~2000→present | Web tables; per-firm pages (e.g. `RISK.USFIN-MR.MES`). Programmatic export not officially documented → scrape/parse | Free, open-access (academic) | https://vlab.stern.nyu.edu/srisk |
| 2 | **FRED (St. Louis Fed) + FRED API** | 800k+ macro/financial series: stress indices (OFRFSI, NFCI, STLFSI4, KCFSI, CFSI), VIX, Treasury yields, credit spreads (BAA-AAA, ICE BofA OAS), interbank/repo spreads | Daily–monthly; long history | **REST API**, JSON/XML; free API key. CSV per series | Free (US Gov data; some series have provider terms) | https://fred.stlouisfed.org/docs/api/fred/ |
| 3 | **OFR Financial Stress Index** | The OFR FSI + its 33 indicator series, 5 categories, 3 regions | Daily; multi-year history | "Download all data" (CSV); also FRED `OFRFSI` | Public domain (US Treasury/OFR) | https://www.financialresearch.gov/financial-stress-index/ |
| 4 | **FDIC BankFind Suite — Failures API** ⭐ | **Real failed/assisted banks 1934→present**: NAME, CERT, CITYST, FAILDATE, **QBFASSET** (total assets), **QBFDEP** (total deposits), **COST** (est. loss), RESTYPE (resolution type) | Per-institution **event** rows | **REST API** JSON/CSV (`Accept` header / `format=csv`); also bulk download | Public domain (US Gov) | https://api.fdic.gov/banks/docs |
| 5 | **FDIC BankFind Suite — Financials/Institutions** | Quarterly balance-sheet & income data for **every** US insured bank (assets, equity, leverage, ROA, capital ratios); institution registry | Per-bank, **quarterly** | REST API JSON/CSV; bulk download | Public domain (US Gov) | https://banks.data.fdic.gov/bankfind-suite/api |
| 6 | **FFIEC Central Data Repository (CDR) — Call Reports** | Bank Call Reports (FFIEC 031/041/051): full balance sheet, capital, loans, off-balance-sheet | Per-bank, **quarterly** | Bulk download (caret-delimited TXT/zip); web UI | Public domain (US Gov) | https://cdr.ffiec.gov/public/ |
| 7 | **FFIEC NIC — FR Y-9C / FR Y-15** ⭐ | **FR Y-9C** consolidated holding-company financials; **FR Y-15** *Systemic Risk Report*: size, **interconnectedness**, substitutability, complexity, cross-jurisdictional activity, short-term wholesale funding (the GSIB indicators) | Holding-company, **quarterly** (Y-9C back to 2000); Y-15 snapshots | Financial Data Download (caret-delimited TXT); Y-15 snapshot files | Public domain (US Gov) | https://www.ffiec.gov/npw/FinancialReport/FinancialDataDownload |
| 8 | **ECB Data Portal / Statistical Data Warehouse** | CISS/SovCISS stress; bank balance-sheet (BSI), MFI interest rates, cross-border/exposure stats | Country/aggregate; daily–monthly | **SDMX REST API**; CSV / SDMX-ML | Free (ECB; attribution) | https://data.ecb.europa.eu/ — API: https://data-api.ecb.europa.eu |
| 9 | **BIS Data Portal** | Locational & **consolidated banking statistics** (cross-border claims by country/sector — direct exposure-network inputs), credit, debt securities | Country×country / sector; quarterly | **SDMX REST API**; CSV/zip bulk | Free (BIS terms) | https://data.bis.org/ — Bulk: https://data.bis.org/bulkdownload |

⭐ = highest-value rows for this project (real default events + GSIB interconnectedness inputs).

---

## Part D — How each source maps to our model parameters

- **`p_i` (per-institution distress/default propensity).**
  - Best ready-made proxies: **SRISK%** and **LRMES/MES** from V-Lab (firm-level, weekly) — already
    a "probability-weighted severity of distress."
  - Bottom-up: **FDIC Call Reports / FR Y-9C** leverage and capital ratios → distance-to-default
    style PDs; **credit spreads / CDS** from FRED as market-implied PD proxies.
  - Ground truth: **FDIC failures** gives realised default labels to calibrate/validate base rates.
- **`J_ij` (pairwise couplings).**
  - **Diebold-Yilmaz** variance-decomposition matrix and **Billio et al.** Granger/PCA networks are
    *directly* estimable from free equity returns and give a firm×firm coupling matrix.
  - **FR Y-15** interconnectedness line items (intra-financial assets/liabilities) and **BIS
    consolidated banking statistics** (cross-border claims) give *balance-sheet* exposure edges.
  - **ECB CISS** portfolio-theory aggregation is a conceptual template for "co-stress" weighting.
- **Tail / event validation.**
  - **FDIC failed-bank list** = the rare-event ground truth our generator is meant to populate;
    failure clustering by year (2008–2010 wave) is exactly the correlated-cascade signal.
  - **Stress indices** (OFR FSI, NFCI, CISS) flag the stressed regimes where `J_ij` should be
    strongest — useful for regime-conditional calibration.

---

## Part E — Concrete pull recipes (verified syntax)

- **FDIC failures (real default events) → CSV, no key:**
  `https://api.fdic.gov/banks/failures?fields=NAME,CERT,CITYST,FAILDATE,QBFDEP,QBFASSET,COST,RESTYPE&filters=FAILYR:[2007 TO 2010]&limit=10000&format=csv&download=true&filename=bank-failures`
  (Confirmed field names: NAME, CERT, CITYST, FAILDATE, QBFASSET, QBFDEP, COST, RESTYPE; Elastic
  query-string filter syntax with `[]` inclusive ranges.)
- **FRED series → JSON, free key:**
  `https://api.stlouisfed.org/fred/series/observations?series_id=NFCI&api_key=YOUR_KEY&file_type=json`
  (Swap `series_id` for OFRFSI, STLFSI4, KCFSI, VIXCLS, BAMLC0A0CM, etc.)
- **ECB CISS → CSV via Data Portal API:** dataset `CISS`, e.g. key
  `CISS.D.U2.Z0Z.4F.EC.SS_CIN.IDX` through https://data-api.ecb.europa.eu (SDMX, CSV/SDMX-ML).
- **BIS consolidated banking statistics → zipped CSV bulk:** https://data.bis.org/bulkdownload .
- **FFIEC FR Y-9C / FR Y-15 → caret-delimited TXT bulk:**
  https://www.ffiec.gov/npw/FinancialReport/FinancialDataDownload (Y-9C back to 2000) and FR Y-15
  snapshots https://www.ffiec.gov/npw/FinancialReport/FRY15Reports .

---

## Part F — Caveats / unverified

- V-Lab **bulk** download / official API: the live SRISK tables are clearly free and browsable, but
  a documented programmatic export endpoint was **not** found; plan on HTML parsing of per-portfolio
  pages (e.g. `https://vlab.stern.nyu.edu/srisk/RISK.USFIN-MR.MES`). **[UNVERIFIED: official CSV/API]**
- OFR FSI exposes a "Download all data" button; the exact static CSV filename/path was not captured
  here, but the data is reliably mirrored as FRED series **OFRFSI**. **[UNVERIFIED: direct CSV path]**
- FDIC API: registration/key is **optional** (anonymous access works); rate limits may apply to heavy
  use.
- Hedge-fund return inputs to Billio et al. (TASS/HFR) and intraday data for Diebold-Yilmaz can be
  proprietary; the **bank/insurer** portions are reproducible with free equity data.

---

### Source list (verifiable)
- SRISK paper: https://academic.oup.com/rfs/article/30/1/48/2669965
- Measuring Systemic Risk (SES/MES): https://academic.oup.com/rfs/article-abstract/30/1/2/2682977
- Capital Shortfall (AER 2012): https://dx.doi.org/10.2139/ssrn.1611229
- CoVaR (AER 2016): https://www.aeaweb.org/articles?id=10.1257/aer.20120555
- Billio et al. (JFE 2012): https://www.sciencedirect.com/science/article/abs/pii/S0304405X11002868
- Diebold-Yilmaz (J.Econometrics 2014): https://ideas.repec.org/a/eee/econom/v182y2014i1p119-134.html
- V-Lab SRISK: https://vlab.stern.nyu.edu/srisk
- FRED API: https://fred.stlouisfed.org/docs/api/fred/
- OFR FSI: https://www.financialresearch.gov/financial-stress-index/
- Chicago Fed NFCI: https://www.chicagofed.org/research/data/nfci/about
- St. Louis Fed STLFSI4: https://fred.stlouisfed.org/series/STLFSI4
- Kansas City Fed KCFSI: https://www.kansascityfed.org/data-and-trends/kansas-city-financial-stress-index/
- Cleveland Fed CFSI (discontinued): https://fred.stlouisfed.org/series/CFSI
- ECB CISS: https://data.ecb.europa.eu/data/datasets/CISS/data-information
- FDIC BankFind API: https://api.fdic.gov/banks/docs
- FDIC failures explorer: https://banks.data.fdic.gov/bankfind-suite/failures
- FFIEC CDR Call Reports: https://cdr.ffiec.gov/public/
- FFIEC NIC Financial Data Download (Y-9C/Y-15): https://www.ffiec.gov/npw/FinancialReport/FinancialDataDownload
- BIS Data Portal: https://data.bis.org/
