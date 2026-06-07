# Crisis 2008 — Asian & African institutions affected by the Global Financial Crisis

A roster and daily equity-price snapshot for institutions outside the US/UK/EU
core that were materially affected by the 2007–2009 Global Financial Crisis. It
exists to broaden the globe visualisation (which was sparse across Africa and
Asia) and to give the Part-A pipeline a geographically diverse, crisis-relevant
set of real institutions.

## Files

- `affected_institutions.csv` — curated roster. Columns: `id, name, ticker,
  yahoo_ticker, type` (bank/insurer), `country, region, hq_lat, hq_lon,
  crisis_role` (one-line factual note), `source`.
- `prices/<id>.csv` — daily adjusted close per institution, `date,adjclose`,
  fetched over the crisis window (2007-01-01 → 2010-12-31).
- `fetch_log.json` — what the fetch retrieved (obs counts, date ranges, errors).

Regenerate prices with:

```bash
uv run python scripts/fetch_crisis_equities.py
```

## Provenance

- **Roster:** hand-curated from public reporting; each row carries a `source`
  tag and a one-line `crisis_role`. Headquarters coordinates are approximate
  (city level), for map placement only.
- **Prices:** Yahoo Finance daily chart API (adjusted close), same keyless,
  browser-User-Agent method the project uses for `banks/equity_corr`. Licensed
  for personal/research use; treat as a reproducible snapshot, not redistribution.

## What the research found (Asia & Africa)

**Asia**
- **MUFG (Japan)** — invested US$9bn in Morgan Stanley (Oct 2008), a key rescue.
- **Nomura (Japan)** — acquired Lehman Brothers' Asia-Pacific and European/
  Middle-East operations after the Sep 2008 collapse.
- **Mizuho (Japan)** — subprime-related losses; took part in a Merrill Lynch
  capital raise.
- **ICBC (China)** — held US subprime securities; completed a 20% (~US$5.5bn)
  stake in South Africa's Standard Bank in 2008.
- **Bank of China (China)** — among the largest US subprime/MBS exposures of the
  Chinese banks.
- **Ping An (China, insurer)** — wrote down its stake in Fortis (~€2.3bn loss).
- **DBS (Singapore)** — caught in the Lehman "Minibond" retail fallout and
  regional credit stress.
- **ICICI Bank (India)** — share-price slump and a depositor-confidence scare in
  Oct 2008.
- **Shinhan / Korean banks** — acute USD-funding and won pressure; Korea
  Development Bank's lapsed talks to buy Lehman immediately preceded its collapse
  (KDB itself is state-owned and unlisted, so Shinhan stands in here).

**Africa** (direct bank exposure was limited; the channel was mostly markets,
commodities, and capital flows)
- **Standard Bank (South Africa)** — ICBC took a 20% stake; the JSE and the rand
  sold off sharply. South African banks were otherwise largely shielded (sound
  regulation, little subprime).
- **First Bank of Nigeria / FBN Holdings** — operated through the 2008–09
  Nigerian banking crisis, where the GFC compounded a margin-loan and oil-price
  shock and led to CBN bank rescues in 2009. *(No Yahoo series available for the
  Nigerian Exchange, so `prices/FBN.csv` is absent — see `fetch_log.json`.)*

## Sources

- Nomura's Lehman acquisitions — Nomura Holdings Form 6-K filings, FY2008
  (SEC EDGAR, CIK 0001163653).
- ICBC ↔ Standard Bank 20% stake — Goldman Sachs firm history
  (`goldmansachs.com/our-firm/history/moments/2007-icbc-stake-in-standard-bank`);
  ICBC press release; Euromoney (2018 retrospective).
- South African banks' resilience; cross-border channels — BIS Papers No. 54
  (`bis.org/publ/bppdf/bispap54w.pdf`).
- Nigerian 2008–09 banking crisis — IMF eLibrary, "The Nigerian Banking Crisis
  of 2008–2009"; BIS Review r110124c (Sanusi).

URLs verified during this session (2026-06-07). Equity series are best-effort
from a keyless endpoint.
