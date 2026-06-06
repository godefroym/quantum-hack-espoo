# Section 01 — Interbank / Financial-Network Contagion and Cascade Models

**Scope.** Literature on how shocks propagate through networks of mutually-exposed financial
institutions: clearing-vector models, threshold/cascade models, feedback-centrality measures,
and the empirical-network and bounds literatures. The emphasis of this annotation is on the
**DATA each work uses and where it comes from**, because for our project (synthetic correlated
default-scenario generator + cascade simulator) the binding real-world constraint is that the
bilateral-exposure data needed to fit cascade models is overwhelmingly **confidential
central-bank supervisory data**, not public. Where a study *does* use public or
commercially-available data, that is flagged explicitly, because those are the datasets we
could realistically touch.

**Verification note.** Citations, venues, DOIs/URLs and the data-source facts below were read
from primary sources (publisher pages, author-hosted PDFs, central-bank working papers, and the
Glasserman–Young 2016 JEL survey, which itself catalogs the data provenance of the field).
Items I could not confirm against a primary source are marked `[UNVERIFIED]`.

---

## Headline takeaway on data provenance (read this first)

> The single most important fact for our project: **bilateral interbank-exposure matrices are
> almost never public.** They are confidential supervisory reports or credit-register data held
> by central banks / banking supervisors. Christian Upper's 2011 survey states plainly that
> "the most reliable sources of information on bilateral exposures in the interbank market tend
> to be reports provided by banks to their supervisors or credit registers," and Glasserman &
> Young (2016) note that "empirical work is limited by the confidentiality of interbank
> transactions and the low frequency of financial crises."

Consequences:
1. Most landmark cascade papers are either **(a) purely theoretical** (Eisenberg–Noe;
   Acemoglu–Ozdaglar–Tahbaz-Salehi), **(b) calibrated to *synthetic* random networks**
   (Gai–Kapadia), or **(c) empirical but on confidential national supervisory data** released
   only to the authors under central-bank agreements (Cont–Moussa–Santos for Brazil; the
   dozens of country studies catalogued by Upper).
2. The handful of works using **public** data are valuable as calibration anchors: Elliott–
   Golub–Jackson use BIS Quarterly Review sovereign-claims tables; Glasserman–Young (2015) use
   European Banking Authority stress-test disclosures; Battiston et al. (DebtRank) use the
   Bloomberg-released US Federal Reserve emergency-loan dataset plus Bureau van Dijk Orbis
   equity data.
3. When only **aggregate** interbank totals are public (each bank's total interbank assets and
   liabilities), the field reconstructs the bilateral matrix with **maximum-entropy** methods —
   directly analogous to what our generative "plausibility model" is doing: inferring a joint
   structure consistent with marginals.

---

## Key papers (annotated)

### 1. Eisenberg & Noe (2001) — Clearing payment vectors
- **Citation.** Larry Eisenberg, Thomas H. Noe. "Systemic Risk in Financial Systems."
  *Management Science* 47(2): 236–249, 2001.
- **Method.** Models a financial system as a network of nominal liabilities `L_ij`; proves via a
  fixed-point argument (Tarski) that a **clearing payment vector** — payments respecting limited
  liability, debt priority, and proportional (pro-rata) sharing in default — always exists and is
  generically unique. The foundational "who-pays-whom after default" engine that almost every
  later cascade simulator (including ours) uses implicitly.
- **Data.** **None — purely theoretical/mathematical.** No empirical interbank dataset. The paper
  establishes existence/uniqueness and comparative statics; any numbers are illustrative toy
  examples.
- **Public/proprietary.** N/A (no data).
- **Link.** DOI: 10.1287/mnsc.47.2.236.819 · SSRN abstract 173249
  (https://papers.ssrn.com/sol3/papers.cfm?abstract_id=173249).

### 2. Gai & Kapadia (2010) — Contagion in financial networks
- **Citation.** Prasanna Gai, Sujit Kapadia. "Contagion in financial networks." *Proceedings of
  the Royal Society A* 466(2120): 2401–2423, 2010. (Also Bank of England Working Paper No. 383,
  2009.)
- **Method.** Analytical + simulation model of default cascades on a network with arbitrary
  degree distribution. A bank fails when its **capital buffer** is exhausted by losses on
  defaulted interbank counterparties; uses generating-function / percolation techniques. Headline
  result: financial networks are **"robust-yet-fragile"** — contagion is a low-probability event,
  but conditional on occurring it is often system-wide.
- **Data.** **No real bank data.** The network is a *synthetic* directed random graph
  (Poisson / Erdős–Rényi degree distribution); results are Monte-Carlo averages over many drawn
  networks. This is a stylized-calibration paper, not an empirical one.
- **Baseline parameters (verified from the model family).** Network is directed Erdős–Rényi with
  a tunable average degree (the connectivity/contagion "window" is explored as the average degree
  is varied, with contagion appearing roughly in a low-average-degree band and dying out as the
  network becomes very dense — the robust-yet-fragile knife-edge). Standard values used in the
  paper and its direct descendants: **N = 1000 banks**, **capital buffer ≈ 4% of total assets**,
  **interbank assets ≈ 20% of total assets**, **zero recovery** on interbank assets on impact
  (worst-case LGD = 100% at the moment of default). A widely-used descendant (Hurd/Kobayashi
  formulation, arXiv:1312.6804) makes the structure explicit: `N = 1000`, capital ratio
  `γ = w/(total assets)`, interbank-to-total-assets ratio `θ ≈ 0.2–0.3`, Erdős–Rényi directed
  graphs averaged over many draws, and a "crisis" defined as **≥ 5% of banks failing**. The
  `4%` buffer / `20%` interbank / `zero-recovery` triple as the *original* Gai–Kapadia baseline is
  the commonly-cited convention `[UNVERIFIED against the original PDF — publisher and BoE PDFs
  returned 403; confirmed only that the model is synthetic Erdős–Rényi with a capital-buffer
  solvency condition]`.
- **Public/proprietary.** N/A — synthetic data; fully reproducible.
- **Link.** DOI: 10.1098/rspa.2009.0410 · BoE WP 383:
  https://www.bankofengland.co.uk/working-paper/2010/contagion-in-financial-networks

### 3. Battiston, Puliga, Kaushik, Tasca & Caldarelli (2012) — DebtRank
- **Citation.** Stefano Battiston, Michelangelo Puliga, Rahul Kaushik, Paolo Tasca, Guido
  Caldarelli. "DebtRank: Too Central to Fail? Financial Networks, the FED and Systemic Risk."
  *Scientific Reports* 2: 541, 2012.
- **Method.** Introduces **DebtRank**, a feedback-centrality measure (PageRank-like, but
  distress-propagating and walk-avoiding so impact is not double-counted) quantifying the fraction
  of total network value potentially affected by the distress/default of a node. Reframes
  "too-big-to-fail" as **"too-central-to-fail."**
- **Data.** This is one of the most data-transparent landmark papers. Two datasets combined:
  - **US Federal Reserve emergency-loan program** (~**USD 1.2 trillion**, the "FED Discount
    Window" and related crisis facilities), daily exposures, **Aug 2007 – Jun 2010** (1,004
    days), covering **407 institutions** (analysis focuses on the 22 that borrowed > USD 5bn on
    average). Crucially, the authors state this was "the first data set, publicly available, on the
    daily financial exposures between a central bank and a large set of institutions" — they used
    the version **released by Bloomberg** after consolidating the FED's original disclosure.
  - **Equity cross-holdings** from the **Orbis database (Bureau van Dijk)**, equity shares one
    firm holds in another as of Q4 2007; plus daily market-capitalization data.
- **Public/proprietary.** **FED-loan data: PUBLIC** (Bloomberg/FED disclosure — this is the rare
  public bilateral central-bank-to-bank exposure set). **Orbis equity data: PROPRIETARY**
  (Bureau van Dijk commercial subscription).
- **Link.** DOI: 10.1038/srep00541 ·
  https://pmc.ncbi.nlm.nih.gov/articles/PMC3412322/ (open access)

### 4. Acemoglu, Ozdaglar & Tahbaz-Salehi (2015) — Systemic risk and stability in financial networks
- **Citation.** Daron Acemoglu, Asuman Ozdaglar, Alireza Tahbaz-Salehi. "Systemic Risk and
  Stability in Financial Networks." *American Economic Review* 105(2): 564–608, 2015. (NBER WP
  18727, 2013.)
- **Method.** Theoretical model of `n` banks over three dates linked by standard debt contracts.
  Proves a **phase transition**: for *small* shocks, denser/more-diversified interconnection is
  stabilizing (risk-sharing); beyond a shock-size threshold, the *same* dense connections become
  the propagation channel and the system is fragile. Formalizes "robust-yet-fragile" with explicit
  thresholds and shows complete vs. ring networks are extremal.
- **Data.** **None — purely theoretical.** No empirical calibration; results are analytical
  theorems with stylized illustrative networks.
- **Public/proprietary.** N/A.
- **Link.** DOI: 10.1257/aer.20130456 · open PDF:
  https://economics.mit.edu/sites/default/files/publications/Systemic%20Risk%20and%20Stability%20in%20Financial%20Networks..pdf
  · NBER WP: https://www.nber.org/system/files/working_papers/w18727/w18727.pdf

### 5. Elliott, Golub & Jackson (2014) — Financial networks and contagion
- **Citation.** Matthew Elliott, Benjamin Golub, Matthew O. Jackson. "Financial Networks and
  Contagion." *American Economic Review* 104(10): 3115–3153, 2014.
- **Method.** Cascades through **cross-holdings** (equity/debt claims of organizations on one
  another). Derives a non-inflated "market value" of each org and shows cascades depend
  non-monotonically on two structural axes: **integration** (depth of exposure to counterparties)
  and **diversification** (number of counterparties). Failures are discontinuous: crossing a
  threshold triggers a discrete value drop (failure cost), which can tip others.
- **Data.** Empirical illustration ("proof of concept," their words) using **cross-holdings of
  debt among six European countries: France, Germany, Greece, Italy, Portugal, Spain.** Exact
  source, quoted: *"Data on the cross-holdings are for the end of December 2011 from the BIS (Bank
  for International Settlements) Quarterly Review (Table 9B)... the consolidated foreign claims of
  banks from one country on debt obligations of another country."*
- **Public/proprietary.** **PUBLIC and FREE.** BIS Quarterly Review detailed tables are openly
  published; the paper even gives the access URL (BIS Quarterly Review, June 2012,
  http://www.bis.org/publ/qtrpdf/r_qa1206.pdf). Replication package on openICPSR (project 112694).
- **Link.** DOI: 10.1257/aer.104.10.3115 · author PDF:
  https://bengolub.net/wp-content/uploads/2020/05/financial_networks.pdf ·
  replication: https://www.openicpsr.org/openicpsr/project/112694

### 6. Upper (2011) — Survey: simulation methods for interbank contagion
- **Citation.** Christian Upper. "Simulation methods to assess the danger of contagion in
  interbank markets." *Journal of Financial Stability* 7(3): 111–125, 2011.
- **Method.** Survey/meta-analysis of ~15 counterfactual-simulation studies that estimate
  domino-default contagion from interbank exposures. Catalogs methodologies, data sources, and the
  sensitivity of results to assumptions. **The single best source for data-provenance facts in
  this whole field.**
- **Data (about the field's data).** Confirms the central fact: best bilateral-exposure data are
  **confidential supervisory reports / credit registers** (e.g., explicitly Hungary, Italy,
  Mexico, Belgium, Germany, Switzerland — some only the largest exposures). Where bilateral data
  are unavailable, studies use the **maximum-entropy method** to reconstruct the bilateral matrix
  from each bank's aggregate interbank assets/liabilities (marginals). The main *commercial*
  alternative named is **Bankscope** (Bureau van Dijk), but it gives an incomplete picture and
  mixes in central-bank exposures. Notable data-rich systems: the **Austrian National Bank's
  Systemic Risk Monitor** (Boss et al.).
- **Calibration conventions reported.** Loss-given-default in simulations is typically assumed in
  the **~50–60%** range (results are highly LGD-sensitive). Reported magnitudes of contagious
  losses (as a share of banking-system assets) where contagion does occur: Belgium ~20% of total
  assets, Italy ~16%, UK ~16% — "substantial but below apocalyptic." Capital/Tier-1 minimums
  around the Basel 8% are used as failure thresholds.
- **Public/proprietary.** Survey itself is a published article; the *underlying* studies are
  mostly confidential-data based (that is the survey's recurring caveat).
- **Link.** DOI: 10.1016/j.jfs.2011.06.002 · open PDF:
  https://staff.fnwi.uva.nl/p.j.c.spreij/winterschool/11Upper.pdf

### 7. Cont, Moussa & Santos (2013) — Network structure and systemic risk in banking systems
- **Citation.** Rama Cont, Amal Moussa, Edson Bastos e Santos. "Network Structure and Systemic
  Risk in Banking Systems." Chapter 13 in *Handbook on Systemic Risk* (Fouque & Langsam, eds.),
  Cambridge University Press, 2013. (Working-paper / SSRN version 2010, SSRN 1733528.) Companion
  empirical paper: Cont, Moura & Santos, "The Brazilian Interbank Network Structure and Systemic
  Risk," Banco Central do Brasil Working Paper No. 219, 2010.
- **Method.** Defines a **Contagion Index** (expected systemic loss triggered by an
  institution's default, accounting for network position) and analyzes how heterogeneity and
  counterparty concentration — not just balance-sheet size — drive systemic importance. Argues
  capital requirements should be **exposure-dependent**, targeting central institutions.
- **Data.** **All mutual exposures + capital reserves of financial institutions in Brazil**, at
  **6 dates: Jun 2007, Dec 2007, Mar 2008, Jun 2008, Sep 2008, Nov 2008.** Provided by the
  **Central Bank of Brazil (Banco Central do Brasil)**; ~**400 financial institutions/
  conglomerates**; Tier-1 capital per BCB Resolutions 3,444 / 3,490 (Basel I/II).
- **Network statistics (calibration-grade, verified).** **Sparse, directed, scale-free**: in- and
  out-degree distributions are **heavy-tailed / power-law** (exponent γ ≈ 2.7–2.8); **mean in-
  and out-degree ≈ 8.5** (i.e., a typical bank has roughly 8–9 counterparties out of hundreds —
  very low density). Default contagion uses a recovery rate `r_j` per defaulting conglomerate
  (losses to creditor i are `(1 − r_j)·ℓ_ij`).
- **Public/proprietary.** **PROPRIETARY / CONFIDENTIAL** central-bank supervisory data — accessed
  by the authors under Banco Central do Brasil arrangements; not publicly downloadable. (The
  *aggregate* statistics and BCB working paper are public, but the bilateral matrix is not.)
- **Link.** HAL: https://hal.science/hal-00912018v1 · BCB WP 219:
  https://www.bcb.gov.br/pec/wps/ingl/wps219.pdf · author PDF:
  http://rama.cont.perso.math.cnrs.fr/pdf/ContMoussaSantos.pdf

### 8. Glasserman & Young (2015) — How likely is contagion in financial networks?
- **Citation.** Paul Glasserman, H. Peyton Young. "How likely is contagion in financial
  networks?" *Journal of Banking & Finance* 50: 383–399, 2015. (OFR Working Paper 0009, 2013.)
- **Method.** Derives **analytical upper bounds** on the magnitude of network-spillover contagion
  using only **node-level** data — asset size, leverage, and a **"financial connectivity"** term
  (the fraction of a bank's liabilities held by *other financial* institutions vs. the
  nonfinancial sector) — without needing the full bilateral matrix. Spillover is largest when the
  originating node is big, highly leveraged, and highly financially-connected. The bounds are
  distribution-robust (hold for beta/exponential/normal shocks). Conclusion: **pure
  interbank-spillover (domino) contagion is hard to generate** at realistic leverage/connectivity;
  amplification needs additional channels (fire sales, funding runs).
- **Data.** Empirical illustration uses the **European banking system, from European Banking
  Authority (2011)** stress-test / transparency disclosures (node-level size, capital, leverage,
  interbank-liability shares).
- **Public/proprietary.** **PUBLIC and FREE.** EBA stress-test/transparency datasets are openly
  published bank-by-bank.
- **Link.** DOI: 10.1016/j.jbankfin.2014.02.006 · OFR WP open PDF:
  https://www.financialresearch.gov/working-papers/files/OFRwp0009_GlassermanYoung_HowLikelyContagionFinancialNetworks.pdf

### 9. Glasserman & Young (2016) — Contagion in Financial Networks (JEL survey)
- **Citation.** Paul Glasserman, H. Peyton Young. "Contagion in Financial Networks." *Journal of
  Economic Literature* 54(3): 779–831, 2016.
- **Method.** Authoritative survey of the interconnectedness/systemic-fragility literature: how
  network structure interacts with **leverage, size, common exposures, and short-term funding**,
  plus a critical assessment of network-centrality metrics. (Together with Upper 2011, the best
  meta-source for data provenance.)
- **Data / provenance findings (verified quotes).** "Empirical work is limited by the
  confidentiality of interbank transactions and the low frequency of financial crises." "The most
  detailed analyses of financial networks have used confidential data on short-term (mostly
  overnight) lending" — citing Soramäki et al. (2007, **Fedwire**), Bech & Atalay (2010, US fed
  funds), Afonso–Kovner–Schoar (2011), Gabrieli & Georg (2014, euro area), Blasques et al. On
  network structure: real interbank networks are **sparse with a core–periphery / tiered**
  structure (e.g., Austria, Boss et al. 2004: ~**20% of banks form the core**, three-tier
  organization). On recovery/losses: discusses empirical recovery functions; notes Lehman
  creditors recovered ~**28%** of what they were owed, and that across the 25 largest failed banks
  bankruptcy costs ranged **0.33%–13.19% of book assets (median 5.69%)**. On regulatory
  conventions: the Basel large-exposure cap limits a single exposure to **25% of Tier-1 capital**
  (**15%** between G-SIBs). Overall verdict: standard network-centrality measures have *not* yet
  shown a compelling empirical link to stability, and direct domino contagion is less important
  empirically than funding/fire-sale channels.
- **Public/proprietary.** Survey is published; underlying empirical studies it reviews are mostly
  confidential-data based.
- **Link.** DOI: 10.1257/jel.20151228 · open PDF (LSE):
  https://researchonline.lse.ac.uk/68681/1/Young_Contagion%20in%20financial.pdf

### 10. Huang, Vodenska, Havlin & Stanley (2013) — Bank-asset fire-sale cascades
- **Citation.** Xuqing Huang, Irena Vodenska, Shlomo Havlin, H. Eugene Stanley.
  "Cascading Failures in Bi-partite Graphs: Model for Systemic Risk Propagation."
  *Scientific Reports* 3: 1219, 2013.
- **Method.** Builds a bipartite network between banks and asset classes. An exogenous asset
  write-down reduces bank balance-sheet values; banks fail when marked-to-market assets cross a
  randomized liability barrier; failed banks liquidate their portfolios, causing further
  asset-price deductions and failures. This is a **common-asset / fire-sale channel**, not a
  direct bilateral interbank-exposure channel.
- **Data.** 2007 US commercial-bank balance sheets from WRDS/CBBSD: 7,846 banks and 13 asset
  categories. The model is evaluated against FDIC failures from 2008–2011; 278 of the 371 failed
  banks could be matched to the balance-sheet dataset. The paper reports construction and land
  development loans and nonfarm nonresidential real-estate loans as the strongest crisis drivers.
- **Public/proprietary.** **Bank-level CBBSD/WRDS inputs are proprietary.** The FDIC failure list
  is public. Our implementation therefore uses the paper's published asset categories and average
  weights to create a clearly-labelled synthetic test system.
- **Project role.** Optional robustness engine. It consumes the same binary initial-default
  scenarios as the primary fixed-point cascade, allowing the generators to be compared under a
  second classical contagion mechanism without changing their sample format.
- **Implementation.** `src/systemic_risk/simulator/huang.py`; mathematical conventions in
  `docs/huang_simulation.md`.
- **Link.** DOI: 10.1038/srep01219 ·
  https://www.nature.com/articles/srep01219 · arXiv:1210.4973.

---

## Supporting / catalogued empirical studies (country → data source)

These are the per-country interbank-network studies repeatedly cited by Upper (2011), Cont et al.
(2013), Glasserman–Young (2016), and the review arXiv:1710.11512. Nearly all rely on **national
central-bank supervisory data (confidential)**; the chief exception is the Italian **e-MID**
overnight market, which is **commercially available**.

| Country | Study (representative) | Data source | Public/Proprietary |
|---|---|---|---|
| Austria | Boss et al. (2004); Austrian National Bank "Systemic Risk Monitor" | Oesterreichische Nationalbank (central bank) | Confidential (supervisory) |
| Germany | Upper & Worms (2004) | Deutsche Bundesbank | Confidential (supervisory) |
| Belgium | Degryse & Nguyen (2007) | National Bank of Belgium | Confidential (supervisory) |
| Netherlands | van Lelyveld & Liedorp (2006) | De Nederlandsche Bank | Confidential (supervisory) |
| Italy | Mistrulli (2007); Iori et al. (2008); Bargigli et al. | Banca d'Italia (supervisory) / **e-MID** market | Mostly confidential; **e-MID = commercial (e-MID S.p.A., Milan)** |
| UK | Wells (2004); Langfield et al. (2014) | Bank of England (supervisory) | Confidential (supervisory) |
| US | Furfine (2003); Soramäki et al. (2007); Bech & Atalay (2010) | **Fedwire** / fed funds (Federal Reserve) | Confidential (supervisory) |
| Brazil | Cont, Moura & Santos (2010); Cont–Moussa–Santos (2013) | Banco Central do Brasil | Confidential (supervisory) |
| Mexico | Martínez-Jaramillo et al. | Banco de México | Confidential (supervisory) |
| Japan | Imakubo & Soejima | Bank of Japan | Confidential (supervisory) |
| Euro area | Gabrieli & Georg (2014) | ECB short-term refinancing auctions | Confidential (supervisory) |

---

## Per-paper data-provenance table (the deliverable's core)

| Paper | Dataset | Provider / Source | FREE or PROPRIETARY | URL |
|---|---|---|---|---|
| Eisenberg & Noe (2001) | none (theoretical) | — | N/A | https://papers.ssrn.com/sol3/papers.cfm?abstract_id=173249 |
| Gai & Kapadia (2010) | synthetic Erdős–Rényi random networks | self-generated (Monte Carlo) | FREE (reproducible) | https://www.bankofengland.co.uk/working-paper/2010/contagion-in-financial-networks |
| Battiston et al. (2012) DebtRank | US FED emergency-loan exposures (~$1.2T, 407 inst., 2007–2010) | US Federal Reserve, via Bloomberg disclosure | FREE / PUBLIC | https://pmc.ncbi.nlm.nih.gov/articles/PMC3412322/ |
| Battiston et al. (2012) DebtRank | equity cross-holdings (Q4 2007) | Orbis — Bureau van Dijk | PROPRIETARY | https://pmc.ncbi.nlm.nih.gov/articles/PMC3412322/ |
| Acemoglu et al. (2015) | none (theoretical) | — | N/A | https://www.nber.org/system/files/working_papers/w18727/w18727.pdf |
| Elliott, Golub & Jackson (2014) | European sovereign-debt cross-holdings, 6 countries, Dec 2011 | BIS Quarterly Review, Table 9B (consolidated foreign claims) | FREE / PUBLIC | http://www.bis.org/publ/qtrpdf/r_qa1206.pdf |
| Upper (2011) survey | meta-analysis (no new data) | — | N/A (published article) | https://staff.fnwi.uva.nl/p.j.c.spreij/winterschool/11Upper.pdf |
| Cont, Moussa & Santos (2013) | all bilateral interbank exposures + capital, Brazil, 6 dates 2007–08 | Banco Central do Brasil (supervisory) | PROPRIETARY / CONFIDENTIAL | https://www.bcb.gov.br/pec/wps/ingl/wps219.pdf |
| Glasserman & Young (2015) | European banking system node-level (size/leverage/connectivity) | European Banking Authority (2011) | FREE / PUBLIC | https://www.financialresearch.gov/working-papers/files/OFRwp0009_GlassermanYoung_HowLikelyContagionFinancialNetworks.pdf |
| Glasserman & Young (2016) survey | meta-analysis (no new data) | — | N/A (published article) | https://researchonline.lse.ac.uk/68681/1/Young_Contagion%20in%20financial.pdf |

---

## FREE / PUBLIC datasets identified (usable for calibration)

| Dataset | What it is | Provider | URL | Notes |
|---|---|---|---|---|
| **BIS Quarterly Review — detailed tables** (incl. consolidated banking statistics, foreign claims) | Country-to-country aggregate bank claims; sovereign-debt cross-holdings | Bank for International Settlements | https://www.bis.org/statistics/ ; example tables https://www.bis.org/publ/qtrpdf/r_qa1206.pdf | Used by Elliott–Golub–Jackson. Aggregate (country-level), not bank-bilateral — good for inter-*country* coupling priors. |
| **European Banking Authority stress-test / transparency data** | Bank-by-bank capital, leverage, exposures, RWAs | EBA | https://www.eba.europa.eu/risk-and-data-analysis/risk-analysis/eu-wide-stress-testing | Used by Glasserman–Young (2015). Node-level (per bank), public, machine-readable. Strong source for marginals `p_i` and capital buffers. |
| **US Federal Reserve emergency-loan dataset (Bloomberg/FED)** | Daily central-bank-to-bank crisis exposures, 2007–2010, 407 institutions | Federal Reserve (disclosed via Bloomberg) | https://pmc.ncbi.nlm.nih.gov/articles/PMC3412322/ (described); FED disclosures via federalreserve.gov | Used by DebtRank. Rare *bilateral* (central-bank↔bank) public exposure data. |
| **e-MID Italian overnight interbank market** | Bilateral overnight interbank trades, Italy | e-MID S.p.A., Milan | http://www.e-mid.it/ | **Commercially available** (not free, but obtainable without a central bank). The main non-confidential bilateral interbank dataset in the literature. |
| **Fedwire payment data** | Interbank payment flows (US) | Federal Reserve | (research-access; not openly downloadable) | Used by Soramäki et al. (2007). Listed for completeness — generally restricted. |

**For our generator specifically:** the realistic free pipeline is EBA (per-bank marginals `p_i`,
sizes, capital buffers) + BIS (country-level coupling structure for `J_ij` priors), with the
bilateral interbank matrix filled in by **maximum-entropy reconstruction** (Upper 2011) — which
is conceptually the same move as our plausibility model inferring couplings from marginals.

---

## Realistic parameter values for calibrating a synthetic generator

Pulled from the sources above; these are the numbers a synthetic correlated-default generator +
cascade simulator should target to be defensible.

**Network topology / density**
- Real interbank networks are **sparse and scale-free** (NOT Erdős–Rényi). Degree distributions
  are **heavy-tailed / power-law**, typical exponent **γ ≈ 2.0–3.0** (Brazil: γ ≈ 2.7–2.8).
- **Brazil (Cont et al.):** ~400 institutions, **mean in/out degree ≈ 8.5** → density on the
  order of ~8.5/400 ≈ **~2%** (very sparse; most bank pairs have no direct exposure).
- Empirical interbank networks across studies: **average degree ≈ 15–23**, **clustering
  coefficient ≈ 0.1–0.9**, **average path length ≈ 2.3–2.6** (small-world). `[from JASSS
  20(4):15 summary of empirical studies]`
- **Core–periphery structure** is the norm: a small dense core (**~20% of banks**, per Austria/
  Boss et al.) intermediating a sparse periphery. A realistic `J_ij` should be tiered, not
  homogeneous.
- For purely-stylized random-network baselines (Gai–Kapadia family): **N = 1000**, directed
  Erdős–Rényi, average degree swept across a range to locate the robust-yet-fragile contagion
  window. Use these only as a sanity-check null model, not as the realistic case.

**Capital buffers / leverage (failure thresholds)**
- Gai–Kapadia-family baseline: **capital buffer ≈ 4% of total assets**; descendants also use
  **~10%** capital ratios. Basel **Tier-1 minimum ≈ 8%** is the common regulatory failure
  threshold used in simulations.
- Regulatory single-name exposure caps (good priors for max edge weight): **≤ 25% of Tier-1
  capital** per counterparty; **≤ 15%** between global systemically-important banks (Basel large-
  exposure standard).

**Interbank-asset share**
- **Interbank assets ≈ 20% of total assets** is the standard Gai–Kapadia-family baseline (range
  ~**20–40%** explored). This sets how much of a bank's balance sheet is exposed to the network
  channel vs. external assets.

**Loss-given-default / recovery**
- Simulation convention: **LGD ≈ 50–60%** (Upper 2011); results are highly LGD-sensitive.
- Worst-case on-impact: **zero recovery (LGD = 100%)** is the standard conservative Gai–Kapadia
  assumption at the moment of default.
- Empirical anchor: **Lehman creditors recovered ≈ 28%** (LGD ≈ 72%); failed-bank bankruptcy
  costs alone **0.33%–13.19% of book assets, median 5.69%** (Glasserman–Young 2016).

**Cascade-outcome thresholds (for defining a "severe"/tail scenario)**
- A common definition of a systemic "crisis" in this literature: **≥ 5% of banks fail**
  (Hurd/Kobayashi); contagion-trigger thresholds in the ~10%-of-banks range also appear.
- Country contagious-loss magnitudes where contagion occurs: **~16–20% of total banking-system
  assets** (Belgium ~20%, Italy ~16%, UK ~16%) — a realistic target band for the loss
  distribution's right tail.

**Marginals `p_i` (default probabilities)**
- Not standardized across these network papers (they take exposures as given and shock them).
  For marginals, draw on rating-implied PDs (the DebtRank/credit literature) — e.g., investment-
  grade annual PDs well under 1%, sub-investment-grade rising to tens of percent — rather than on
  the cascade papers, which are about the *coupling* (`J_ij`) side, not the marginals.

---

### Sources consulted (primary)
Royal Society (Gai–Kapadia landing) · Bank of England WP 383 · Scientific Reports / PMC
(DebtRank) · AEA / MIT / NBER (Acemoglu et al.) · AEA / Golub author PDF / openICPSR (Elliott et
al.) · Journal of Financial Stability / UvA-hosted PDF (Upper) · HAL / Banco Central do Brasil WP
219 / Cont author PDF (Cont et al.) · Journal of Banking & Finance / OFR WP (Glasserman–Young
2015) · Journal of Economic Literature / LSE Research Online (Glasserman–Young 2016) · review
arXiv:1710.11512 (Kobayashi et al.) · arXiv:1312.6804 (Hurd-style threshold model) · JASSS
20(4):15 (empirical network-statistic ranges).
