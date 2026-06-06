# 02 — Credit Default Correlation & Portfolio Credit Risk Models

**Scope.** Survey of the foundational credit-default-correlation and portfolio-credit-risk literature, with emphasis on **(a) what data each work uses, (b) where that data comes from, (c) whether it is FREE/PUBLIC or PROPRIETARY/PAID, and (d) realistic numeric values for marginal default probabilities `p_i` and asset/default correlations** that can calibrate this repo's Ising/Boltzmann plausibility model

```
Π(x) = Σ_i ln(p_i) x_i + Σ_{i<j} J_ij x_i x_j        x ∈ {0,1}^n
```

where `p_i` ≈ one-year marginal default probabilities and `J_ij` ≈ pairwise default couplings.

> **Calibration bottom line (read first).** Marginal one-year default probabilities `p_i` are well-pinned by rating-agency studies: roughly **Aaa/AAA ≈ 0%, Baa/BBB ≈ 0.2-0.3%, Ba/BB ≈ 1-1.5%, B ≈ 3-8%, Caa-C/CCC ≈ 15-30%**. Pairwise *default* correlations (the right scale for binary `x_i x_j`) are **small and positive**, typically **~0.001-0.03**, larger within an industry than across, and larger in recessions. The Basel/Vasicek *asset* correlation (a latent-Gaussian-factor correlation, NOT the same object as default correlation) is **0.12-0.24 for corporates**, **0.15 for mortgages**, **0.04 for revolving retail**. Use asset correlation if you adopt a Gaussian-copula/factor link; use the smaller default-correlation numbers if `J_ij` is meant to reproduce raw co-default rates.

---

## 1. The two "correlations" — do not conflate them

A recurring trap in this literature: there are two distinct quantities.

- **Default (event) correlation** `ρ_D = corr(1_{default i}, 1_{default j})` — the Pearson correlation of the *binary* default indicators. This is the object most directly analogous to the coupling between bits `x_i, x_j` in the repo's plausibility model. It is **small** (single-digit percent at most) because defaults are rare; for two issuers each defaulting with prob `p`, even perfect comonotonicity gives bounded covariance.
- **Asset (latent factor) correlation** `ρ_A` — the correlation of the continuous latent "asset-value" variables in a Merton/Gaussian-copula model. A given `ρ_A` (e.g. 0.20) maps to a *much smaller* default correlation `ρ_D` after thresholding at the (deep) default barrier. Basel's 0.12-0.24 numbers are `ρ_A`.

Mapping (single Gaussian factor): latent `V_i = √ρ_A · Z + √(1-ρ_A) · ε_i`, default if `V_i < Φ⁻¹(p_i)`. Then joint default prob = bivariate normal CDF `Φ_2(Φ⁻¹(p_i), Φ⁻¹(p_j); ρ_A)`, and `ρ_D` follows. **This Φ⁻¹ thresholding is exactly Li (2000)'s Gaussian copula and is the standard analytic recipe to turn `(p_i, ρ_A)` into the repo's `J_ij`.**

---

## 2. Key models / papers — annotations

### 2.1 Li (2000) — Gaussian copula for default correlation
- **Citation.** David X. Li, "On Default Correlation: A Copula Function Approach," *Journal of Fixed Income*, Vol. 9, No. 4 (2000), pp. 43-54. SSRN working-paper version: abstract_id=187289. ([pm-research](https://www.pm-research.com/content/iijfixinc/9/4/43), [SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=187289), [open PDF mirror](https://www.ressources-actuarielles.net/EXT/ISFA/1226.nsf/8d48b7680058e977c1256d65003ecbb5/34e84cb615c8b4eac12575fe006a9759/$FILE/li.defaultcorrelation.pdf))
- **Method.** Defines default correlation as the correlation of *survival times*; uses a copula to glue marginal survival distributions into a joint law. Shows the CreditMetrics asset-correlation approach is **equivalent to a Gaussian (normal) copula**.
- **DATA.** Marginal default/survival curves are bootstrapped from **market credit spreads** — "risky bond prices or asset swap spreads" (and in practice **CDS spreads**); the dependence parameter is imported as an **asset correlation** (à la CreditMetrics, i.e. from equity correlations).
- **SOURCE / PROVIDER.** Bond/asset-swap/CDS quotes from dealers and pricing vendors; correlation input from CreditMetrics-style equity factor data.
- **PUBLIC vs PROPRIETARY.** Method is public; the **inputs are proprietary** (live CDS/bond quotes, equity-correlation matrices). **PAID.**
- **Why it matters for us.** This is the analytic engine that converts `(p_i, ρ_A)` into co-default probabilities — i.e. the closed form for `J_ij` under a Gaussian dependence assumption. Also the cautionary tale: post-2008, the model was blamed for **underestimating tail/systemic co-movement**, partly because correlation was calibrated to *CDS price co-movement in benign times* rather than to historical joint-default data (Li himself: "very few people understand the essence of the model"). ([David X. Li — Wikipedia](https://en.wikipedia.org/wiki/David_X._Li))

### 2.2 Vasicek single-factor / ASRF (1987, 2002) — the Basel kernel
- **Citation.** Oldrich A. Vasicek, "Probability of Loss on a Loan Portfolio," KMV working paper (1987); published as **"Loan Portfolio Value," *Risk* 15(12), Dec 2002, pp. 160-162.** ([Risk.net](https://www.risk.net/risk-management/credit-risk/1500333/loan-portfolio-value), [Bank of Greece PDF](https://www.bankofgreece.gr/MediaAttachments/Vasicek.pdf))
- **Method.** Merton structural model + a single systematic factor; in the large, well-diversified limit the portfolio loss rate converges to the closed-form **Vasicek distribution**, governed by `(PD, ρ_A)`. "Portfolio-invariant," which is why Basel adopted it.
- **DATA.** The 1987/2002 papers are **theoretical** (no dataset); the model is fed by a firm's own `PD` estimates and an assumed/estimated asset correlation `ρ_A`. KMV calibrated `ρ_A` from its proprietary equity-derived asset-value model.
- **SOURCE / PROVIDER.** KMV (now **Moody's Analytics**) asset-value/EDF data for calibration.
- **PUBLIC vs PROPRIETARY.** Model public; KMV/EDF calibration data **PROPRIETARY/PAID**.
- **Numbers for us.** Gives the exact latent-factor structure the repo's `J_ij` can mirror; `ρ_A` in the **0.10-0.25** range is the empirically and regulatorily relevant band.

### 2.3 Gordy (2003) + Basel II/III IRB — regulatory asset correlations (PRIMARY NUMERIC SOURCE)
- **Citation.** Michael B. Gordy, "A risk-factor model foundation for ratings-based bank capital rules," *Journal of Financial Intermediation* 12(3), 2003, pp. 199-232. Operationalized in **BCBS, "An Explanatory Note on the Basel II IRB Risk Weight Functions" (2005)** ([BIS PDF](https://www.bis.org/bcbs/irbriskweight.pdf)) and restated verbatim in national rules, e.g. **OSFI CAR (2026) Chapter 5** ([OSFI](https://www.osfi-bsif.gc.ca/en/guidance/guidance-library/capital-adequacy-requirements-car-2026-chapter-5-credit-risk-internal-ratings-based-approach)).
- **Method.** Proves the ASRF capital formula is the asymptotic, portfolio-invariant limit; capital = `LGD · [Φ((Φ⁻¹(PD) + √ρ · Φ⁻¹(0.999))/√(1-ρ)) − PD] · maturity_adj`, at the **99.9%** confidence level.
- **DATA.** The supervisory `ρ` values were **calibrated by the Basel Committee using supervisory/G10 bank internal data and time-series of default rates** (the exact proprietary inputs are not public); the **formulas themselves are fully public**.
- **PUBLIC vs PROPRIETARY.** **Formulas FREE/PUBLIC** (BIS, OSFI, EBA). Underlying calibration data proprietary to BCBS/supervisors.
- **EXACT FORMULAS & NUMBERS (verified against OSFI CAR 2026 Ch.5 and BIS):**
  - **Corporate / sovereign / bank:** `R = 0.12·(1−e^(−50·PD))/(1−e^(−50)) + 0.24·(1−(1−e^(−50·PD))/(1−e^(−50)))` → ranges **0.12 (high PD) to 0.24 (low PD)**.
  - **SME firm-size adjustment** (sales `S` in €M, 5-50M, here 7.5-75M per CAR): subtract `0.04·(1−(S−7.5)/67.5)` from the corporate `R`.
  - **Residential mortgage:** fixed **R = 0.15** (0.22 if repayment materially depends on property cash flows).
  - **Qualifying revolving retail (QRRE):** fixed **R = 0.04**.
  - **Other retail:** `R = 0.03·(1−e^(−35·PD))/(1−e^(−35)) + 0.16·(1−(1−e^(−35·PD))/(1−e^(−35)))` → **0.03 to 0.16**.
  - **Maturity adjustment** `b = (0.11852 − 0.05478·ln(PD))²`.

### 2.4 CreditMetrics (J.P. Morgan, 1997) — equity correlations as a default-correlation proxy
- **Citation.** Gupton, Finger & Bhatia, *CreditMetrics — Technical Document*, J.P. Morgan, 1997 (now stewarded by MSCI/RiskMetrics). ([MSCI PDF](https://www.msci.com/documents/10199/93396227-d449-4229-9143-24a94dab122f), [intro PDF](http://marshallinside.usc.edu/dietrich/RiskMetricsIntrotoCreditMetrics.pdf))
- **Method.** Merton-style rating-transition model: simulate joint *credit-quality migrations* by thresholding correlated latent asset returns; portfolio value distribution from rating-migration + spread revaluation.
- **DATA.** (i) **Rating-transition and default matrices** from rating agencies (S&P/Moody's); (ii) crucially, because joint default data is too sparse, it **uses EQUITY-return correlations as a proxy for asset/credit-quality correlations**, mapped through an industry/country factor model.
- **SOURCE / PROVIDER.** Agency transition matrices (S&P, Moody's); equity prices and the CreditMetrics industry/country correlation dataset (J.P. Morgan / MSCI-RiskMetrics).
- **PUBLIC vs PROPRIETARY.** Methodology document **FREE**; the equity-correlation dataset and agency transition matrices are **PROPRIETARY/PAID**.
- **Why it matters.** This is the canonical justification for the repo's whole premise: *you cannot read joint-default correlations off history, so you infer the dependence structure from a proxy and generate scenarios.* The "equity proxy" is exactly a way of setting `J_ij` when co-default data is absent.

### 2.5 Duffie & Singleton (1999) — intensity / reduced-form models
- **Citation.** Darrell Duffie & Kenneth J. Singleton, "Modeling Term Structures of Defaultable Bonds," *Review of Financial Studies* 12(4), 1999, pp. 687-720. ([Oxford Academic](https://academic.oup.com/rfs/article-abstract/12/4/687/1578719), [JSTOR](https://www.jstor.org/stable/2645962))
- **Method.** Default arrives as the first jump of a hazard-rate (intensity) process; defaultable claims priced by discounting at a default-adjusted short rate `r + λ·LGD` ("recovery of market value").
- **DATA.** Calibrated to **corporate and sovereign bond/credit-spread term structures** (and later CDS spreads); the paper is mainly methodological with illustrative term-structure data.
- **SOURCE / PROVIDER.** Bond prices / credit spreads from market data vendors; sovereign yield curves.
- **PUBLIC vs PROPRIETARY.** Method public; spread/CDS inputs **PROPRIETARY/PAID** (some sovereign curves free).
- **Relevance.** Gives `p_i` a dynamic, spread-implied footing (`p ≈ 1−e^(−λ·T)`); correlation enters via correlated intensities (`λ_i`), the continuous-time analogue of correlated `p_i`. Foundational for the *frailty* literature (§2.8).

### 2.6 CreditRisk+ (Credit Suisse First Boston, 1997) — actuarial / Poisson
- **Citation.** *CreditRisk+: A Credit Risk Management Framework*, Credit Suisse Financial Products, 1997. ([WIAS PDF](https://www.wias-berlin.de/people/schoenma/hrs_JOR.pdf), [studylib](https://studylib.net/doc/9088140/credit-risk-plus))
- **Method.** Actuarial: number of defaults ~ Poisson with a **stochastic default rate** driven by Gamma-distributed sector factors; correlation is induced purely through shared, volatile sector intensities (no asset values). Closed-form loss distribution.
- **DATA NEEDED.** Per-obligor **default rate**, **default-rate volatility**, **sector weights**, **exposure**, **recovery**.
- **SOURCE / PROVIDER.** Default rates & volatilities from rating-agency histories (Moody's/S&P) or internal data; the document itself ships **illustrative agency-based default-rate and volatility figures**.
- **PUBLIC vs PROPRIETARY.** Framework **FREE**; calibration inputs typically agency data (**PAID**) or internal.
- **Relevance.** The Gamma-mixed-Poisson "default-rate volatility" is an alternative, parsimonious way to inject correlation — analogous to making the repo's `p_i` themselves random/shared rather than adding explicit `J_ij`.

### 2.7 de Servigny & Renault (2002/2004) — empirical default correlations (PRIMARY EMPIRICAL `J_ij` SOURCE)
- **Citation.** Arnaud de Servigny & Olivier Renault, "Default Correlation: Empirical Evidence," S&P working paper, 2002 (later in *The Standard & Poor's Guide to Measuring and Managing Credit Risk*, McGraw-Hill, 2004). ([Semantic Scholar](https://www.semanticscholar.org/paper/Default-correlation:-empirical-evidence-Servigny-Renault/aae251436d0e3b489951c0d38463d71106755675), [Risk.net summary](https://www.risk.net/risk-management/credit-risk/1530250/correlation-evidence))
- **Method.** Directly estimate empirical default correlations from a large rating history; benchmark against equity-implied correlations.
- **DATA.** **Standard & Poor's CreditPro** rating + default database (issuer ratings and defaults; multi-decade).
- **SOURCE / PROVIDER.** **S&P Global (CreditPro / S&P rating database).**
- **PUBLIC vs PROPRIETARY.** **PROPRIETARY/PAID** (CreditPro is a paid product). Summary findings are public.
- **KEY FINDINGS / NUMBERS.** (i) Empirical default correlations are **small and positive**; (ii) **within-industry > between-industry**; (iii) **higher in recessions than expansions** — but for *investment grade* the recession uplift is tiny (de Servigny–Renault report investment-grade default correlation only ~**0.01 percentage point** higher in recessions than in growth years); (iv) the **equity-default correlation link is obscured by statistical noise**, and risk-free rates have little measurable effect. This validates modeling `J_ij` as small, positive, sector-structured, and regime-dependent.

### 2.8 Das, Duffie, Kapadia & Saita (2007) — defaults cluster *more* than a factor model implies
- **Citation.** S. Das, D. Duffie, N. Kapadia, L. Saita, "Common Failings: How Corporate Defaults Are Correlated," *Journal of Finance* 62(1), 2007, pp. 93-117. ([RePEc](https://ideas.repec.org/a/bla/jfinan/v62y2007i1p93-117.html), [NBER w11961](https://www.nber.org/papers/w11961))
- **Method.** Tests the "doubly stochastic" hypothesis (defaults independent given common factors). Rejects it — there is **excess default clustering** (contagion / **frailty** = unobserved common factors). Follow-up: Duffie et al., "Frailty Correlated Default," *J. Finance* 2009.
- **DATA.** **U.S. public firms, 1979-2004**; default and covariate data, with funding/data from **Moody's** (Moody's Default Risk Service + Moody's KMV EDFs).
- **SOURCE / PROVIDER.** **Moody's / Moody's KMV.**
- **PUBLIC vs PROPRIETARY.** **PROPRIETARY/PAID** (Moody's DRS / KMV EDFs).
- **Relevance — direct support for this repo.** This is the empirical case that *Gaussian/factor models underestimate joint tail defaults*. It motivates a richer generator (the repo's QCBM/Boltzmann model) that can capture excess co-default mass beyond a single-factor `ρ_A` — i.e. heavier `J_ij`-driven tails than ASRF.

### 2.9 Lopez (2004) — how asset correlation depends on PD and size
- **Citation.** Jose A. Lopez, "The empirical relationship between average asset correlation, firm probability of default, and asset size," *Journal of Financial Intermediation* 13(2), 2004, pp. 265-283 (FRBSF WP 2002-05). ([RePEc/FRBSF](https://ideas.repec.org/p/fip/fedfwp/2002-05.html), [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S1042957303000457))
- **Method.** Within the ASRF frame, estimate average asset correlation as a function of PD and asset size.
- **DATA.** **Moody's KMV** asset-value/EDF data, **year-end 2000**, U.S./Japanese/European firms.
- **SOURCE / PROVIDER.** **Moody's KMV (now Moody's Analytics).**
- **PUBLIC vs PROPRIETARY.** **PROPRIETARY/PAID.**
- **KEY FINDING (justifies Basel's functional form).** Average **asset correlation DECREASES with PD** and **INCREASES with asset size** — exactly the shape Basel hard-codes (0.24 → 0.12 as PD rises). Empirically asset correlations land broadly in the **~0.10-0.25** band, consistent with Basel's 0.12-0.24.

---

## 3. Provenance table (paper → dataset → provider → access → URL)

| Work | Dataset used | Provider | FREE / PAID | URL |
|---|---|---|---|---|
| Li (2000), copula | Credit/CDS spreads → marginals; asset correlation input | Dealers/vendors; CreditMetrics equity factors | PAID (inputs) | https://www.pm-research.com/content/iijfixinc/9/4/43 |
| Vasicek (1987/2002) ASRF | Theoretical; KMV asset-value calib. | KMV / Moody's Analytics | PAID (calib.) | https://www.risk.net/risk-management/credit-risk/1500333/loan-portfolio-value |
| Gordy (2003) / Basel IRB | Supervisory bank/default data → fixed ρ | BCBS / national supervisors | **FREE (formulas)** | https://www.bis.org/bcbs/irbriskweight.pdf |
| CreditMetrics (1997) | Agency transition matrices + **equity correlations (proxy)** | S&P/Moody's; J.P. Morgan / MSCI | PAID (data); FREE doc | https://www.msci.com/documents/10199/93396227-d449-4229-9143-24a94dab122f |
| Duffie & Singleton (1999) | Corp/sovereign bond & credit-spread curves | Market data vendors | PAID (inputs) | https://academic.oup.com/rfs/article-abstract/12/4/687/1578719 |
| CreditRisk+ (1997) | Default rates + default-rate volatilities | Rating agencies / internal | PAID (data); FREE doc | https://www.wias-berlin.de/people/schoenma/hrs_JOR.pdf |
| de Servigny & Renault (2002) | **S&P CreditPro** rating/default DB | **S&P Global** | PAID | https://www.semanticscholar.org/paper/Default-correlation:-empirical-evidence-Servigny-Renault/aae251436d0e3b489951c0d38463d71106755675 |
| Das–Duffie–Kapadia–Saita (2007) | US firms 1979-2004; defaults + EDFs | **Moody's / Moody's KMV** | PAID | https://www.nber.org/papers/w11961 |
| Lopez (2004) | Moody's KMV EDF/asset data, YE2000 | **Moody's KMV** | PAID | https://ideas.repec.org/p/fip/fedfwp/2002-05.html |
| Lucas (1995) | Moody's defaults 1970-1993 | **Moody's** | PAID | https://www.researchgate.net/publication/228679968_Default_correlation_Empirical_evidence |

---

## 4. FREE / PUBLIC datasets and resources

| Resource | What it gives | FREE? | URL |
|---|---|---|---|
| **Basel IRB formulas** (BIS explanatory note; EBA; OSFI CAR) | Exact asset-correlation & capital formulas (0.12-0.24, 0.15, 0.04, 0.03-0.16) | **FREE** | https://www.bis.org/bcbs/irbriskweight.pdf · https://www.osfi-bsif.gc.ca/en/guidance/guidance-library/capital-adequacy-requirements-car-2026-chapter-5-credit-risk-internal-ratings-based-approach |
| **Rating-agency annual default studies** (summary tables) | One-year & cumulative default rates by rating; transition matrices. S&P "Default, Transition & Recovery" and Moody's "Annual Default Study" are published yearly; **summary tables are free to read** even though the underlying databases (CreditPro, DRS) are paid. | **FREE (tables)** | https://www.spglobal.com/ratings/en/regulatory/article/default-transition-and-recovery-2025-annual-global-corporate-default-and-rating-transition-study-s101673333 |
| **Moody's "Corporate Default and Recovery Rates, 1920-xxxx"** (historic Special Comments, PDF) | Long-run one-year & cumulative default rates by rating, 1920 onward | **FREE** (older issues posted publicly) | https://www.bu.edu/econ/files/2015/01/Moodys_Default_1920-2004.pdf |
| **FRED (St. Louis Fed)** | Moody's Aaa & Baa seasoned yields (`AAA`, `BAA`, `DAAA`, `DBAA`), Baa-10Y spread (`BAA10Y`), **ICE BofA US High-Yield OAS (`BAMLH0A0HYM2`)** and rating-bucket OAS series — proxies for market-implied credit risk and a free way to get spread co-movement | **FREE** | https://fred.stlouisfed.org/series/BAA · https://fred.stlouisfed.org/series/BAMLH0A0HYM2 |
| **NUS-CRI Probability of Default** | Daily point-in-time PDs (1m-5y) for 90,000+ listed firms; "public-good" credit measure | **FREE** (3,000 firms open; full set with credentials) | https://nuscri.org/ · https://d.nuscri.org/ |
| **AssetCorr (R/CRAN)** | Open-source estimators of asset correlation from default-rate time series (method of moments, asymptotic ML) — useful to fit `ρ_A`/`J_ij` from free agency default series | **FREE (code)** | https://cran.r-project.org/package=AssetCorr |
| **Li (2000) open mirror; Vasicek (2002) Bank-of-Greece mirror** | Full text of foundational papers | **FREE** | https://www.ressources-actuarielles.net/EXT/ISFA/1226.nsf/8d48b7680058e977c1256d65003ecbb5/34e84cb615c8b4eac12575fe006a9759/$FILE/li.defaultcorrelation.pdf · https://www.bankofgreece.gr/MediaAttachments/Vasicek.pdf |

**Proprietary/paid (for completeness):** S&P **CreditPro**/RatingsDirect; Moody's **Default Risk Service (DRS)** & **Moody's Analytics/KMV EDF**; **IHS Markit / S&P Global CDS pricing** (single-name CDS spreads, subscription, also via WRDS/FactSet) — https://wrds-www.wharton.upenn.edu/pages/about/data-vendors/markit/ .

---

## 5. Realistic numeric values to calibrate `p_i` and `J_ij`

### 5.1 Marginal one-year default probabilities `p_i` (by rating)

| Rating (S&P / Moody's) | Typical **1-yr** default prob `p_i` | Source / note |
|---|---|---|
| AAA / Aaa | ~**0.00%** (0.000) | Moody's 1920-2006: Aaa 0.000%; S&P max-observed 0% ([efalken table](https://www.efalken.com/banking/html's/defaultcurves.htm)) |
| AA / Aa | ~**0.02-0.06%** | Moody's 1920-2006: Aa 0.062%; S&P max-yr 0.38% |
| A / A | ~**0.05-0.1%** | Moody's 1920-2006: A 0.074% |
| BBB / Baa | ~**0.2-0.3%** | Moody's 1920-2006: Baa **0.296%**; S&P 1981-96 avg ~0.2%, max-yr 1.02% |
| BB / Ba | ~**1.0-1.8%** | S&P 1981-96 avg **1.1%**, Moody's **1.8%**; S&P max-yr 4.22% |
| B / B | ~**3.4-8%** | S&P 1981-96 avg **4.8%**, Moody's **8.1%**; S&P max-yr 13.84% |
| CCC-C / Caa-C | ~**15-30%** | S&P 1981-96 avg **16.4%**; S&P max-yr **49.28%** |
| Investment grade (agg.) | ~**0.1%** | — |
| Speculative grade (agg.) | ~**3.8-4.2%** | S&P 3.8%, Moody's 4.2% (1981-96) |

*Notes:* averages differ across providers/windows; **recession-year `p_i` can be 2-5× the long-run average** (e.g. spec-grade default rates spiked toward ~10-12% in 2001 and 2009 — useful for stress `p_i`). For dynamic `p_i`, intensity models give `p ≈ 1 − e^(−λT)` with `λ` from spreads. Primary public sources: [S&P annual study](https://www.spglobal.com/ratings/en/regulatory/article/default-transition-and-recovery-2025-annual-global-corporate-default-and-rating-transition-study-s101673333), [Moody's 1920-2004](https://www.bu.edu/econ/files/2015/01/Moodys_Default_1920-2004.pdf), [CFI summary](https://corporatefinanceinstitute.com/resources/fixed-income/investment-grade-bonds/).

### 5.2 Correlations — two scales

**(A) Asset / latent-factor correlation `ρ_A`** (use if `J_ij` is set via a Gaussian-copula thresholding of `p_i`):

| Context | `ρ_A` value | Source |
|---|---|---|
| Basel corporate/bank/sovereign | **0.12 → 0.24** (decreasing in PD) | [BIS IRB note](https://www.bis.org/bcbs/irbriskweight.pdf), [OSFI CAR](https://www.osfi-bsif.gc.ca/en/guidance/guidance-library/capital-adequacy-requirements-car-2026-chapter-5-credit-risk-internal-ratings-based-approach) |
| Basel residential mortgage | **0.15** (0.22 if property-cash-flow dependent) | OSFI CAR ¶79 |
| Basel qualifying revolving retail | **0.04** | OSFI CAR ¶80 |
| Basel other retail | **0.03 → 0.16** | OSFI CAR ¶81 |
| Empirical (Lopez 2004) | broadly **~0.10-0.25**, ↓ with PD, ↑ with size | [Lopez](https://ideas.repec.org/p/fip/fedfwp/2002-05.html) |

**(B) Default (event) correlation `ρ_D`** (use directly if `J_ij` reproduces raw co-default rates of the binary `x_i`):

| Context | `ρ_D` value | Source / note |
|---|---|---|
| General magnitude | **~0.001-0.03** (small, positive) | de Servigny & Renault; rare-event arithmetic |
| Within-industry vs across | within > across (across often near 0) | [de Servigny & Renault](https://www.risk.net/risk-management/credit-risk/1530250/correlation-evidence) |
| Investment grade, recession uplift | only ~**+0.01 pp** vs expansion | de Servigny & Renault (2004) |
| Speculative / B-rated | larger; rises sharply at longer horizons (e.g. B-rated 2-yr default correlation ~**0.16** in Lucas-type estimates vs ~0.04 implied by a thin factor model) | [Lucas 1995](https://www.researchgate.net/publication/228679968_Default_correlation_Empirical_evidence), [Das et al. discussion] |
| Excess clustering vs factor model | defaults cluster **more** than doubly-stochastic factor models predict (frailty/contagion) | [Das–Duffie–Kapadia–Saita 2007](https://www.nber.org/papers/w11961) |

**Practical defaults for this repo.** A defensible starting calibration: set `p_i` from the §5.1 rating table for each institution; set `J_ij` via Gaussian-copula thresholding with `ρ_A ≈ 0.20` for same-sector pairs and `ρ_A ≈ 0.05-0.10` cross-sector (decreasing toward the 0.12 floor for the riskiest names), which yields small positive default correlations `ρ_D ~ 10^-3-10^-2` consistent with the empirical evidence — then **deliberately fatten the tail** (extra `J_ij` mass / a shared frailty factor) to honor Das et al.'s finding of excess co-default clustering, which is precisely the regime where a generative/quantum sampler earns its keep.

---

## 6. Caveats & [UNVERIFIED] flags
- Agency average default rates depend heavily on **provider, weighting (issuer- vs value-weighted), and sample window**; the §5.1 numbers are representative, not canonical. The S&P "max one-year" figures are *worst observed years*, not averages.
- The **exact proprietary calibration data** behind Basel's 0.12/0.24 (and the precise empirical default-correlation tables in de Servigny & Renault and Lucas) sit behind paywalls; specific cell-level values beyond those quoted here are **[UNVERIFIED]** without the paid sources/full PDFs (several academic PDFs could not be machine-read in this pass).
- The `ρ_A` ↔ `ρ_D` mapping is model-dependent (single-factor Gaussian assumed); other copulas (t, Clayton) give materially fatter joint tails — relevant to the repo's thesis that Gaussian dependence understates systemic co-defaults.
