# Research Foundation

Five literature surveys ground this project's data sourcing and synthetic generation. The full
annotated surveys live in `research/sections/`; this file synthesizes the cross-cutting conclusions
that drive what data we pull and how we generate scenarios.

| # | Section | Focus |
|---|---------|-------|
| 01 | [`sections/01_network_contagion.md`](sections/01_network_contagion.md) | Interbank / network contagion & cascade models |
| 02 | [`sections/02_credit_default_correlation.md`](sections/02_credit_default_correlation.md) | Credit default correlation & portfolio credit risk |
| 03 | [`sections/03_statistical_mechanics_ising.md`](sections/03_statistical_mechanics_ising.md) | Statistical-mechanics / Ising / max-entropy (theoretical core) |
| 04 | [`sections/04_systemic_risk_measures.md`](sections/04_systemic_risk_measures.md) | Systemic-risk measures & public data infrastructure |
| 05 | [`sections/05_quantum_finance.md`](sections/05_quantum_finance.md) | Quantum-finance precedents |

---

## 1. Our Ising/Boltzmann plausibility model has direct academic precedent

The plausibility model in `scenario_generation.md` is not novel framing — it is an established
correlated-default model:

- **Filiz, Guo, Morton & Sturmfels (2012), *Graphical Models for Correlated Defaults*** (arXiv:0809.1393).
  Their joint default law is *literally* our Π(x): `p_w = (1/Z) exp(Σ η_i w_i + Σ η_uv w_u w_v)`, so
  η_i ↔ our field and η_uv ↔ `J_ij`. **This is the load-bearing classical citation.**
- **Molins & Vives (2005)** (arXiv:cond-mat/0401378) and **Kitsukawa, Mori & Hisakado (2006)**
  (arXiv:physics/0603040) — mean-field (infinite-range) Ising credit model with closed-form
  field→P_default and coupling→ρ_default. Crucially this gives an **exact loss distribution at any n
  (including 54)** for the homogeneous case → a validation oracle for both the classical sampler and
  the QCBM.
- **Schneidman et al. (2006), *Nature*** — canonical Boltzmann-learning template; "weak pairwise
  correlations imply strongly correlated network states" = exactly the systemic-risk thesis that
  independent-Bernoulli sampling undercounts joint catastrophes.

## 2. Generator recipe (concrete, citable)

From §03 (+ §02). This is the method Phase-2 generation implements:

- **Field ↔ marginal:** use `h_i = logit(p_i) = ln(p_i/(1-p_i))` (the {0,1}-basis max-ent field; the
  doc's "ln p_i" is its rare-event approximation). Once `J ≠ 0`, marginals drift, so **refit fields by
  Boltzmann learning** (`Δh_i ∝ p_i − ⟨x_i⟩_model`) to restore `⟨x_i⟩ = p_i`. *(Filiz 2012 Cor. 5;
  Schneidman 2006.)*
- **Coupling ↔ correlation:** fast inverse-Ising — naive-MF `J_ij ≈ −(C⁻¹)_ij` or TAP; pseudo-likelihood
  / Boltzmann learning when correlations are strong. *(Bury 2013; Nguyen–Zecchina–Berg 2017.)*
- **Coupling ↔ exposure (our framing):** reconstruct the exposure graph with the **density-corrected
  gravity model** from per-node size data, then `J_ij ∝ w_ij`. *(Cimini et al. 2015.)*
- **Sampling by size:** `n ≤ ~20` → **exact enumeration** of 2ⁿ (compute Z, exact tails); `~20–40` →
  Gibbs/Metropolis MCMC; **`n = 54` → exact is impossible (2⁵⁴ ≈ 1.8×10¹⁶)**, use MCMC with
  parallel tempering / cluster moves (critical slowing-down near the correlation-driven first-order
  transition). The homogeneous Ising loss distribution stays **closed-form at any n** → ground truth.
  *(Nguyen–Zecchina–Berg 2017; Molins–Vives 2005.)* This MCMC bottleneck at scale is precisely what the
  QCBM/qGAN loader targets *(Zoufal–Lucchi–Woerner 2019).*

## 3. Calibration parameters (with sources)

**Marginal one-year default probabilities `p_i`** (S&P / Moody's annual default studies):

| Rating | 1-yr PD | | Rating | 1-yr PD |
|---|---|---|---|---|
| AAA/Aaa | ~0.00% | | BB/Ba | ~1.0–1.8% |
| AA/Aa | ~0.02–0.06% | | B | ~3.4–8% |
| A | ~0.05–0.1% | | CCC–C | ~15–30% (worst yr ~49%) |
| BBB/Baa | ~0.2–0.3% | | Spec-grade agg. | ~3.8–4.2% |

**Correlations — two distinct objects** (key finding, §02):
- *Asset/latent `ρ_A`* (use with Gaussian-copula Φ⁻¹ link): Basel corporate **0.12→0.24**, mortgage 0.15,
  revolving retail 0.04; empirical ~0.10–0.25.
- *Default/event `ρ_D`* (use if `J_ij` reproduces raw co-default rates): **~0.001–0.03**, within-industry
  > cross-industry, larger uplift in recessions for spec-grade. The two are linked by default-barrier
  thresholding — do not conflate them.

**Network topology & thresholds** (§01): real interbank networks are **sparse, scale-free
(γ ≈ 2–3), core-periphery (~20% core) — NOT Erdős–Rényi**; mean degree ~8–23, ~2% density at ~400 nodes.
Capital buffers ~4% (Gai–Kapadia) to 8% (Basel Tier-1 min); single-counterparty cap ≤25% of Tier-1
(≤15% between G-SIBs); interbank-asset share ~20% of total assets; LGD ~50–60% (worst case 100%).
"Severe"/systemic threshold ≈ ≥5% of institutions failing; observed contagious losses ~16–20% of
system assets → a realistic right-tail target band.

## 4. Data provenance: the hard truth

**Real bilateral interbank exposure matrices are almost universally confidential supervisory data**
(Bundesbank, OeNB, BoE, Banxico, Banco Central do Brasil, DNB, NBB, …) — not obtainable. The rare
public exceptions: the US Fed emergency-loan dataset (Bloomberg/Fed, 407 institutions, 2007–10),
EU sovereign cross-holdings (BIS Quarterly Review Table 9B, used by Elliott–Golub–Jackson), and e-MID
(Italian interbank, *commercial*, not free).

The field's standard workaround when only aggregates/marginals are public is **maximum-entropy /
gravity-model reconstruction** of the bilateral matrix (Upper 2011; Cimini 2015) — conceptually
*identical* to our model inferring `J_ij` couplings. This is both a methodological justification and a
narrative gift: we are doing the accepted thing, and the quantum generator is the natural next step.

## 5. Free / public data shortlist (Phase-2 sourcing targets)

Ranked by value to us. Full details + verified URLs in §04 (and §02 for credit).

| Priority | Source | What it gives us | Access |
|---|---|---|---|
| ⭐⭐ | **FDIC BankFind Failures API** | Real bank-failure events 1934→present (name, date, assets, deposits, est. loss, resolution); 2008–10 clustering = ground-truth correlated-cascade tail | Free, keyless CSV/JSON |
| ⭐ | **NYU V-Lab SRISK** | Firm-level SRISK%, LRMES, leverage (weekly) → ready-made distress-propensity `p_i` | Free (HTML scrape; no bulk API) |
| ⭐ | **FRED API** | Stress indices (OFRFSI, NFCI, STLFSI4, KCFSI), VIX, Treasury yields, credit spreads (BAA-AAA, ICE BofA HY OAS `BAMLH0A0HYM2`) → macro state + market-implied PD proxies | Free (API key) |
| ⭐ | **FFIEC FR Y-15 + FR Y-9C** | GSIB interconnectedness/size/complexity line items (intra-financial assets/liabilities) → balance-sheet inputs for `J_ij` | Free bulk (caret-delimited TXT) |
| | **FDIC Financials / FFIEC Call Reports** | Quarterly per-bank balance sheets, capital ratios → bottom-up `p_i` (distance-to-default) | Free REST CSV / bulk |
| | **BIS consolidated banking statistics** + Quarterly Review Table 9B | Cross-border claims by country/sector; EU sovereign cross-holdings → sector-level exposure edges | Free SDMX API / PDF |
| | **Moody's "Corporate Default & Recovery Rates 1920–2004"** | PD-by-rating tables (free PDF mirror) | Free PDF |
| | **NUS-CRI** | Public-good PDs for 90k+ firms | Free |
| | **ECB Data Portal** | CISS co-stress index, MFI balance-sheet stats | Free SDMX |

Tooling (not data): **ConIII** (inverse-Ising solver, github.com/eltrompetero/coniii), **NEMtropy**
(config-model network reconstruction), **AssetCorr** (R, asset-correlation estimators),
**Qiskit Finance** (credit-risk GCI + IQAE, qGAN loader tutorials).

## 6. Honest quantum claims (§05)

| Claim | Status | Key citation |
|---|---|---|
| Entangled QCBM/qGAN **loading** of the correlated default distribution | **Well-supported in principle on small synthetic problems** (3–4 qubits, hardware); no proof of efficient loading at scale, no provable classical-sampling hardness for our circuit | Zoufal–Lucchi–Woerner 2019; Benedetti 2019; Liu–Wang 2018 |
| **QAE** quadratic tail-risk speedup (P(severe), CVaR) | **Well-supported asymptotically; not at advantage scale** (advantage needs ~thousands of logical qubits, fault-tolerant) | Brassard 2002; Woerner–Egger 2019; Egger et al. 2021; IQAE Grinko 2021 |
| **Grover / Dürr–Høyer** worst-case scenario search | **Real in the oracle model; weakest claim — keep "byproduct, do not oversell"** | Dürr–Høyer 1996 |

State-preparation can erase QAE's quadratic advantage "if not handled carefully" — stated almost
verbatim by **Egger et al. (2021)**, the closest analogue (QAE for correlated-default credit risk,
12 qubits simulated). This is *why* the entangled loader is load-bearing rather than decorative.
Notably, **no prior work loads a real high-dimensional correlated default distribution** — that gap is
our angle.
