# Progress Log — Data Foundation (Phases 1–2)

**Date:** 2026-06-06
**Scope:** Gathering/producing data for the quantum systemic-stress project — academic literature,
sourcing free real datasets, and generating synthetic scenarios via the Ising/Boltzmann method of
`scenario_generation.md`.

**One-line status:** Literature ✅ done · Data sourcing ✅ done · Synthetic generation 🟡 code complete
but **the n=54 sampler is broken** (validation failed) and the dataset/script/test deliverables (#11)
are not yet written.

| Phase | Status | Notes |
|---|---|---|
| 1 — Literature review | ✅ Done | 5 surveys + synthesis in `research/` |
| 2a — Source free real data | ✅ Done | Files in `data/external/` |
| 2b — Synthetic Boltzmann generator | 🟡 Code complete, **NOT working at n=54** | Bug isolated to MCMC sampler + coupling-scale calibration |
| 2b — Datasets/script/tests/README (#11) | ⬜ Not started | No `data/synthetic/` outputs yet |

---

## Phase 1 — Literature review ✅

Five parallel surveys, then synthesized. Read first; it drove everything below.

- `research/README.md` — executive synthesis (precedent, generator recipe, calibration numbers,
  free-data shortlist, honest quantum-claims table).
- `research/sections/01_network_contagion.md` … `05_quantum_finance.md` — full annotated surveys.

**Conclusions that shaped Phase 2:**
1. Real bilateral interbank exposures are **confidential** everywhere; the field's accepted workaround
   (max-entropy / gravity reconstruction from public marginals) **is** our `J_ij` inference — so the
   synthetic generator is the standard move, not a fallback.
2. Our Π(x) has a direct precedent: **Filiz–Guo–Morton–Sturmfels (2012)**; and the homogeneous
   mean-field Ising credit model (**Molins–Vives 2005**) gives a **closed-form loss distribution at
   any n incl. 54** → our validation oracle.
3. Concrete calibration: `h_i = logit(p_i)` refit by Boltzmann learning; `J_ij` from gravity-reconstructed
   exposures; exact enumeration ≤20, MCMC + parallel tempering at 54.

---

## Phase 2a — Free real data sourced ✅

All under `data/external/` with `CATALOG.md`, `fetch.sh` (reproducible), `README.md`.

| Dataset | File | Rows | Calibrates |
|---|---|---|---|
| FDIC bank failures 2000–2024 | `fdic/failures.csv` | 587 | real default **events** |
| FDIC bank failures 1980–2024 | `fdic/failures_1980_2024.csv` | 3,548 | **events** — captures both S&L (1988–92) and GFC (2009–11) clusters |
| Moody's 1-yr PD by rating | `ratings/moodys_pd_by_rating.csv` | 30 | **`p_i`** (primary) — wired into the generator |
| FRED credit spreads | `fred/{AAA,BAA,BAA10Y}.csv` | — | macro-state / market-implied PD proxy |
| FRED stress indices | `fred/{STLFSI4,KCFSI,NFCI,ANFCI,VIXCLS}.csv` | — | macro-state (cover 2008 + COVID) |
| ECB systemic stress | `ecb/ciss_euro_area.csv` | 12,119 | macro-state (EU) |

**Caveats / documented-only (verified URLs in `CATALOG.md`, not fetched):**
- `fred/BAMLH0A0HYM2.csv` (HY OAS) — keyless endpoint serves **2023+ only** (FRED licensing clip);
  full 1996+ needs a free FRED API key. BAA–AAA spread covers 2008/2020 in the meantime.
- **NYU V-Lab SRISK** (`p_i`) — client-rendered; needs headless scrape. Stub at `vlab/srisk.csv`.
- **FFIEC FR Y-15 / FR Y-9C** (`J_ij`) — HTTP 403 bot challenge; field→`J_ij` mapping documented.
- **BIS consolidated banking statistics** (`J_ij`) — 85 MB bulk verified reachable, not committed.
- **OFRFSI** — FRED id 404s on keyless endpoint; use OFR portal or API key.

**Mapping:** `p_i` ← Moody's PD (+ V-Lab/FR Y-9C documented) · `J_ij` ← FR Y-15 + BIS (documented;
real bilateral data is confidential → gravity reconstruction) · **events** ← FDIC · **macro-state** ← FRED + ECB.

---

## Phase 2b — Synthetic Boltzmann generator 🟡

### Implemented (code complete, imports clean, existing suite green: 8 passed)

- `src/systemic_risk/models/ising.py` — `IsingModel` (energy/`log_weight`, exact enumeration + `Z` +
  moments for n≤20, Gibbs MCMC, parallel tempering), `LossDistribution` (tail prob, CVaR). `MAX_EXACT_N=20`.
- `src/systemic_risk/models/calibration.py` — `logit_fields`, `fit_fields_boltzmann`,
  `couplings_from_correlation` (naive-MF/TAP inverse-Ising), `couplings_from_exposure` (gravity),
  `couplings_from_spec`, `mean_field_marginals`, etc.
- `src/systemic_risk/models/mean_field_oracle.py` — `MeanFieldIsingOracle` (closed-form homogeneous
  loss distribution, exact at any n; `from_targets` solver), `total_variation_distance`.
- `src/systemic_risk/generators/ising_boltzmann.py` — `IsingBoltzmannGenerator` (`ScenarioGenerator`):
  couplings from spec, fields refit by Boltzmann learning, auto coupling-scale via mean-field linear
  response, size-keyed sampler. Registered in `generators/__init__.py`.
- `src/systemic_risk/data/synthetic.py` — **`make_scalable_system(n≤54)`**: programmatic institution
  mix, scale-free / core-periphery gravity topology (γ≈2–3), per-rating PDs (**loads the real Moody's
  CSV** if present, else literature defaults), 4–8% Tier-1 buffers, ≤25%-of-Tier-1 single-counterparty
  cap, ~20% interbank-asset share. (`make_synthetic_system`, capped at 20, left intact.)

### Validation — ran the pipeline end-to-end (the step the agent never did)

| Check | Result | Verdict |
|---|---|---|
| **A.** Oracle vs exact enumeration, homogeneous n=10 | TV = 4.3×10⁻¹⁶ | ✅ PASS — energy fn + oracle correct |
| **B.** Generator @ n=12 (exact sampler), marginal recovery | MAE = 3.0×10⁻³ (target 0.0083) | ✅ PASS — small-n path works |
| **C.** n=54 parallel-tempering vs oracle, *known-good* homogeneous (h,J) | TV = **0.675**; marginal **0.375 vs 0.040** | ❌ FAIL — sampler broken |
| **D.** Full generator @ n=54 (realistic system) | emp_agg **0.967 vs 0.012**; coupling_scale **57.7** | ❌ FAIL — whole system always collapses |

### Root-cause (isolated)

- The **MCMC sampler (Gibbs / parallel tempering, n>20) is wrong**: check C feeds it the *same*
  `IsingModel(fields, couplings)` construction that check A validates to machine precision via exact
  enumeration, yet sampling overshoots massively (0.375 vs 0.040 marginal). So the physics/energy
  function is correct and the **bug lives in the sampler** — likely parallel-tempering replica handling
  (collecting from a hot replica instead of the β=1 chain) or the Gibbs conditional. *To investigate:
  `IsingModel._gibbs_sweep` / `_parallel_tempering` / `sample` in `models/ising.py`.*
- Secondary: **`IsingBoltzmannGenerator` coupling-scale auto-calibration blows up at n=54**
  (scale 57.7 vs 5.4 at n=12) — the mean-field linear-response bisection in
  `_resolve_coupling_scale` runs to its cap. Even with a fixed sampler this would over-couple.

### Not done (#11)

No `data/synthetic/` datasets · no `scripts/generate_synthetic_data.py` · no `tests/test_ising.py` ·
no `data/synthetic/README.md`. Blocked on the n=54 fix (datasets at scale would be garbage today).

---

## Repo changes

```
new:       research/  data/  src/systemic_risk/models/  src/systemic_risk/generators/ising_boltzmann.py
modified:  src/systemic_risk/data/synthetic.py  data/__init__.py  generators/__init__.py
tests:     8 passed (existing) — no new tests yet
```

## Next steps (priority order)

1. **Fix the MCMC sampler** (`models/ising.py` Gibbs/parallel-tempering) → re-run check C until
   TV(MCMC@54, oracle) ≲ 0.03 and marginal ≈ target.
2. **Fix coupling-scale calibration** at n=54 in `IsingBoltzmannGenerator._resolve_coupling_scale`.
3. **Finish #11**: generate datasets for n ∈ {12, 20, 30, 54} (`spec.json`, `samples.npz`,
   `diagnostics.json`), `scripts/generate_synthetic_data.py`, `tests/test_ising.py`,
   `data/synthetic/README.md`; re-run A–D as the test suite.
4. **Later:** integrate documented-only data (V-Lab `p_i`, FR Y-15/BIS `J_ij`); wire the generator
   into `evaluation/harness.py` alongside the Bernoulli/copula baselines; QCBM loader → QAE.

## Phase 3 — Part A: real data & exposure network ✅ (2026-06-06)

Built the **data-and-network** layer: one real dataset → a frozen canonical spec → a legible
community plot, consumed by B/C/D without loss. New package `src/systemic_risk/data_network/`.

**Real dataset (the anchor).** `data/external/banks/gsib_roster.csv` — 28 real, publicly
listed G-SIB / large banks (public S&P ratings + FY2023 total assets, per-row provenance).
`data/external/banks/equity_corr.csv` — the **real 28×28 daily equity-return correlation
matrix** (755 obs, 2021-06-01→2024-06-01) fetched from Yahoo Finance and committed for
reproducibility (refresh: `scripts/build_system_spec.py --refresh-equity`).

**Pipeline.** `sources/{roster,equity_returns,synthetic}` → `clean` (normalize/reconcile,
rating→whole-letter) → `estimate` (marginals from Moody's Exhibit-17 PD table; correlation
from equity returns; interbank asset/liability totals + Tier-1 buffers from total assets) →
`reconstruct` (bilateral exposures, pluggable `max_entropy` RAS / `min_density` Anand-style;
real bilateral data is confidential) → `cluster` (greedy modularity + perturbation-ARI
stability) → `assemble` → `validate`.

**Canonical spec.** `data_network/spec.py`: frozen `EmpiricalLayer` (ground truth) +
`ReconstructedLayer` (swappable edges + method tag) + `FeatureSchema` (field meanings +
per-consumer visibility) + `Provenance` (source, fit params, SHA-256 content hash), wrapped
in `NetworkSpec` with `to_json`/`from_json` (lossless), `view_for(consumer)`, and
`to_system_spec()` → the **existing flat `SystemSpec`** that B/C/D already consume (chosen to
blend in without breaking the consumer contract).

**End-to-end result** (`scripts/build_system_spec.py`):

| Check | Result |
|---|---|
| Round-trip lossless (`NetworkSpec` + flat `SystemSpec`) | ✅ |
| Communities | **3 stable** — N. America / Europe-UK-LatAm / Japan; mean ARI **0.955** |
| B/C/D contract (copula fits, cascade runs, views enforce visibility) | ✅ |
| Tests | `tests/test_data_network.py` — 19 new; **full suite 27 passed** |

Artifacts in `outputs/data_network/`: `network_spec.json`, `system_spec.json`/`.npz`,
`community_network.png`. Docs updated: `data/external/CATALOG.md`, `data/external/README.md`,
repo-root `README.md` (§ *The real exposure network*).

## Reproduce the validation

```bash
uv run pytest -q                       # existing suite (8 passed)
# end-to-end A–D smoke test: see the "Validation" section — fit IsingBoltzmannGenerator on
# make_scalable_system(n) and compare against MeanFieldIsingOracle.
```
