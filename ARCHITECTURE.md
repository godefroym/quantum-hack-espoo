# Architecture: Quantum Systemic Stress Scenario Discovery

End-to-end pipeline for quantum-native systemic stress testing. One generator proposes
correlated default scenarios; one deterministic simulator evaluates contagion; one harness
compares quantum against classical on identical inputs. Quantum advantage lives on two
surfaces — **generation** (loading the correlated distribution) and **calculation** (estimating
tail risk over it).

```
DATA  →  GENERATION  →  CONTAGION  →  EVALUATION
roster/        Born machine     exposure cascade    metrics (P_severe, CVaR, tail)
equity corr    + classical      + Huang fire-sale   QAE tail-risk estimator
ratings→PD     baselines        (shared contract)   classical-vs-quantum harness
```

The chain is `SystemSpec → ScenarioGenerator.sample() → ContagionChannel.simulate() → compute_metrics()`.
Every stage is deterministic and seeded.

---

## 1. Data gathering

Two sources of `SystemSpec` (the canonical system object: `node_names`, `node_types`,
`exposure_matrix` `W[i,j]`=loss to i if j defaults, `capital_buffers`, `marginal_default_probs`,
`target_pairwise_corr`, `clusters`, `metadata`). Defined in `src/systemic_risk/spec.py`.

### Real 28-bank network (`src/systemic_risk/data_network/`)
Pipeline: `roster → clean → estimate → reconstruct → cluster → assemble → validate`.

- **Roster** — `data/external/banks/gsib_roster.csv`: 28 G-SIBs + 10 corporates (ticker, country,
  S&P rating, FY2023 total assets). Curated from public filings.
- **Correlation** — the *only* genuinely-real network signal: daily equity log-returns,
  Yahoo Finance, 2021-06-01 → 2024-06-01 (556 trading days), correlated into an n×n matrix
  (`equity_corr.csv`). Used as the latent-Gaussian correlation for generators and as the graph
  for community detection.
- **Marginals (PD)** — S&P ratings → Moody's 1-year default rates (Exhibit 17,
  `moodys_pd_by_rating.csv`). Through-the-cycle baseline mean PD ≈ **0.23%**.
- **Exposures** — *not observed*. Reconstructed from interbank totals (20% of assets) via
  max-entropy (RAS/IPF) or min-density, capped at 25% of buffer per counterparty (Basel
  large-exposure rule), then risk-adjusted (LGD, maturity/rollover, wrong-way, substitutability)
  into an effective-loss matrix (`edge_metrics.py`).

> **Explicit limitation (load-bearing):** the real 28-bank net is *not hardware-loadable as-is*.
> It has ~365 strong entangler pairs (~2600 two-qubit gates → full decoherence), and its baseline
> PDs (~0.23%) sit **below the QPU readout noise floor (~2.7%)** — on hardware they are pure noise.
> Two mitigations are applied, and both are stated, not hidden: (a) **sparsify** to the strongest
> ~40 edges (drops 89% of strong pairs); (b) apply a **2008-style stress transform**
> (`data_network/stress.py`): a monotone logit-shift lifting mean PD to ~15% (GFC-realized range,
> anchored to the 3.85× BAA–AAA spread blowout) while preserving the correlation graph and risk
> ordering.

### Synthetic networks (`src/systemic_risk/data/`)
For scaling to the 54-qubit hardware target without a 54-node roster, and for controlled tests.
- `make_scalable_system(n≤54)` — scale-free / core-periphery topology (density-corrected gravity
  model, Cimini 2015), per-rating PDs, Basel-consistent balance sheets.
- `make_huang_2008_style_system` — bank×asset balance sheets over 13 real asset classes (Moody's
  portfolio weights) for the fire-sale channel.
- `make_clustered_system` — planted-cluster two-factor Vasicek instances with known ground truth.

---

## 2. Data generation (scenario generators)

Every generator implements `fit(spec) → sample(n, seed) → (n_samples, n)` binary matrix.
Row = one scenario; column i = institution i; 1 = defaults. **Fairness invariant: all generators
are calibrated to the same marginals + pairwise joints** — so any difference in tail behavior comes
from correlation structure, not from cheating on the first two moments.
(`src/systemic_risk/generators/`)

### Classical baselines (the comparison set)
| Generator | Model | Captures |
|---|---|---|
| **Bernoulli** | independent `p_i` | marginals only, zero correlation |
| **Gaussian copula** | latent Gaussian, thresholds `Φ⁻¹(p_i)`, ρ root-solved to joints | marginals + pairwise corr |
| **Student-t copula** | as Gaussian, t(df=4) latent | + heavier tail dependence |
| **Ising/Boltzmann** | `P(x) ∝ exp(Σhᵢxᵢ + ΣJᵢⱼxᵢxⱼ)`, fields by Boltzmann learning, couplings from exposure | + higher-order structure |

### Quantum generator — Entangled Born machine (QCBM)
`EntangledBornMachineGenerator` (`quantum_born_machine.py`). **One qubit per institution**,
`|1⟩`=default. Output bitstring = one scenario. `P(x) = |⟨x|U(θ)|0⟩|²`.

**Angles are set analytically — no gradient training:**
- Marginal: `θᵢ = 2·arcsin(√pᵢ)` ⟹ `P(qubitᵢ=1) = pᵢ` exactly.
- Entanglement: controlled-RY per dependency edge, angle from closed-form covariance inversion
  `α = 2·arcsin(√(pⱼ + covᵢⱼ/(pᵢ(1−pᵢ)))) − θⱼ`. Correlations are carried by **amplitude mixing
  only** (RY+CRY); Z-diagonal phase gates would be inert under Z-basis measurement.
- Optional light Newton calibration (≤30 steps) to absorb interference where edges share qubits.

**Scaling without 2ⁿ blowup:** the entangler graph is partitioned (Kernighan-Lin) into blocks
≤22 qubits; cross-block edges stay classical. For homogeneous systems two closed-form loaders
(`SymmetricIsingLoader`, `GHZBlend`) give the exact loss-count law at **any n** via an (n+1)-term
sum — no statevector at all. A separately-trained **qGAN** variant exists (trained in sim,
validated on hardware) per the project memory.

---

## 3. Processing (contagion simulators)

Both channels emit the shared `CascadeOutcome` protocol (`final_defaults`, `failure_count`,
`rounds_to_convergence`, `systemic_collapse`, …) so evaluation treats them identically.
(`src/systemic_risk/simulator/`)

### Exposure cascade (`cascade.py`) — default
Deterministic fixed-point contagion. From initial defaults, each round only the *newly-failed*
frontier transmits loss: `cumulative_losses += (W·LGD) @ frontier`; a node fails when cumulative
loss crosses its capital buffer. Iterates to convergence (no new failures / all failed). No
randomness.

### Huang fire-sale (`huang.py`) — alternative channel
Bank×asset contagion via price impact, not direct exposure. Failed banks liquidate holdings →
asset prices drop by `α · market_share` → surviving banks' asset values fall below distress
barriers → more failures. Optional stochastic distress barrier (η).

---

## 4. Classical comparison (the harness)

`EvaluationHarness` (`evaluation/harness.py`): fit each generator → sample (default 200k) →
push through *one* contagion channel → `compute_metrics`. Same channel for all generators, so
metric deltas isolate the generator's correlation modeling.

**Metrics** (`evaluation/metrics.py`): `marginal_rmse`, `pairwise_joint_rmse`,
`excess_coskewness_rms` (vs Gaussian-copula reference), `aggregate_tail_dependence`,
`p_severe_cascade` = `P(failures ≥ ⌈0.5n⌉)`, `tail_mean_{1,5,10}pct`,
`cascade_count_cvar_{95,99}` = E[failures | worst 5%/1%].

**Validated results** (real community n≈14, 200k samples; `outputs/demonstration_comparison.csv`):
- All generators match marginals/joints to RMSE ~1e-4 (fairness holds).
- **Higher-order separation:** excess co-skewness — entangled **0.480** vs Gaussian 0.253 (1.9×);
  the Gaussian foil's excess co-skewness *decays with N*, the entangled's persists.
- **Tail movement:** `p_severe` entangled 3.55e-4 vs Gaussian 1.9e-4 (1.9×).
- **n=54 exactness** (`outputs/scaling_oracle_validation.csv`): TV distance between the entangled
  loss-count law and the exact mean-field oracle = **0.0 at machine precision** for n ∈ {8,16,24,32,54},
  with no 2⁵⁴ statevector on either side.

---

## 5. Quantum advantage — calculation surface (tail risk via QAE)

`QAETailRiskEstimator` (`evaluation/qae_tail_risk.py`): the QCBM loads `P(x)`; the cascade becomes
a reversible oracle marking severe scenarios; **Quantum Amplitude Estimation** (MLAE) estimates
the tail amplitude `a = P(cascade ≥ threshold)` and CVaR.

**Where the advantage is — query complexity, not wall clock:**
- Monte Carlo to relative error ε: `N_MC = (1−a)/(ε²a)` = **O(1/(ε²·a))**.
- QAE: `N_QAE ≈ π/(2ε√(a(1−a)))` = **O(1/(ε·√a))**.
- Quadratic in ε, and the `1/a` → `1/√a` improvement makes the gap **widen in the deep tail** —
  exactly where systemic risk lives.

**Measured** (`outputs/qae_query_advantage.csv`, ε=10%): speedup 10× at a=0.2 → **201× at a=0.001**.
MLAE on the real oracle (`qae_measured_advantage.csv`): 5.4× at ε=10% → 26× at ε=1%.

> **Explicit honesty clause (in the code's own docstring):** the estimator is an **exact
> statevector simulation** — enumerating 2ⁿ scenarios is exponential, so it runs only at small n
> (validated at n=12). **No wall-clock speedup is claimed.** What is claimed and verified: (1) the
> QAE estimate matches exact and Monte-Carlo values within 3σ Fisher (`qae_equivalence.csv` — every
> P_severe and CVaR estimate lands in-CI); (2) the reported **oracle-query count** is the
> hardware-relevant cost and beats MC quadratically. The advantage is a *query-complexity* result,
> realized on hardware, not a simulator timing.

---

## 6. Hardware status (IBM, `ibm_boston`)

Run on real QPUs and recorded under `outputs/ibm_quantum/` and
`outputs/real_cluster_mixture_stress_hw/`:
- 4q / 20q / 30q synthetic loaders; real 28-bank sparsified loader (40 kept / 338 dropped pairs,
  marginal RMSE 0.033, kept-edge joint RMSE 0.009); qGAN trained-circuit validation.
- **48-entity 2008-stress run** (28 banks + 10 corporates), partitioned into 4 blocks
  [15,13,10,10] qubits, 200k shots each, mean PD lifted 0.26% → 15%. Cross-cluster correlation
  recovered (β≈0.905) and cascade reconciled: `P(severe)` **9.15% → 25.5%**, mean cascade
  **8.92 → 18.38** vs the Gaussian reference.

**What's done:** both generation (QCBM loader, hardware-validated) and calculation (QAE of
P_severe/CVaR over the QCBM-loaded oracle, exact-sim validated) exist end-to-end.
**What remains:** running the *calculation* surface on a 54-qubit machine.

---

## File map
| Concern | Path |
|---|---|
| System object | `src/systemic_risk/spec.py` |
| Real network pipeline | `src/systemic_risk/data_network/` |
| Synthetic / Huang data | `src/systemic_risk/data/` |
| Generators (classical + QCBM) | `src/systemic_risk/generators/` |
| Quantum circuits / loaders | `src/systemic_risk/generators/quantum/` |
| Contagion simulators | `src/systemic_risk/simulator/` |
| Metrics + harness + QAE | `src/systemic_risk/evaluation/` |
| Canonical run / smoke / hardware | `scripts/run_demonstration.py`, `run_mvp.py`, `run_ibm_quantum_*.py` |
| Results | `outputs/` |
