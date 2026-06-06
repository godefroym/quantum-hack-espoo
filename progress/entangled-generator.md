# Progress Log — Entangled Generator

**Date:** 2026-06-06
**Branch:** `alex/entanglement-generator`
**Scope:** The project's core IP — the entangled, quantum-native scenario generator — built on
top of the classical data foundation (see `data-foundation.md`).
**Status of this document:** rewritten as a single coherent log after an independent, adversarial
audit that reproduced every headline number and ran a confound-free re-test of the central claim.
The audit's job was to *break* the prior agent's "all three criteria PASS" self-report; what
survived and what did not is recorded honestly below.

> **⚠️ Known weakness (documented this stage, deliberately not patched).** The headline
> demonstration (`scripts/run_demonstration.py`) still scores **Criterion 3 ("material to risk")**
> against an *under-correlated* Gaussian foil (foil realizes mean corr ≈ 0.147 vs the entangled
> generator's ≈ 0.427), which inflates the cascade-tail effect to ≈ 7.6×. The confound-free re-test
> (`scripts/_audit_matched_foil.py`), which calibrates the foil to the entangled generator's **own
> realized marginals and correlation**, shows the true matched-second-order effect is only **≈ 1.4×
> on deep co-default mass and a tie on count-CVaR_99.9**. Criterion 3 is therefore **PARTIAL, not
> PASS**: the *direction* is real (consistent with Criterion 2), but the demonstrated *magnitude* is
> largely a second-order correlation-mismatch artifact. We chose to record this openly rather than
> re-engineer the demo in this stage — the fix (restate the demo's Criterion-3 verdict against a
> correlation-matched foil) is tracked as **Next steps #1**. Criteria 1 and 2 are unaffected and
> stand. See the full **Criterion 3** section and **Honest caveats** below.

## The problem this stage solves

Build an entanglement-structured generator of correlated default scenarios that is an **honest
drop-in replacement** for the classical baseline: it reproduces the same marginals and pairwise
correlations, so that any difference in downstream results comes from one place only — the
**higher-order joint structure** that entanglement can carry and second-order classical models
cannot.

What counts as solving it (the three success criteria, unchanged from the plan):

1. **Honest comparison** — its first- and second-order statistics match the *strongest* classical
   generator within tolerance, so the two are genuinely interchangeable at that level. (Beating a
   weak baseline proves nothing.)
2. **Genuine higher-order structure** — it carries joint-tail dependence that the best classical
   model cannot reproduce even when calibrated to those same marginals and correlations.
3. **Material to risk** — that structure must measurably move the systemic-risk outcome (the
   contagion tail), not merely show up as a static distributional difference. A richer joint that
   leaves the tail unchanged is not an advantage.

---

## What was built this stage

The whole pipeline is wired end-to-end on the **real-data foundation**, with one runnable
demonstration that renders an explicit, numeric per-criterion verdict:

```
real 28-bank G-SIB network            (data_network.build_system_spec)
   -> community detection + plot, and the verified block-separability limit (Part 1)
real largest community (n = 14)        (the criterion spec, a single fully-simulated block)
   -> all generators -> deterministic cascade -> 2nd-order + higher-order + tail metrics
   -> per-criterion verdict (the headline real-data test, Part 2)
n = 54 synthetic + mean-field oracle   (the scale story, Part 3)
```

Components touched/added:

- **`data_network` → generator linkage.** `build_system_spec()` (real 28-bank G-SIB offline
  snapshot) and `build_synthetic_system_spec(n)` feed a `SystemSpec` straight into every generator.
- **`EntangledBornMachineGenerator`** (`src/systemic_risk/generators/quantum_born_machine.py`) —
  computational-basis Born machine, one qubit per institution. `RY` marginals + amplitude-mixing
  controlled-`RY` entanglers on dependency-graph edges; per-edge angles from closed-form
  single-pair covariance inversion, then a light Newton calibration against the *exact statevector*
  moments. Block-separable by community once `n > max_block_qubits` (default 22). An exchangeable
  (`SymmetricIsingLoader`) path gives a closed-form loss-count law at any `n`.
- **Higher-order / tail tooling** (`src/systemic_risk/evaluation/joint_structure.py`) —
  `excess_coskewness` (sampled connected third cumulant minus the closed-form **Gaussian-copula
  reference at the sample's own marginals+correlation**) and tail-dependence statistics, plus
  `cascade_count_cvar`. Surfaced through `compute_metrics`.
- **Mean-field oracle** (`src/systemic_risk/models/mean_field_oracle.py`) — closed-form homogeneous
  Ising loss-count law, the exact ground truth for the n = 54 validation.
- **Demonstration + scaling scripts** (`scripts/run_demonstration.py`,
  `scripts/run_scaling_experiment.py`, `scripts/run_mvp.py`; shared logic in `scripts/_demo/`).

---

## Verified per-criterion verdict (audit numbers)

All numbers below were **reproduced by the auditor**, not copied from the prior run. The headline
demonstration reproduces within sampling noise; the n = 54 oracle reproduces exactly.

### Criterion 1 — Honest comparison → **PASS (with an honest split, fairly disclosed)**

The demonstration is careful and the characterization is fair. Two regimes:

- **Heterogeneous real community (n = 14).** The entangled generator does **not** reach the
  achievable (Fréchet) correlation ceiling; it *ties* the strongest classical model. Reproduced:
  corr-RMSE-to-ceiling **entangled 0.245 vs Student-t 0.242** (Gaussian 0.475, Bernoulli 0.621);
  entangled marginal-RMSE 3.5e-4. So "ties the strongest classical, far beats the Gaussian" is
  accurate; "clean drop-in" would not be, and the demo does not claim it here.
- **Exchangeable target at the same tiny credit marginal (the clean case).** Reproduced: at
  marginal ≈ 0.0022, equicorrelation target 0.5, the entangled generator lands at corr **0.508
  (RMSE 0.011)** while **every classical model misses**: Gaussian copula collapses to 0.071,
  Student-t reaches only 0.267, and — a detail worth recording — the Ising/Boltzmann model with
  its auto coupling-scale **overshoots to comonotone (corr 1.000)**, jumping the first-order
  transition. So at real-credit marginals the entangled generator is the *only* generator that
  reproduces a target equicorrelation; this is a genuine, clean drop-in where the target is
  exchangeable.

Verdict: PASS is justified. The honest caveat (already in the demo) stands: on a fully
heterogeneous real correlation matrix the single-control CRY ansatz cannot satisfy all conflicting
per-edge correlations at once, so it does not reach the ceiling there — it only ties the best
classical. The clean drop-in is real but is demonstrated on the exchangeable spec, not the
heterogeneous one.

### Criterion 2 — Genuine higher-order structure → **PASS (robust)**

This is the strongest of the three claims and it survives every adversarial check.

- **The discriminator is a genuine matched-moment statistic.** Confirmed directly: a *pure
  Gaussian-copula* sample, with its excess scored against a Gaussian reference built from that
  sample's **own** empirical marginals+correlation, gives raw co-skewness ≈ 1.94 but **excess**
  co-skewness that decays toward zero with N (0.40 at N = 1e5 → 0.13 at N = 1e6). So the statistic
  subtracts off the second-order content and is not re-measuring correlation.
- **The separation is not finite-sample noise.** On a spec where the Gaussian copula and the
  entangled generator are *provably* matched at second order (the homogeneous credit spec), the
  Gaussian's excess co-skewness decays as ≈ 1/√N (0.178 → 0.085 → 0.044 → 0.024 over N =
  5e4…3.2e6) while the entangled generator's **plateaus at ≈ 3.1**. The structure is real and
  stable; the foil's signal is vanishing noise.
- **It holds even against a foil matched to the entangled generator's OWN realized correlation**
  (see the confound test below): excess co-skewness **≈ 1.9 (entangled) vs ≈ 0.23 (latent-matched
  Gaussian), ~8× and stable**, with the Gaussian decaying with N (0.58 → 0.10) and the entangled
  flat. Reproduced across three seeds.

Verdict: PASS. The entangled generator carries beyond-second-order joint structure that a
moment-matched Gaussian copula structurally cannot.

### Criterion 3 — Material to risk → **PARTIAL (directionally real, magnitude largely confounded)**

This is where the audit changes the story. The demonstration reports the entangled generator
moving the cascade tail by large factors **vs the Gaussian-copula foil** (`p_severe` 0.0014 vs
0.0002 ≈ 7.6×; CVaR_99.9 11.2 vs 4.8). Those numbers reproduce. **But the comparison is
confounded:** in that same demo table the Gaussian foil realizes mean correlation **0.147** while
the entangled generator realizes **0.427** (both are printed in `demonstration_comparison.csv`).
The foil is not just missing higher-order structure — it is far *less correlated*, because a third
of the real targets are Fréchet-infeasible and the Gaussian copula additionally under-realizes
correlation through threshold discretization at these tiny marginals. A heavier cascade tail is the
expected consequence of higher *second-order* correlation alone, so the 7.6× cannot be attributed
to higher-order structure as demonstrated.

**The confound-free re-test (`scripts/_audit_matched_foil.py`).** The auditor estimated the
entangled generator's **own** realized marginals `p*` and realized pairwise correlation matrix
`C*` from a 400k sample (`C*` is Fréchet-feasible at `p*`, 0% violations, with headroom), then
built classical foils calibrated to *those*: a **latent-matched Gaussian copula** (latent
correlation solved so the sampled Pearson correlation lands on `C*`) and an **Ising/Boltzmann**
model (global coupling scale bisected so its sampled mean correlation hits `C*`'s mean). Equal-N
(200k) samples were run through the **same** real cascade. Results (representative; stable over
seeds):

| metric (matched 2nd order) | Entangled | Gaussian (matched) | Ising (matched) |
|---|---|---|---|
| realized corr mean | 0.435 | **0.440** | 0.412 |
| corr-RMSE to `C*` | 0.028 | **0.029** | 0.167 |
| excess co-skewness (rms) | 1.92 | 0.23 | 2.69 |
| deep `p(K ≥ half)` | 0.00157 | 0.00107 | 0.00102 |
| cascade tail-mean (1%) | 2.62 | 2.41 | 2.39 |
| cascade-count CVaR_99.9 | 11.57 | **11.59** | 14.0 |

What this shows, plainly:

- **Against the genuinely matched Gaussian copula, the cascade-tail gap largely collapses.** The
  deep `p(K ≥ half)` ratio falls from the demo's ~7.6× to **~1.4×** (stable: 1.38–1.44× over
  seeds), the 1% tail-mean ratio to ~1.09×, and **CVaR_99.9 essentially ties** (11.57 vs 11.59).
  So the headline magnitude in criterion 3 was mostly a second-order (correlation-mismatch)
  artifact. A *residual* real effect remains: at matched correlation the entangled generator still
  puts ~40% more probability mass on whole-community co-default — consistent with criterion 2, but
  far smaller than advertised, and not visible in the deeper count-CVaR.
- **The Ising/Boltzmann foil is not a valid "second-order" foil and underlines the point.** It is
  itself a non-elliptical, higher-order model, so it *also* carries large excess co-skewness (2.69)
  and a *heavier* deep tail and count-CVaR than the entangled generator. It tied or beat the
  entangled generator on tail metrics — i.e. "moves the cascade tail" is not unique to entanglement
  among classical generators once you allow a non-elliptical one. Criterion 3's claim is properly
  read as *"vs the best moment-matched **elliptical** model,"* which is the Gaussian copula.

Verdict: downgrade from PASS to **PARTIAL**. The direction is real and consistent with criterion 2
(entanglement does put extra mass on systemic co-default), but the demonstration's magnitude is
inflated by an uncontrolled correlation mismatch in its chosen foil, and on the deepest tail
statistic (count-CVaR_99.9) the matched Gaussian foil ties. Even within the demo's own table the
better-correlated classical foil (Student-t, realized corr 0.380) already shrinks the `p_severe`
gap to ~2× and the CVaR_99.9 gap to 9.8 vs 11.2 — corroborating the confound from a second angle.

---

## Honest caveats and limitations

- **Block-separability above `max_block_qubits` (default 22).** On the full 28-bank network the
  heterogeneous ansatz splits into its 3 communities: within-cluster correlation is captured
  (generated 0.416 vs target 0.645) but **cross-cluster correlation collapses to ≈ 0** (generated
  0.004 vs target 0.465). The whole-network spec is therefore *not* a valid drop-in spec; the demo
  shows this rather than hiding it and tests the criteria on a single community.
- **Fréchet infeasibility of real targets.** ~32% of the real off-diagonal target correlations
  exceed the Fréchet bound for Bernoulli marginals at these tiny default probabilities — unreachable
  by *any* binary generator. Match is (correctly) scored against the achievable ceiling, not the
  nominal target.
- **Where criterion 1 is clean vs only a tie.** Clean drop-in: the **exchangeable** target at
  real-credit marginal. Tie-with-the-best-classical only: the **heterogeneous** real community.
- **Criterion 3 magnitude is foil-dependent.** The large headline factors require the under-
  correlated nominal-target Gaussian foil. At genuinely matched correlation the move is modest
  (~1.4× on deep co-default mass, tie on count-CVaR_99.9). This is the most important honesty point
  in this log.
- **The cascade module is correct but the tail is shallow at these marginals.** `p_severe` /
  `p(K ≥ half)` events occur at ~1e-3, so even N = 200k gives only ~hundreds of tail events;
  fixed-α CVaR_95/99 can place its VaR at a zero count (which is why CVaR_99.9 and `p(K ≥ half)`
  are the faithful deep-tail statistics). Numbers below ~1e-4 should be read as order-of-magnitude.
- **No code bug found.** The metrics, cascade, oracle, and generator code paths the demo exercises
  are internally consistent; the auditor did not modify `src/`. The issue with criterion 3 is one
  of *experimental design / framing* (the foil's correlation was not matched), not a defect in the
  statistics or the simulator. The realized correlations are disclosed in the artifact CSV, but the
  criterion-3 verdict text presents the 7.6× without flagging that the foil is far less correlated —
  that framing should be corrected to the matched-foil numbers.

---

## n = 54 oracle / scaling evidence (reproduced exactly)

- The heterogeneous n = 54 synthetic fit runs as **48 community blocks of ≤ 7 qubits** — it never
  forms the 2^54 statevector.
- In the exactly-solvable homogeneous limit the entangled permutation-symmetric loader reproduces
  the closed-form mean-field Ising loss-count law **to machine precision**:
  **TV(entangled, oracle) = 0.00e+00** at n = 8, 16, 24, 32, **and 54**, with marginal error 0 and
  default-correlation error 0 at every size (`scripts/run_scaling_experiment.py`). This is the
  evidence that the *generation* extrapolates to the hardware target — for the exchangeable
  (homogeneous) family. It does **not** establish exact fidelity for a fully heterogeneous 54-qubit
  target (which is block-separable, hence the cross-cluster caveat above).

---

## How to reproduce

```bash
uv sync                                       # core + dev deps
uv run python scripts/run_demonstration.py    # headline per-criterion verdict + artifacts (~15s)
uv run python scripts/run_scaling_experiment.py  # size scaling + n=54 oracle table (TV ≈ 0)
uv run python scripts/_audit_matched_foil.py  # the confound-free criterion-3 re-test
uv run pytest -q                              # 63 passed
uv run --extra quantum pytest -q              # 63 passed (Qiskit backend)
```

Artifacts land in `outputs/` (`demonstration_comparison.csv` — note the `corr_mean_gen` column,
`demonstration_verdict.txt`, `demonstration_crisis_card.md`, `real_network_communities.png`,
`scaling_*.csv`).

---

## Bottom line (auditor's assessment)

- **Criterion 1 (honest comparison): supported.** Fairly characterized, with the clean-vs-tie split
  disclosed. Reproduced.
- **Criterion 2 (higher-order structure): fully supported.** The excess-co-skewness discriminator is
  a genuine matched-moment statistic; the separation persists under N-scaling and against a foil
  matched to the entangled generator's own realized correlation. This is the project's real,
  defensible quantum-distinguishable result.
- **Criterion 3 (material to risk): partially supported.** Real in direction (entanglement adds
  systemic co-default mass) but the demonstrated magnitude is largely a correlation-mismatch
  confound; at matched second order the cascade-tail move shrinks to ~1.4× on deep co-default mass
  and ties on count-CVaR_99.9. The demo's headline criterion-3 numbers should be restated against a
  correlation-matched foil.

Net: the central scientific claim — *entanglement carries higher-order joint structure a
moment-matched classical model cannot* — **stands**. The *risk-materiality* claim stands only in a
weaker, honestly-bounded form and needs the matched-foil framing to be defensible.

---

## Next steps

1. **Fix the criterion-3 framing in the demo** to use a correlation-matched foil (the latent-matched
   Gaussian copula from `scripts/_audit_matched_foil.py`), and report the matched-foil tail numbers
   alongside (or instead of) the nominal-target ones. Promote `_audit_matched_foil.py` into the
   demonstration once stable.
2. **QAE oracle / amplitude estimation.** Turn the cascade into a reversible oracle and use quantum
   amplitude estimation for `P(severe)` / CVaR with the quadratic, deep-tail-amplified speedup; add
   amplitude amplification for worst-case scenario search. (The state-loader half — the entangled
   QCBM — is what this stage delivered.)
3. **Heterogeneous fidelity at scale.** The exact-at-54 result is for the homogeneous family. Either
   extend the ansatz so cross-cluster correlation survives above `max_block_qubits`, or state
   plainly that hardware scaling is exact only for exchangeable targets and approximate (block-
   separable) for heterogeneous ones.
4. **Hardware.** Run the entangled loader on the incoming 54-qubit device and compare sampled
   loss-count statistics to the oracle.
