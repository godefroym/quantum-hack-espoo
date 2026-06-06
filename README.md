# Quantum Systemic Stress Scenario Discovery

A quantum-assisted stress-testing MVP. An entanglement-structured generator proposes correlated financial default scenarios; a deterministic classical simulator evaluates the resulting contagion cascades. The quantum advantage has two surfaces: **generating** the correlated scenario distribution and **calculating** the tail risk over it.

## Problem

Banks, insurers, funds, corporates, sovereigns, and market utilities are linked by directed financial exposures. Stress testing asks which plausible initial default scenarios trigger severe downstream cascades. Hand-designed scenarios are sparse, independent sampling misses correlated shocks, and copulas are strong baselines but may not reach the same crisis tail — and the cascades that matter most are the deep-tail events you have almost no historical data on.

## Approach

A generator-agnostic benchmark that keeps the comparison honest — same synthetic network, same marginal default probabilities, same pairwise dependency targets, same deterministic cascade simulator, **different scenario generators**:

- Independent Bernoulli baseline
- Gaussian copula baseline
- Student-t copula baseline
- Entangled generator (quantum circuit Born machine target; classical Ising/Boltzmann sampler as the current stand-in)

> Headline: under matched marginals and pairwise dependencies, which generator reaches the most severe plausible contagion tails?

## Where The Quantum Advantage Is

Each qubit is one entity (`|0>` survives, `|1>` initially defaults). `Ry` rotations encode individual default tendencies; entangling gates on the exposure/dependency graph make linked institutions sample from a non-factorized joint distribution.

**Generation.** A quantum circuit Born machine samples `x` with probability `|<x|U(θ)|0>|²`. Its entangling layers encode a correlated, classically-hard-to-sample default distribution, and the *same* circuit doubles as the state-loader for the step below — so entanglement is load-bearing, not decorative.

**Calculation.** The cascade becomes a reversible oracle `|x>|0> -> |x>|severity(cascade(x)) >= s>`. Evaluated over the loaded distribution in superposition, it unlocks:

- **Quantum Amplitude Estimation** of tail risk — `P(severe cascade)` and CVaR of cascade size — with a quadratic speedup that *grows in the deep tail*: estimating a rare probability `a` to relative error `ε` costs about `O(1/(ε·√a))` queries versus `O(1/(ε²·a))` for classical Monte Carlo.
- **Amplitude amplification** to surface rare severe scenarios in about `O(1/√a)` iterations (versus `O(1/a)` classical draws) — the basis for quantum worst-case scenario search.

Unified pipeline:

```text
A = U_QCBM (load P(x)) -> U_severity (cascade oracle) -> mark severe
  => QAE : quantify P(severe), CVaR        (quadratic, deep-tail amplified)
  => AA  : discover the worst plausible scenario
```

**Implemented vs. designed.** The classical baselines, the cascade simulator, and the comparison harness are implemented today. The Born-machine loader and the QAE cascade oracle are the quantum layer this MVP is structured around.

## What This Does Not Claim

- A single cascade has no quantum speedup — the advantage is over the scenario distribution (estimation) and the search space (discovery).
- The cascade is not quantum-simulated except as an oracle; reverse-stress-test optimization (QAOA) is heuristic; no quantum-linear-algebra (HHL) advantage is claimed.

## Project Structure

```text
src/systemic_risk/
  spec.py                 # SystemSpec validation and JSON/NPZ IO
  data/                   # deterministic synthetic network generation
  generators/             # Bernoulli, copula, and entangled generators
  simulator/              # deterministic fixed-point cascade engine
  evaluation/             # metrics and comparison harness
  visualization/          # graph plots and crisis cards
  utils/
scripts/
  run_mvp.py
  run_scaling_experiment.py
tests/
app/
notebooks/
```

## Run The MVP

This project is managed with [uv](https://docs.astral.sh/uv/) ([install it](https://docs.astral.sh/uv/getting-started/installation/)). `uv` provisions the pinned Python interpreter (`.python-version`), creates the virtual environment, and installs the locked dependencies (`uv.lock`) for you.

```bash
uv sync                              # core + dev dependencies into .venv
uv run python scripts/run_mvp.py     # run the benchmark
```

`uv run` executes inside the managed environment, so there is nothing to activate.

Expected outputs in `outputs/`: `network.png`, `comparison.csv`, one crisis card per generator, and `synthetic_system.json`.

Run tests:

```bash
uv run pytest
```

### Optional extras

The app and quantum layers are opt-in extras, kept out of the base install:

```bash
# Streamlit + Plotly dashboard
uv run --extra app streamlit run app/streamlit_app.py

# Real Qiskit backend for the entangled generator (otherwise a classical fallback sampler is used)
uv run --extra quantum python scripts/run_mvp.py

# Install every extra into .venv
uv sync --all-extras
```
