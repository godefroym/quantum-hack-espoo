# Quantum Systemic Stress Scenario Discovery

A quantum-assisted stress-testing MVP. An entanglement-structured generator proposes correlated financial default scenarios; a deterministic classical simulator evaluates the resulting contagion cascades. The quantum advantage has two surfaces: **generating** the correlated scenario distribution and **calculating** the tail risk over it.

## Problem

Banks, insurers, funds, corporates, sovereigns, and market utilities are linked by directed financial exposures. Stress testing asks which plausible initial default scenarios trigger severe downstream cascades. Hand-designed scenarios are sparse, independent sampling misses correlated shocks, and copulas are strong baselines but may not reach the same crisis tail — and the cascades that matter most are the deep-tail events you have almost no historical data on.

## Approach

A generator-agnostic benchmark that keeps the comparison honest — same synthetic network, same marginal default probabilities, same pairwise dependency targets, same deterministic cascade simulator, **different scenario generators**:

- Independent Bernoulli baseline
- Gaussian copula baseline
- Student-t copula baseline
- Entangled Born-machine generator (exact NumPy/Qiskit statevector locally, IBM Runtime hardware path)

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

**Implemented vs. designed.** The classical baselines, cascade simulator, comparison harness, and
small-block Born-machine circuit are implemented. The circuit can be emulated exactly or submitted
to IBM hardware. The reversible cascade oracle and QAE remain future work.

## What This Does Not Claim

- A single cascade has no quantum speedup — the advantage is over the scenario distribution (estimation) and the search space (discovery).
- The cascade is not quantum-simulated except as an oracle; reverse-stress-test optimization (QAOA) is heuristic; no quantum-linear-algebra (HHL) advantage is claimed.

## Project Structure

```text
src/systemic_risk/
  spec.py                 # flat SystemSpec validation and JSON/NPZ IO (the B/C/D contract)
  data/                   # deterministic synthetic network generation
  data_network/           # PART A: real data -> canonical layered NetworkSpec -> SystemSpec
    sources/              #   roster (real anchor), equity_returns (Yahoo), synthetic (scaling)
    clean / estimate      #   normalize/reconcile; marginals, correlation, balance-sheet totals
    reconstruct           #   bilateral exposures: max_entropy (RAS) | min_density (pluggable)
    cluster / assemble    #   community detection + stability; layer assembly
    validate              #   round-trip + cluster-stability + B/C/D contract conformance
  generators/             # Bernoulli, copula, and entangled generators
    quantum/layout.py     # dependency clustering, sparse entanglers, circuit layers
  simulator/              # deterministic cascade, exogenous shocks, LGD, round diagnostics
  evaluation/             # metrics and comparison harness
  visualization/          # graph, cascade, dependency, and entanglement-layout plots
  utils/
scripts/
  run_mvp.py
  run_scaling_experiment.py
  build_system_spec.py    # PART A end-to-end: build + validate + render the real network
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

Expected outputs in `outputs/`: `network.png`, `comparison.csv`, one crisis card per generator, and
`real_system.json`.

Run tests:

```bash
uv run pytest
```

## The real exposure network (Part A)

The benchmark above runs on a calibrated *synthetic* network. Part A builds the **real
anchor**: one real dataset → a frozen, canonical spec → a legible network plot, consumed by
B/C/D without loss.

```bash
uv run python scripts/build_system_spec.py                    # build + validate + render
uv run python scripts/build_system_spec.py --method min_density   # sparse reconstruction
uv run python scripts/build_system_spec.py --refresh-equity   # re-fetch the correlation
```

Outputs land in `outputs/data_network/`: `network_spec.json` (the layered `NetworkSpec`),
`system_spec.json` / `.npz` (the flat `SystemSpec` for B/C/D), and `community_network.png`.

**Pipeline** (`src/systemic_risk/data_network/`):

```text
roster (28 real banks)  ─┐
equity returns (Yahoo) ──┼─► estimate ─► reconstruct ─► cluster ─► assemble ─► validate
Moody's PD table       ──┘   p_i, corr,   bilateral W   communities  NetworkSpec  round-trip,
                             totals       (max-entropy   (+ stability)  ──► flat   stability,
                                           | min-density)                SystemSpec  B/C/D
```

- **Nodes** — a curated roster of 28 real, publicly listed G-SIB / large banks
  (`data/external/banks/gsib_roster.csv`).
- **Marginals `p_i`** — each bank's public S&P rating → 1-year PD via the committed Moody's
  Exhibit-17 table.
- **Correlation** — real **daily equity-return** correlation (755 obs, 2021–2024;
  `data/external/banks/equity_corr.csv`). This is the genuine network signal: it drives
  community detection and is the latent asset-return correlation the copula baselines
  threshold into correlated defaults.
- **Edges** — bilateral exposures are **reconstructed** from per-node interbank totals
  (real bilateral matrices are confidential — the field-standard move), pluggable between
  `max_entropy` (RAS/IPF, dense) and `min_density` (Anand-style, sparse).
- **Communities** — greedy-modularity detection on the correlation graph; the committed
  snapshot yields three stable communities — **North America**, **Europe/UK/LatAm**, **Japan**
  (mean ARI ≈ 0.96 under perturbation).

**Canonical layered spec.** `NetworkSpec` separates the *empirical* layer (frozen ground
truth: marginals, correlation, balance-sheet totals) from the *reconstructed* layer
(swappable bilateral edges + method tag), with a documented `FeatureSchema` (field meanings +
per-consumer visibility) and `Provenance` (source, fit params, content hash). It round-trips
losslessly (`to_json`/`from_json`) and assembles down into the flat `SystemSpec` via
`to_system_spec()`. `view_for("generator" | "simulator" | "visualization")` returns only the
fields each consumer is allowed to see.

**Check everything** (the A end-to-end test — load raw → emit a valid spec → round-trip →
stable clusters → B/C/D conformance):

```bash
uv run pytest tests/test_data_network.py -q
```

The flat spec records whether `target_pairwise_corr` is a latent Gaussian
correlation or a binary default-event correlation. This keeps the Gaussian
copula baseline and the entangled generator on the same marginal and pairwise
default targets.

The shared cascade keeps the original binary-scenario API and also supports
named scenarios, direct exogenous losses, scalar or edge-level LGD, explicit
failure rounds, cumulative-loss diagnostics, and convergence reporting.

### Dependency clustering and entanglement layout

Shay's clustering layer combines binary-default correlation and symmetrised
exposure strength, detects deterministic threshold-connected clusters, selects
sparse intra-cluster entanglers, and schedules them into collision-free circuit
layers:

```bash
uv run python examples/run_clustering_layout.py
```

The example writes `outputs/entanglement_layout.png` and
`outputs/dependency_matrix.png`. The layout can also drive the Born machine
directly with `EntangledBornMachineGenerator(layout_strategy="clustered")`;
the original strongest-edge strategy remains the default for reproducibility.

The earlier standalone `contagion/` prototype is not a second runtime package:
its simulator, metrics, and plots were integrated into
`systemic_risk.simulator`, `systemic_risk.evaluation`, and
`systemic_risk.visualization`. New contagion-related work should extend those
canonical modules rather than recreate the old package.

### Optional classical fire-sale extension

The primary benchmark uses the fixed-point exposure cascade. A supplementary
Huang et al. bank-asset engine evaluates the same binary initial-default
scenarios under common-asset liquidation and price-impact feedback:

```bash
uv run python scripts/run_huang_2008_demo.py
uv run python scripts/compare_generators_huang.py
```

See [`docs/huang_simulation.md`](docs/huang_simulation.md) for the equations,
adapter, scope, and differences from the primary cascade engine. The comparison
script writes calibration, cascade-distribution, and tail plots under
`outputs/huang_generator_comparison/`.

### Optional extras

The app and quantum layers are opt-in extras, kept out of the base install:

```bash
# Streamlit + Plotly dashboard
uv run --extra app streamlit run app/streamlit_app.py

# Install Qiskit plus IBM Runtime support
uv sync --extra quantum

# Dry-run the explicit IBM hardware smoke test (does not submit a job)
uv run --extra quantum python scripts/run_ibm_quantum_test.py

# Install every extra into .venv
uv sync --all-extras
```

### IBM Quantum hardware

IBM Quantum exposes hardware through Qiskit Runtime. Create an IBM Quantum Platform account and API
key, then save it interactively without exposing it in shell history:

```bash
uv run --extra quantum python scripts/configure_ibm_quantum.py
```

Alternatively, provide:

```bash
export IBM_QUANTUM_TOKEN="..."
# Optional for accounts with an explicit service instance:
export IBM_QUANTUM_INSTANCE="..."
```

Submit the four-qubit smoke test only after reviewing the dry-run:

```bash
uv run --extra quantum python scripts/run_ibm_quantum_test.py --submit

# Or select a backend explicitly:
uv run --extra quantum python scripts/run_ibm_quantum_test.py \
  --backend <backend-name> --qubits 8 --max-degree 2 --shots 4096 --submit
```

The script transpiles the fitted `RY + CRY` circuit to the backend ISA, executes it with
`SamplerV2`, and writes samples plus hardware-versus-ideal errors to `outputs/ibm_quantum/`.
Submitting may consume plan allocation and wait in the provider queue. Credentials are never read
from repository files.
