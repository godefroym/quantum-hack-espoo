# Quantum Systemic Stress Scenario Discovery

A quantum-assisted stress-testing MVP. An entanglement-structured generator proposes correlated financial default scenarios; a deterministic classical simulator evaluates the resulting contagion cascades. The quantum advantage has two surfaces: **generating** the correlated scenario distribution and **calculating** the tail risk over it.

## Problem

Banks, insurers, funds, corporates, sovereigns, and market utilities are linked by directed financial exposures. Stress testing asks which plausible initial default scenarios trigger severe downstream cascades. Hand-designed scenarios are sparse, independent sampling misses correlated shocks, and copulas are strong baselines but may not reach the same crisis tail — and the cascades that matter most are the deep-tail events you have almost no historical data on.

## Approach

A generator-agnostic benchmark that keeps the comparison honest — same network (a real 28-bank G-SIB exposure network, see *The real exposure network* below), same marginal default probabilities, same pairwise dependency targets, same deterministic cascade simulator, **different scenario generators**:

- Independent Bernoulli baseline
- Gaussian copula baseline
- Student-t copula baseline
- Entangled generator (quantum circuit Born machine, run on an exact statevector backend by default; Qiskit optional)

> Headline: under matched marginals and pairwise dependencies, which generator reaches the most severe plausible contagion tails?

## Where The Quantum Advantage Is

Each qubit is one entity (`|0>` survives, `|1>` initially defaults). `Ry` rotations encode individual default tendencies; entangling gates on the exposure/dependency graph make linked institutions sample from a non-factorized joint distribution.

**Generation.** A quantum circuit Born machine samples `x` with probability `|<x|U(θ)|0>|²`. Its entangling layers encode a correlated, classically-hard-to-sample default distribution, and the *same* circuit doubles as the state-loader for the step below — so entanglement is load-bearing, not decorative.

**Calculation.** The cascade becomes a reversible oracle `|x>|0> -> |x>|severity(cascade(x)) >= s>`. Evaluated over the loaded distribution in superposition, it unlocks:

- **Quantum Amplitude Estimation** of tail risk — `P(severe cascade)` and CVaR of cascade size — with a quadratic speedup that *grows in the deep tail*: estimating a rare probability `a` to relative error `ε` costs about `O(1/(ε·√a))` queries versus `O(1/(ε²·a))` for classical Monte Carlo.
- **Amplitude amplification** to surface rare severe scenarios in about `O(1/√a)` iterations (versus `O(1/a)` classical draws) — the basis for quantum worst-case scenario search.

```text
A = U_QCBM (load P(x)) -> U_severity (cascade oracle) -> mark severe
  => QAE : quantify P(severe), CVaR        (quadratic, deep-tail amplified)
  => AA  : discover the worst plausible scenario
```

**Implemented vs. designed.** The classical baselines, the cascade simulator, the entangled Born-machine generator, and the comparison harness are implemented today — and so is the QAE *calculation* surface, as an **exact classical statevector simulation**: amplitude estimation reads `P(severe)` / CVaR off the QCBM-loaded cascade oracle (`uv run python scripts/run_qae_tail_risk.py`). It is verified to agree with the classical Monte-Carlo answer (equivalence) and to reach a target accuracy in quadratically fewer *oracle queries* in the deep tail (advantage). What remains is running it on hardware — the simulation is exponential in `n`, but the construction (one qubit per institution + cascade-comparison ancillas, the QCBM as loader) is hardware-ready. No wall-clock speedup is claimed; the advantage is in oracle-query count.

## What This Does Not Claim

- A single cascade has no quantum speedup — the advantage is over the scenario distribution (estimation) and the search space (discovery).
- The cascade is not quantum-simulated except as an oracle; reverse-stress-test optimization (QAOA) is heuristic; no quantum-linear-algebra (HHL) advantage is claimed.

## Run

```text
src/systemic_risk/
  spec.py                 # flat SystemSpec validation and JSON/NPZ IO (the B/C/D contract)
  edge_metrics.py         # directed risk-adjusted edge weights (LGD, maturity, wrong-way, …)
  data/                   # deterministic synthetic network generation
  data_network/           # PART A: real data -> canonical layered NetworkSpec -> SystemSpec
    sources/              #   roster (real anchor), equity_returns (Yahoo), synthetic (scaling)
    clean / estimate      #   normalize/reconcile; marginals, correlation, balance-sheet totals
    reconstruct           #   bilateral exposures: max_entropy (RAS) | min_density (pluggable)
    cluster / assemble    #   community detection + stability; layer assembly
    validate              #   round-trip + cluster-stability + B/C/D contract conformance
  generators/             # Bernoulli, copula, and entangled generators
  simulator/              # deterministic cascade, exogenous shocks, LGD, round diagnostics
  evaluation/             # metrics and comparison harness
  visualization/          # graph plots (incl. community plot) and crisis cards
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

This project is managed with [uv](https://docs.astral.sh/uv/) ([install it](https://docs.astral.sh/uv/getting-started/installation/)). `uv` provisions the pinned Python interpreter (`.python-version`), creates the virtual environment, and installs the locked dependencies (`uv.lock`) for you. `uv run` executes inside that environment.

```bash
uv sync                                      # core + dev dependencies
uv run python scripts/run_demonstration.py   # CANONICAL end-to-end run (per-criterion verdict)
uv run python scripts/run_mvp.py             # fast smoke test / dashboard feed
uv run pytest                                # tests
```

**Entry points.** One canonical run, one smoke test, the rest specialized:

| Script | Role |
|---|---|
| `run_demonstration.py` | **Canonical** — full per-criterion verdict on the real community + the n=54 scale story |
| `run_mvp.py` | **Smoke test** — fast cascade comparison on the real network; feeds the dashboard |
| `build_system_spec.py` | Part A — build + validate + render the real exposure network |
| `compare_real_institutions_quantum.py` | 28-bank or 38-institution Gaussian vs ideal/IBM QCBM comparison |
| `run_scaling_experiment.py` | Size/scale study + the n=54 mean-field oracle table |
| `run_qae_tail_risk.py` | Calculation surface — QAE of `P(severe)`/CVaR vs Monte Carlo (equivalence + oracle-query advantage) |
| `run_huang_2008_demo.py`, `compare_generators_huang.py` | Optional fire-sale contagion channel (see below) |

All entry points write to `outputs/` (e.g. `run_mvp.py` → `network.png`, `comparison.csv`, `real_system.json`, one crisis card per generator).

The real-institution comparison has one implementation with two scopes. The 38-qubit scope
contains the same 28 banks plus 10 corporates; it is not a separate or concatenated model.

```bash
# 28 banks, noiseless MPS reference
uv run --extra quantum python scripts/compare_real_institutions_quantum.py --scope banks

# 38 institutions, noiseless MPS reference
uv run --extra quantum python scripts/compare_real_institutions_quantum.py --scope all

# Experimental backend-aware graph: add native/routed relations up to depth 50
uv run --extra quantum python scripts/compare_real_institutions_quantum.py \
  --scope all --entanglement-layout topology --backend <backend-name> --max-depth 50

# Explicit IBM submission; large requests are split at the backend shot limit
uv run --extra quantum python scripts/compare_real_institutions_quantum.py \
  --scope all --shots 1000000 --backend <backend-name> --submit
```

The default `chain` layout is the low-noise hardware baseline. The experimental `topology`
layout maps institutions onto a compact native backend subgraph, then greedily adds important
non-native dependencies while enforcing `--max-depth`. It can encode more pairwise relations,
but the additional routing and two-qubit gates may cost more fidelity than they add expressivity.

## The real exposure network (Part A)

The benchmark runs on a **real anchor**: one real dataset → a frozen, canonical spec → a legible network plot, consumed by the generators / cascade / harness without loss.

```bash
uv run python scripts/build_system_spec.py                      # build + validate + render
uv run python scripts/build_system_spec.py --method min_density # sparse reconstruction
uv run python scripts/build_system_spec.py --refresh-equity     # re-fetch the correlation
```

Outputs land in `outputs/data_network/`: `network_spec.json` (the layered `NetworkSpec`), `system_spec.json` / `.npz` (the flat `SystemSpec`), and `community_network.png`.

```text
roster (28 banks + 10 corporates) ─┐
equity returns (Yahoo) ────────────┼─► estimate ─► reconstruct ─► risk-adjust ─► cluster ─► assemble ─► validate
Moody's PD table       ────────────┘   p_i, corr,   bilateral W   effective W   communities  NetworkSpec  round-trip,
                                       totals       (max-entropy   (LGD/maturity (+ stability)  ──► flat   stability,
                                                     | min-density) /wrong-way…)                 SystemSpec  B/C/D
```

- **Nodes** — a curated roster of **38** real, publicly listed entities: 28 G-SIB / large
  banks **and 10 non-financial corporates** (Apple, ExxonMobil, Toyota, Volkswagen, Boeing,
  Petrobras, …) (`data/external/banks/gsib_roster.csv`, with a `node_type` column).
- **Marginals `p_i`** — each entity's public S&P rating → 1-year PD via the committed Moody's
  Exhibit-17 table.
- **Correlation** — real **daily equity-return** correlation (755 obs, 2021–2024;
  `data/external/banks/equity_corr.csv`). This is the genuine network signal: it drives
  community detection and is the latent asset-return correlation the copula baselines
  threshold into correlated defaults.
- **Edges** — bilateral exposures are **reconstructed** from per-node totals (real bilateral
  matrices are confidential — the field-standard move), pluggable between `max_entropy`
  (RAS/IPF, dense) and `min_density` (Anand-style, sparse). Corporates **borrow** from banks
  but do not lend interbank, so the graph has directed **bank → corporate** edges (a bank
  loses if a corporate it financed defaults).
- **Edge weights are risk-adjusted** (`src/systemic_risk/edge_metrics.py`): each *directed*
  notional is scaled into an **effective loss** by loss-given-default (recovery / seniority /
  collateralization), maturity / rollover stress, wrong-way risk (correlation-conditional),
  and concentration / substitutability. The cascade propagates this effective matrix; the raw
  notional is kept for audit. "Mutual exposure" is no longer flattened — `A`'s loss if `B`
  fails ≠ the reverse.
- **Communities** — greedy-modularity detection on the correlation graph; the committed
  snapshot yields five stable communities along **region × sector** lines — North-American
  banks, Europe/UK, Japan, an energy/LatAm cluster (Petrobras, ExxonMobil, Brazilian banks),
  and a US-tech cluster (Apple, Microsoft, …) — mean ARI ≈ 0.85 under perturbation.

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

## Dependency clustering and entanglement layout

The contagion simulator stays classical, deterministic, and generator-agnostic.
This clustering layer does a different job: it decides which institutions are
"close enough" to deserve entanglement structure in the quantum generator and
layout.

The rule of thumb is simple:

- strong positive correlation or strong mutual exposure -> same cluster -> candidate entanglement edge
- weak dependency, cross-cluster, or distant institutions -> keep classical -> no expensive entanglement needed

In practice, the layout combines binary-default correlation and symmetrised
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

The point is to use entanglement only where systemic-risk dependencies are
dense and meaningful, while keeping weak or distant relationships classical so
the quantum layout stays sparse, interpretable, and financially motivated.

The earlier standalone `contagion/` prototype is not a second runtime package:
its simulator, metrics, and plots were integrated into
`systemic_risk.simulator`, `systemic_risk.evaluation`, and
`systemic_risk.visualization`. New contagion-related work should extend those
canonical modules rather than recreate the old package.

## Optional fire-sale extension

The primary benchmark uses the fixed-point exposure cascade. A supplementary Huang et al. bank-asset engine evaluates the same binary initial-default scenarios under common-asset liquidation and price-impact feedback. See [`docs/huang_simulation.md`](docs/huang_simulation.md) for the equations, adapter, and scope.

```bash
uv run python scripts/run_huang_2008_demo.py        # single fire-sale cascade
uv run python scripts/compare_generators_huang.py   # generator comparison under the fire-sale channel
```

## Optional extras

```bash
uv run --extra app streamlit run app/streamlit_app.py   # Streamlit dashboard
uv run --extra quantum python scripts/run_mvp.py        # real Qiskit backend (else exact statevector)
uv sync --all-extras                                    # install every extra
```

## Project structure

```text
src/systemic_risk/
  spec.py                 # flat SystemSpec validation and JSON/NPZ IO (the consumer contract)
  data/                   # synthetic network generation + Huang bank-asset adapter
  data_network/           # Part A: real data -> canonical layered NetworkSpec -> SystemSpec
  generators/             # Bernoulli, copula, Ising/Boltzmann, and entangled Born-machine generators
  simulator/              # deterministic exposure cascade + Huang fire-sale engine
  evaluation/             # metrics and comparison harness
  visualization/          # graph plots (incl. community plot), cascade plot, crisis cards
scripts/                  # entry points (see table above) + shared _demo/ helpers
tests/  app/  notebooks/
research/                 # literature synthesis grounding the data and generator design
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
  --backend <backend-name> --qubits 8 --max-degree 2 --shots 100000 --submit
```

The script transpiles the fitted `RY + CRY` circuit to the backend ISA, executes it with
`SamplerV2`, and writes samples plus hardware-versus-ideal errors to `outputs/ibm_quantum/`.
Submitting may consume plan allocation and wait in the provider queue. Credentials are never read
from repository files.
