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

This project is managed with [uv](https://docs.astral.sh/uv/) ([install it](https://docs.astral.sh/uv/getting-started/installation/)), which provisions the pinned interpreter, virtual environment, and locked dependencies for you. `uv run` executes inside that environment — nothing to activate.

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
| `run_scaling_experiment.py` | Size/scale study + the n=54 mean-field oracle table |
| `run_qae_tail_risk.py` | Calculation surface — QAE of `P(severe)`/CVaR vs Monte Carlo (equivalence + oracle-query advantage) |
| `run_huang_2008_demo.py`, `compare_generators_huang.py` | Optional fire-sale contagion channel (see below) |

All entry points write to `outputs/` (e.g. `run_mvp.py` → `network.png`, `comparison.csv`, `real_system.json`, one crisis card per generator).

## The real exposure network (Part A)

The benchmark runs on a **real anchor**: one real dataset → a frozen, canonical spec → a legible network plot, consumed by the generators / cascade / harness without loss.

```bash
uv run python scripts/build_system_spec.py                      # build + validate + render
uv run python scripts/build_system_spec.py --method min_density # sparse reconstruction
uv run python scripts/build_system_spec.py --refresh-equity     # re-fetch the correlation
```

Outputs land in `outputs/data_network/`: `network_spec.json` (the layered `NetworkSpec`), `system_spec.json` / `.npz` (the flat `SystemSpec`), and `community_network.png`.

- **Nodes** — a curated roster of 28 real, publicly listed G-SIB / large banks (`data/external/banks/gsib_roster.csv`).
- **Marginals `p_i`** — each bank's public S&P rating → 1-year PD via the committed Moody's Exhibit-17 table.
- **Correlation** — real daily equity-return correlation (755 obs, 2021–2024; `data/external/banks/equity_corr.csv`). This is the genuine network signal: it drives community detection and is the latent asset-return correlation the copula baselines threshold into correlated defaults.
- **Edges** — bilateral exposures are **reconstructed** from per-node interbank totals (real bilateral matrices are confidential — the field-standard move), pluggable between `max_entropy` (RAS/IPF, dense) and `min_density` (Anand-style, sparse).
- **Communities** — greedy-modularity detection on the correlation graph; the committed snapshot yields three stable communities (North America, Europe/UK/LatAm, Japan; mean ARI ≈ 0.96 under perturbation).

The pipeline lives in `src/systemic_risk/data_network/`; see that package's README for the layered `NetworkSpec` (frozen empirical layer + swappable reconstructed layer + provenance) and `data/external/CATALOG.md` for full per-dataset provenance and licensing. End-to-end checks (round-trip, cluster stability, downstream contract):

```bash
uv run pytest tests/test_data_network.py -q
```

## Dependency clustering and entanglement layout

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
  --backend <backend-name> --qubits 8 --max-degree 2 --shots 4096 --submit
```

The script transpiles the fitted `RY + CRY` circuit to the backend ISA, executes it with
`SamplerV2`, and writes samples plus hardware-versus-ideal errors to `outputs/ibm_quantum/`.
Submitting may consume plan allocation and wait in the provider queue. Credentials are never read
from repository files.
