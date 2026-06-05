# Quantum Systemic Stress Scenario Discovery

A quantum-assisted stress-testing MVP that uses an entanglement-structured parameterized circuit interface to generate correlated financial default scenarios, then evaluates systemic contagion with a classical cascade simulator.

## Problem

Banks, insurers, funds, corporates, sovereigns, and market utilities are linked by directed financial exposures. Stress testing asks which plausible initial default scenarios can trigger severe downstream cascades. Hand-designed scenarios are useful but sparse; independent sampling misses correlated shocks; basic copulas are strong baselines but may not explore the same crisis tail.

## Approach

The MVP keeps the comparison honest:

- Same synthetic financial network.
- Same marginal default probabilities.
- Same pairwise dependency targets.
- Same deterministic classical contagion simulator.
- Different scenario generators.

Implemented generators:

- Independent Bernoulli baseline.
- Gaussian copula baseline.
- Student-t copula baseline.
- `EntangledPQCGenerator`, a quantum-native scenario generator interface with a runnable Born-inspired fallback sampler for environments without Qiskit.

Headline comparison:

> Under matched marginals and pairwise dependencies, which generator samples the most severe plausible contagion tails?

## What Is Quantum-Native

Each qubit represents one financial entity:

- `|0>` means the entity survives the initial shock.
- `|1>` means the entity initially defaults.

Single-qubit `Ry` rotations encode individual default tendencies. Entangling interactions follow the exposure/dependency graph, so linked institutions are sampled from a non-factorized joint distribution. The generated samples are binary default scenarios, not final cascade states.

## What This Does Not Claim

- The quantum circuit does not simulate the full financial cascade.
- Entanglement is not treated as literal financial contagion.
- No quantum advantage is claimed.
- The goal is to benchmark whether a quantum-native generator changes the tail of sampled systemic crises under the same evaluator.

## Project Structure

```text
src/systemic_risk/
  spec.py                 # SystemSpec validation and JSON/NPZ IO
  data/                   # deterministic synthetic network generation
  generators/             # Bernoulli, copula, and PQC-style generators
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

```bash
python -m pip install -r requirements.txt
python scripts/run_mvp.py
```

Expected outputs in `outputs/`:

- `network.png`
- `comparison.csv`
- one crisis card per generator
- `synthetic_system.json`

Run tests:

```bash
pytest
```

Optional Streamlit app:

```bash
python -m pip install streamlit
streamlit run app/streamlit_app.py
```

## Scientific Framing

The quantum role is scenario generation, not direct cascade simulation. A generator proposes correlated initial defaults. The classical simulator then evaluates the true fixed-point cascade over the same exposure matrix and capital buffers for every generator. This makes the MVP generator-agnostic and keeps the benchmark focused on crisis-tail sampling rather than on changing the contagion engine.
