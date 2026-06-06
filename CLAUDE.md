# Quantum Systemic Stress Scenario Discovery

Quantum-native systemic stress testing. An entangled Born-machine generator proposes correlated default scenarios; a deterministic cascade simulator evaluates contagion. Quantum advantage on two surfaces:

- **Generation** — an entangled QCBM samples the correlated default distribution and serves as the QAE state-loader (one qubit per institution; angles set analytically from marginals and the exposure graph).
- **Calculation** — the cascade becomes a reversible oracle; QAE estimates tail risk (`P(severe)`, CVaR) with a quadratic, deep-tail-amplified speedup; amplitude amplification finds the worst plausible scenario.

**Direction:** classical baselines, cascade, and comparison harness are implemented; building the QCBM loader and QAE oracle, and scaling generation to a **54-qubit quantum computer (access incoming)**.

## Run

```bash
uv sync                              # core + dev deps into .venv
uv run python scripts/run_mvp.py     # benchmark -> outputs/
uv run pytest                        # tests
```

Extras:

```bash
uv run --extra app streamlit run app/streamlit_app.py   # dashboard
uv run --extra quantum python scripts/run_mvp.py        # Qiskit backend
uv sync --all-extras
```

## Layout

`src/systemic_risk/`: `spec.py` (SystemSpec), `data/` (synthetic network), `generators/` (Bernoulli, copula, entangled), `simulator/` (cascade), `evaluation/` (metrics + harness), `visualization/`. Entry points in `scripts/`; tests in `tests/`.
