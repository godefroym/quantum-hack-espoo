# Quantum Systemic Stress Scenario Discovery

Quantum-native systemic stress testing. An entangled Born-machine generator proposes correlated default scenarios; a deterministic cascade simulator evaluates contagion. Quantum advantage on two surfaces:

- **Generation** — an entangled QCBM samples the correlated default distribution and serves as the QAE state-loader (one qubit per institution; angles set analytically from marginals and the exposure graph).
- **Calculation** — the cascade becomes a reversible oracle; QAE estimates tail risk (`P(severe)`, CVaR) with a quadratic, deep-tail-amplified speedup; amplitude amplification finds the worst plausible scenario.

**Direction:** the classical baselines, the entangled QCBM generator, both contagion channels (exposure cascade + Huang fire-sale, unified behind one harness), and the comparison harness are implemented and validated end-to-end — n=54 generation reproduces the exact mean-field oracle loss distribution. The remaining quantum layer is the QAE cascade oracle (the QCBM state-loader half already exists). Hardware access to a **54-qubit machine is incoming**.

## Run

```bash
uv sync                                      # core + dev deps into .venv
uv run python scripts/run_demonstration.py   # canonical end-to-end run -> outputs/
uv run python scripts/run_mvp.py             # fast smoke test / dashboard feed
uv run pytest                                # tests
```

Extras:

```bash
uv run --extra app streamlit run app/streamlit_app.py   # Streamlit dashboard
uv run --extra quantum python scripts/run_mvp.py        # Qiskit backend (else exact statevector)
uv sync --all-extras
```

## Layout

`src/systemic_risk/`: `spec.py` (SystemSpec), `data/` (synthetic network + Huang adapter), `data_network/` (Part A: real 28-bank network), `generators/` (Bernoulli, copula, Ising, entangled Born machine), `simulator/` (exposure cascade + Huang fire-sale, sharing one `CascadeOutcome` contract), `evaluation/` (metrics + harness; one harness drives either contagion channel), `visualization/`. Entry points in `scripts/` (`run_demonstration.py` canonical, `run_mvp.py` smoke test, rest specialized — see README table); tests in `tests/`.
