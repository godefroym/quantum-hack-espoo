# Solution Architecture — Quantum vs Classical Scenario Generation

## Project overview (short)

This project compares a set of classical scenario generators (independent Bernoulli, Gaussian and Student-t copulas, pairwise Ising/Boltzmann) against an entanglement-structured quantum / quantum-inspired generator (Born-machine / PQC). The goal is to assess which generator produces more severe plausible contagion tails when fed into the same deterministic contagion simulator. The experiment isolates the scenario generator as the single varying component.

## Main solution diagram

```mermaid
flowchart LR
  %% Inputs
  subgraph Inputs[Input / Data]
    RAW[ECB / real financial data
    (conceptual)]
    ROSTER[data/external/banks/gsib_roster.csv]
    CORR[data/external/banks/equity_corr.csv]
    BALANCE[balance sheets / totals]
  end
  ## Slide-ready solution storyboard
  
  This document summarizes the experimental setup and embeds a presentation-ready storyboard showing how classical and quantum scenario generators feed the same deterministic contagion simulator.
  
  **Slide**
  - **Image:** see [docs/diagrams/solution_explainer_slide.svg](docs/diagrams/solution_explainer_slide.svg)
  
  **What the diagram shows**
  - **Input:** real financial data (exposures, balance sheets, correlations).
  - **Shared SystemSpec:** canonical network and parameters built once and held fixed.
  - **Parallel generators:** a classical baseline and a quantum/quantum-inspired generator produce scenarios in the same sample format.
  - **Shared pipeline:** both outputs merge to the same deterministic contagion simulator and evaluation harness.
  
  **Where quantum is involved**
  - Quantum involvement is limited to the scenario generation branch (clustering → entanglement layout → PQC/Born-machine sampling). The simulator, metrics, and SystemSpec remain classical and identical between arms.
  
  **What is shared**
  - SystemSpec, deterministic contagion simulator rules, evaluation metrics, and the scenario sample format are identical for both arms. This enforces a controlled, fair comparison.
  
  **Why this is fair**
  - By fixing SystemSpec and the simulator, any differences in downstream systemic-risk outcomes can be attributed to differences in the scenario generators.
  
  **TL;DR**
  - One canonical financial system, two generators (classical vs quantum), one shared simulator and evaluation — compare results fairly.
  
  If you want, I can regenerate the slide PNG at a different DPI or tweak font sizes/wording for a specific slide deck.

  ## Project overview (short)

  Compare a classical scenario-generation pipeline vs an entanglement-structured quantum/quantum-inspired pipeline for systemic-risk scenario discovery. Both generators consume the same calibrated `SystemSpec` and feed the same deterministic contagion simulator; the generator is the only experimental variable.

  ---

  **Slide-ready solution storyboard**

  The main slide is generated at `docs/diagrams/solution_explainer_slide.svg` (and PNG). It illustrates the experimental setup: inputs → shared `SystemSpec` → two parallel generators (classical and quantum) → same scenario format → shared deterministic contagion simulator → evaluation → fair comparison.

  ![Solution storyboard](docs/diagrams/solution_explainer_slide.svg)

  Mermaid fallback (source): `docs/diagrams/solution_explainer_slide.mmd`

  ---

  ## Where quantum is involved

  Quantum is only used in the scenario generation branch. Specifically:
  - Clustering / entanglement layout that maps dependency clusters to qubits.
  - The quantum scenario generator (PQC / Born-machine) that samples entanglement-structured scenarios.
  - Quantum sampling yields stress scenarios in the same format as the classical generator so that downstream evaluation is fair.

  ## What is shared / non-quantum

  The following components are shared and non-quantum:
  - `SystemSpec` (nodes, exposures, thresholds, marginals, pairwise correlations)
  - Scenario format (the representation of sampled stress scenarios)
  - Deterministic contagion simulator (`simulator/cascade.py`) — generator-agnostic cascade model
  - Metrics and evaluation harness (`evaluation/metrics.py`, `evaluation/harness.py`)

  ## Why this architecture is fair

  Both the classical and quantum pipelines consume the identical `SystemSpec` and produce stress scenarios in the same format. The same deterministic simulator and evaluation metrics are applied to both sets of scenarios. Because only the generator differs, the experimental variable is isolated: any downstream differences in tail contagion outcomes can be attributed to the scenario generator.

  ## TL;DR for slides

  - Two parallel generators (classical vs quantum) consume the same `SystemSpec`.
  - Both produce scenarios in the same format and feed the identical simulator.
  - Evaluation uses shared metrics and harness so comparisons are apples-to-apples.
  - The only experimental variable is the scenario generator.
  - This isolates the generator effect on tail contagion outcomes.

  RAW --> Pre[Preprocess / Build SystemSpec]
  ROSTER --> Pre
  CORR --> Pre
  ```mermaid
  flowchart LR
    Problem["Compare classical vs quantum\nsystemic-risk scenario generation"]

    subgraph Data[Data / Inputs]
      ECB["ECB / financial data"]
      NET["Exposure network"]
      BAL["Balance sheets"]
      CORR["Correlation inputs"]
    end

    Data --> Spec["Build shared SystemSpec\n(nodes, marginals, correlations, exposures)"]

    Spec --> Classical["CLASSICAL / Non‑Quantum"]
    Spec --> Quantum["QUANTUM / Quantum‑Inspired"]

    subgraph ClassicalLane[CLASSICAL]
      direction TB
      G1["Gaussian copula\nclassical sampling"]
      G1 --> S1["Classical stress\nscenarios (same format)"]
    end

    subgraph QuantumLane[QUANTUM]
      direction TB
      QL["Clustering / Entanglement\nlayout (layout.py)"]
      QG["Quantum scenario\ngenerator (Born‑machine)"]
      QL --> QG --> S2["Quantum stress\nscenarios (same format)"]
    end

    S1 --> Simulator["Shared Deterministic\nContagion Simulator (cascade.py)"]
    S2 --> Simulator

    Simulator --> Metrics["Metrics & Tail Measures\n(CVaR, tail probs)"]
    Metrics --> Eval["Evaluation Harness\n(compare generators fairly)"]
    Eval --> Output["Fair Generator Comparison\n(comparison.csv, plots, cards)"]

    %% Styling groups
    classDef input fill:#f3f9ff,stroke:#3b82c4;
    classDef shared fill:#eefdf3,stroke:#059669;
    classDef classical fill:#fff7ed,stroke:#d97706;
    classDef quantum fill:#f0f9ff,stroke:#0369a1,stroke-width:2px;
    class Problem input;
    class Data input;
    class Spec shared;
    class Simulator,MET,Metrics,Eval shared;
    class ClassicalLane classical;
    class QuantumLane quantum;
  ```

  Data --> Spec --> Classical --> Simulator --> Metrics
  Spec --> Layout --> Quantum --> Simulator
```

## Where quantum is involved vs not involved

- Quantum: `generators/quantum/*` (layout, ansatz, statevector, qiskit_backend) and `generators/quantum_born_machine.py`. Quantum involvement is limited to scenario generation and layout; sampling can be emulated locally or submitted to IBM Runtime (optional).
- Classical: `generators/gaussian_copula.py`, `generators/student_t_copula.py`, `generators/bernoulli.py`, `generators/ising_boltzmann.py` — these implement the non-quantum baselines.
- Shared (non-quantum): `src/systemic_risk/simulator/cascade.py`, `src/systemic_risk/simulator/huang.py` and the evaluation harness (`src/systemic_risk/evaluation/harness.py`) — the contagion simulator and metrics are identical for both branches.

Why this separation matters: Quantum resources and modelling choices only affect how scenarios are drawn and structured. By keeping the simulator and evaluation shared, the experiment isolates generator impact on tail risk.

## Architecture rationale

- Two branches exist to create a fair controlled experiment: the classical branch provides established benchmark generators; the quantum branch tests whether entanglement-structured generators create deeper-tail correlated defaults. Both branches accept the same calibrated `SystemSpec` so downstream differences in cascade severity are attributable to the generator only.
- Clustering/entanglement layout localises quantum structure to economically meaningful clusters, keeping circuits shallow and interpretable while capturing concentrated dependency structures.

## TL;DR for slides (what to say aloud)

- Controlled experiment: same SystemSpec and same deterministic simulator; only the scenario generator changes.
- Classical branch: Gaussian / Student-t copula and other baselines generate correlated default scenarios.
- Quantum branch: entanglement-structured Born-machine / PQC generates non-factorized joint scenarios guided by clustering/layout.
- Shared simulator: both feeds go into the same contagion engine and metrics pipeline; this isolates generator effects.
- Outcome: compare tail probabilities, CVaR, and worst-case scenarios to assess whether entanglement-style generators reach deeper crisis tails.

---

Rendering tip

To render the main mermaid file `docs/diagrams/solution_explainer.mmd` to PNG/SVG (example using `npx`):

```bash
npx --yes @mermaid-js/mermaid-cli -i docs/diagrams/solution_explainer.mmd -o docs/diagrams/solution_explainer.svg
npx --yes @mermaid-js/mermaid-cli -i docs/diagrams/solution_explainer.mmd -o docs/diagrams/solution_explainer.png
```

Use these images in slides; the `.mmd` can be pasted into Mermaid Live Editor for quick tweaks.
