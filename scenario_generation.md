# Scenario Generation as the Adjacent Problem

## The Plausibility Model as a Boltzmann Distribution

The plausibility model

$$\Pi(x) = \sum_i \ln p_i \, x_i + \sum_{i < j} J_{ij} \, x_i x_j$$

is a Boltzmann/Ising distribution over default configurations; a correlated multivariate Bernoulli with pairwise couplings. *Sampling* from it (not optimizing it) is genuinely hard, and it is the engine behind "draw me plausible correlated stress scenarios."

The core problem with systemic risk is that the events you most need to understand are the ones you have almost no data on. A severe cascade (many entities failing together) happens rarely or never in your historical record. If you try to estimate "how likely is a big cascade" by counting how often it happened in the past, you are dividing by a sample that contains zero or one tail events. You cannot measure a 1-in-500 catastrophe from 30 years of data.

A generative model fixes this by learning the *structure* of defaults, the marginals $p_i$ and the couplings $J_{ij}$, rather than memorizing which defaults happened. Once you have that structure, you can draw unlimited new scenarios that never occurred historically but are fully consistent with the dependency pattern: plausible combinations of co-failures that the data implies are possible even though they have not materialized yet.

That gives you three things:

- **It populates the tail.** You get synthetic-but-plausible extreme scenarios to estimate rare-event probabilities, which raw history cannot supply.

- **It preserves correlation.** If you sampled each entity independently from its own $p_i$, you would drastically *underestimate* joint catastrophes, because the whole danger of systemic risk is that correlated clusters fail *together*. The $J_{ij}$ couplings make the generator produce those co-failures at realistic rates.

- **It gives you arbitrarily many draws.** Statistical estimation of any risk number needs averaging over many samples; a generator is an unlimited sample source, and it is also the thing that feeds the quantum estimation step described below.

> **One-line version:** you cannot read rare correlated catastrophes off sparse history, so you fit a model to the dependency structure and let it generate the catastrophes for you.

---

## Quantum Circuit Born Machines

A **quantum circuit Born machine (QCBM)** prepares $U(\theta)|0\rangle$ and measures in the computational basis, so each shot yields a scenario $x$ with probability

$$|\langle x | U(\theta) | 0 \rangle|^2.$$

The entangling layers of $U(\theta)$ *are* what produce the statistical correlations among the default bits. This claim is operationally true: you sample $x$ and can directly check that defaults $i$ and $j$ co-occur at the rate $J_{ij}$ implies. The entanglement in the circuit shapes the correlation in the output, and the output is the thing you use.

---

## Closing the Loop with QAE

The central caveat of the QAE risk-estimation route is that you must load the plausibility distribution into the state-preparation unitary $A$, and arbitrary state preparation can be expensive enough to erase the quadratic speedup.

The entangled generative model **is** that loader. Train a QCBM, or a qGAN (Zoufal, Lucchi & Woerner demonstrated exactly this for quantum risk analysis: a quantum generator that learns and loads a distribution specifically to feed amplitude estimation), so that $A$ prepares the plausibility distribution, then run QAE on top to estimate tail risk. Entanglement enters precisely at $A$, and it solves the state-loading bottleneck rather than being decorative.

The integrated picture across the full pipeline is then:

| Component | Role | Where entanglement lives |
|---|---|---|
| **Grover / Dürr–Høyer** | *Find* the worst plausible scenario | Byproduct - do not oversell |
| **QAE** | *Quantify* tail risk given a loaded distribution | Inherited from state prep |
| **QCBM / qGAN** | *Generate and load* the correlated scenario distribution | Load-bearing: this is an honest claim |

Entanglement does genuine, verifiable work at the generation and loading step, which is also the step that enables the quadratic QAE advantage downstream.
