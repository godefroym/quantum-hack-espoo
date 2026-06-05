# Reframing the Finance Problem onto Proven Quantum Advantages

A proven quantum advantage attaches to a **computational task type**, not to a problem domain. Optimisation — the original $\arg\max_x F(x)$ — has no proven quantum advantage in general, which is exactly why a QAOA formulation could never be proven. The move, therefore, is to change *the question asked* about the contagion problem until it lands on a task type that does carry a proven speedup.

Three reframings work, in decreasing order of how cleanly they fit.

---

## 1. Optimisation → Estimation (QAE)

**Reframe.** Instead of "find the single worst plausible seed," ask a risk-measure question:

- $P(F \ge X)$ — probability a cascade exceeds size $X$ under the plausibility distribution (tail probability)
- $\mathbb{E}[F]$ — expected cascade size
- **VaR / CVaR** of cascade losses — the loss quantile and the expected loss beyond it

**Mechanism.** All of these are *estimation* problems. Quantum Amplitude Estimation gives error $\epsilon$ in $O(1/\epsilon)$ oracle calls versus Monte Carlo's $O(1/\epsilon^2)$. VaR is recovered by wrapping QAE in a bisection search over the threshold; CVaR follows similarly. This is exactly the construction in the established quantum-finance literature (Woerner–Egger; Egger et al. on credit risk).

**Proven status.** ✅ **Proven quadratic speedup** over Monte Carlo, in the query-complexity model.

**Why it is also the right question.** "How likely is a systemic event" is the quantity risk committees and regulators actually act on — arguably more useful than a single worst-case scenario. The reframe is honest, not a contrivance.

**Catch.** The speedup is real only if state preparation is cheap. This is where the entangled generative loader (QCBM / qGAN) sits underneath, learning and loading the plausibility distribution into the QAE state-prep unitary $A$.

---

## 2. Search (Grover / Dürr–Høyer)

**Reframe.** Keep the argmax, but pose the decision version: *does a plausible seed of size $\le k$ exist that causes a cascade $\ge M$?* Sweep $M$ upward to recover the maximum.

**Mechanism.** Use the exact, classically-cheap $F$ as a reversible oracle and run Dürr–Høyer quantum maximum finding (iterated Grover). Initialise as a Dicke superposition over strings with $|x| \le k$, so the search space is $\binom{n}{k}$ from the start and the cost is $O\!\left(\sqrt{\binom{n}{k}}\right)$ oracle calls. Because $F$ is evaluated exactly inside the oracle, the true cascade dynamics are preserved — no linearisation.

**Proven status.** ✅ **Proven quadratic speedup** — but against *brute-force enumeration*, not against strong classical heuristics.

**Catch.** Simulated annealing, branch-and-bound, and MILP are far better than brute force, and there is no proven advantage over those. It is a genuine proven query-complexity result, but the practical baseline you would benchmark against is tougher than the one the proof assumes.

---

## 3. Graph / Markov Process (Quantum Walks)

**Reframe.** The problem is natively a graph problem. Recast contagion as a process on the exposure graph and ask a graph-property question. Two map on, each requiring a model adjustment:

- **Hitting time (Szegedy walk).** Model contagion as a stochastic diffusion with "systemic collapse" as an absorbing state; ask for the hitting time or absorption probability.
- **Connectivity (span-program walk).** Simplify contagion to a percolation model — an entity fails if connected to the seed through active exposure edges — and ask whether a giant failed component forms.

**Mechanism.** Szegedy quantum walks give a quadratic speedup on the hitting time of a Markov chain; st-connectivity quantum walks give proven query speedups for connectivity.

**Proven status.** ✅ **Proven quadratic query speedup** for these graph tasks.

**Catches.**
- You give up the exact threshold-accumulation dynamics in exchange for a diffusion or percolation model the proven algorithm accepts.
- Szegedy's speedup is stated for *reversible* chains, whereas real contagion is monotone and irreversible (defaults do not reverse), so the reversibility condition needs care or reformulation.

Worth pursuing only if a diffusion/percolation contagion model is acceptable for the purpose.

---

## Comparison

| Reframing | Task type | Quantum method | Speedup | Proven? | Main caveat |
|---|---|---|---|---|---|
| **1. Estimation** | Estimate a risk measure | QAE (+ generative loader) | Quadratic over Monte Carlo | ✅ Proven (query model) | State-prep cost; needs the loader |
| **2. Search** | Decide / locate worst seed | Grover / Dürr–Høyer | Quadratic over brute force | ✅ Proven (query model) | Not proven vs. strong classical heuristics |
| **3. Graph** | Graph property / hitting time | Quantum walk (Szegedy / span program) | Quadratic over classical walk | ✅ Proven (query model) | Requires simplified contagion model; reversibility condition |

---

## Recommendation

Lead with **Reframing 1 (QAE)**: the one proven, practically meaningful, domain-appropriate speedup, and it absorbs the generative-model work as its state-prep front end. Keep **Reframing 2 (Grover)** as a secondary exhibit for the "find the scenario" framing, and present **Reframing 3 (quantum walks)** as the research-direction option if a diffusion contagion model is acceptable.

The deciding question: does the end user want a **number** (probability or severity of a systemic event → QAE) or a **scenario** (the specific seed set to stress-test → Grover)? The proven-advantage path is far cleaner for the number, so if both are acceptable, framing the deliverable as a risk measure is what lets you stand behind the word "proven."
