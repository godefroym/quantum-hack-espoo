# 05 — Quantum Algorithms for Finance/Risk and Quantum Generative Loading

Literature survey supporting the quantum claims in this repo. Scope: the works our
two quantum surfaces rest on — **(a) generation/loading** (entangled QCBM / qGAN as a
state-loader) and **(b) calculation** (Quantum Amplitude Estimation for tail risk;
Grover / Dürr–Høyer for worst-case search).

The repo's own claims live in `scenario_generation.md` and `README.md`. The single
load-bearing claim there is: *"The central caveat of the QAE risk-estimation route is
that you must load the plausibility distribution into the state-preparation unitary A,
and arbitrary state preparation can be expensive enough to erase the quadratic speedup
… The entangled generative model is that loader."* This survey checks that claim, and
the QAE/Grover claims, against the primary literature.

Method note: details below were extracted from arXiv (HTML/ar5iv), publisher abstract
pages, and the Qiskit Finance docs. Where a specific number could not be re-confirmed
from a text-readable source it is flagged [UNVERIFIED]. Nothing here is fabricated;
all DOIs/arXiv IDs were returned by search or fetched directly.

---

## A. The two foundational risk papers (the citations we lean on)

### A1. Woerner & Egger (2019) — "Quantum risk analysis"
- **Citation:** S. Woerner, D. J. Egger. *Quantum risk analysis.* npj Quantum
  Information **5**, 15 (2019). DOI: 10.1038/s41534-019-0130-6. arXiv:1806.06893.
  Links: https://www.nature.com/articles/s41534-019-0130-6 · https://arxiv.org/abs/1806.06893
- **What it demonstrates:** A gate-based quantum algorithm that uses Quantum
  Amplitude Estimation (QAE) to compute risk measures — **Value at Risk (VaR)** and
  **Conditional Value at Risk (CVaR)** — over a loaded loss distribution, faster than
  classical Monte Carlo. This is the canonical "QAE for risk" reference and the direct
  ancestor of our CVaR/P(severe) tail-risk claim.
- **Distribution / data:** Synthetic / parametric. Worked examples are small portfolios
  (e.g. a two-asset toy model) with the loss distribution loaded analytically into the
  state-prep unitary; not real market data.
- **Speedup + caveat:** With the shortest circuit, convergence is O(M^-2/3), already
  beating classical MC's O(M^-1/2); allowing circuit depth to grow polynomially pushes
  it toward the optimal O(M^-1), i.e. a **near-quadratic speedup**. Caveat: the result
  is asymptotic and assumes the distribution is already loaded — state preparation cost
  is exactly the thing the follow-on credit-risk paper warns about (see A2).
- **Hardware:** Primarily simulation plus small illustrative circuits; this is an
  algorithm/methodology paper, not a hardware-advantage demonstration.

### A2. Egger, García Gutiérrez, Cahué Mestre & Woerner (2020/21) — "Credit Risk Analysis using Quantum Computers"
- **Citation:** D. J. Egger, R. García Gutiérrez, J. Cahué Mestre, S. Woerner.
  *Credit Risk Analysis Using Quantum Computers.* IEEE Transactions on Computers
  **70**(12), 2136–2145 (2021). DOI: 10.1109/TC.2020.3038063. arXiv:1907.03044.
  Links: https://ar5iv.labs.arxiv.org/html/1907.03044 · https://doi.org/10.1109/TC.2020.3038063
- **What it demonstrates:** QAE applied to credit-portfolio risk under the **Gaussian
  Conditional Independence (GCI)** model (the structure used in the Basel II IRB
  approach). It estimates **Economic Capital Requirement = VaR_α[L] − E[L]** via a
  bisection search wrapping QAE. This is the closest published analogue to *our* setup:
  a correlated multivariate-Bernoulli default model with latent correlation, exactly
  the family our Ising/Boltzmann Π(x) lives in.
- **Distribution / data:** Synthetic / parametric — illustrative two-asset (and small
  portfolio) examples with specified default probabilities, loss-given-defaults, and
  sensitivities; the example circuit is **12 qubits, classically simulated** (no
  physical-hardware results in the paper).
- **Speedup:** Quadratic: QAE error ~ O(1/M) vs classical MC O(1/√M).
- **★ The load-bearing caveat (verbatim):** *"The ability to efficiently construct the
  uncertainty model is a crucial part in QAE-based algorithms, and if not handled
  carefully can diminish the potential quantum advantage."* They mitigate by using
  ancillas to cut the loading circuit depth from O(nzK) to O(log K). **This is the
  exact statement our repo cites in spirit** — state preparation can erase the speedup,
  which is why the entangled generator-as-loader is the non-decorative part of the
  story. The paper also notes QPE can be omitted from QAE (foreshadowing IQAE/MLQAE).

> Takeaway for us: A1 + A2 jointly support our QAE tail-risk claim **and** they are the
> primary sources for the caveat we must state. We should cite A2 specifically when we
> say "state prep can erase the quadratic advantage," because it says so almost word
> for word and it operates on the same GCI/correlated-default model family we use.

---

## B. The loader citation (load-bearing for "entangled QCBM/qGAN as state-loader")

### B1. Zoufal, Lucchi & Woerner (2019) — "Quantum Generative Adversarial Networks for learning and loading random distributions"
- **Citation:** C. Zoufal, A. Lucchi, S. Woerner. *Quantum Generative Adversarial
  Networks for learning and loading random distributions.* npj Quantum Information
  **5**, 103 (2019). DOI: 10.1038/s41534-019-0223-2. arXiv:1904.00043.
  Links: https://www.nature.com/articles/s41534-019-0223-2 · https://ar5iv.labs.arxiv.org/html/1904.00043
- **What it demonstrates:** A hybrid quantum-classical **qGAN** that learns a
  probability distribution *given only data samples* and loads it into a quantum state
  in **O(poly(n))** gates, versus the **O(2^n)** gates that exact/arbitrary state
  preparation can require. The trained generator is then used directly as the
  state-prep unitary for amplitude estimation. **This is the precise mechanism our repo
  invokes: a trained quantum generator that doubles as the QAE state-loader.**
- **Distribution / data:** Trained on samples from **synthetic parametric**
  distributions — a **log-normal** (μ=1, σ=1), a **triangular** (l=0, u=7, m=2), and a
  **bimodal** mixture — each on a **3-qubit** generator (2^3 = 8 discretized values).
  Not real financial data; the distributions are stand-ins for an asset/loss model.
- **Hardware:** Yes — real quantum hardware. The generator was trained/run using the
  **IBM Q Boeblingen 20-qubit** chip (the qGAN itself uses 3 qubits + a discriminator).
- **Downstream + speedup:** The loaded state feeds QAE to price a **European call
  option**, achieving Grover-type error scaling **ε = O(1/N)** vs classical MC
  **O(1/√N)** — i.e. the quadratic speedup, contingent on the poly-size loader.
- **Why it matters here:** It is the empirical existence proof that an *entangled,
  trained* quantum generator can serve as an *efficient* loader for QAE. That is
  exactly our "entanglement is load-bearing, not decorative" claim. **Caveat to state
  honestly:** it is *approximate* loading of *low-dimensional synthetic* distributions
  (3 qubits, 8 bins); it does not prove efficient loading of an arbitrary
  high-dimensional correlated default distribution, and does not claim end-to-end
  advantage on real data.

---

## C. Option pricing on real hardware (the most-cited end-to-end QAE finance demo)

### C1. Stamatopoulos, Egger, Sun, Zoufal, Iten, Shen & Woerner (2020) — "Option Pricing using Quantum Computers"
- **Citation:** N. Stamatopoulos, D. J. Egger, Y. Sun, C. Zoufal, R. Iten, N. Shen,
  S. Woerner. *Option Pricing using Quantum Computers.* Quantum **4**, 291 (2020).
  DOI: 10.22331/q-2020-07-06-291. arXiv:1905.02666.
  Links: https://quantum-journal.org/papers/q-2020-07-06-291/ · https://ar5iv.labs.arxiv.org/html/1905.02666
- **What it demonstrates:** End-to-end QAE pricing of **vanilla, multi-asset, and
  path-dependent (barrier) options**, with the actual circuit constructions for the
  state-prep and payoff operators — the engineering recipe for "load distribution →
  apply payoff/oracle → QAE."
- **Distribution / data:** Synthetic / parametric — **Black-Scholes-Merton**, i.e. the
  underlying follows a **log-normal** spot-price distribution loaded into the state.
- **Speedup:** Amplitude Estimation gives O(M^-1) error vs classical MC O(M^-1/2) — the
  quadratic speedup.
- **Hardware:** Real device — **IBM Q Tokyo**. The European-call hardware demo used
  **3 qubits** and used **maximum-likelihood amplitude estimation (no QPE)** plus a
  two-qubit-gate error-mitigation scheme to get usable results on NISQ hardware.
- **Why it matters here:** It is the template for our "cascade-as-oracle then QAE"
  pipeline, and it is concrete evidence that the *only* QAE finance results on real
  hardware so far are tiny (≈3 qubits, toy/synthetic distributions). Useful for keeping
  our hardware claims modest.

---

## D. NISQ-friendly amplitude estimation (what we'd actually run)

### D1. Suzuki, Uno, Raymond, Tanaka, Onodera & Yamamoto (2020) — "Amplitude estimation without phase estimation" (MLQAE)
- **Citation:** Y. Suzuki, S. Uno, R. Raymond, T. Tanaka, T. Onodera, N. Yamamoto.
  *Amplitude estimation without phase estimation.* Quantum Information Processing
  **19**, 75 (2020). DOI: 10.1007/s11128-019-2565-2. arXiv:1904.10246.
  Links: https://link.springer.com/article/10.1007/s11128-019-2565-2 · https://arxiv.org/abs/1904.10246
- **What it demonstrates:** **Maximum-Likelihood QAE (MLQAE)** — drops the expensive
  controlled-QPE machinery, instead combining measurements from circuits with different
  numbers of Grover/amplitude-amplification steps via MLE. Asymptotically attains near
  the optimal (quadratic) quantum speedup with much shorter circuits.
- **Data / speedup / hardware:** Numerical demonstrations; the method (not a finance
  problem) is what Stamatopoulos C1 used on IBM Q Tokyo. No qubit-count headline of
  its own; the point is drastically reduced depth/width vs canonical QAE.

### D2. Grinko, Gacon, Zoufal & Woerner (2021) — "Iterative Quantum Amplitude Estimation" (IQAE)
- **Citation:** D. Grinko, J. Gacon, C. Zoufal, S. Woerner. *Iterative Quantum
  Amplitude Estimation.* npj Quantum Information **7**, 52 (2021).
  DOI: 10.1038/s41534-021-00379-1. arXiv:1912.05559.
  Links: https://www.nature.com/articles/s41534-021-00379-1 · https://arxiv.org/abs/1912.05559
- **What it demonstrates:** **IQAE** — also removes QPE, "only based on Grover's
  Algorithm, which reduces the required number of qubits and gates," with a rigorous
  proof of a **quadratic speedup up to a double-logarithmic factor** vs classical MC,
  and small constant overhead. Empirically needs fewer samples than other QAE variants
  at the same accuracy/confidence.
- **Why D1/D2 matter here:** Canonical (Brassard) QAE needs QPE → many qubits + deep
  circuits, infeasible near-term. MLQAE/IQAE are the variants we should claim, and IQAE
  is the **default in Qiskit Finance's credit-risk tutorial** (see G). They keep the
  quadratic speedup honest while being NISQ-plausible.

### D3. Brassard, Høyer, Mosca & Tapp (2002) — "Quantum Amplitude Amplification and Estimation"
- **Citation:** G. Brassard, P. Høyer, M. Mosca, A. Tapp. *Quantum Amplitude
  Amplification and Estimation.* Contemporary Mathematics **305**, 53–74 (2002).
  arXiv:quant-ph/0005055. Links: https://arxiv.org/abs/quant-ph/0005055
- **What it demonstrates:** The **foundational** result. Amplitude amplification finds a
  marked item with O(1/√a) applications of A and A† (generalizing Grover), and amplitude
  estimation reads off a to additive error with a quadratic improvement over sampling.
  Every QAE-for-risk paper above is a specialization of this. This is the correct
  primary citation for "the quadratic speedup" itself.

---

## E. Quantum Circuit Born Machines (our generator's native model)

### E1. Benedetti, Garcia-Pintos, Perdomo, Leyton-Ortega, Nam & Perdomo-Ortiz (2019) — "A generative modeling approach for benchmarking and training shallow quantum circuits"
- **Citation:** M. Benedetti, D. Garcia-Pintos, O. Perdomo, V. Leyton-Ortega, Y. Nam,
  A. Perdomo-Ortiz. *A generative modeling approach for benchmarking and training
  shallow quantum circuits.* npj Quantum Information **5**, 45 (2019).
  DOI: 10.1038/s41534-019-0157-8. arXiv:1801.07686.
  Links: https://www.nature.com/articles/s41534-019-0157-8 · https://ar5iv.labs.arxiv.org/html/1801.07686
- **What it demonstrates:** A **QCBM** (prepare U(θ)|0⟩, measure in computational basis,
  sample x ∝ |⟨x|U(θ)|0⟩|²) trained as a generative model on real hardware, plus the
  hardware-independent **qBAS score** for benchmarking. States explicitly that
  **"entanglement is a key ingredient in encoding the patterns of this data set"** —
  direct support for our "entangling layers produce the correlations" claim.
- **Distribution / data:** The **Bars-and-Stripes (BAS)** dataset — a structured,
  correlated discrete distribution (synthetic). The hardware run used **BAS(2,2)**
  (a 2×2 grid, 6 valid patterns).
- **Hardware:** Real device — **4 qubits on a trapped-ion (ion-trap) quantum computer
  at the University of Maryland.**
- **Why it matters here:** Direct precedent that an *entangled* QCBM learns a
  *correlated* discrete distribution on hardware, and that entanglement is what carries
  the correlation. Caveat: tiny (4 qubits, 6 patterns), trained to match a known
  distribution; it is *not* evidence of classically-hard sampling at useful scale.

### E2. Liu & Wang (2018) — "Differentiable Learning of Quantum Circuit Born Machines"
- **Citation:** J.-G. Liu, L. Wang. *Differentiable learning of quantum circuit Born
  machines.* Physical Review A **98**, 062324 (2018). DOI: 10.1103/PhysRevA.98.062324.
  arXiv:1804.04168. Links: https://link.aps.org/doi/10.1103/PhysRevA.98.062324 · https://arxiv.org/abs/1804.04168
- **What it demonstrates:** A practical **gradient-based** QCBM training method using the
  kerneled **Maximum Mean Discrepancy (MMD)** loss, runnable on near-term devices. This
  is the standard recipe for actually *training* a QCBM (vs the qGAN adversarial route).
- **Distribution / data:** Synthetic — **Bars-and-Stripes** and **Gaussian-mixture**
  distributions, simulated with deep circuits.
- **Why it matters here:** Gives us the concrete training objective (MMD) for our QCBM
  loader as an alternative to the qGAN of B1. Note the "can exhibit quantum advantages
  for probabilistic generative modeling" framing is an *aspiration*, not a proof.

> Born-machine expressivity caveat (for honesty): the claim that QCBM output
> distributions can be classically hard to sample is supported in the literature
> (e.g. the "Born supremacy" line of work, npj QI 6, 86 (2020),
> https://www.nature.com/articles/s41534-020-00288-9), but it is a *conjectured/
> conditional* hardness for specific circuit families, not a guarantee for our
> particular Π(x)-targeted circuit. We can say "entangling layers produce the
> correlations" (operationally true, verifiable from samples — and our repo already
> frames it this way); we should NOT claim our specific loader is provably
> classically-hard to sample.

---

## F. Worst-case scenario search (Grover / Dürr–Høyer)

### F1. Dürr & Høyer (1996) — "A Quantum Algorithm for Finding the Minimum"
- **Citation:** C. Dürr, P. Høyer. *A Quantum Algorithm for Finding the Minimum.*
  arXiv:quant-ph/9607014 (1996; rev. 1999). Links: https://arxiv.org/abs/quant-ph/9607014
- **What it demonstrates:** Finds the index of the minimum entry in a size-N table in
  **O(√N)** oracle queries with high probability (within a constant of the proven lower
  bound), via repeated Grover search with a decreasing threshold.
- **Why it matters here:** This is the correct primary citation for our "find the worst
  plausible scenario" step — minimizing/maximizing cascade severity over scenarios is a
  Grover-style search, O(1/√a) vs O(1/a) classical.
- **★ Honesty flag (matches the repo):** The √N speedup is over an **unstructured**
  search with **oracle access**, and assumes the scenarios are loadable/queryable in
  superposition. Our repo already correctly tags Grover/Dürr–Høyer as **"Byproduct — do
  not oversell."** Keep it that way: the worst-case-search advantage is real in the
  oracle model but is the weakest of our three claims in practice (constant-factor
  overheads, oracle/loading assumptions, no problem-structure exploitation).

---

## G. Open-source code we can reuse

- **Qiskit Finance** (qiskit-community) — the most directly reusable. Implements the
  exact algorithms above with worked tutorials:
  - Landing / install: https://qiskit-community.github.io/qiskit-finance/
  - Quantum Amplitude Estimation primer: https://qiskit-community.github.io/qiskit-finance/tutorials/00_amplitude_estimation.html
  - **Credit Risk Analysis (GCI model + Iterative AE)** — closest to our problem:
    https://qiskit-community.github.io/qiskit-finance/tutorials/09_credit_risk_analysis.html
  - European call pricing: https://qiskit-community.github.io/qiskit-finance/tutorials/03_european_call_option_pricing.html
  - **Option pricing with qGANs** (loader + QAE end-to-end):
    https://qiskit-community.github.io/qiskit-finance/tutorials/10_qgan_option_pricing.html
  - All QAE variants (canonical / IQAE / MLQAE) share the `AmplitudeEstimator`
    interface (in `qiskit-algorithms`), so we can swap estimators on one
    `EstimationProblem` — i.e. prototype with IQAE/MLQAE directly.
- **Liu & Wang QCBM reference implementation** (gradient/MMD training):
  https://github.com/GiggleLiu/QuantumCircuitBornMachine
- **Community QCBM (Benedetti-style) implementation:**
  https://github.com/akastellakis/Quantum_Circuit_Born_Machine

> Practical path: reuse Qiskit Finance's GCI credit-risk + IQAE pipeline as the
> calculation backbone (it already does VaR/CVaR-style economic-capital via QAE on a
> correlated-default model), and reuse the qGAN-option-pricing tutorial's
> loader→QAE wiring. Swap the loader for our QCBM (trained via MMD à la Liu–Wang or
> adversarially à la Zoufal) targeting Π(x).

---

## H. Honest-claims digest (map to our three quantum surfaces)

| Our claim | Verdict | Primary support | Required caveat |
|---|---|---|---|
| **Entangled QCBM/qGAN as the state-loader** (load-bearing) | **Well-supported in principle, on small synthetic problems** | Zoufal–Lucchi–Woerner (B1): qGAN loads a distribution in O(poly n) and feeds QAE, on IBM hardware. Benedetti (E1) + Liu–Wang (E2): entangled QCBMs learn correlated discrete distributions; entanglement carries the correlation. | Demonstrations are tiny (3–4 qubits, 8 bins / BAS(2,2)) and *approximate*. No proof of efficient loading of an arbitrary high-dim correlated default distribution. Don't claim provable classical-sampling hardness for our specific circuit. |
| **QAE quadratic speedup for tail risk (P(severe), CVaR)** | **Well-supported asymptotically; not yet demonstrated at advantage scale** | Brassard et al. (D3) foundational; Woerner–Egger (A1) for VaR/CVaR; Egger et al. (A2) for correlated-default economic capital; IQAE/MLQAE (D1,D2) for NISQ feasibility. | Speedup is asymptotic and *conditional on efficient state preparation*. **State prep can erase the advantage — said explicitly by Egger et al. (A2)** and echoed by Herman et al. survey (Sec. I). Real-hardware QAE finance results are ~3 qubits on synthetic distributions (C1). |
| **Grover / Dürr–Høyer worst-case scenario search** | **Real in the oracle model; weakest claim — keep as "byproduct," don't oversell** | Dürr–Høyer (F1): O(√N) minimum finding; Brassard et al. (D3): O(1/√a) amplitude amplification. | Unstructured-search √-speedup with oracle/loading assumptions and constant-factor overheads; exploits no problem structure. Repo already labels this "do not oversell" — correct. |

**Cross-cutting caveat, well-sourced (use this when we discuss limits):** the
state-preparation / data-loading bottleneck. Egger et al. (A2): efficient construction
of the uncertainty model is crucial and *"if not handled carefully can diminish the
potential quantum advantage."* Herman et al. (2023 Nature Reviews Physics survey,
arXiv:2307.11230, https://www.nature.com/articles/s42254-023-00603-1): generic data
loading scales with the data size, embedding classical data is a component whose
resource cost must be reduced "or the speedup advantage may be negated," and the
overhead of error correction "could prevent the quadratic speedup from being
realizable" — end-to-end commercially-relevant quantum advantage in finance is not yet
achieved on current hardware. The Chakrabarti et al. resource-estimate paper (Quantum
5, 463 (2021), arXiv:2012.03819, https://quantum-journal.org/papers/q-2021-06-01-463/)
makes the scale concrete: useful derivative-pricing advantage needs **~8k logical
qubits and T-depth ~5.4×10^7** — i.e. fault-tolerant, not NISQ.

This is *consistent* with our repo's framing: our loader claim is precisely the answer
to that bottleneck (an entangled generator that *is* the efficient A), which is why it
is the honest, load-bearing part of the story while the calculation-side speedup should
be stated as asymptotic-and-conditional and the Grover search as a byproduct.

---

## I. Additional context sources (for breadth / framing, verified)

- **Herman, Googin, Liu, Sun, Galda, Safro, Pistoia & Alexeev (2023)** — *Quantum
  computing for finance.* Nature Reviews Physics **5**, 450–465 (2023).
  DOI: 10.1038/s42254-023-00603-1. arXiv:2307.11230.
  https://www.nature.com/articles/s42254-023-00603-1 — authoritative survey;
  source for the state-preparation-bottleneck and fault-tolerance caveats in Sec. H.
- **Chakrabarti, Krishnakumar, Mazzola, Stamatopoulos, Woerner & Zeng (2021)** — *A
  Threshold for Quantum Advantage in Derivative Pricing.* Quantum **5**, 463 (2021).
  DOI: 10.22331/q-2021-06-01-463. arXiv:2012.03819.
  https://quantum-journal.org/papers/q-2021-06-01-463/ — resource estimate showing
  QAE-for-finance advantage is a fault-tolerant-era goal (~8k logical qubits).
- **Born-supremacy / Ising Born machine** — npj Quantum Information **6**, 86 (2020),
  https://www.nature.com/articles/s41534-020-00288-9 — conditional classical-hardness
  arguments for Born-machine sampling (use carefully; conjectural for our circuit).
