# 03 — Statistical Mechanics / Ising / Maximum-Entropy / Boltzmann Models of Systemic Risk

**Scope.** This section surveys the statistical-mechanics lineage that underpins our "plausibility model"

$$\Pi(x) \;=\; \sum_i \ln p_i \, x_i \;+\; \sum_{i<j} J_{ij}\, x_i x_j, \qquad x \in \{0,1\}^n,$$

which is exactly a **pairwise maximum-entropy (Ising / Boltzmann) distribution over binary default configurations** — a correlated multivariate Bernoulli with pairwise couplings. We cover four threads, all of which feed our synthetic generator and our claims about quantum advantage in *sampling* this distribution:

1. **Pairwise max-ent (Ising) models of market co-movement** (Bury; and the foundational Schneidman–Bialek neuroscience method that defines the inference machinery everyone reuses).
2. **Ising / graphical / spin-glass models of correlated default and credit contagion** (Filiz–Guo–Morton–Sturmfels; Molins–Vives; Kitsukawa–Mori–Hisakado; Emonti–Fontana "Jungle"/"Dandelion").
3. **Maximum-entropy reconstruction of interbank/financial networks** (Cimini–Squartini–Garlaschelli–Gabrielli; Mastromatteo–Zarinelli–Marsili; Squartini–Garlaschelli; the Squartini et al. *Physics Reports* review), i.e. where the $J_{ij}$ come from when the exposure graph is only partially observed.
4. **The physics-of-financial-networks umbrella + the inverse-Ising methods toolbox** (Bardoscia et al.; Nguyen–Zecchina–Berg), which tell us how to fit $h_i, J_{ij}$ and which sampling method is feasible at a given $n$.

A note on conventions used throughout. Two equivalent spin conventions appear in the literature: **binary** $x_i\in\{0,1\}$ (default indicator; our convention) and **±1 spins** $\sigma_i\in\{-1,+1\}$. Mapping: $\sigma_i = 2x_i-1$. Fields and couplings transform between conventions but the physics is identical. Our $\Pi(x)$ is already in the $\{0,1\}$ "log-linear / auto-logistic" form, where the diagonal field is literally $\ln p_i$-like (see methods digest).

---

## A. Pairwise maximum-entropy models of co-movement

### A1. Bury (2013), *Market structure explained by pairwise interactions*
- **Citation.** T. Bury, "Market structure explained by pairwise interactions," *Physica A: Statistical Mechanics and its Applications* **392**(6), 1375–1385 (2013). Preprint: arXiv:1210.8380.
  - arXiv: https://arxiv.org/abs/1210.8380
  - Journal/RePEc record: https://ideas.repec.org/a/eee/phsmap/v392y2013i6p1375-1385.html
- **Companion paper (same author, more methodology detail).** T. Bury, "Statistical pairwise interaction model of stock market," *European Physical Journal B* **86**, 89 (2013). Preprint: arXiv:1206.4420. https://arxiv.org/abs/1206.4420
- **Method (1–2 sentences).** Binarize daily stock returns to spins $s_i=\pm1$ (up/down) and fit a pairwise maximum-entropy (Ising) model that reproduces the observed single-stock means and pairwise correlations; show the pairwise model captures ~98% of the multi-information (so higher-order terms add little), and use the inferred coupling matrix as a dissimilarity/structure measure for the market that reproduces clustering and order–disorder ("crash") transitions.
- **DATA.** Daily (and some minute-resolution) index constituents: BEL20, AEX, DAX, CAC40 (European), Dow Jones (30 names), S&P 100 (US); plus the Onnela NYSE return set. Sizes are $n\sim20$–100 names; samples of ~1,050–4,800 trading days (and ~30,000 intraday points for Dow).
- **SOURCE / provider.** **Yahoo! Finance** (free) for the indices; the **Onnela dataset** from `jponnela.com` (free). 
- **PUBLIC/FREE vs PROPRIETARY.** **FREE / PUBLIC** (Yahoo! Finance historical quotes; Onnela's posted research dataset).
- **Verifiable link.** arXiv:1210.8380 (above); companion arXiv:1206.4420.
- **Methodology we can reuse (important).**
  - **Model.** $p(s)=Z^{-1}\exp\!\big(\tfrac12\sum_{i\ne j}J_{ij}s_is_j+\sum_i h_i s_i\big)$, with $h_i$ conjugate to the mean orientations $\langle s_i\rangle$ and $J_{ij}$ conjugate to the pairwise correlations $\langle s_i s_j\rangle$ (the standard max-ent duality $\partial S/\partial h_i = -\langle s_i\rangle$, etc.).
  - **Coupling ← correlation (fast, closed form).** Uses **second-order / TAP mean-field inversion**: at leading order the inferred couplings are minus the inverse of the connected-correlation (covariance) matrix, with a TAP correction $\;(C^{-1})_{ij} \approx -J_{ij} - J_{ij}^2\,q_i q_j$. This is the cheap, $O(n^3)$ route from a measured correlation matrix to $J_{ij}$.
  - **Data-requirement rule of thumb.** Reliable inversion needs sample length $T>20\,n$ with $n>10$ — a useful constraint when we calibrate from limited history.

### A2. Schneidman, Berry, Segev & Bialek (2006) — the canonical inverse-Ising method (cross-domain, foundational)
- **Citation.** E. Schneidman, M. J. Berry II, R. Segev, W. Bialek, "Weak pairwise correlations imply strongly correlated network states in a neural population," *Nature* **440**, 1007–1012 (2006).
  - Publisher record: https://weizmann.elsevierpure.com/en/publications/weak-pairwise-correlations-imply-strongly-correlated-network-stat
  - Bialek lecture notes summarizing the method: https://www.princeton.edu/~wbialek/rome/lecture3.htm
- **Why it is here.** It is the most-cited template for *fitting* a pairwise max-ent model to binary data; the finance papers (Bury, and the credit-default papers below) inherit this machinery. The key takeaway transferable to us: **a pairwise model is the unique max-ent distribution consistent with measured first and second moments**, and "weak pairwise correlations imply strongly correlated network states" — i.e. small $J_{ij}$ still produce large joint-tail (many-units-on-together) probabilities. That is precisely the systemic-risk message: independent-Bernoulli sampling massively underestimates joint catastrophes.
- **DATA.** Multi-electrode recordings of vertebrate **retinal ganglion neurons** (biological, not financial). Not relevant as a finance dataset; cited only for method.
- **Methodology we can reuse.**
  - **Model.** $P(\{s_i\})=Z^{-1}\exp\!\big(\sum_i h_i s_i+\tfrac12\sum_{i\ne j}J_{ij}s_is_j\big)$.
  - **Inference = Boltzmann learning.** Choose $h_i,J_{ij}$ so the model's $\langle s_i\rangle$ and $\langle s_i s_j\rangle$ match the data, by iterative gradient ascent on the log-likelihood: $\Delta h_i\propto \langle s_i\rangle_{\text{data}}-\langle s_i\rangle_{\text{model}}$, $\Delta J_{ij}\propto \langle s_is_j\rangle_{\text{data}}-\langle s_is_j\rangle_{\text{model}}$, with the model moments evaluated by **Monte Carlo (Metropolis/Gibbs)** at each step. For small $n$ ($\lesssim20$) the moments are computed by exact enumeration instead.

---

## B. Ising / graphical / spin-glass models of correlated default

### B1. Filiz, Guo, Morton & Sturmfels (2012), *Graphical models for correlated defaults* — **closest match to our $\Pi(x)$**
- **Citation.** I. O. Filiz, X. Guo, J. Morton, B. Sturmfels, "Graphical Models for Correlated Defaults," *Mathematical Finance* **22**(4), 621–644 (2012). Preprint: arXiv:0809.1393.
  - arXiv: https://arxiv.org/abs/0809.1393
  - Journal: https://onlinelibrary.wiley.com/doi/10.1111/j.1467-9965.2011.00499.x
  - (Follow-up on dynamics: Filiz et al., "Non-existence of Markovian time dynamics for graphical models of correlated default," *Queueing Systems* (2012), arXiv:1008.2226.)
- **Method.** Put a graph $G=(V,E)$ on firms; the joint default vector $w\in\{0,1\}^M$ follows an **Ising/log-linear graphical model**, and they prove (via algebraic geometry) the model can represent *any* prescribed single-firm marginals and pairwise correlation matrix, with maximum-likelihood calibration and closed-form loss-distribution formulas.
- **DATA.** **No real data** — illustrated on synthetic graphs (a 3-firm triangle; a 12-firm, 3-sector example) and compared analytically against the one-factor Gaussian copula. (So it is a *modeling/theory* anchor, not a data source.)
- **PUBLIC/FREE vs PROPRIETARY.** N/A (no dataset).
- **Verifiable link.** arXiv:0809.1393.
- **Methodology we can reuse — this is the exact functional form of $\Pi(x)$.**
  - **Model (their Eq. 1).** $\;p_w(\eta)=\dfrac{1}{Z}\exp\!\Big(\sum_{i\in V}\eta_i\,w_i+\sum_{(u,v)\in E}\eta_{uv}\,w_u w_v\Big)$, with partition function $Z=\sum_{w\in\{0,1\}^M}\exp(\cdots)$. **Identify $\eta_i\leftrightarrow\ln p_i$ (the diagonal field) and $\eta_{uv}\leftrightarrow J_{uv}$.** This is literally our $\Pi(x)$ in the $\{0,1\}$ basis.
  - **Field ← marginal mapping.** *Implicit, not closed form.* They prove (Corollary 5) the MLE $\hat\eta$ is the **unique** parameter vector whose model marginals equal the empirical marginals $P_\bullet$. So $\eta_i$ are *not* simply $\ln p_i$ once couplings are on — they are the values that make $\langle x_i\rangle_{\text{model}}=p_i$ after accounting for $Z$. (Our $\ln p_i$ field is the *independent-baseline* / weak-coupling value; under strong $J$ it must be refit. See methods digest, "field correction.")
  - **Calibration.** Maximum likelihood = max-entropy subject to marginal constraints; solved by **Iterative Proportional Fitting** (Darroch–Ratcliff) for small models, **convex optimization / quasi-Newton (L-BFGS)** for larger, and **pseudolikelihood** for sparse graphs.
  - **Loss distribution.** Computed by **exact enumeration / explicit summation** (their Prop. 6); in the single-sector case it decomposes into a mixture of independent binomials (Prop. 7); multi-period via a Markov transition matrix. **No MCMC** is used (because they keep $n$ small / exploit sector exchangeability).

### B2. Molins & Vives (2005), *Long-range Ising model for credit risk* — the mean-field (infinite-range) analytic route
- **Citation.** J. Molins, E. Vives, "Long range Ising model for credit risk modeling in homogeneous portfolios," *AIP Conf. Proc.* 779, 156 (2005) / preprint arXiv:cond-mat/0401378. Later: J. Molins, E. Vives, "Model risk on credit risk," *Risk and Decision Analysis* (2016), arXiv:1502.06984.
  - arXiv: https://arxiv.org/abs/cond-mat/0401378
  - Follow-up: https://arxiv.org/abs/1502.06984
- **Method.** Model a homogeneous credit portfolio as a **finite-size, infinite-range (mean-field) Ising model** with uniform coupling $J/N$ and external field $H$; via the maximum-entropy principle this is the adequate model when default correlations are included, and exact analysis reveals a **first-order-like transition** (a sudden jump in portfolio risk as correlation rises) that standard credit models miss.
- **DATA.** Synthetic / illustrative parameter studies (homogeneous portfolio with prescribed $P_d,\rho_d$); not fit to a named real dataset.
- **PUBLIC/FREE vs PROPRIETARY.** N/A (no dataset).
- **Methodology we can reuse.**
  - **Hamiltonian.** $\;H_{\text{Ising}}=-\dfrac{J}{N}\sum_{i<j}\sigma_i\sigma_j-H\sum_i\sigma_i$, $\sigma_i\in\{-1,+1\}$ (default $\leftrightarrow$ a chosen sign).
  - **Field ← marginal, coupling ← correlation.** The **marginal default probability $P_d$ is set by the external field $H$** (through the equilibrium magnetization $\langle\sigma\rangle$), and the **default correlation $\rho_d$ is set by the coupling $J$**. Because the model is homogeneous/mean-field, both $P_d$ and $\rho_d$ and the normalization $Z$ are obtained **exactly** (finite-$N$ enumeration over the magnetization, or saddle-point/self-consistency in the large-$N$ limit). Useful for us as a **closed-form, exchangeable special case** to validate the generator: pick $(P_d,\rho_d)$, solve for $(H,J)$, and check the cascade tail against the analytic loss distribution.

### B3. Kitsukawa, Mori & Hisakado (2006) — long-range Ising for tranches; calibrated to a *real* index
- **Citation.** K. Kitsukawa, S. Mori, M. Hisakado, "Evaluation of Tranche in Securitization and Long-range Ising Model," *Physica A* **368**, 191–206 (2006). Preprint: arXiv:physics/0603040.
  - arXiv: https://arxiv.org/abs/physics/0603040
  - Related: S. Mori, K. Kitsukawa, M. Hisakado, "Correlation Structures of Correlated Binomial Models and Implied Default Distribution," *J. Phys. Soc. Jpn.* (2008), arXiv:physics/0609093. https://arxiv.org/abs/physics/0609093
- **Method.** Use the finite-$N$ long-range Ising model (coupling $J/N$, field $H$) as a homogeneous credit-portfolio model; derive perturbative closed forms for $P_d$, $\rho_d$, and $Z$ as functions of $(N,J,H)$, then price CDO tranches via the cumulative loss distribution.
- **DATA.** The companion 2008 paper compares model-implied default distributions against **iTraxx-CJ (iTraxx Japan) tranche** market quotes; the 2006 paper is primarily analytic with illustrative parameters.
- **SOURCE / provider.** iTraxx index tranche quotes (index administered by **Markit / IHS Markit**, now **S&P Global**).
- **PUBLIC/FREE vs PROPRIETARY.** **PROPRIETARY** (iTraxx tranche quotes are commercial market data).
- **Methodology we can reuse.** Same $(H\!\to\!P_d,\ J\!\to\!\rho_d)$ mean-field mapping as B2, but with **explicit perturbative expressions for $Z$, $P_d$, $\rho_d$** in $(N,J,H)$ — handy closed-form checks. Computation is analytic; the loss CDF follows from the exact homogeneous-portfolio distribution.

### B4. Emonti & Fontana (2025), *Negative correlations in Ising models of credit risk* — recent, explicit field/coupling↔moment formulas
- **Citation.** C. Emonti, R. Fontana, "Negative correlations in Ising models of credit risk," arXiv:2502.21199 (2025), Politecnico di Torino.
  - arXiv: https://arxiv.org/abs/2502.21199 (HTML: https://arxiv.org/html/2502.21199)
- **Method.** Work in the **"Jungle model" = Ising model** family for defaults; study a star-graph special case ("Dandelion model") and derive when default correlations can be driven **negative**, with closed-form moment relations.
- **DATA.** Synthetic only (illustrative $p=0.4$, $N=100$); no real portfolio.
- **PUBLIC/FREE vs PROPRIETARY.** N/A.
- **Methodology we can reuse.**
  - **Model.** $P(\ell_0,\ldots,\ell_n)=Z^{-1}\exp\!\big(\alpha_0\ell_0+\alpha\sum_i\ell_i+\beta\sum_i\ell_0\ell_i\big)$ over binary defaults $\ell$.
  - **Field ← marginal.** $\alpha_0,\alpha$ are fixed by the marginal default constraints $\mathbb{E}[L_0]=p_0$, $\mathbb{E}[L_i]=p$.
  - **Coupling ← correlation.** $\beta$ sets the pairwise interaction; the central-node default correlation is $\rho=\dfrac{q-p^2}{p(1-p)}$ with $q=\mathbb{E}[L_0L_i]$ (this is just the Pearson correlation of two Bernoullis — directly invertible to set $J$ from a target $\rho$). Computation is analytic.

---

## C. Maximum-entropy reconstruction of interbank / financial networks (where $J_{ij}$ comes from)

This thread answers: *given only partial exposure data (e.g. bank total assets/liabilities, or node degrees), how do you reconstruct the exposure/adjacency matrix that defines the couplings?* Our generator's $J_{ij}$ are meant to be set from "the exposure graph" — these are the methods to build/complete that graph.

### C1. Cimini, Squartini, Garlaschelli & Gabrielli (2015), *Systemic Risk Analysis on Reconstructed Economic and Financial Networks* — **the reconstruction workhorse**
- **Citation.** G. Cimini, T. Squartini, D. Garlaschelli, A. Gabrielli, "Systemic Risk Analysis on Reconstructed Economic and Financial Networks," *Scientific Reports* **5**, 15758 (2015). Preprint: arXiv:1411.7613.
  - Open access: https://pmc.ncbi.nlm.nih.gov/articles/PMC4623768/
  - Journal: https://www.nature.com/articles/srep15758
  - arXiv: https://arxiv.org/abs/1411.7613
- **Method.** **Fitness-induced configuration model (FiCM) / density-corrected gravity model (dc-GM):** use only node-specific "fitness" (e.g. total assets/liabilities) plus the overall link density to assign a probabilistic topology, then place weights — recovering realistic *sparse* networks (unlike naive MaxEnt, which yields a dense, fully-connected graph and underestimates contagion).
- **DATA.** (i) **World Trade Web (WTW)**, year ~2000; (ii) **e-MID** Italian interbank deposit market, ~1999.
- **SOURCE / provider.** WTW from **UN COMTRADE** (trade flows); e-MID from **e-MID S.p.A.** (Electronic Market for Interbank Deposits, Milan).
- **PUBLIC/FREE vs PROPRIETARY.** WTW/**COMTRADE = PUBLIC/FREE**; **e-MID = PROPRIETARY** (commercial/regulated interbank transaction data; access restricted for confidentiality).
- **Methodology we can reuse — the canonical coupling-from-fitness recipe.**
  - **Link probability (their key formula).** $\;p_{ij}=\dfrac{z\,\chi_i\,\psi_j}{1+z\,\chi_i\,\psi_j}$, where $\chi_i,\psi_j$ are node fitnesses (e.g. in-/out-strength proxies) and $z$ is a single global parameter fixed by matching the **expected total number of links to the known density**.
  - **Weights.** Where a link exists, expected weight $\;w_{ij}\approx \dfrac{\chi_i\psi_j\,W}{z\,\chi_i\psi_j}$ (degree-corrected gravity), with $W$ the total network weight — i.e. **strengths are preserved while topology stays sparse.**
  - **Use for us.** When we only have per-institution size data, build the exposure/adjacency matrix with this $p_{ij}$ (Bernoulli per edge) and gravity weights, then set $J_{ij}$ proportional to $w_{ij}$. This is the principled way to turn "one number per bank" into the coupling matrix our $\Pi(x)$ needs.

### C2. Squartini & Garlaschelli (2011), *Analytical maximum-likelihood method to detect patterns in real networks* — the underlying ERG/configuration-model engine
- **Citation.** T. Squartini, D. Garlaschelli, "Analytical maximum-likelihood method to detect patterns in real networks," *New J. Phys.* **13**, 083001 (2011). Preprint: arXiv:1103.0701.
  - arXiv: https://arxiv.org/abs/1103.0701
  - Journal: https://iopscience.iop.org/article/10.1088/1367-2630/13/8/083001
- **Method.** Exponential-random-graph (ERG) / **canonical configuration model** solved analytically by maximum likelihood: fit one "fitness" Lagrange multiplier per node so that **expected degrees equal observed degrees**.
- **DATA.** Multiple empirical networks for validation, including economic/trade networks; primarily a *methods* paper.
- **Methodology we can reuse.**
  - **Binary undirected configuration model link probability.** $\;p_{ij}=\dfrac{x_i x_j}{1+x_i x_j}$ (canonical/maximum-entropy form; this is the $z\to1$, $\chi=\psi=x$ specialization of C1's $p_{ij}$). The $x_i$ are set by ML so that $\sum_{j\ne i}p_{ij}=k_i^{\text{obs}}$ (observed degree).
  - **Why it matters.** This is the rigorous foundation behind C1 and is implemented in the open `NEMtropy` Python package (https://nemtropy.readthedocs.io/) — directly usable to generate/complete our synthetic exposure graphs.

### C3. Mastromatteo, Zarinelli & Marsili (2012), *Reconstruction of financial network for robust estimation of systemic risk*
- **Citation.** I. Mastromatteo, E. Zarinelli, M. Marsili, "Reconstruction of financial networks for robust estimation of systemic risk," *J. Stat. Mech.* (2012) P03011. Preprint: arXiv:1109.6210.
  - arXiv: https://arxiv.org/abs/1109.6210
  - SSRN: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1934766
- **Method.** Shows naive MaxEnt (dense) reconstruction **understates contagion**, and proposes a **message-passing (cavity) algorithm** to sample the *ensemble of sparse networks* consistent with the known marginals (row/column sums of the exposure matrix), giving a "maximally fragile" upper bound on contagion risk.
- **DATA.** **Synthetic** ensembles engineered to mimic real-network features (sparsity, heterogeneity); no proprietary dataset.
- **PUBLIC/FREE vs PROPRIETARY.** N/A (synthetic).
- **Methodology we can reuse.** **Belief-propagation / message-passing over network ensembles** is the scalable way to *draw many plausible exposure matrices* under partial-information constraints — conceptually parallel to our generator drawing many plausible default configurations. Key cautionary takeaway: **do not default to the dense MaxEnt graph; sparsity changes tail risk a lot.**

### C4. Squartini, Caldarelli, Cimini, Gabrielli & Garlaschelli (2018), *Reconstruction methods for networks* — the review/menu
- **Citation.** T. Squartini, G. Caldarelli, G. Cimini, A. Gabrielli, D. Garlaschelli, "Reconstruction methods for networks: the case of economic and financial systems," *Physics Reports* **757**, 1–47 (2018). Preprint: arXiv:1806.06941. https://arxiv.org/abs/1806.06941
- **Use.** Comprehensive comparison of reconstruction methods (dense MaxEnt/IPF, configuration models, fitness/dc-GM, message-passing); confirms **dc-GM systematically wins "horse races"** for interbank reconstruction. Our single best reference for *choosing* a coupling-graph reconstruction method.

---

## D. Umbrella review + inverse-Ising methods toolbox

### D1. Bardoscia et al. (2021), *The physics of financial networks* — field-defining review
- **Citation.** M. Bardoscia, P. Barucca, S. Battiston, F. Caccioli, G. Cimini, D. Garlaschelli, F. Saracco, T. Squartini, G. Caldarelli, "The physics of financial networks," *Nature Reviews Physics* **3**, 490–507 (2021). Preprint: arXiv:2103.05623.
  - Journal: https://www.nature.com/articles/s42254-021-00322-5
  - arXiv: https://arxiv.org/abs/2103.05623
  - Open repository (UZH/ZORA): https://www.zora.uzh.ch/id/eprint/208604/
- **Use / what it gives us.** The authoritative survey tying together (a) max-ent network reconstruction (configuration models, fitness/dc-GM, ERGM), (b) contagion dynamics (DebtRank, Eisenberg–Noe), and (c) the statistical-physics ensemble viewpoint. It also enumerates the **standard empirical datasets** of the field and their (mostly proprietary) provenance — see provenance table below. Note: it frames contagion mostly via epidemic/threshold-cascade language rather than an explicit Ising Hamiltonian, but the ensemble/max-ent thread is exactly ours.
- **Datasets named (providers; nearly all restricted):** e-MID (Italian interbank); Austrian interbank (OeNB); US Federal Funds (Federal Reserve); Mexican banking system (Banco de México); Belgian interbank (NBB); **Dutch Interbank Network (DNB — De Nederlandsche Bank)**; UK/EU OTC-derivatives trade repositories (regulatory); International Trade Network (COMTRADE, public).

### D2. Nguyen, Zecchina & Berg (2017), *Inverse statistical problems: from the inverse Ising problem to data science* — the how-to-fit-and-sample reference
- **Citation.** H. C. Nguyen, R. Zecchina, J. Berg, "Inverse statistical problems: from the inverse Ising problem to data science," *Advances in Physics* **66**(3), 197–261 (2017). Preprint: arXiv:1702.01522. https://arxiv.org/abs/1702.01522
- **Use.** The definitive review of **how to infer $J_{ij}$ from correlations** and the tradeoffs of each method. It is our methods backbone for the coupling-from-correlation map and for choosing a sampler.
- **Methods catalog (with our usage notes):**
  - **Naive mean-field (nMF):** $J_{ij}\approx -(C^{-1})_{ij}$ for $i\ne j$ (inverse of the connected-correlation/covariance matrix). $O(n^3)$, instant, but degrades at low temperature / strong correlation.
  - **TAP:** adds the Onsager reaction term; $J_{ij}$ solves a quadratic involving $C^{-1}$. Slightly better, still cheap.
  - **Sessak–Monasson small-correlation expansion;** **independent-pair approximation** — other closed-form upgrades.
  - **Pseudo-likelihood maximization (PLM):** consistent, scalable, the de-facto standard for accurate inference; needs the full configurations, not just moments.
  - **Boltzmann machine learning (exact gradient):** most accurate, but each step needs the model moments by MCMC — feasible only at modest $n$.
  - **Adaptive cluster expansion; Bethe/Bethe-Peierls (message passing)** for sparse graphs.

---

## E. PROVENANCE TABLE — {work → dataset → provider → FREE/PROPRIETARY → URL}

| Work | Dataset | Provider | FREE / PROPRIETARY | URL |
|---|---|---|---|---|
| Bury 2013 (1210.8380 / 1206.4420) | BEL20, AEX, DAX, CAC40, Dow Jones, S&P 100 daily/minute returns | **Yahoo! Finance** | **FREE/PUBLIC** | https://finance.yahoo.com |
| Bury 2013 | Onnela NYSE return set | J.-P. Onnela (academic posting) | **FREE/PUBLIC** | https://www.jponnela.com |
| Schneidman et al. 2006 | Retinal ganglion neuron spike recordings | Weizmann/lab (biological) | N/A to finance (academic) | https://weizmann.elsevierpure.com/en/publications/weak-pairwise-correlations-imply-strongly-correlated-network-stat |
| Filiz et al. 2012 | None (synthetic 3- & 12-firm examples) | — | N/A | https://arxiv.org/abs/0809.1393 |
| Molins & Vives 2005 | None (synthetic homogeneous portfolio) | — | N/A | https://arxiv.org/abs/cond-mat/0401378 |
| Kitsukawa et al. 2006 / Mori et al. 2008 | iTraxx-CJ (Japan) index tranche quotes | **Markit / IHS Markit (now S&P Global)** | **PROPRIETARY** | https://arxiv.org/abs/physics/0603040 |
| Emonti & Fontana 2025 | None (synthetic) | — | N/A | https://arxiv.org/abs/2502.21199 |
| Cimini et al. 2015 | World Trade Web (~2000) | **UN COMTRADE** | **FREE/PUBLIC** | https://comtradeplus.un.org |
| Cimini et al. 2015 | e-MID Italian interbank (~1999) | **e-MID S.p.A.** | **PROPRIETARY** | https://www.e-mid.it |
| Squartini & Garlaschelli 2011 | Various empirical (incl. trade) networks | mixed (mostly public) | mostly **FREE** | https://arxiv.org/abs/1103.0701 |
| Mastromatteo et al. 2012 | Synthetic network ensembles | — | N/A | https://arxiv.org/abs/1109.6210 |
| Bardoscia et al. 2021 (review) | e-MID; Austrian (OeNB); US Fed Funds; Mexican (Banxico); Belgian (NBB); **Dutch (DNB)**; EU OTC repos; COMTRADE | central banks / regulators / UN | **mostly PROPRIETARY**; COMTRADE **FREE** | https://arxiv.org/abs/2103.05623 |

---

## F. FREE / PUBLIC datasets to actually use (names + URLs)

- **UN COMTRADE — World Trade Web / International Trade Network.** Bilateral trade flows between countries; the standard *public* testbed for fitness/dc-GM reconstruction. https://comtradeplus.un.org (legacy: https://comtrade.un.org)
- **Yahoo! Finance historical equity prices.** Free constituent-level daily returns to binarize for a Bury-style co-movement Ising model (e.g. S&P 100, DAX, CAC40). https://finance.yahoo.com (programmatic via the community `yfinance` package)
- **Onnela NYSE return dataset.** Cleaned multi-year NYSE daily returns used in econophysics correlation/structure studies. https://www.jponnela.com
- **NEMtropy (software, not data).** Open Python implementation of the configuration-model / fitness max-ent reconstruction (Squartini–Garlaschelli family) — lets us *generate* synthetic exposure graphs and fit $p_{ij}$. https://nemtropy.readthedocs.io/  (PyPI: `NEMtropy`)
- **ConIII (software).** Open Python package for solving inverse-Ising / pairwise max-ent models (fields & couplings from data). arXiv:1801.08216; https://github.com/eltrompetero/coniii

> Reality check on interbank data: **the actual bilateral interbank exposure matrices are essentially all proprietary/regulator-only** (e-MID, DNB, OeNB, Banxico, Fed, NBB). For a hackathon, the *public* path is: (a) build the network synthetically with NEMtropy/dc-GM from public node sizes, or (b) use COMTRADE as a real, public, fitness-reconstructable network, then overlay our default/cascade model. This matches our repo's existing **synthetic network generator** design.

---

## G. METHODS DIGEST FOR OUR GENERATOR (the important part)

Our $\Pi(x)=\sum_i \ln p_i\, x_i+\sum_{i<j}J_{ij}x_i x_j$ is a pairwise max-ent / Boltzmann model over $x\in\{0,1\}^n$. Here is the recommended, source-attributed recipe for each piece.

### G1. Field ↔ marginal mapping ($h_i$ from $p_i$)
- **Independent baseline (weak coupling), closed form.** If $J=0$, then $x_i$ are independent Bernoulli$(p_i)$ and the field that reproduces marginal $p_i$ in the $\{0,1\}$ form is the **logit**:
  $$h_i^{(0)}=\operatorname{logit}(p_i)=\ln\frac{p_i}{1-p_i}.$$
  Our "$\ln p_i$" is the small-$p_i$ approximation of this (since $\ln\frac{p_i}{1-p_i}\approx \ln p_i$ for rare defaults). **Use $\operatorname{logit}(p_i)$ for correctness**, especially if any $p_i$ is not tiny. *(Standard Bernoulli/auto-logistic identity; the auto-logistic/log-linear form is exactly Filiz et al. 2012, Eq. 1; the logit field is the canonical $\{0,1\}$ max-ent field.)*
- **Coupling-corrected field (do this once $J\ne0$).** With couplings on, the *marginal* $\langle x_i\rangle$ no longer equals $\sigma(h_i^{(0)})$, so the fields must be **refit so model marginals match the targets $p_i$** — this is the unique-MLE statement of **Filiz et al. 2012 (Cor. 5)** and the Boltzmann-learning update of **Schneidman et al. 2006** ($\Delta h_i\propto p_i-\langle x_i\rangle_{\text{model}}$). Practically: initialize $h_i=\operatorname{logit}(p_i)$, then run a few Boltzmann-learning iterations (or a mean-field correction) to restore the marginals. For the **homogeneous/mean-field case**, $h_i$ (the field $H$) is obtained exactly from $P_d$ via the self-consistent magnetization (**Molins–Vives 2005; Kitsukawa et al. 2006**).

### G2. Coupling ↔ {correlation, exposure} mapping ($J_{ij}$)
Two regimes, depending on whether we have correlations or an exposure graph:

- **(a) $J_{ij}$ from a target correlation matrix (inverse-Ising).**
  - **Fast / default:** **naive mean-field**, $J_{ij}\approx-(C^{-1})_{ij}$ ($i\ne j$), $C$ = connected-correlation matrix; **TAP** for a better estimate (quadratic in $C^{-1}$). *(Bury 2013 uses exactly the 2nd-order/TAP inversion; catalogued in Nguyen–Zecchina–Berg 2017.)*
  - **Accurate / scalable:** **pseudo-likelihood maximization (PLM)** or **Boltzmann learning** when correlations are strong (mean-field breaks down at "low temperature"). *(Nguyen–Zecchina–Berg 2017.)*
  - **Single-pair sanity check:** for two Bernoullis, $\rho_{ij}=\dfrac{q_{ij}-p_ip_j}{\sqrt{p_i(1-p_i)p_j(1-p_j)}}$ with $q_{ij}=\langle x_ix_j\rangle$ — invert to set a pairwise $J$. *(Emonti–Fontana 2025.)*
- **(b) $J_{ij}$ from an exposure/adjacency matrix (our repo's framing).** Reconstruct the exposure graph with the **fitness / density-corrected gravity model**:
  $$p_{ij}=\frac{z\,\chi_i\,\psi_j}{1+z\,\chi_i\,\psi_j},\qquad w_{ij}\propto \chi_i\psi_j,$$
  ($\chi,\psi$ = node sizes/strengths; $z$ fixed by overall density), then set $J_{ij}\propto w_{ij}$ (stronger mutual exposure → larger positive coupling → more co-default). *(Cimini et al. 2015; reduces to Squartini–Garlaschelli 2011 configuration model $p_{ij}=x_ix_j/(1+x_ix_j)$ when $z=1,\chi=\psi=x$. dc-GM is the recommended choice per the Squartini et al. 2018 review.)* If only marginals (row/col sums) are known and sparsity matters for tail risk, use **message-passing reconstruction** (**Mastromatteo et al. 2012**) instead of the dense MaxEnt graph.
  - **Sign caveat:** real exposure-driven couplings are positive (co-failure), but star/hub topologies can induce *negative* pairwise correlations among leaves — keep the cascade simulator, not just $\Pi$, as the arbiter of joint tail probabilities. *(Emonti–Fontana 2025.)*

### G3. Sampling methods by system size $n$ (feasibility, incl. up to $n=54$)
Sampling $\Pi(x)$ is the genuinely hard step (the partition function $Z=\sum_{x\in\{0,1\}^n}$ has $2^n$ terms). Recommendation by size, attributed:

| $n$ | Exact / partition function | Recommended classical sampler | Notes / source |
|---|---|---|---|
| $\le \sim$20 | **Exact enumeration** of all $2^n$ states ($2^{20}\approx10^6$) — compute $Z$, marginals, exact tail probabilities | exact (no sampling needed) | Filiz et al. 2012 compute loss distributions by exact summation; Schneidman et al. fit small $n$ by enumeration. |
| $\sim$20–40 | Enumeration infeasible ($2^{40}\approx10^{12}$) | **Gibbs / Metropolis MCMC** (single-spin-flip), with thinning; Boltzmann learning for fitting | Standard Ising MC; method of Schneidman et al. 2006 / Nguyen–Zecchina–Berg 2017. |
| $\sim$40–54+ | Hopeless exactly ($2^{54}\approx1.8\times10^{16}$) | **MCMC (Gibbs/Metropolis, or cluster/parallel-tempering for strong coupling)**; **mean-field/TAP** for moments; **message-passing** if sparse | This is precisely where the *classical* sampler is expensive and where the **QCBM is pitched as the quantum sampler/state-loader** (one qubit per institution, $n=54$ qubits). Classical fallback: parallel tempering to cross the first-order transition flagged by Molins–Vives 2005. |
| Homogeneous / mean-field $\Pi$ (uniform $J$) | **Exact at any $n$** via the magnetization (1-D sum over number-of-defaults) | none needed | Molins–Vives 2005; Kitsukawa et al. 2006 — use as an exact $n=54$ validation oracle for the generator. |

**Key feasibility statements for $n=54$.**
- *Classically:* exact enumeration is impossible; **MCMC is the only general option, and it is slow/biased near the correlation-driven first-order transition** (critical slowing down) — use **parallel tempering / cluster moves** and expect long autocorrelation times. Mean-field/TAP gives moments cheaply but not faithful tail samples. (Nguyen–Zecchina–Berg 2017; Molins–Vives 2005 for the transition.)
- *The exchangeable special case is a free exact check:* if we set a uniform field and uniform $J$, the loss distribution at $n=54$ is computable exactly by summing over the 55 possible default counts (Molins–Vives 2005; Kitsukawa et al. 2006) — ideal ground truth to validate both the classical MCMC generator and the QCBM.
- *Quantum link (cross-section, for context):* the entangled QCBM/qGAN is exactly the "learn-and-load the correlated distribution into a state for amplitude estimation" idea of **Zoufal, Lucchi & Woerner, "Quantum Generative Adversarial Networks for Learning and Loading Random Distributions," *npj Quantum Information* 5, 103 (2019), arXiv:1904.00043** (https://arxiv.org/abs/1904.00043) — our $\Pi(x)$ is the target distribution it would load. (Detailed quantum treatment belongs in the quantum section; noted here only because it is *the* downstream consumer of this sampling problem.)

### G4. One-paragraph implementation recommendation
Set $h_i=\operatorname{logit}(p_i)$; build the coupling matrix either from a correlation target via TAP/PLM (regime a) or, matching our repo, from a fitness/dc-GM exposure reconstruction $p_{ij}=z\chi_i\psi_j/(1+z\chi_i\psi_j)$ with $J_{ij}\propto w_{ij}$ (regime b); run a short Boltzmann-learning correction so model marginals hit the target $p_i$; sample with exact enumeration for $n\le20$, Gibbs/Metropolis (parallel-tempered if couplings are strong) for $n\gtrsim20$; and **validate against the exact homogeneous-Ising loss distribution** (uniform $h,J$) at the same $n$, which is closed-form at any size including $n=54$.

---

## H. Verification / honesty notes
- All citations, venues, and arXiv IDs above were retrieved and cross-checked against arXiv/journal/RePEc records during this survey. Formula extractions for **Filiz et al. (Eq. 1, Cor. 5, Props. 6–7)**, **Bury (TAP inversion, $I_2/I_n\approx98\%$, Yahoo!/Onnela data)**, **Cimini et al. (dc-GM $p_{ij}$, WTW+e-MID)**, **Molins–Vives (mean-field $H_{\text{Ising}}$, $H\!\to\!P_d$, $J\!\to\!\rho_d$)**, and **Emonti–Fontana (Dandelion $P$, $\rho=(q-p^2)/p(1-p)$)** are taken from the papers' own text (ar5iv/PMC/HTML renderings).
- The **binary configuration-model link probability $p_{ij}=x_ix_j/(1+x_ix_j)$** is the standard canonical/maximum-entropy form (Park–Newman; Squartini–Garlaschelli 2011) and the $z=1$ specialization of the dc-GM formula explicitly extracted from Cimini et al. 2015 — stated as established, not novel.
- The **logit field $h_i=\operatorname{logit}(p_i)$** is the elementary Bernoulli/auto-logistic identity (the $\{0,1\}$-basis max-ent field), consistent with Filiz et al.'s log-linear form; presented as standard.
- Nothing above is marked [UNVERIFIED]; where a paper had **no dataset** (Filiz, Molins–Vives, Emonti–Fontana, Mastromatteo) this is stated explicitly rather than inferred.
