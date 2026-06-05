# quantum-hack-espoo

## Problem: Maximal Plausible Cascade Set

**Graph.** Financial exposure network $G=(V,E,w)$ with entities $V=\{1,\dots,n\}$ (banks, funds, CCPs, corporates, sovereigns), exposure weights $w_{ij}\ge0$ (loss to $i$ if $j$ defaults), and capital buffers $c_i>0$.

**Cascade.** For seed $x\in\{0,1\}^n$, default propagates until fixed point $s(x)$, where $i$ fails iff $\sum_{j\in N(i)} w_{ij}\,s_j > c_i$. Cascade size: $F(x)=\sum_i s_i(x)$.

**Plausibility.** $\Pi(x)=\sum_{i:x_i=1}\ln p_i + \sum_{i<j} J_{ij}\,x_i x_j$, with $p_i$ the marginal default probability and $J_{ij}$ rewarding empirically correlated co-failure.

**Objective.** Worst *plausible* nightmare, not worst possible:

$$x^\star=\arg\max_{x\in\{0,1\}^n} F(x)\quad\text{s.t.}\quad \Pi(x)\ge\tau,\;\;\textstyle\sum_i x_i\le k.$$

Budget $k$ = simultaneous shocks; threshold $\tau$ = believability.

**Hardness.** $x\mapsto F(x)$ is a nonlinear, nonlocal fixed point over a combinatorial feasible set; the decision version is NP-hard (embeds influence-maximization).

**QUBO / QAOA.** Linearizing propagation and folding constraints as penalties:

$$H = -\sum_i s_i + \lambda_1\big(\tau-\Pi(x)\big)_+ + \lambda_2\big(\textstyle\sum_i x_i - k\big)_+^2,$$

a quadratic pseudo-Boolean cost Hamiltonian $H_C$ over $n$ qubits (one per entity).

**Entanglement is the encoding.** Each quadratic term $w_{ij},J_{ij}$ generates a two-qubit gate $e^{-i\gamma\theta_{ij}Z_iZ_j}$ on the linked pair. The circuit's entanglement graph *is* $G$: qubits $i,j$ entangle iff entities $i,j$ are financially coupled. Separable subsystems = independent clusters; entangled subsystems = correlated risk. The state's correlations are the contagion correlations.

> **Caveat.** $H_C$ captures few propagation rounds; the true $F(x)$ is the full iterated fixed point. We restrict to the linearized threshold model.

# Easy words 

Risk assessment over financial entities. Do not need to stick exactly with the above formulation.
