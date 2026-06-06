# Huang Bank-Asset Cascade

This module implements the bank-asset fire-sale mechanism in:

X. Huang, I. Vodenska, S. Havlin, and H. E. Stanley, "Cascading
Failures in Bi-partite Graphs: Model for Systemic Risk Propagation,"
Scientific Reports 3, 1219 (2013), DOI: 10.1038/srep01219.

The synthetic data helper uses the paper's 13 asset categories and reported
average portfolio weights. It does not reproduce the proprietary WRDS bank
observations used in the paper.

## Role In This Project

The primary MVP benchmark remains the deterministic fixed-point exposure
cascade in `simulator/cascade.py`. This Huang implementation is an optional
robustness engine for a second classical contagion channel: overlapping
portfolios and fire-sale price impact.

It does not change the generator comparison. B and C still produce the same
binary initial-default vectors and must match the same marginal and pairwise
targets.

## State

Let:

- \(B_{i,m}\) be the pre-shock book value of asset class \(m\) held by bank \(i\).
- \(L_i\) be bank \(i\)'s liabilities.
- \(A_m=\sum_i B_{i,m}\) be the initial market value of asset class \(m\).
- \(q_m(t)\in[0,1]\) be its price factor relative to the pre-shock price.

The marked-to-market asset value of bank \(i\) is:

\[
V_i(t)=\sum_m B_{i,m}q_m(t).
\]

## Initial Shock

An exogenous shock specifies \(p_m\in[0,1]\):

\[
q_m(0)=p_m.
\]

For example, \(p_m=0.6\) is a 40 percent write-down of asset class \(m\).

## Distress Barrier

For each bank, draw one tolerance:

\[
r_i\sim U(0,\eta), \qquad 0\leq\eta\leq0.5.
\]

Bank \(i\) fails when:

\[
V_i(t)<(1-r_i)L_i.
\]

Marginalizing over \(r_i\) gives the paper's piecewise failure probability:

\[
P_i(V_i,L_i)=
\begin{cases}
0,&V_i\geq L_i,\\
\frac{L_i-V_i}{\eta L_i},
  &(1-\eta)L_i<V_i<L_i,\\
1,&V_i\leq(1-\eta)L_i.
\end{cases}
\]

When \(\eta=0\), failure is deterministic at \(V_i<L_i\).

The implementation samples \(r_i\) once per simulation. This preserves the
paper's failure probability while avoiding repeated random trials for a bank
whose balance sheet has not changed.

## Fire-Sale Impact

When banks fail, liquidation reduces the market value of every asset they
hold. The code uses the cumulative, order-independent form of the paper's
per-bank deduction:

\[
q_m(t+1)=\max\left(
0,\,
p_m-\alpha_m\,
\frac{\sum_{i\in F(t+1)}B_{i,m}}{A_m}
\right),
\]

where \(F(t)\) is the cumulative set of failed banks and
\(\alpha_m\in[0,1]\) controls market illiquidity.

- \(\alpha_m=0\): liquidation has no price impact.
- \(\alpha_m=1\): a failed bank's entire market share is deducted.

The process alternates between bank revaluation and asset price deductions
until no additional bank fails.

## Scenario-Generator Bridge

`simulate_huang_scenarios` accepts binary arrays of shape
`(n_scenarios, n_banks)`. A one means that the bank is an exogenous initial
default. This is an extension of the paper that allows the Bernoulli, Gaussian
copula, Student-t copula, and quantum generators to use the same fire-sale
contagion engine.

`bank_asset_to_system_spec` builds the common generator specification from the
same bank-asset balance sheets:

\[
W_{ij}^{\mathrm{first\ order}}
=
\sum_m B_{i,m}\alpha_m\frac{B_{j,m}}{A_m}.
\]

This is the immediate marked-to-market loss to bank \(i\) if bank \(j\)
liquidates. Pairwise default targets are derived from a positive-semidefinite
cosine-overlap kernel over portfolio weights. Marginal default probabilities
can be supplied from external data; the demonstration otherwise calibrates a
transparent equity-ratio vulnerability score to a requested system-wide mean.

The adapter is an experimental bridge, not a claim that these heuristic
marginals reconstruct the historical 2008 default law.

## Example

```bash
uv run python scripts/run_huang_2008_demo.py
uv run python scripts/compare_generators_huang.py
```
