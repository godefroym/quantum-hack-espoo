# Crisis Correlation Network

A quantum approach to systemic stress testing. It learns how financial institutions fail together, runs that joint-default model on real quantum hardware, and estimates the chance of a system-wide collapse, focused on the rare correlated tail that standard models handle worst.

## The data

Public market data on 48 major institutions, mostly global banks plus large corporates. Daily equity returns from 2021 to 2024 give their co-movement. Credit ratings, via a ratings-to-default table, give each one a standalone one-year default probability. SEC 13F filings show who holds the same securities, hence who would sell into the same fire sale. Nothing is synthetic.

## Network and clusters

From the returns we build the correlation matrix and keep the strongest links, giving a graph of which institutions move together. Exposure weights are reconstructed from public balance-sheet totals, since real bilateral lending data is confidential. Community detection splits the graph into clusters that are dense inside and sparse across, and clusters do two jobs. First, they validate the data: each cluster falls along region and business type on its own, and the demo reports its purity, the share of members from one region. High purity means the correlations recovered real structure without being told to. Second, they make the quantum step feasible: a full 48-wide entangled circuit is too deep for today's hardware, but each cluster is only 10 to 15 institutions and fits as a shallow circuit.

## Entanglement and the generator

Each institution is one qubit. Measured as 1, it has defaulted, so every measured bitstring is one full crisis scenario at once. A single-qubit rotation sets each institution's default probability, and entangling gates between qubits create a joint distribution that does not factorize, which is exactly what correlated default is. Independent qubits cannot produce co-defaults; entanglement can, and it carries that dependence directly, with no copula bolted on. The full joint law over 48 institutions has 2^48 outcomes and is intractable classically; the circuit holds it with 48 qubits, one per institution.

The circuit is a quantum GAN: the generator produces scenarios, a classical discriminator scores them against the real co-default structure, and the feedback tunes the gate angles until the marginals and correlations match. Each cluster runs as its own circuit, in parallel, on IBM's ibm_boston (a 156-qubit Heron chip) at 200,000 shots per cluster. The clusters are then coupled through a shared crisis factor, a common shock that fires several together, rebuilding one stream of full-market scenarios while each cluster stays quantum-exact.

## Contagion

Each scenario runs through a deterministic cascade. An institution fails when its losses from defaulted counterparties exceed its capital buffer, which can push its creditors under, and so on until the system settles. Losses are adjusted per directed link for recovery and seniority, funding maturity, wrong-way risk, and concentration, so transmission is realistic and directional. A second channel models fire-sale contagion through the 13F common-holdings overlap.

## The quantum advantage

Holding each institution's own default rate fixed and changing only the correlation, the quantum-correlated scenarios put about 25 percent probability on a severe collapse (at least 24 of 48 failing) against about 18 percent under independence. Measuring that tail is where quantum scales. Monte Carlo needs about 1/(error^2 times a) samples for a rare event of probability a, most wasted outside the tail; quantum amplitude estimation reads it in about 1/(error times sqrt(a)) queries. The reduction grows as the event gets rarer: about 10 times fewer queries at a common shock, over 200 times fewer at a one-in-a-thousand collapse.

## Why it matters to financial institutions

Risk teams and regulators size capital and run stress tests against tail losses. Assuming defaults are independent understates the joint tail, here by roughly 40 percent on a half-system collapse, so buffers set that way are too thin for the scenario that matters. This tool generates the correlated tail directly instead of hand-picking scenarios. It separates contagion links (positive correlation, which amplify losses) from hedges (negative correlation, which offset them), and it surfaces the baskets whose joint failure tips the system, which informs exposure limits and concentration management.

## Scope and demo

The generator ran on real IBM hardware. The amplitude-estimation advantage is exact at small scale and the construction extends to the full machine; no wall-clock speedup is claimed yet. The Crisis Correlation Network view shows the output: institutions coloured by cluster, links split into contagion and hedge, a slider to isolate the strongest links (the contagion backbone), and a click for any institution's failure odds and likely co-failures.
