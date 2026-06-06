# `data_network/` — Part A: data and exposure network

Owns sourcing and shaping the real-world input into the canonical system spec: nodes
(institutions), weighted edges (reconstructed exposures), per-node marginals, the pairwise
correlation matrix, community structure, a documented feature schema, and provenance.

## Pipeline

```
sources/  ->  clean  ->  estimate  ->  reconstruct  ->  cluster  ->  assemble  ->  validate
```

| Module | Role |
|---|---|
| `sources/roster.py` | the real anchor: 38 real entities — 28 banks + 10 corporates (`data/external/banks/gsib_roster.csv`) |
| `sources/equity_returns.py` | real daily equity-return correlation (Yahoo) + committed snapshot |
| `sources/synthetic.py` | calibrated-synthetic source, lifts `make_scalable_system(n≤54)` into a `NetworkSpec` |
| `sources/holdings_13f.py` | 13F holdings → portfolio-overlap (common-asset / fire-sale) network: **streaming sampler** (carves one quarter out of the ~5 GB `holdings.csv`), matrix, cosine + statistically-validated overlap, liquidity-weighted & directed fire-sale matrices, CRSP illiquidity |
| `clean.py` | normalization + ID reconciliation; `node_type` (bank/corporate); S&P rating → whole-letter PD bucket |
| `estimate.py` | marginals `p_i` (Moody's PD table), correlation (equity), interbank/borrowing totals + buffers |
| `reconstruct.py` | bilateral exposures — `max_entropy` (RAS/IPF) \| `min_density` (Anand-style), pluggable |
| `cluster.py` | greedy-modularity community detection + perturbation-ARI stability |
| `assemble.py` | layers → risk-adjusted effective edges → content-hashed `NetworkSpec` → flat `SystemSpec` |
| `validate.py` | round-trip + cluster-stability + B/C/D contract conformance |
| `spec.py` | `EmpiricalLayer`, `ReconstructedLayer`, `FeatureSchema`, `Provenance`, `NetworkSpec` |

The reconstructed bilateral notional is risk-adjusted into a **directed effective-loss matrix**
by the shared `systemic_risk.edge_metrics` module — each edge scaled by loss-given-default
(recovery / seniority / collateralization), maturity / rollover stress, wrong-way risk, and
concentration / substitutability. The cascade propagates this effective matrix. Corporates
borrow from banks but do not lend interbank, so the graph carries directed **bank → corporate**
exposures.

## The canonical spec

`NetworkSpec` is the frozen source-of-truth object:

- **`EmpiricalLayer`** — frozen ground truth (marginals, equity correlation, balance-sheet
  totals, capital buffers, categorical node attributes). Round-trips exactly.
- **`ReconstructedLayer`** — swappable bilateral exposures + `method` tag + `method_params`.
- **`FeatureSchema`** — every field's meaning, level, dtype, and *which consumer may read it*.
- **`Provenance`** — source string, fit params, SHA-256 content hash.

It round-trips losslessly (`to_json` / `from_json`), exposes consumer-scoped projections
(`view_for("generator" | "simulator" | "visualization")`), and **assembles down** into the
existing flat `systemic_risk.spec.SystemSpec` (`to_system_spec()`) that parts B/C/D consume —
so this layer blends in without changing the consumer contract.

## Run & check

```bash
# build the real network spec end-to-end (build + validate + render the community plot)
uv run python scripts/build_system_spec.py
uv run python scripts/build_system_spec.py --method min_density   # sparse reconstruction
uv run python scripts/build_system_spec.py --refresh-equity       # re-fetch the correlation

# the A end-to-end test: load raw -> valid spec -> round-trip -> stable clusters -> B/C/D
uv run pytest tests/test_data_network.py -q
```

```python
from systemic_risk.data_network import build_network_spec, build_system_spec
from systemic_risk.data_network.validate import validate_spec

nspec = build_network_spec()                 # frozen, content-hashed NetworkSpec
print(validate_spec(nspec).ok)               # True
system = nspec.to_system_spec()              # the flat SystemSpec B/C/D consume

# scaling source (beyond the real roster, up to the 54-qubit target)
from systemic_risk.data_network.sources import synthetic_network_spec
big = synthetic_network_spec(n=54, seed=11)
```

## 13F portfolio-overlap (common-asset / fire-sale) layer

A *separate* contagion channel whose nodes are **13F asset managers** (not the bank roster).
The raw `holdings.csv` (~5 GB, keyed by CRSP `permno`) is streamed once into a small
single-quarter slice, then turned into validated-overlap + fire-sale matrices:

```bash
uv run python scripts/build_13f_overlap.py --rdate 2008-09-30 --top 250
uv run pytest tests/test_holdings_13f.py -q
```

Outputs → `outputs/data_network/overlap_13f/` (matrices `.npz`, `summary.json`, backbone plot).
See `data/external/holdings_13f/README.md`. It is kept standalone (different node universe)
rather than fused into `NetworkSpec`.

## Notes on honesty

- Real **bilateral** interbank exposures are confidential everywhere; reconstructing the
  matrix from public per-node marginals is the field-standard move (Upper & Worms 2004;
  Anand, Craig & von Peter 2015), not a shortcut.
- Equity-return correlation is an **asset / latent** correlation, not a realised default
  correlation. The copula baselines threshold it by the marginals to produce co-default
  structure (Merton / single-factor ASRF reading).
- Total-assets figures in the roster are approximate public values used only as a relative
  balance-sheet *scale* for reconstruction — never as marginals.
