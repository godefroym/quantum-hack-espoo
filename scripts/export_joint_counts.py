"""Export the cross-cluster n-bit joint default-scenario counts (CSV) and a histogram.

The joint is the reconciled global sample from the stressed hardware run: each shot is an
n-bit correlated default scenario (1 = institution defaults post-cascade input), where n and
the cluster count are read from the run's reconciled NPZ (no fixed network size assumed).
NOTE: this joint is classically reconciled from the per-cluster device samples, not a native
n-qubit device measurement -- see the run's analysis notes.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from systemic_risk.data_network import build_network_spec

OUTDIR = ROOT / "outputs" / "real_cluster_mixture_stress_hw"

g = np.load(OUTDIR / "reconciled_global_stress.npz")
joint = g["reconciled"]            # (n_shots, n) int {0,1}
labels = g["labels"]               # cluster id per column
n_shots, n = joint.shape
severe = int(np.ceil(0.5 * n))     # "severe" = at least half the system defaults

# --- qubit legend: which bank/company each bit position represents ---
# Bit order in every bitstring below is positional: leftmost char = qubit 0,
# rightmost = qubit n-1, matching the node order of build_network_spec().
emp = build_network_spec().empirical
# Blocs are derived from each cluster's dominant member region(s), so this stays
# correct for any partition (k, sizes) rather than the old hand-labelled k=3 map.
from collections import Counter as _Counter
_node_region = [emp.node_attributes[t].get("region", "?") for t in emp.node_ids]
_k = int(labels.max()) + 1
BLOC = {}
for _c in range(_k):
    _regs = [_node_region[i] for i in range(n) if int(labels[i]) == _c]
    _top = "+".join(r for r, _ in _Counter(_regs).most_common(2))
    BLOC[_c] = f"{chr(65 + _c)}:{_top}"
legend = []
for i, ticker in enumerate(emp.node_ids):
    a = emp.node_attributes[ticker]
    legend.append({
        "qubit": i,
        "ticker": ticker,
        "name": a.get("name", ""),
        "node_type": a.get("node_type", ""),
        "business_type": a.get("business_type", ""),
        "region": a.get("region", ""),
        "cluster": int(labels[i]),
        "bloc": BLOC.get(int(labels[i]), str(int(labels[i]))),
    })

legend_csv = OUTDIR / "qubit_legend.csv"
with legend_csv.open("w") as fh:
    fh.write("qubit,ticker,name,node_type,business_type,region,cluster,bloc\n")
    for r in legend:
        fh.write(f'{r["qubit"]},{r["ticker"]},"{r["name"]}",{r["node_type"]},'
                 f'{r["business_type"]},{r["region"]},{r["cluster"]},{r["bloc"]}\n')
(OUTDIR / "qubit_legend.json").write_text(json.dumps(legend, indent=2))

# --- per-bitstring joint counts -> CSV ---
keys = ["".join(map(str, row)) for row in joint]
counts = Counter(keys)
rows = counts.most_common()        # sorted by count desc
csv_path = OUTDIR / "joint_counts.csv"
ticker_order = " ".join(r["ticker"] for r in legend)
with csv_path.open("w") as fh:
    fh.write(f"# bit order (left->right, qubit 0->{n - 1}): {ticker_order}\n")
    fh.write("# see qubit_legend.csv for the bank/company behind each position\n")
    fh.write("bitstring,count,probability,n_defaults\n")
    for bs, ct in rows:
        fh.write(f"{bs},{ct},{ct / n_shots:.8f},{bs.count('1')}\n")

# --- histograms ---
defaults_per_shot = joint.sum(axis=1)            # Hamming weight per scenario
hist = np.bincount(defaults_per_shot, minlength=n + 1)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# (a) THE correlated-failure histogram: how many institutions fail together per scenario
axes[0].bar(np.arange(n + 1), hist, color="#b3202c", width=0.9)
mean_def = defaults_per_shot.mean()
p_severe = (defaults_per_shot >= severe).mean()
axes[0].axvline(mean_def, color="k", ls="--", lw=1,
                label=f"mean = {mean_def:.1f} defaults")
axes[0].axvline(severe, color="#444", ls=":", lw=1.2,
                label=f"severe ≥{severe}  (P={p_severe:.2f})")
axes[0].set_xlabel(f"number of institutions defaulting in a scenario (of {n})")
axes[0].set_ylabel("shot count")
axes[0].set_title("(a) Cross-cluster joint: correlated-failure distribution")
axes[0].legend(frameon=False)

# (b) top joint scenarios by count (literal joint counts)
topN = 25
top = rows[:topN]
labels = [f"{bs.count('1')}d" for bs, _ in top]   # label by #defaults (38-bit string too long)
axes[1].bar(range(topN), [ct for _, ct in top], color="#2c5fb3")
axes[1].set_xticks(range(topN))
axes[1].set_xticklabels(labels, rotation=90, fontsize=7)
axes[1].set_xlabel(f"top {topN} joint scenarios (labelled by # defaults)")
axes[1].set_ylabel("shot count")
axes[1].set_title(f"(b) Most frequent individual {n}-bit joint scenarios")

fig.suptitle(
    "Stressed real-network (2008 regime) — cross-cluster joint default scenarios, "
    f"{n_shots:,} shots, {len(counts):,} distinct",
    fontsize=12,
)
fig.tight_layout(rect=(0, 0.1, 1, 0.96))
# bloc membership footnote so the figure documents what the qubits are
bloc_members = {b: [r["ticker"] for r in legend if r["bloc"] == b] for b in sorted(BLOC.values())}
foot = "   |   ".join(f"{b}: {', '.join(ts)}" for b, ts in bloc_members.items())
fig.text(0.5, 0.02, foot, ha="center", va="bottom", fontsize=7.5, wrap=True)
png_path = OUTDIR / "joint_counts_hist.png"
fig.savefig(png_path, dpi=140)

print(f"shots={n_shots:,}  distinct_joint_scenarios={len(counts):,}")
print(f"mean defaults/scenario={mean_def:.2f}  P(severe>={severe})={p_severe:.3f}  "
      f"max defaults in any shot={defaults_per_shot.max()}")
print(f"CSV  -> {csv_path}  ({len(rows):,} rows)")
print(f"PNG  -> {png_path}")
