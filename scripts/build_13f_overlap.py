"""Build the 13F portfolio-overlap (common-asset / fire-sale) network for one quarter.

Streams the ~5 GB ``data/external/13F/holdings.csv`` once into a small single-quarter slice,
builds the institution x asset holdings matrix, and computes the overlap layer:

    * statistically-validated overlap network (Gualdi-Cimini et al. 2016)
    * weighted cosine portfolio similarity
    * liquidity-weighted + directed fire-sale matrices (CRSP market-cap illiquidity)

Artifacts -> ``outputs/data_network/overlap_13f/``: ``overlap_13f.npz`` (all matrices +
institution ids), ``summary.json``, and ``overlap_network.png``.

Run:
    uv run python scripts/build_13f_overlap.py                       # 2008-09-30, top 250
    uv run python scripts/build_13f_overlap.py --rdate 2008-12-31 --top 300
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
MPL_CACHE = ROOT / "outputs" / ".matplotlib"
MPL_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from systemic_risk.data_network.sources.holdings_13f import (
    cosine_overlap,
    crsp_illiquidity,
    directed_fire_sale_matrix,
    holdings_matrix,
    liquidity_weighted_overlap,
    load_or_sample_holdings,
    rdate_to_yearqtr,
    validated_overlap_network,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the 13F portfolio-overlap network.")
    parser.add_argument("--rdate", default="2008-09-30", help="quarter-end report date")
    parser.add_argument("--top", type=int, default=250, help="keep top-N filers by AUM")
    parser.add_argument("--min-positions", type=int, default=3,
                        help="drop assets held by fewer than this many filers")
    parser.add_argument("--min-assets", type=int, default=5,
                        help="drop filers holding fewer than this many assets")
    parser.add_argument("--alpha", type=float, default=0.01, help="SVN significance level")
    parser.add_argument("--out", default=str(ROOT / "outputs" / "data_network" / "overlap_13f"))
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading / sampling holdings for rdate={args.rdate} (top {args.top} filers) ...")
    panel = load_or_sample_holdings((args.rdate,), top_institutions=args.top)
    mat = holdings_matrix(panel, quarter=args.rdate,
                          min_positions=args.min_positions, min_assets_held=args.min_assets)
    print(f"  matrix: {mat.n_institutions} institutions x {mat.n_assets} assets")

    # --- overlap measures ------------------------------------------------- #
    svn, pvalues = validated_overlap_network(mat, alpha=args.alpha)
    cos = cosine_overlap(mat)
    n_links = int(svn.sum() // 2)
    density = svn.sum() / (mat.n_institutions * (mat.n_institutions - 1))
    print(f"  validated overlap links: {n_links}  (density {density:.3f})")

    # --- fire-sale (CRSP illiquidity if available) ------------------------ #
    try:
        ill = crsp_illiquidity(mat, yearqtr=rdate_to_yearqtr(args.rdate))
        ill_source = "CRSP 1/market-cap"
    except FileNotFoundError:
        ill = None
        ill_source = "equal-weight (no CRSP file)"
    lwo = liquidity_weighted_overlap(mat, illiquidity=ill)
    fire = directed_fire_sale_matrix(mat, illiquidity=ill)
    print(f"  fire-sale illiquidity: {ill_source}; directed asymmetry "
          f"{'yes' if not np.allclose(fire, fire.T) else 'no'}")

    # --- save artifacts --------------------------------------------------- #
    npz_path = out_dir / "overlap_13f.npz"
    np.savez_compressed(
        npz_path,
        institutions=np.array(mat.institutions),
        assets=np.array(mat.assets),
        validated_overlap=svn,
        overlap_pvalues=pvalues,
        cosine_overlap=cos,
        liquidity_weighted_overlap=lwo,
        directed_fire_sale=fire,
        holdings=mat.H,
    )
    aum = mat.H.sum(axis=1)
    summary = {
        "rdate": args.rdate,
        "yearqtr": rdate_to_yearqtr(args.rdate),
        "n_institutions": mat.n_institutions,
        "n_assets": mat.n_assets,
        "validated_links": n_links,
        "validated_density": round(float(density), 4),
        "illiquidity_source": ill_source,
        "mean_cosine_overlap": round(float(cos[~np.eye(len(cos), dtype=bool)].mean()), 4),
        "top_filers_by_aum": [
            {"cik": mat.institutions[i], "aum": float(aum[i])}
            for i in np.argsort(-aum)[:10]
        ],
        "source": "EDGAR-Parsing holdings.csv (CRSP permno-keyed); arXiv:1603.05914 method",
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    plot_path = out_dir / "overlap_network.png"
    _plot_overlap_network(svn, cos, aum, mat.institutions, args.rdate, plot_path)

    print(f"\nSaved:\n  {npz_path}\n  {out_dir / 'summary.json'}\n  {plot_path}")


def _plot_overlap_network(svn, cosine, aum, institutions, rdate, path,
                          backbone_pct: float = 2.0) -> None:
    """Draw the overlap network's *strongest-overlap backbone*, coloured by community.

    The validated network is near-complete during the crisis (every big filer overlaps with
    every other), so drawing all links is a hairball. Instead keep only the top
    ``backbone_pct`` percent of edges by **cosine overlap weight** among validated pairs — the
    strongest co-holding relationships — which exposes the cluster structure. Communities are
    detected on that backbone; node size ~ AUM.
    """
    import matplotlib.pyplot as plt
    import networkx as nx

    n = len(institutions)
    iu = np.triu_indices(n, 1)
    weights = np.where(svn[iu] > 0, cosine[iu], 0.0)   # cosine weight on validated pairs only
    positive = weights[weights > 0]
    cutoff = np.percentile(positive, 100 - backbone_pct) if positive.size else 1.0

    graph = nx.Graph()
    graph.add_nodes_from(range(n))
    for a, b, w in zip(iu[0], iu[1], weights):
        if w >= cutoff and w > 0:
            graph.add_edge(int(a), int(b), weight=float(w))

    palette = ["#2f6f9f", "#b24a3c", "#6f8f3a", "#9f6b2f", "#8f4d6f",
               "#3a8f8f", "#7a5fb0", "#b08a2f", "#4f4f4f", "#c0507a"]
    color = ["#cccccc"] * n
    if graph.number_of_edges():
        for k, comm in enumerate(nx.community.greedy_modularity_communities(graph, weight="weight")):
            for node in comm:
                color[node] = palette[k % len(palette)]

    pos = nx.spring_layout(graph, seed=11, k=0.5, weight="weight")
    fig, ax = plt.subplots(figsize=(11, 8))
    sizes = 50 + 650 * (aum / aum.max())
    nx.draw_networkx_edges(graph, pos, ax=ax, alpha=0.20, edge_color="#777777")
    nx.draw_networkx_nodes(graph, pos, ax=ax, node_color=color, node_size=sizes,
                           linewidths=0.4, edgecolors="#1f1f1f")
    ax.set_title(f"13F portfolio-overlap backbone (top {backbone_pct:.0f}% co-holdings) — {rdate}")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=170)


if __name__ == "__main__":
    main()
