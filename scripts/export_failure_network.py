"""Build the failure-network page data from the latest hardware run.

Reads the cluster-mixture stress run in
``outputs/real_cluster_mixture_stress_hw/`` (``joint_counts.csv`` weighted
bitstrings + ``qubit_legend.csv`` institution metadata), reduces it to the
prototype's sufficient statistics, and writes ``data.json`` straight into the
embedded visualisation. The run is 48 qubits (48-bit bitstrings), so the
reduction is done here in numpy rather than the 32-bit TypeScript path.

Outputs: prototyping-for-quantum/prototyping-for-quantum/src/data.json

Usage:
    uv run python scripts/export_failure_network.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "outputs" / "real_cluster_mixture_stress_hw"
LEGEND = RUN / "qubit_legend.csv"
COUNTS = RUN / "joint_counts.csv"
OUT = ROOT / "prototyping-for-quantum" / "prototyping-for-quantum" / "src" / "data.json"

SHIP_PHI_FLOOR = 0.02
HEDGE_PHI_FLOOR = 0.03
MIN_SUPPORT = 0.02
MAX_ITEMSET = 4

REGION_NAME = {
    "US": "North America",
    "CA": "North America",
    "UK": "UK & Europe",
    "EU": "UK & Europe",
    "JP": "Japan",
    "LATAM": "LatAm",
    "APAC": "Asia-Pacific",
    "OCEANIA": "Oceania",
    "MEA": "Middle East",
    "AFRICA": "Africa",
}

# headquarters coordinates [lat, lon]
HQ = {
    "JPM": (40.755, -73.976), "BAC": (35.227, -80.843), "C": (40.721, -74.009),
    "WFC": (37.791, -122.402), "GS": (40.715, -74.013), "MS": (40.764, -73.979),
    "USB": (44.977, -93.271), "PNC": (40.441, -79.994), "TFC": (35.224, -80.840),
    "BK": (40.711, -74.011), "STT": (42.353, -71.055), "HSBC": (51.505, -0.019),
    "BCS": (51.504, -0.017), "LYG": (51.513, -0.091), "DB": (50.114, 8.671),
    "SAN": (40.404, -3.873), "BBVA": (43.263, -2.935), "ING": (52.372, 4.896),
    "UBS": (47.370, 8.541), "MUFG": (35.680, 139.764), "SMFG": (35.686, 139.764),
    "MFG": (35.686, 139.770), "NMR": (35.690, 139.773), "RY": (43.648, -79.380),
    "TD": (43.648, -79.381), "BNS": (43.648, -79.378), "ITUB": (-23.561, -46.656),
    "BBD": (-23.532, -46.792), "AAPL": (37.335, -122.009), "MSFT": (47.640, -122.130),
    "XOM": (32.835, -96.949), "WMT": (36.366, -94.209), "BA": (38.886, -77.092),
    "T": (32.778, -96.797), "TM": (35.083, 137.156), "VWAGY": (52.428, 10.787),
    "SIEGY": (48.137, 11.575), "PBR": (-22.905, -43.176), "ICBC": (39.915, 116.404),
    "DBS": (1.283, 103.851), "ICICI": (19.076, 72.877), "KB": (37.566, 126.978),
    "CBA": (-33.867, 151.207), "WBC": (-33.867, 151.206), "RJHI": (24.713, 46.675),
    "QNB": (25.286, 51.531), "SBK": (-26.196, 28.047), "FSR": (-26.146, 28.045),
}


def load_legend() -> list[dict]:
    return list(csv.DictReader(LEGEND.open(encoding="utf-8")))


def load_counts(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (X: (M,n) 0/1 float, counts: (M,) float)."""
    bitstrings: list[str] = []
    counts: list[int] = []
    with COUNTS.open(encoding="utf-8") as f:
        for line in f:
            if line.startswith("#") or line.startswith("bitstring"):
                continue
            bs, c = line.split(",", 2)[:2]
            bs = bs.strip()
            if len(bs) != n:
                continue
            bitstrings.append(bs)
            counts.append(int(c))
    blob = np.frombuffer("".join(bitstrings).encode("ascii"), dtype=np.uint8)
    X = (blob.reshape(len(bitstrings), n) == ord("1")).astype(np.float64)
    return X, np.asarray(counts, dtype=np.float64)


def eigenvector_centrality(pos_phi: np.ndarray) -> np.ndarray:
    n = pos_phi.shape[0]
    v = np.full(n, 1 / np.sqrt(n))
    for _ in range(300):
        nv = pos_phi @ v
        norm = np.linalg.norm(nv) or 1.0
        v = nv / norm
    v = np.abs(v)
    return v / (v.max() or 1.0)


def apriori(X: np.ndarray, counts: np.ndarray, total: float, marg: np.ndarray) -> list[dict]:
    n = X.shape[1]
    out: list[dict] = []
    prev = [[i] for i in range(n) if marg[i] >= MIN_SUPPORT]
    prev_set = {tuple(p) for p in prev}
    for size in range(2, MAX_ITEMSET + 1):
        cands: list[list[int]] = []
        for a in range(len(prev)):
            for b in range(a + 1, len(prev)):
                A, B = prev[a], prev[b]
                if A[: size - 2] != B[: size - 2] or A[size - 2] >= B[size - 2]:
                    continue
                cand = A + [B[size - 2]]
                if all(
                    tuple(cand[:k] + cand[k + 1 :]) in prev_set for k in range(len(cand))
                ):
                    cands.append(cand)
        if not cands or len(cands) > 8000:
            break
        cur: list[list[int]] = []
        for cand in cands:
            mask = (X[:, cand] > 0.5).all(axis=1)
            support = float(counts[mask].sum() / total)
            if support >= MIN_SUPPORT:
                cur.append(cand)
                exp = float(np.prod(marg[cand]))
                out.append(
                    {
                        "members": cand,
                        "labels": [],
                        "support": support,
                        "lift": support / exp if exp > 0 else 0.0,
                        "size": len(cand),
                    }
                )
        prev = cur
        prev_set = {tuple(p) for p in prev}
        if not prev:
            break
    return out


def main() -> None:
    legend = load_legend()
    n = len(legend)
    tickers = [r["ticker"] for r in legend]

    X, counts = load_counts(n)
    total = float(counts.sum())
    marg = (counts @ X) / total
    joint = ((X.T * counts) @ X) / total

    denom = np.sqrt(np.outer(marg * (1 - marg), marg * (1 - marg)))
    with np.errstate(divide="ignore", invalid="ignore"):
        phi = np.where(denom > 0, (joint - np.outer(marg, marg)) / denom, 0.0)
    np.fill_diagonal(phi, 0.0)

    pos_phi = np.clip(phi, 0, None)
    centrality = eigenvector_centrality(pos_phi)

    # communities = the run's own clusters (qubit_legend cluster/bloc columns)
    clusters = [int(r["cluster"]) for r in legend]
    blocs = [r["bloc"] for r in legend]
    community = np.array(clusters, dtype=int)

    # edges
    edges = []
    phi_vals = []
    for i in range(n):
        for j in range(i + 1, n):
            p = float(phi[i, j])
            floor = SHIP_PHI_FLOOR if p > 0 else HEDGE_PHI_FLOOR
            if abs(p) < floor:
                continue
            phi_vals.append(p)
            lo, hi = (i, j) if marg[i] <= marg[j] else (j, i)
            pij = float(joint[i, j])
            edges.append(
                {
                    "source": lo,
                    "target": hi,
                    "phi": round(p, 5),
                    "lift": round(pij / (marg[i] * marg[j]), 4) if marg[i] * marg[j] > 0 else 0.0,
                    "pBoth": round(pij, 5),
                    "pSource": round(float(marg[lo]), 5),
                    "pTarget": round(float(marg[hi]), 5),
                    "confST": round(pij / marg[lo], 4) if marg[lo] > 0 else 0.0,
                    "confTS": round(pij / marg[hi], 4) if marg[hi] > 0 else 0.0,
                }
            )

    # nodes
    nodes = []
    for i, r in enumerate(legend):
        region = r["region"]
        coord = HQ.get(r["ticker"])
        nodes.append(
            {
                "id": i,
                "label": r["ticker"],
                "sector": region,
                "sectorName": REGION_NAME.get(region, region),
                "pd": round(float(marg[i]), 5),
                "marginal": round(float(marg[i]), 5),
                "centrality": round(float(centrality[i]), 4),
                "community": int(community[i]),
                "lat": coord[0] if coord else None,
                "lon": coord[1] if coord else None,
            }
        )

    # community summaries = the run's clusters, named by their bloc; "purity" is
    # the share of the most common region within the cluster
    def bloc_name(bloc: str) -> tuple[str, str]:
        if ":" in bloc:
            code, rest = bloc.split(":", 1)
            return code, rest.replace("+", " + ")
        return bloc, bloc

    communities = []
    for cid in sorted(set(clusters)):
        members = [i for i in range(n) if clusters[i] == cid]
        code, name = bloc_name(blocs[members[0]])
        regs: dict[str, int] = {}
        for m in members:
            regs[nodes[m]["sector"]] = regs.get(nodes[m]["sector"], 0) + 1
        _, dom_count = max(regs.items(), key=lambda kv: kv[1])
        communities.append(
            {
                "id": cid,
                "members": members,
                "size": len(members),
                "dominantSector": code,
                "dominantSectorName": name,
                "purity": dom_count / len(members),
                "avgMarginal": float(np.mean([marg[m] for m in members])),
            }
        )

    # itemsets
    itemsets = apriori(X, counts, total, marg)
    for it in itemsets:
        it["labels"] = [tickers[m] for m in it["members"]]
    itemsets.sort(key=lambda it: it["support"], reverse=True)
    itemsets = itemsets[:30]

    n_def = X.sum(axis=1)
    p_no = float(counts[n_def == 0].sum() / total)
    pos_sorted = sorted(v for v in phi_vals if v > 0)
    neg_sorted = sorted(-v for v in phi_vals if v < 0)

    def q(arr: list[float], qq: float, default: float) -> float:
        if not arr:
            return default
        return float(np.quantile(arr, qq))

    meta = {
        "n": n,
        "shots": int(total),
        "factorNames": [],
        "pNoFailure": p_no,
        "expectedFailures": float(marg.sum()),
        "maxAbsPhi": float(np.max(np.abs(phi))),
        "posPhiScale": q(pos_sorted, 0.97, 0.1),
        "negPhiScale": q(neg_sorted, 0.97, 0.05),
        "defaultThreshold": 0.2,
        "note": (
            f"Pairwise phi co-failure correlations from {int(total):,} shots of a "
            f"cluster-mixture stress run on IBM hardware ({n} institutions). Edge "
            f"hue = sign/strength; brightness runs driver->dependent."
        ),
    }

    data = {"meta": meta, "nodes": nodes, "edges": edges, "itemsets": itemsets, "communities": communities}
    OUT.write_text(json.dumps(data))
    print(f"wrote {OUT}")
    print(f"  n={n} shots={int(total)} edges={len(edges)} blocs={len(communities)} "
          f"itemsets={len(itemsets)} maxPhi={meta['maxAbsPhi']:.3f}")
    top = sorted(range(n), key=lambda i: -centrality[i])[:5]
    print("  top centrality:", [f"{tickers[i]} {centrality[i]:.2f}" for i in top])


if __name__ == "__main__":
    main()
