"""Derive a FINER hardware-ready community partition over the REAL exposure network.

Twin of ``scripts/derive_real_network_clusters.py`` but with a SMALLER per-device
size band of [5, 8] nodes instead of [10, 15]. Motivation: the 14/14/10 hardware run
showed marginal/correlation error scales with circuit DEPTH -- the shallow 10-qubit
block (depth ~105) recovered marginals well (RMSE ~0.02) while the deep 14-qubit
blocks (depth ~600-900) collapsed to noise (RMSE ~0.20-0.23). Smaller, shallower
clusters should therefore improve per-block fidelity. This script re-derives the
partition for that smaller band and persists it to a DISTINCT artifact so the prior
14/14/10 partition is never clobbered.

Feasibility of the band over n=38
---------------------------------
Splitting 38 into k parts each in [5, 8] is arithmetically possible for:
  k=5  (5*5=25 <= 38 <= 5*8=40)  -- requires sizes near 8
  k=6  (6*5=30 <= 38 <= 6*8=48)
  k=7  (7*5=35 <= 38 <= 7*8=56)  -- requires sizes near 5-6
k=4 caps at 4*8=32 < 38; k=8 needs >= 8*5=40 > 38. So feasible k in {5,6,7}.
We still explore k in [3, 8] and report modularity for each, but only k in {5,6,7}
can satisfy the band. Among in-band partitions we pick the highest weighted modularity.

Everything else (Ward seed, size-constrained greedy refinement, modularity scoring,
artifact + per-cluster .npz emission) is reused from the [10,15] script unchanged.

Read/compute only -- nothing is submitted to hardware.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import networkx as nx
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from systemic_risk.data_network import build_network_spec  # noqa: E402
from systemic_risk.generators.quantum.budget_clustering import (  # noqa: E402
    dependency_for_clustering,
    discover_clusters,
)

SIZE_MIN, SIZE_MAX = 5, 8
K_RANGE = range(3, 9)  # [3, 8]


def dep_graph(dependency: np.ndarray) -> nx.Graph:
    n = dependency.shape[0]
    g = nx.Graph()
    g.add_nodes_from(range(n))
    for i, j in zip(*np.triu_indices(n, 1)):
        w = float(dependency[i, j])
        if w > 0:
            g.add_edge(int(i), int(j), weight=w)
    return g


def modularity(graph: nx.Graph, labels: np.ndarray) -> float:
    comm: dict[int, set[int]] = {}
    for node, lab in enumerate(labels):
        comm.setdefault(int(lab), set()).add(int(node))
    return float(nx.community.modularity(graph, list(comm.values()), weight="weight"))


def sizes(labels: np.ndarray) -> list[int]:
    _, counts = np.unique(labels, return_counts=True)
    return sorted((int(c) for c in counts), reverse=True)


def relabel_desc(labels: np.ndarray) -> np.ndarray:
    """Relabel 0..k-1 by descending size, tie-broken by smallest member index."""
    groups: dict[int, list[int]] = {}
    for node, lab in enumerate(labels):
        groups.setdefault(int(lab), []).append(int(node))
    order = sorted(groups.values(), key=lambda m: (-len(m), m[0]))
    out = np.full(len(labels), -1, dtype=int)
    for cid, members in enumerate(order):
        for node in members:
            out[node] = cid
    return out


def ward_cut(dependency: np.ndarray, k: int) -> np.ndarray:
    """Ward-linkage cut of the correlation-distance tree into exactly ``k`` clusters."""
    dist = np.clip(1.0 - np.abs(dependency), 0.0, 1.0)
    dist = 0.5 * (dist + dist.T)
    np.fill_diagonal(dist, 0.0)
    z = linkage(squareform(dist, checks=False), method="ward")
    return relabel_desc(fcluster(z, t=k, criterion="maxclust") - 1)


def band_violation(labels: np.ndarray, k: int) -> int:
    """Total nodes by which clusters are over SIZE_MAX or under SIZE_MIN."""
    v = 0
    for c in range(k):
        s = int(np.sum(labels == c))
        v += max(0, s - SIZE_MAX) + max(0, SIZE_MIN - s)
    return v


def rebalance_to_band(graph, labels, k, max_iters=5000):
    """Greedy boundary swaps: first satisfy the [SIZE_MIN, SIZE_MAX] band, then maximise modularity."""
    labels = labels.copy()
    for _ in range(max_iters):
        cur_viol = band_violation(labels, k)
        cur_mod = modularity(graph, labels)
        best = None  # (viol, mod, new_labels)
        for node in range(len(labels)):
            src = int(labels[node])
            if int(np.sum(labels == src)) <= SIZE_MIN:
                continue  # never empty a cluster below the floor
            for dst in range(k):
                if dst == src or int(np.sum(labels == dst)) >= SIZE_MAX:
                    continue
                new = labels.copy()
                new[node] = dst
                key = (-band_violation(new, k), modularity(graph, new))
                if best is None or key > (best[0], best[1]):
                    best = (key[0], key[1], new)
        if best is None:
            break
        new_viol, new_mod = -best[0], best[1]
        # Accept if it reduces violation, or (once feasible) strictly improves modularity.
        if new_viol < cur_viol or (new_viol == 0 == cur_viol and new_mod > cur_mod + 1e-12):
            labels = best[2]
        else:
            break
    return relabel_desc(labels)


def main() -> None:
    nspec = build_network_spec()
    spec = nspec.to_system_spec()
    node_ids = list(nspec.empirical.node_ids)
    names = spec.node_names
    n = nspec.n

    dependency = dependency_for_clustering(spec)
    graph = dep_graph(dependency)

    # ---- explore k in [3, 8]: Ward seed, then size-constrained refinement -------
    candidates = []
    for k in K_RANGE:
        seed = ward_cut(dependency, k)
        feasible_band = (k * SIZE_MIN) <= n <= (k * SIZE_MAX)
        refined = rebalance_to_band(graph, seed, k) if feasible_band else seed
        candidates.append(
            {
                "k": int(k),
                "band_arithmetically_possible": bool(feasible_band),
                "seed_sizes": sizes(seed),
                "seed_modularity": round(modularity(graph, seed), 5),
                "sizes": sizes(refined),
                "modularity": round(modularity(graph, refined), 5),
                "in_size_band": all(SIZE_MIN <= s <= SIZE_MAX for s in sizes(refined)),
                "labels": refined.tolist(),
            }
        )

    # ---- cross-check: the repo's hard-size-cap budget clusterer (the downstream tool) --
    budget = discover_clusters(spec, max_cluster_size=SIZE_MAX)
    budget_labels = relabel_desc(np.asarray(budget.labels))
    budget_entry = {
        "method": f"budget_discover_clusters(cap={SIZE_MAX})",
        "k": int(budget.n_clusters),
        "sizes": sizes(budget_labels),
        "modularity": round(modularity(graph, budget_labels), 5),
        "in_size_band": all(SIZE_MIN <= s <= SIZE_MAX for s in sizes(budget_labels)),
        "cut_fraction": round(budget.cut_fraction, 5),
        "note": f"average-linkage size-CAP clusterer; respects <={SIZE_MAX} but not the >={SIZE_MIN} floor",
    }

    # ---- choose: highest modularity among in-band candidates -------------------
    feasible = [c for c in candidates if c["in_size_band"]]
    if not feasible:
        raise RuntimeError("No in-band partition found for [%d,%d]" % (SIZE_MIN, SIZE_MAX))
    chosen = max(feasible, key=lambda c: c["modularity"])
    rationale = (
        f"For n={n} entities in band [{SIZE_MIN},{SIZE_MAX}], feasible k in "
        f"{[c['k'] for c in feasible]} (k<5 caps below n; k>7 needs more than n). "
        f"Among in-band partitions the chosen k={chosen['k']} maximises weighted modularity "
        f"({chosen['modularity']}); the size-constrained refinement adjusted the Ward seed "
        f"(seed modularity {chosen['seed_modularity']}) to bring every cluster into band."
    )

    chosen_labels = np.asarray(chosen["labels"], dtype=int)
    clusters = []
    for cid in range(int(chosen_labels.max()) + 1):
        members = [int(i) for i in np.where(chosen_labels == cid)[0]]
        clusters.append(
            {
                "cluster_id": cid,
                "size": len(members),
                "member_indices": members,
                "member_ids": [node_ids[i] for i in members],
                "member_names": [names[i] for i in members],
                "min_marginal_default_prob": round(
                    float(min(spec.marginal_default_probs[i] for i in members)), 6
                ),
                "max_marginal_default_prob": round(
                    float(max(spec.marginal_default_probs[i] for i in members)), 6
                ),
            }
        )

    iu = np.triu_indices(n, 1)
    w = dependency[iu]
    same = chosen_labels[iu[0]] == chosen_labels[iu[1]]
    within_w, cut_w = float(w[same].sum()), float(w[~same].sum())
    cut_fraction = cut_w / (within_w + cut_w) if (within_w + cut_w) > 0 else 0.0

    artifact = {
        "description": (
            "FINER community partition over the REAL exposure network "
            "(systemic_risk.data_network) for the per-device quantum-hardware mixture "
            "decomposition. Every cluster constrained to [5,8] nodes (smaller/shallower "
            "than the 14/14/10 partition, to reduce per-block circuit depth)."
        ),
        "source": "systemic_risk.data_network.build_network_spec().to_system_spec()",
        "content_hash": spec.metadata.get("content_hash", ""),
        "n_nodes": n,
        "node_ids": node_ids,
        "node_names": names,
        "node_types": list(spec.node_types),
        "dependency_basis": "abs(correlation) with exposure fallback (dependency_for_clustering)",
        "size_band": [SIZE_MIN, SIZE_MAX],
        "k_explored": list(K_RANGE),
        "selection_criterion": "max weighted modularity subject to every cluster in [5,8]",
        "method": "Ward-linkage seed + size-constrained greedy refinement",
        "exploration": candidates,
        "budget_clusterer_crosscheck": budget_entry,
        "chosen": {
            "k": chosen["k"],
            "sizes": chosen["sizes"],
            "modularity": chosen["modularity"],
            "seed_modularity": chosen["seed_modularity"],
            "in_size_band": chosen["in_size_band"],
            "within_dependency_weight": round(within_w, 5),
            "cut_dependency_weight": round(cut_w, 5),
            "cut_fraction": round(cut_fraction, 5),
            "rationale": rationale,
        },
        "labels": chosen_labels.tolist(),
        "clusters": clusters,
        "hardware_load_note": (
            "Each cluster loads as its own statevector / device circuit (one qubit per "
            "member). Smaller [5,8] clusters yield shallower blocks than the [10,15] band, "
            "directly attacking the depth-driven error wall observed in the 14/14/10 run. "
            "The per-cluster marginal ranges below remain tiny (~1e-5..1.4e-2), so "
            "sub-noise-floor PDs persist WITHIN blocks regardless of depth."
        ),
        "downstream_load": (
            "partition = json.load(open('outputs/data_network/real_network_partition_small.json')); "
            "clusters = [c['member_indices'] for c in partition['clusters']]  # list of global "
            "node-index lists. Per-cluster .npz files under real_network_clusters_small/ carry "
            "the same 'members' arrays."
        ),
    }

    out_dir = ROOT / "outputs" / "data_network"
    out_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = out_dir / "real_network_partition_small.json"
    artifact_path.write_text(json.dumps(artifact, indent=2), encoding="utf-8")

    npz_dir = out_dir / "real_network_clusters_small"
    npz_dir.mkdir(parents=True, exist_ok=True)
    npz_paths = []
    for c in clusters:
        p = npz_dir / f"cluster{c['cluster_id']}_members.npz"
        np.savez(
            p,
            members=np.asarray(c["member_indices"], dtype=int),
            member_ids=np.asarray(c["member_ids"], dtype=object),
        )
        npz_paths.append(str(p.relative_to(ROOT)))

    # ---- console report -------------------------------------------------------
    print("Real network: n =", n, "entities (28 banks + 10 corporates)")
    print(f"\nExploration over k in [3,8] (weighted modularity; * = every cluster in [{SIZE_MIN},{SIZE_MAX}]):")
    for c in candidates:
        flag = " *" if c["in_size_band"] else ("   (band impossible)" if not c["band_arithmetically_possible"] else "")
        print(f"  k={c['k']}  mod={c['modularity']:.4f}  sizes={c['sizes']}{flag}")
    print(f"  [crosscheck] budget cap={SIZE_MAX} -> k={budget_entry['k']}  mod={budget_entry['modularity']:.4f}  "
          f"sizes={budget_entry['sizes']}")

    print(f"\nCHOSEN: k={chosen['k']}  mod={chosen['modularity']:.4f}  sizes={chosen['sizes']}")
    print("  cut_fraction =", round(cut_fraction, 4),
          "(fraction of dependency mass severed across cluster boundaries)")
    for c in clusters:
        print(f"  cluster {c['cluster_id']} (n={c['size']}, PD {c['min_marginal_default_prob']}"
              f"..{c['max_marginal_default_prob']}): {', '.join(c['member_ids'])}")

    print("\nArtifact:", artifact_path.relative_to(ROOT))
    for p in npz_paths:
        print("  ", p)


if __name__ == "__main__":
    main()
