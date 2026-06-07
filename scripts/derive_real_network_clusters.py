"""Derive a hardware-ready community partition over the REAL exposure network.

The downstream quantum-hardware mixture pipeline (``scripts/run_cluster_mixture*.py``)
loads the network as a set of small, independently-loadable clusters
(``discover_clusters(spec, ...).clusters`` -- a list of global-index lists, one per
device). This script derives that decomposition from the *real* data-network
(``systemic_risk.data_network``), not the synthetic planted-cluster system, and persists
a clean artifact the next stage can read programmatically.

Why k=3 is forced
-----------------
The real roster is n=38 entities and the per-device band is [10, 15] nodes. Splitting 38
into k parts each in [10,15] is only arithmetically possible for **k=3** (k=2 caps at
2*15=30 < 38; k=4 needs at least 4*10=40 > 38). We still *explore* k in [2, 6] and report
the community quality (weighted modularity) of each, but only k=3 can satisfy the band.

Method
------
1. Build the real ``NetworkSpec`` (28 banks + 10 corporates) and flat ``SystemSpec`` with
   the existing ``data_network`` machinery; cluster on the repo's own dependency matrix
   (``dependency_for_clustering``: |correlation| with an exposure fallback).
2. For each k, seed with a **Ward-linkage** dendrogram cut (Ward balances cluster sizes
   far better than the average linkage used for the size-*cap* clusterer, which chains on
   this dense, near-uniform correlation graph). Score by weighted modularity.
3. For the feasible k (=3), run a **size-constrained greedy refinement**: move boundary
   nodes from over-full to under-full clusters, each move chosen to first drive every
   cluster into [10,15] and then to maximise modularity. This is the same weakest-link
   spirit as the repo's Kernighan-Lin ``split_oversize_group``, generalised to a two-sided
   band rather than a one-sided cap.
4. Choose the in-band partition with the highest modularity, persist the artifact and
   per-cluster member-index ``.npz`` files the hardware harness ingests directly.

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

SIZE_MIN, SIZE_MAX = 10, 15
K_RANGE = range(2, 7)  # [2, 6]


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

    # ---- explore k in [2, 6]: Ward seed, then size-constrained refinement -------
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
        "method": "budget_discover_clusters(cap=15)",
        "k": int(budget.n_clusters),
        "sizes": sizes(budget_labels),
        "modularity": round(modularity(graph, budget_labels), 5),
        "in_size_band": all(SIZE_MIN <= s <= SIZE_MAX for s in sizes(budget_labels)),
        "cut_fraction": round(budget.cut_fraction, 5),
        "note": "average-linkage size-CAP clusterer; respects <=15 but not the >=10 floor",
    }

    # ---- choose: highest modularity among in-band candidates -------------------
    feasible = [c for c in candidates if c["in_size_band"]]
    chosen = max(feasible, key=lambda c: c["modularity"])
    rationale = (
        f"Only k=3 can place all of n={n} entities in [{SIZE_MIN},{SIZE_MAX}] "
        f"(k=2 caps at {2*SIZE_MAX}<{n}; k>=4 needs >={4*SIZE_MIN}>{n}). Among in-band "
        f"partitions the chosen k={chosen['k']} maximises weighted modularity "
        f"({chosen['modularity']}); the size-constrained refinement raised modularity "
        f"above the Ward seed ({chosen['seed_modularity']}) while bringing every cluster "
        f"into band."
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
            "Community partition over the REAL exposure network "
            "(systemic_risk.data_network) for the per-device quantum-hardware mixture "
            "decomposition. Every cluster constrained to [10,15] nodes."
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
        "selection_criterion": "max weighted modularity subject to every cluster in [10,15]",
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
            "member). The full real network is dense (every pair correlated) and many "
            "default probs sit below QPU noise floors; decomposing into <=15-qubit blocks "
            "is what makes a hardware run plausible. The per-cluster marginal ranges below "
            "are tiny (~1e-4..1e-3), so sub-noise-floor PDs remain a concern WITHIN blocks "
            "-- inspect/rescale before submitting."
        ),
        "downstream_load": (
            "partition = json.load(open('outputs/data_network/real_network_partition.json')); "
            "clusters = [c['member_indices'] for c in partition['clusters']]  # list of global "
            "node-index lists, the form discover_clusters(...).clusters returns and "
            "run_cluster_mixture_hardware.py consumes. Per-cluster .npz files under "
            "real_network_clusters/ carry the same 'members' arrays."
        ),
    }

    out_dir = ROOT / "outputs" / "data_network"
    out_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = out_dir / "real_network_partition.json"
    artifact_path.write_text(json.dumps(artifact, indent=2), encoding="utf-8")

    npz_dir = out_dir / "real_network_clusters"
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
    print(f"\nExploration over k in [2,6] (weighted modularity; * = every cluster in [{SIZE_MIN},{SIZE_MAX}]):")
    for c in candidates:
        flag = " *" if c["in_size_band"] else ("   (band impossible)" if not c["band_arithmetically_possible"] else "")
        print(f"  k={c['k']}  mod={c['modularity']:.4f}  sizes={c['sizes']}{flag}")
    print(f"  [crosscheck] budget cap=15 -> k={budget_entry['k']}  mod={budget_entry['modularity']:.4f}  "
          f"sizes={budget_entry['sizes']} (violates >=10 floor)")

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
