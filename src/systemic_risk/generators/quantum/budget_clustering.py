"""Budget-respecting cluster discovery for the per-device mixture roadmap.

We partition institutions into clusters so each cluster fits on a small quantum
device (a hard qubit budget, ``max_cluster_size``) and can be loaded and sampled
independently, with the weaker cross-cluster links handled classically afterwards.
A *good* partition keeps the strong pairwise dependency mass WITHIN clusters and
cuts as little weight as possible ACROSS cluster boundaries -- and never returns a
cluster larger than the qubit budget.

Two cooperating pieces:

* :func:`budget_clusters_from_dependency` -- the primary clusterer. It runs
  average-linkage **hierarchical clustering** on the correlation distance
  ``d = 1 - |dependency|`` (the standard finance treatment of a correlation
  matrix), then takes a **budget-respecting recursive dendrogram cut**: descend the
  tree and *accept any subtree the moment its size fits the cap*. This is the one
  formulation where the hard size cap drops out naturally -- ``scipy``'s
  ``fcluster(..., 'maxclust')`` controls the cluster *count*, not the cluster
  *size*, so we walk the linkage tree ourselves. Any leaf-dense subtree that is
  still oversize when the descent bottoms out is handed to the explicit splitter.

* :func:`split_oversize_group` -- the explicit splitter. It improves on the old
  index-order chop (which severs whatever links happen to straddle an arbitrary
  index boundary) with **recursive Kernighan-Lin bisection** (``networkx``) on the
  within-group dependency weights, so cohesive members stay together and the split
  falls on the *weakest* cut. Used both as the dendrogram-cut fallback and as the
  drop-in replacement for the ansatz partitioner's chop.

:func:`discover_clusters` is the :class:`SystemSpec` entry point: it reads the
spec's dependency matrix (correlation, by magnitude), falls back to the symmetrised
exposure matrix when no correlation is present (matching
:func:`systemic_risk.generators.quantum.ansatz.dependency_edges`), and returns the
size-capped partition.
"""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform

from systemic_risk.spec import SystemSpec


__all__ = [
    "ClusterPartition",
    "dependency_for_clustering",
    "discover_clusters",
    "budget_clusters_from_dependency",
    "split_oversize_group",
]


@dataclass(frozen=True)
class ClusterPartition:
    """A size-capped partition of ``n`` nodes into clusters.

    ``clusters`` is the list of clusters, each a sorted list of global node indices;
    every cluster satisfies ``len(cluster) <= max_cluster_size``. ``labels[i]`` is the
    cluster id of node ``i``. ``within_weight`` / ``cut_weight`` are the total
    dependency weight kept inside clusters and severed across boundaries (upper
    triangle), a direct objective read-out.
    """

    clusters: list[list[int]]
    labels: np.ndarray
    max_cluster_size: int
    within_weight: float
    cut_weight: float

    @property
    def n_clusters(self) -> int:
        return len(self.clusters)

    @property
    def sizes(self) -> list[int]:
        return [len(c) for c in self.clusters]

    @property
    def cut_fraction(self) -> float:
        """Fraction of total dependency weight that lands on cut (cross-cluster) edges."""
        total = self.within_weight + self.cut_weight
        return float(self.cut_weight / total) if total > 0 else 0.0


# ---------------------------------------------------------------------------
# Dependency extraction (magnitude, with exposure fallback)
# ---------------------------------------------------------------------------


def dependency_for_clustering(spec: SystemSpec) -> np.ndarray:
    """Return the symmetric, nonnegative, zero-diagonal dependency matrix to cluster on.

    Uses ``|correlation|`` (signed correlations enter by magnitude -- a strong negative
    dependence still ties two institutions together for loading purposes). When the spec
    carries no correlation signal at all, falls back to the symmetrised, max-normalised
    exposure matrix, exactly matching the fallback in
    :func:`systemic_risk.generators.quantum.ansatz.dependency_edges`.
    """
    dep = np.abs(spec.dependency_matrix())
    if dep.max() <= 0.0 and float(np.sum(spec.exposure_matrix)) > 0.0:
        exposure = spec.exposure_matrix + spec.exposure_matrix.T
        dep = exposure / exposure.max()
    dep = 0.5 * (dep + dep.T)
    np.fill_diagonal(dep, 0.0)
    return dep


# ---------------------------------------------------------------------------
# Objective read-out
# ---------------------------------------------------------------------------


def _within_cut_weight(dependency: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    """Total upper-triangle dependency weight within clusters and cut across them."""
    iu = np.triu_indices(dependency.shape[0], k=1)
    w = dependency[iu]
    same = labels[iu[0]] == labels[iu[1]]
    return float(w[same].sum()), float(w[~same].sum())


# ---------------------------------------------------------------------------
# Explicit weight-respecting splitter (recursive Kernighan-Lin bisection)
# ---------------------------------------------------------------------------


def split_oversize_group(
    dependency: np.ndarray,
    members: list[int],
    max_cluster_size: int,
) -> list[list[int]]:
    """Split ``members`` into ``<= max_cluster_size`` cohesive pieces by edge weight.

    This is the weight-aware replacement for an index-order chop. It recursively
    **Kernighan-Lin bisects** the induced dependency subgraph: KL searches for the
    balanced 2-partition that minimises the cut weight, so the split falls on the
    *weakest* links and strongly-coupled members stay together. Each side is bisected
    again until every piece fits the cap.

    ``members`` are global node indices into ``dependency``; the returned pieces are
    sorted lists of the same global indices and form a disjoint cover of ``members``.
    Pieces already within the cap are returned untouched. A group with no internal edges
    is chopped by index order as a last resort (any split is equally good when nothing
    binds the members).
    """
    if max_cluster_size < 1:
        raise ValueError("max_cluster_size must be >= 1")
    members = sorted(members)
    if len(members) <= max_cluster_size:
        return [members]

    sub = dependency[np.ix_(members, members)]
    graph = nx.Graph()
    graph.add_nodes_from(members)
    for a in range(len(members)):
        for b in range(a + 1, len(members)):
            w = float(sub[a, b])
            if w > 0.0:
                graph.add_edge(members[a], members[b], weight=w)

    if graph.number_of_edges() == 0:
        # Nothing binds these members: any split is equally good -> index-order chunks.
        return [
            members[start : start + max_cluster_size]
            for start in range(0, len(members), max_cluster_size)
        ]

    # KL on a disconnected graph would ignore the components; bisect each component
    # (then recurse) so the natural separation is honoured before any forced cut.
    components = list(nx.connected_components(graph))
    if len(components) > 1:
        pieces: list[list[int]] = []
        for comp in components:
            pieces.extend(split_oversize_group(dependency, sorted(comp), max_cluster_size))
        return pieces

    # Connected and oversize: bisect on minimum cut weight, recurse on each half.
    left, right = nx.algorithms.community.kernighan_lin_bisection(
        graph, weight="weight", seed=0
    )
    # Degenerate KL output (one side empty) -> fall back to a balanced index split.
    if not left or not right:
        mid = len(members) // 2
        left, right = set(members[:mid]), set(members[mid:])

    pieces = []
    pieces.extend(split_oversize_group(dependency, sorted(left), max_cluster_size))
    pieces.extend(split_oversize_group(dependency, sorted(right), max_cluster_size))
    return pieces


# ---------------------------------------------------------------------------
# Primary clusterer: hierarchical clustering + budget-respecting recursive cut
# ---------------------------------------------------------------------------


def _recursive_dendrogram_cut(
    children: np.ndarray,
    n_leaves: int,
    dependency: np.ndarray,
    max_cluster_size: int,
) -> list[list[int]]:
    """Descend a scipy linkage tree, accepting any subtree once it fits the cap.

    ``children`` is the ``(n_leaves - 1, 2)`` left/right child array from
    :func:`scipy.cluster.hierarchy.linkage` (node ``n_leaves + k`` is the ``k``-th
    merge). We collect the leaves under each node bottom-up, then walk from the root:
    a node whose leaf-set fits the cap becomes a cluster; otherwise we descend into its
    two children. Any node that is accepted while still oversize (only possible via the
    guard below) is repaired by :func:`split_oversize_group`.
    """
    # Leaves under every node (0..n_leaves-1 are leaves; n_leaves+k are merges).
    leaves_of: list[list[int]] = [[i] for i in range(n_leaves)]
    for k in range(children.shape[0]):
        a, b = int(children[k, 0]), int(children[k, 1])
        leaves_of.append(leaves_of[a] + leaves_of[b])

    root = 2 * n_leaves - 2 if n_leaves > 1 else 0
    clusters: list[list[int]] = []
    stack = [root]
    while stack:
        node = stack.pop()
        members = leaves_of[node]
        if len(members) <= max_cluster_size:
            clusters.append(sorted(members))
        elif node < n_leaves:
            # A leaf is a single node, so it can never be oversize; guard anyway.
            clusters.append(sorted(members))
        else:
            internal = node - n_leaves
            a, b = int(children[internal, 0]), int(children[internal, 1])
            stack.extend([a, b])

    # Anything still over the cap (should not happen for a binary tree, but keep the
    # cap a hard guarantee regardless) is split explicitly by edge weight.
    repaired: list[list[int]] = []
    for cluster in clusters:
        if len(cluster) <= max_cluster_size:
            repaired.append(cluster)
        else:
            repaired.extend(split_oversize_group(dependency, cluster, max_cluster_size))
    return repaired


def budget_clusters_from_dependency(
    dependency: np.ndarray,
    max_cluster_size: int,
    *,
    linkage_method: str = "average",
) -> ClusterPartition:
    """Partition nodes from a dependency matrix into size-capped cohesive clusters.

    Runs hierarchical clustering on the correlation distance ``1 - |dependency|`` and
    takes a budget-respecting recursive dendrogram cut (see
    :func:`_recursive_dendrogram_cut`), with :func:`split_oversize_group` repairing any
    subtree the cut leaves oversize. ``dependency`` must be a symmetric, nonnegative,
    zero-diagonal matrix (use :func:`dependency_for_clustering` to build one from a spec).

    The hard cap is guaranteed: every returned cluster has ``<= max_cluster_size`` nodes.
    """
    dependency = np.asarray(dependency, dtype=float)
    n = dependency.shape[0]
    if dependency.shape != (n, n):
        raise ValueError("dependency must be a square matrix")
    if max_cluster_size < 1:
        raise ValueError("max_cluster_size must be >= 1")

    if n == 0:
        return ClusterPartition([], np.zeros(0, dtype=int), max_cluster_size, 0.0, 0.0)
    if n == 1:
        return ClusterPartition([[0]], np.zeros(1, dtype=int), max_cluster_size, 0.0, 0.0)

    if n <= max_cluster_size and float(np.sum(dependency)) <= 0.0:
        # No structure and it already fits: one block (singletons would be wasteful).
        clusters = [list(range(n))]
    else:
        # Correlation distance: identical -> 0, independent -> 1. Clamp for safety.
        dist = np.clip(1.0 - np.abs(dependency), 0.0, 1.0)
        dist = 0.5 * (dist + dist.T)
        np.fill_diagonal(dist, 0.0)
        condensed = squareform(dist, checks=False)
        if condensed.size == 0:
            clusters = [list(range(n))]
        else:
            z = linkage(condensed, method=linkage_method)
            clusters = _recursive_dendrogram_cut(
                z[:, :2].astype(int), n, dependency, max_cluster_size
            )

    # Deterministic labelling: larger clusters first, then by smallest member index.
    clusters = sorted(clusters, key=lambda c: (-len(c), c[0]))
    labels = np.full(n, -1, dtype=int)
    for cid, members in enumerate(clusters):
        for node in members:
            labels[node] = cid

    within, cut = _within_cut_weight(dependency, labels)
    return ClusterPartition(
        clusters=clusters,
        labels=labels,
        max_cluster_size=max_cluster_size,
        within_weight=within,
        cut_weight=cut,
    )


def discover_clusters(
    spec: SystemSpec,
    max_cluster_size: int,
    *,
    linkage_method: str = "average",
) -> ClusterPartition:
    """Discover size-capped clusters for a :class:`SystemSpec` (the public entry point).

    Reads the spec's pairwise dependency structure by magnitude (correlation, falling
    back to the symmetrised exposure graph when no correlation is present) and returns a
    partition in which every cluster fits ``max_cluster_size`` (the per-device qubit
    budget) while keeping the strong dependency mass within clusters. The returned
    :class:`ClusterPartition` exposes ``clusters`` (global index lists), ``labels``, and
    the within/cut weight objective.
    """
    dependency = dependency_for_clustering(spec)
    return budget_clusters_from_dependency(
        dependency, max_cluster_size, linkage_method=linkage_method
    )
