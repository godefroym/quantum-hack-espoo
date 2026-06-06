"""Per-cluster default sampling -- the source-agnostic seam of the mixture pipeline.

Each cluster is its own small quantum loader. Right now we *simulate* every cluster's loader
on the exact numpy statevector engine (:func:`sample_cluster_statevector`), but the reconciler
downstream only ever sees a :class:`ClusterSamples` -- a plain container of one cluster's
measured bitstrings plus the global node indices it occupies. That is the explicit
hardware seam:

    # Today (simulated loader):
    clusters = sample_clusters_statevector(spec, partition, n_samples, seed)

    # Later (real per-cluster devices): run each cluster's circuit on its own QPU, collect the
    # measured 0/1 bitstrings, and wrap them -- nothing else in the pipeline changes:
    clusters = [
        cluster_samples_from_bitstrings(member_indices, measured_bits_for_that_device)
        for member_indices, measured_bits_for_that_device in per_device_results
    ]
    reconciled = reconciler.reconcile(clusters, ...)

The within-cluster joint default law carried by a :class:`ClusterSamples` is treated as exact:
the reconciler never modifies a cluster's own samples, it only *couples* clusters together.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from systemic_risk.generators.quantum import ansatz as A
from systemic_risk.generators.quantum.budget_clustering import ClusterPartition
from systemic_risk.generators.quantum.statevector import StateVector, sample_bitstrings
from systemic_risk.spec import SystemSpec
from systemic_risk.utils.validation import ensure_binary_samples


@dataclass(frozen=True)
class ClusterSamples:
    """One cluster's independent default samples, tagged with its global node indices.

    ``members`` are the global institution indices this cluster occupies (sorted, as returned
    by :class:`ClusterPartition`). ``samples`` is an ``(n_samples, len(members))`` 0/1 array in
    ``members`` order: column ``k`` is institution ``members[k]``. ``source`` records where the
    bitstrings came from (``"statevector"`` now, ``"hardware:<backend>"`` later) for the report.

    This is the only object the reconciler consumes, so the same code path serves simulated and
    real per-cluster sample sets. The samples are read as the EXACT within-cluster joint law.
    """

    members: tuple[int, ...]
    samples: np.ndarray
    source: str = "unknown"

    def __post_init__(self) -> None:
        members = tuple(int(m) for m in self.members)
        samples = ensure_binary_samples(np.asarray(self.samples))
        if samples.shape[1] != len(members):
            raise ValueError(
                f"samples has {samples.shape[1]} columns but cluster has "
                f"{len(members)} members"
            )
        object.__setattr__(self, "members", members)
        object.__setattr__(self, "samples", samples)
        object.__setattr__(self, "source", str(self.source))

    @property
    def size(self) -> int:
        return len(self.members)

    @property
    def n_samples(self) -> int:
        return int(self.samples.shape[0])

    @property
    def default_counts(self) -> np.ndarray:
        """Per-sample number of defaults inside this cluster (the cluster severity score)."""
        return self.samples.sum(axis=1)


def cluster_samples_from_bitstrings(
    members,
    bitstrings: np.ndarray,
    *,
    source: str = "hardware",
) -> ClusterSamples:
    """Wrap externally-measured per-cluster bitstrings (the real-hardware entry point).

    ``members`` are the global node indices of the cluster (the order of the columns in
    ``bitstrings``); ``bitstrings`` is the ``(shots, len(members))`` 0/1 array a device
    returned. No within-cluster structure is altered -- this is the literal "reconcile IBM
    machine results locally" hook.
    """
    return ClusterSamples(members=tuple(members), samples=np.asarray(bitstrings), source=source)


def sample_cluster_statevector(
    spec: SystemSpec,
    members,
    n_samples: int,
    *,
    rng: np.random.Generator,
    edge_threshold: float = 0.02,
    calibrate: bool = True,
    calibration_iterations: int = 30,
) -> ClusterSamples:
    """Simulate ONE cluster's entangled loader and draw its independent default samples.

    Builds the per-cluster (RY + CRY) block circuit from the spec's within-cluster marginals
    and covariance (:func:`systemic_risk.generators.quantum.ansatz._block_circuit`), lightly
    calibrates it against the exact statevector moments, then samples the exact Born
    distribution. This is the simulated stand-in for running the cluster on its own device; the
    returned :class:`ClusterSamples` is indistinguishable to the reconciler from a hardware one.

    Only the cluster's own qubits enter the ``2^k`` statevector (``k = len(members) <= budget``),
    so this stays cheap and per-cluster even when the whole system is far past the global
    statevector limit.
    """
    members = sorted(int(m) for m in members)
    p = np.clip(spec.marginal_default_probs, 1e-6, 1.0 - 1e-6)
    cov = A.target_covariance(spec)

    # Edges restricted to this cluster's own qubits (cross-cluster edges are deliberately
    # dropped here -- they are the boundary the classical reconciler will rebuild).
    member_set = set(members)
    global_edges = [
        (i, j)
        for (i, j) in A.dependency_edges(spec, threshold=edge_threshold, within_clusters_only=False)
        if i in member_set and j in member_set
    ]
    circuit = A._block_circuit(members, p, cov, global_edges)
    if calibrate and circuit.size > 1 and circuit.edges:
        circuit = A.calibrate_block(
            circuit, _statevector_block_moments, iterations=calibration_iterations
        )

    if circuit.size == 1:
        prob1 = float(np.sin(circuit.ry[0] / 2.0) ** 2)
        bits = (rng.random(n_samples) < prob1).astype(int)[:, None]
    else:
        state = _build_statevector(circuit.ry, circuit.edges, circuit.cry)
        bits = sample_bitstrings(state.probabilities(), circuit.size, n_samples, rng)
    return ClusterSamples(members=tuple(members), samples=bits, source="statevector")


def sample_clusters_statevector(
    spec: SystemSpec,
    partition: ClusterPartition,
    n_samples: int,
    *,
    seed: int | None = None,
    edge_threshold: float = 0.02,
    calibrate: bool = True,
    calibration_iterations: int = 30,
) -> list[ClusterSamples]:
    """Sample every cluster of a partition INDEPENDENTLY via its own statevector loader.

    Each cluster gets its own RNG stream (derived from ``seed``) so the per-cluster draws are
    genuinely independent -- exactly the situation real per-cluster devices would produce, where
    no cross-cluster co-movement is present in the raw samples. The cross-cluster co-movement is
    rebuilt afterwards by :class:`~systemic_risk.mixture.reconcile.CommonShockReconciler`.
    """
    seed_seq = np.random.SeedSequence(seed)
    child_seeds = seed_seq.spawn(len(partition.clusters))
    out: list[ClusterSamples] = []
    for members, child in zip(partition.clusters, child_seeds):
        out.append(
            sample_cluster_statevector(
                spec,
                members,
                n_samples,
                rng=np.random.default_rng(child),
                edge_threshold=edge_threshold,
                calibrate=calibrate,
                calibration_iterations=calibration_iterations,
            )
        )
    return out


# --------------------------------------------------------------------- statevector helpers
def _build_statevector(
    ry: np.ndarray, edges: list[tuple[int, int]], cry: np.ndarray
) -> StateVector:
    state = StateVector(len(ry))
    for qubit, theta in enumerate(ry):
        state.ry(qubit, float(theta))
    for (control, target), alpha in zip(edges, cry):
        state.cry(control, target, float(alpha))
    return state


def _statevector_block_moments(
    ry: np.ndarray, edges: list[tuple[int, int]], cry: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    state = _build_statevector(ry, edges, cry)
    return state.marginals(), state.pairwise_joint()
