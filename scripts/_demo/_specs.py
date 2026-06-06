"""Demonstration specs and the achievable (Fréchet) correlation ceiling.

Why these specs
---------------
The headline foundation is the **real 28-bank G-SIB network** (``build_system_spec``). But two
verified limitations shape *where* each success criterion can be tested honestly:

1. The heterogeneous entangled ansatz is **block-separable by community** once ``n`` exceeds
   ``max_block_qubits`` (default 22). On the full 28-bank spec it splits into its 3 communities,
   so cross-cluster correlation collapses toward independence (verified: generated cross-cluster
   mean ≈ 0 vs target ≈ 0.47) while within-cluster correlation is captured. A clean
   "interchangeable at 2nd order" claim therefore cannot be made on the whole network.
2. About a third of the real target correlations exceed the **Fréchet bound** for Bernoulli
   marginals at those tiny default probabilities — unreachable by *any* binary generator. Match
   must be judged against the achievable ceiling, not the nominal target.

So the demonstration uses three lenses, each chosen for what it can honestly show:

* :func:`real_full_spec`   -- the whole 28-bank network. Used for the network picture and to
  *demonstrate* limitation (1) with numbers, not to claim a 2nd-order drop-in.
* :func:`real_community_spec` -- the largest detected community (``community_0``, n = 14). Small
  enough that the entangled generator is a **single, fully-simulated block** (no separability
  artifact), so the criterion-1 match is assessed where it is actually meaningful, and criteria
  2 & 3 are evaluated on genuine real-bank exposures.
* :func:`synthetic_scale_spec` + :func:`homogeneous_oracle_spec` -- the n = 54 scale story.

All specs are deterministic. ``build_system_spec`` reads a committed offline snapshot by default;
if it ever needs the network and fails, we fall back to the calibrated-synthetic builder and the
caller surfaces which was used.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from systemic_risk.data_network import build_synthetic_system_spec, build_system_spec
from systemic_risk.spec import SystemSpec


@dataclass(frozen=True)
class SpecBundle:
    """A demonstration spec plus the provenance label and offline-fallback flag to report."""

    spec: SystemSpec
    label: str
    used_fallback: bool


def real_full_spec() -> SpecBundle:
    """Return the full real 28-bank network (offline snapshot), or the synthetic fallback."""
    try:
        spec = build_system_spec()
        return SpecBundle(spec, "real 28-bank G-SIB network (offline snapshot)", False)
    except Exception:  # pragma: no cover - only when the offline snapshot is unavailable
        spec = build_synthetic_system_spec(n=28)
        return SpecBundle(spec, "calibrated-synthetic n=28 (real build unavailable offline)", True)


def real_community_spec(bundle: SpecBundle | None = None) -> SpecBundle:
    """Return the largest detected community of the real network as a standalone spec.

    Selecting one community keeps the spec under ``max_block_qubits`` so the entangled
    generator runs as a single, fully-simulated block — the regime where its 2nd-order match
    is a fair, artifact-free test. The sub-spec carries the real marginals, the real
    within-community correlation block, and the real exposure sub-matrix.
    """
    base = bundle or real_full_spec()
    spec = base.spec
    if spec.clusters is None:
        return SpecBundle(spec, base.label, base.used_fallback)
    clusters = np.asarray(spec.clusters)
    labels, counts = np.unique(clusters, return_counts=True)
    largest = labels[int(np.argmax(counts))]
    idx = np.sort(np.where(clusters == largest)[0])
    sub = _subset(spec, idx, note=f"{base.label} — community '{largest}'")
    label = f"{base.label}, largest community '{largest}' (n={sub.n}, single block)"
    return SpecBundle(sub, label, base.used_fallback)


def synthetic_scale_spec(n: int = 54, seed: int = 11) -> SpecBundle:
    """Return the calibrated-synthetic spec at the quantum-hardware target size ``n``."""
    spec = build_synthetic_system_spec(n=n, seed=seed)
    return SpecBundle(spec, f"calibrated-synthetic n={n} (scale vehicle)", False)


def homogeneous_credit_spec(
    n: int = 14, marginal: float = 0.0022, default_corr: float = 0.5
) -> SystemSpec:
    """Return a small homogeneous spec at a *real-credit* marginal and a feasible equicorrelation.

    This is the exchangeable counterpart of the real community: same tiny default level, but a
    uniform marginal and a single equicorrelation. It is the regime where the entangled
    construction is provably interchangeable at 2nd order — the clean criterion-1 capability test,
    contrasted in the run with the heterogeneous real community where the CRY ansatz cannot satisfy
    all conflicting per-edge correlations at once. Defaults mirror ``community_0`` (n≈14, p≈0.0022).
    """
    corr = np.full((n, n), float(default_corr))
    np.fill_diagonal(corr, 1.0)
    return SystemSpec(
        node_names=[f"H{i:02d}" for i in range(n)],
        node_types=["bank"] * n,
        exposure_matrix=np.zeros((n, n)),
        capital_buffers=np.ones(n),
        marginal_default_probs=np.full(n, float(marginal)),
        target_pairwise_corr=corr,
        clusters=["c0"] * n,
        metadata={"kind": "homogeneous-credit-criterion1", "marginal": marginal,
                  "default_corr": default_corr},
    )


def homogeneous_oracle_spec(
    n: int = 54, marginal: float = 0.02, default_corr: float = 0.25
) -> SystemSpec:
    """Return a homogeneous (uniform-marginal, equicorrelated) spec at size ``n``.

    This is the exactly-solvable limit the entangled construction targets at scale: the
    permutation-symmetric loader reproduces the closed-form mean-field Ising loss-count law to
    machine precision at any ``n`` (no ``2^n`` state). Used only for the n = 54 oracle check.
    """
    corr = np.full((n, n), float(default_corr))
    np.fill_diagonal(corr, 1.0)
    return SystemSpec(
        node_names=[f"I{i:02d}" for i in range(n)],
        node_types=["bank"] * n,
        exposure_matrix=np.zeros((n, n)),
        capital_buffers=np.ones(n),
        marginal_default_probs=np.full(n, float(marginal)),
        target_pairwise_corr=corr,
        clusters=["c0"] * n,
        metadata={"kind": "homogeneous-oracle-validation", "marginal": marginal,
                  "default_corr": default_corr},
    )


def _subset(spec: SystemSpec, idx: np.ndarray, note: str) -> SystemSpec:
    """Extract the index subset of a spec as a fresh, validated ``SystemSpec``."""
    grid = np.ix_(idx, idx)
    corr = None if spec.target_pairwise_corr is None else spec.target_pairwise_corr[grid].copy()
    clusters = None if spec.clusters is None else [spec.clusters[i] for i in idx]
    metadata = {**spec.metadata, "subset_of": note, "subset_indices": idx.tolist()}
    return SystemSpec(
        node_names=[spec.node_names[i] for i in idx],
        node_types=[spec.node_types[i] for i in idx],
        exposure_matrix=spec.exposure_matrix[grid].copy(),
        capital_buffers=spec.capital_buffers[idx].copy(),
        marginal_default_probs=spec.marginal_default_probs[idx].copy(),
        target_pairwise_corr=corr,
        clusters=clusters,
        metadata=metadata,
    )


def frechet_corr_bounds(p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(corr_min, corr_max)`` matrices: the Fréchet bounds on Bernoulli correlation.

    For indicators with marginals ``p_i, p_j`` the co-default probability is confined to
    ``[max(0, p_i+p_j-1), min(p_i, p_j)]`` (the Fréchet–Hoeffding bounds). Converting those
    bounds to Pearson correlation gives, per pair, the strongest negative/positive correlation
    *any* binary model can produce. The diagonal is set to ``±1``.
    """
    p = np.clip(np.asarray(p, dtype=float), 1e-12, 1.0 - 1e-12)
    outer = np.outer(p, p)
    denom = np.sqrt(np.outer(p * (1.0 - p), p * (1.0 - p)))
    upper_joint = np.minimum.outer(p, p)
    lower_joint = np.maximum(np.add.outer(p, p) - 1.0, 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        corr_max = np.where(denom > 0, (upper_joint - outer) / denom, 0.0)
        corr_min = np.where(denom > 0, (lower_joint - outer) / denom, 0.0)
    np.fill_diagonal(corr_max, 1.0)
    np.fill_diagonal(corr_min, -1.0)
    return corr_min, corr_max


def achievable_corr(spec: SystemSpec) -> np.ndarray:
    """Return the target correlation clipped into the per-pair Fréchet-achievable band.

    This is the ceiling a binary generator can be fairly judged against on tiny-marginal credit
    targets: where the nominal target is feasible it is unchanged; where it exceeds the bound it
    is replaced by the closest attainable value.
    """
    if spec.target_pairwise_corr is None:
        return np.eye(spec.n)
    corr_min, corr_max = frechet_corr_bounds(spec.marginal_default_probs)
    return np.clip(spec.target_pairwise_corr, corr_min, corr_max)


def infeasible_fraction(spec: SystemSpec, tol: float = 1e-9) -> float:
    """Return the fraction of off-diagonal target correlations that exceed the Fréchet bound."""
    if spec.target_pairwise_corr is None:
        return 0.0
    _, corr_max = frechet_corr_bounds(spec.marginal_default_probs)
    iu = np.triu_indices(spec.n, k=1)
    target = spec.target_pairwise_corr[iu]
    return float(np.mean(target > corr_max[iu] + tol))
