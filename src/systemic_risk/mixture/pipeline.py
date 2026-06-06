"""End-to-end glue for the classical+quantum mixture pipeline and its honest diagnostics.

This ties the pieces together and provides the comparison metrics the make-or-break experiment
needs, so both the script and the tests share one implementation:

* :func:`attach_cluster_exposures` -- the planted-cluster test instance is a pure
  generation/correlation instance (zero exposure graph). To exercise the GLOBAL cascade we
  overlay a deterministic exposure graph with strong within-cluster lending and weak
  cross-cluster links, leaving marginals/correlations untouched. The same overlay is applied to
  reference, reconciled, and independent scenarios so the cascade comparison is apples-to-apples.
* :func:`default_count_distribution` / :func:`total_variation` / :func:`kl_divergence` -- the
  default-count law and its distance from a reference, the headline tail diagnostic swept over
  coupling strength.
* :func:`cascade_loss_cvar` -- tail cascade loss (CVaR of the post-contagion default count),
  reusing :func:`systemic_risk.evaluation.cascade_count_cvar`.
* :func:`reconciliation_diagnostics` -- the full reconciled-vs-independent-vs-reference report
  block for one coupling level.
"""

from __future__ import annotations

import numpy as np

from systemic_risk.evaluation import cascade_count_cvar, joint_tail_excess
from systemic_risk.simulator import simulate_many
from systemic_risk.spec import SystemSpec


def attach_cluster_exposures(
    spec: SystemSpec,
    labels: np.ndarray,
    *,
    within_exposure: float = 0.35,
    cross_exposure: float = 0.05,
    capital: float = 1.0,
    seed: int = 0,
) -> SystemSpec:
    """Return a copy of ``spec`` with a cluster-aware exposure graph for the cascade.

    ``W[i, j]`` (loss to ``i`` when ``j`` defaults) is set to ``within_exposure`` for same-cluster
    ordered pairs and ``cross_exposure`` for cross-cluster pairs, each jittered deterministically
    so the cascade is not degenerate. Capital buffers are uniform ``capital``. Marginals and the
    target correlation are carried over unchanged -- only the contagion channel is added -- so the
    generation side of the instance is unaffected.

    Cross-cluster exposure is deliberately small relative to within-cluster: contagion mostly
    stays inside a cluster, and a SYSTEMIC cascade needs several clusters stressed at once, which
    is exactly the cross-cluster co-movement the reconciler rebuilds (and the independent baseline
    misses).
    """
    labels = np.asarray(labels, dtype=int)
    n = spec.n
    rng = np.random.default_rng(seed)
    same = labels[:, None] == labels[None, :]
    base = np.where(same, within_exposure, cross_exposure)
    jitter = rng.uniform(0.8, 1.2, size=(n, n))
    W = base * jitter
    np.fill_diagonal(W, 0.0)

    return SystemSpec(
        node_names=spec.node_names,
        node_types=spec.node_types,
        exposure_matrix=W,
        capital_buffers=np.full(n, float(capital)),
        marginal_default_probs=spec.marginal_default_probs,
        target_pairwise_corr=spec.target_pairwise_corr,
        clusters=spec.clusters,
        metadata={**spec.metadata, "exposure_overlay": {
            "within_exposure": float(within_exposure),
            "cross_exposure": float(cross_exposure),
            "capital": float(capital),
        }},
    )


def default_count_distribution(samples: np.ndarray, n: int) -> np.ndarray:
    """Return the normalised pmf of the per-scenario default count over ``0..n``."""
    counts = np.asarray(samples).sum(axis=1).astype(int)
    pmf = np.bincount(counts, minlength=n + 1).astype(float)
    total = pmf.sum()
    return pmf / total if total > 0 else pmf


def tail_count_l1(p: np.ndarray, q: np.ndarray, threshold: int) -> float:
    """``sum_{k >= threshold} |p_k - q_k|`` -- L1 distance on the upper tail of the count pmf.

    The full-distribution TV is dominated by the count-0/1 bulk, which is set by the
    (loader-limited) WITHIN-cluster law shared by every method; the systemically interesting
    discrepancy lives in the upper tail, where cross-cluster co-movement piles up many-at-once
    co-defaults. This restricts the comparison to ``k >= threshold`` so it actually measures the
    tail the reconciliation is for.
    """
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    return float(np.abs(p[threshold:] - q[threshold:]).sum())


def total_variation(p: np.ndarray, q: np.ndarray) -> float:
    """Total-variation distance ``0.5 * sum |p - q|`` between two pmfs of equal length."""
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    return float(0.5 * np.abs(p - q).sum())


def kl_divergence(p: np.ndarray, q: np.ndarray, *, eps: float = 1e-9) -> float:
    """``KL(p || q)`` with ``q`` smoothed by ``eps`` to stay finite where ``q`` has no mass."""
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float) + eps
    q = q / q.sum()
    mask = p > 0
    return float(np.sum(p[mask] * np.log(p[mask] / q[mask])))


def cascade_loss_cvar(
    samples: np.ndarray,
    cascade_spec: SystemSpec,
    *,
    alpha: float = 0.95,
    max_eval: int | None = None,
    seed: int = 0,
) -> tuple[float, np.ndarray]:
    """Run the global cascade on ``samples`` and return ``(CVaR_alpha, post_cascade_counts)``.

    ``cascade_spec`` must carry the exposure overlay (see :func:`attach_cluster_exposures`).
    ``CVaR_alpha`` is the conditional-value-at-risk of the post-contagion default count -- the
    tail cascade loss. ``max_eval`` subsamples for speed; the subsample is deterministic in
    ``seed``.
    """
    samples = np.asarray(samples, dtype=int)
    if max_eval is not None and samples.shape[0] > max_eval:
        rng = np.random.default_rng(seed)
        idx = rng.choice(samples.shape[0], size=max_eval, replace=False)
        samples = samples[idx]
    results = simulate_many(samples, cascade_spec)
    counts = np.array([r.failure_count for r in results], dtype=int)
    return cascade_count_cvar(counts, alpha=alpha), counts


def reconciliation_diagnostics(
    reference: np.ndarray,
    reconciled: np.ndarray,
    independent: np.ndarray,
    labels: np.ndarray,
    n: int,
    *,
    cascade_spec: SystemSpec | None = None,
    tail_fraction: float = 0.5,
    cvar_alpha: float = 0.95,
    cascade_max_eval: int | None = 4000,
) -> dict:
    """Compare reconciled and independent global joints against the ground-truth reference.

    Reports, for each of {reconciled, independent}, the absolute error against the reference on:
    the marginals (RMSE), the cross-cluster co-default correlation (mean), the default-count law
    (total variation + KL to the reference), the joint-tail excess, and -- when ``cascade_spec``
    is supplied -- the cascade tail loss (CVaR of the post-contagion count). The honest headline
    is that the reconciled errors are smaller than the independent ones, especially in the tail.
    """
    from systemic_risk.mixture.reconcile import _binary_corr, _mean_cross_block

    labels = np.asarray(labels, dtype=int)
    ref_pmf = default_count_distribution(reference, n)
    ref_marg = reference.mean(axis=0)
    ref_cross = _mean_cross_block(_binary_corr(reference), labels)
    ref_jte = joint_tail_excess(reference, tail_fraction)
    tail_threshold = int(np.ceil(tail_fraction * n))

    def block(samples: np.ndarray) -> dict:
        pmf = default_count_distribution(samples, n)
        cross = _mean_cross_block(_binary_corr(samples), labels)
        entry = {
            "marginal_rmse_vs_ref": float(
                np.sqrt(np.mean((samples.mean(axis=0) - ref_marg) ** 2))
            ),
            "cross_cluster_corr": float(cross),
            "cross_cluster_corr_abs_err_vs_ref": float(abs(cross - ref_cross)),
            "count_tv_vs_ref": total_variation(pmf, ref_pmf),
            "count_kl_vs_ref": kl_divergence(ref_pmf, pmf),
            "tail_count_l1_vs_ref": tail_count_l1(pmf, ref_pmf, tail_threshold),
            "joint_tail_excess": float(joint_tail_excess(samples, tail_fraction)),
            "joint_tail_excess_abs_err_vs_ref": float(
                abs(joint_tail_excess(samples, tail_fraction) - ref_jte)
            ),
        }
        return entry

    out = {
        "reference": {
            "cross_cluster_corr": float(ref_cross),
            "joint_tail_excess": float(ref_jte),
            "mean_default_count": float(reference.sum(axis=1).mean()),
        },
        "reconciled": block(reconciled),
        "independent": block(independent),
    }

    if cascade_spec is not None:
        ref_cvar, ref_counts = cascade_loss_cvar(
            reference, cascade_spec, alpha=cvar_alpha, max_eval=cascade_max_eval
        )
        out["reference"]["cascade_count_cvar"] = float(ref_cvar)
        out["reference"]["mean_cascade_count"] = float(ref_counts.mean())
        for key, samples in (("reconciled", reconciled), ("independent", independent)):
            cvar, counts = cascade_loss_cvar(
                samples, cascade_spec, alpha=cvar_alpha, max_eval=cascade_max_eval
            )
            out[key]["cascade_count_cvar"] = float(cvar)
            out[key]["cascade_count_cvar_abs_err_vs_ref"] = float(abs(cvar - ref_cvar))
            out[key]["mean_cascade_count"] = float(counts.mean())

    return out
