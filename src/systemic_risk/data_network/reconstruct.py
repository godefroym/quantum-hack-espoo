"""Bilateral exposure reconstruction (pluggable).

Real bilateral interbank exposures are confidential everywhere (see
``research/sections/04_systemic_risk_measures.md`` §4 and ``data/external/CATALOG.md``), so
the field's accepted move is to *reconstruct* the matrix from public per-node marginals.
This module implements two standard, swappable methods that both honour the asset (row) and
liability (column) totals:

- ``max_entropy`` — Upper & Worms (2004) RAS/IPF: the densest, least-informative matrix
  consistent with the marginals (every bank lends a little to every other);
- ``min_density`` — Anand, Craig & von Peter (2015) style: a sparse matrix that spreads each
  bank's exposures over as few counterparties as possible (more realistic concentration).

Convention: ``W[i, j]`` is creditor ``i``'s claim on debtor ``j`` — the loss to ``i`` if
``j`` defaults — so row sums are interbank assets and column sums are interbank liabilities,
exactly the cascade engine's input.
"""

from __future__ import annotations

import numpy as np

from systemic_risk.data_network.spec import ReconstructedLayer


def _check_feasible(assets: np.ndarray, liabilities: np.ndarray) -> None:
    if assets.shape != liabilities.shape or assets.ndim != 1:
        raise ValueError("assets and liabilities must be 1-D and the same length")
    if np.any(assets < 0) or np.any(liabilities < 0):
        raise ValueError("totals must be nonnegative")
    if not np.isclose(assets.sum(), liabilities.sum(), rtol=1e-6, atol=1e-9):
        raise ValueError("asset total must equal liability total for a feasible matrix")


def max_entropy(
    assets: np.ndarray,
    liabilities: np.ndarray,
    max_iter: int = 2000,
    tol: float = 1e-9,
) -> np.ndarray:
    """RAS / iterative proportional fitting with a zero diagonal (no self-exposure).

    Seeds with the independence estimate ``a_i l_j / total`` and alternately rescales rows to
    the asset totals and columns to the liability totals, holding the diagonal at zero.
    """
    assets = np.asarray(assets, dtype=float)
    liabilities = np.asarray(liabilities, dtype=float)
    _check_feasible(assets, liabilities)
    n = assets.size
    total = assets.sum()
    if total <= 0:
        return np.zeros((n, n))

    X = np.outer(assets, liabilities) / total
    np.fill_diagonal(X, 0.0)
    for _ in range(max_iter):
        row_sums = X.sum(axis=1)
        row_scale = np.divide(assets, row_sums, out=np.ones_like(assets),
                              where=row_sums > 0)
        X *= row_scale[:, None]
        np.fill_diagonal(X, 0.0)
        col_sums = X.sum(axis=0)
        col_scale = np.divide(liabilities, col_sums, out=np.ones_like(liabilities),
                              where=col_sums > 0)
        X *= col_scale[None, :]
        np.fill_diagonal(X, 0.0)
        if (np.max(np.abs(X.sum(axis=1) - assets)) < tol
                and np.max(np.abs(X.sum(axis=0) - liabilities)) < tol):
            break
    return X


def min_density(
    assets: np.ndarray,
    liabilities: np.ndarray,
    **_: float,
) -> np.ndarray:
    """Sparse minimum-density reconstruction (deterministic greedy).

    Repeatedly places an exposure between the creditor with the most unplaced lending
    capacity and the debtor with the most unmet borrowing need, filling the smaller of the
    two each time. This honours the marginals on as few edges as possible — the sparse
    counterpoint to the dense max-entropy estimate.
    """
    assets = np.asarray(assets, dtype=float).copy()
    liabilities = np.asarray(liabilities, dtype=float).copy()
    _check_feasible(assets, liabilities)
    n = assets.size
    W = np.zeros((n, n))
    eps = 1e-12 * max(assets.sum(), 1.0)
    # Cap the number of placements; each placement exhausts one row or column.
    for _ in range(4 * n + 5):
        i = int(np.argmax(assets))
        if assets[i] <= eps:
            break
        masked = liabilities.copy()
        masked[i] = -np.inf  # no self-exposure
        j = int(np.argmax(masked))
        if liabilities[j] <= eps:
            break
        amount = min(assets[i], liabilities[j])
        W[i, j] += amount
        assets[i] -= amount
        liabilities[j] -= amount
    np.fill_diagonal(W, 0.0)
    return W


_METHODS = {"max_entropy": max_entropy, "min_density": min_density}


def reconstruct(
    method: str,
    assets: np.ndarray,
    liabilities: np.ndarray,
    single_counterparty_cap: np.ndarray | None = None,
    record: dict[str, float] | None = None,
    **method_kwargs: float,
) -> ReconstructedLayer:
    """Dispatch to a named method and wrap the result in a ``ReconstructedLayer``.

    ``method_kwargs`` are forwarded to the chosen method; ``record`` holds extra provenance
    values to store on the layer (not passed to the method). If ``single_counterparty_cap``
    (per-creditor) is given, each exposure ``W[i, j]`` is capped at that creditor's limit
    (Basel large-exposure rule).
    """
    if method not in _METHODS:
        raise ValueError(f"unknown reconstruction method {method!r}; "
                         f"choose from {sorted(_METHODS)}")
    W = _METHODS[method](assets, liabilities, **method_kwargs)
    recorded = dict(method_kwargs)
    recorded.update(record or {})
    if single_counterparty_cap is not None:
        cap = np.asarray(single_counterparty_cap, dtype=float)
        W = np.minimum(W, cap[:, None])
        np.fill_diagonal(W, 0.0)
        recorded["single_counterparty_cap_applied"] = 1.0
    return ReconstructedLayer(edge_matrix=W, method=method, method_params=recorded)
