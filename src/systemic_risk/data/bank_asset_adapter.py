from __future__ import annotations

import numpy as np
from scipy.optimize import brentq

from systemic_risk.bank_asset_spec import BankAssetSystemSpec
from systemic_risk.spec import SystemSpec


def bank_asset_to_system_spec(
    bank_asset_spec: BankAssetSystemSpec,
    *,
    alpha: float | np.ndarray = 0.08,
    marginal_default_probs: np.ndarray | None = None,
    mean_default_probability: float = 0.04,
    vulnerability_slope: float = 1.0,
    correlation_floor: float = 0.002,
    overlap_correlation_scale: float = 0.05,
) -> SystemSpec:
    """Adapt a bank-asset system to the common generator SystemSpec.

    The exposure matrix is the first-order loss to bank i caused by bank j's
    liquidation under the Huang price-impact rule:

        W[i, j] = sum_m B[i, m] alpha[m] B[j, m] / A[m].

    Pairwise default targets are a PSD correlation kernel derived from common
    portfolio overlap. Marginals can be supplied externally; otherwise a
    transparent equity-ratio vulnerability heuristic is calibrated to the
    requested system-wide mean default probability.
    """
    alpha_by_asset = _resolve_alpha(bank_asset_spec, alpha)
    p = _resolve_marginals(
        bank_asset_spec,
        marginal_default_probs,
        mean_default_probability,
        vulnerability_slope,
    )
    overlap = _portfolio_cosine_similarity(bank_asset_spec.portfolio_weights)
    corr = _overlap_correlation(
        overlap,
        floor=correlation_floor,
        scale=overlap_correlation_scale,
    )
    exposure_matrix = _first_order_liquidation_losses(
        bank_asset_spec,
        alpha_by_asset,
    )
    dominant_assets = np.argmax(bank_asset_spec.portfolio_weights, axis=1)
    clusters = [
        f"asset:{bank_asset_spec.asset_names[asset_idx]}"
        for asset_idx in dominant_assets
    ]

    return SystemSpec(
        node_names=bank_asset_spec.bank_names,
        node_types=["bank"] * bank_asset_spec.n_banks,
        exposure_matrix=exposure_matrix,
        capital_buffers=bank_asset_spec.equity,
        marginal_default_probs=p,
        target_pairwise_corr=corr,
        clusters=clusters,
        metadata={
            "name": f"Generator view of {bank_asset_spec.metadata.get('name', 'bank-asset system')}",
            "source_model": "Huang bank-asset fire-sale adapter",
            "alpha_by_asset": alpha_by_asset.tolist(),
            "mean_default_probability": float(p.mean()),
            "marginal_method": (
                "provided"
                if marginal_default_probs is not None
                else "equity-ratio vulnerability calibrated to requested mean"
            ),
            "pairwise_method": "PSD cosine-overlap correlation kernel",
            "exposure_method": "first-order liquidation loss",
        },
    )


def _first_order_liquidation_losses(
    spec: BankAssetSystemSpec,
    alpha_by_asset: np.ndarray,
) -> np.ndarray:
    market_share = spec.holdings / spec.market_values
    losses = (spec.holdings * alpha_by_asset) @ market_share.T
    np.fill_diagonal(losses, 0.0)
    return losses


def _portfolio_cosine_similarity(weights: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(weights, axis=1)
    normalized = np.divide(
        weights,
        norms[:, None],
        out=np.zeros_like(weights),
        where=norms[:, None] > 0,
    )
    similarity = normalized @ normalized.T
    np.fill_diagonal(similarity, 1.0)
    return np.clip(similarity, 0.0, 1.0)


def _overlap_correlation(
    similarity: np.ndarray,
    *,
    floor: float,
    scale: float,
) -> np.ndarray:
    if floor < 0 or scale < 0 or floor + scale >= 1:
        raise ValueError("correlation_floor and overlap scale must be nonnegative and sum to < 1")
    n = similarity.shape[0]
    corr = (
        (1 - floor - scale) * np.eye(n)
        + floor * np.ones((n, n))
        + scale * similarity
    )
    np.fill_diagonal(corr, 1.0)
    return corr


def _resolve_marginals(
    spec: BankAssetSystemSpec,
    supplied: np.ndarray | None,
    target_mean: float,
    slope: float,
) -> np.ndarray:
    if supplied is not None:
        p = np.asarray(supplied, dtype=float)
        if p.shape != (spec.n_banks,):
            raise ValueError("marginal_default_probs must have shape (n_banks,)")
        if np.any((p <= 0) | (p >= 1)):
            raise ValueError("marginal_default_probs must lie strictly in (0, 1)")
        return p.copy()

    min_probability = 0.002
    max_probability = 0.25
    if not min_probability <= target_mean <= max_probability:
        raise ValueError(
            f"mean_default_probability must lie in [{min_probability}, {max_probability}]"
        )
    if slope < 0:
        raise ValueError("vulnerability_slope must be nonnegative")

    equity_ratio = spec.equity / spec.total_assets
    standard_deviation = float(equity_ratio.std())
    if standard_deviation == 0 or slope == 0:
        return np.full(spec.n_banks, target_mean)
    vulnerability = -(equity_ratio - equity_ratio.mean()) / standard_deviation

    def probabilities(intercept: float) -> np.ndarray:
        raw = 1.0 / (1.0 + np.exp(-np.clip(intercept + slope * vulnerability, -40, 40)))
        return np.clip(raw, min_probability, max_probability)

    intercept = brentq(
        lambda value: float(probabilities(value).mean() - target_mean),
        -40.0,
        40.0,
    )
    return probabilities(intercept)


def _resolve_alpha(
    spec: BankAssetSystemSpec,
    alpha: float | np.ndarray,
) -> np.ndarray:
    values = np.asarray(alpha, dtype=float)
    if values.ndim == 0:
        values = np.full(spec.n_assets, float(values))
    if values.shape != (spec.n_assets,):
        raise ValueError("alpha must be a scalar or have shape (n_assets,)")
    if np.any((values < 0) | (values > 1)):
        raise ValueError("alpha values must lie in [0, 1]")
    return values
