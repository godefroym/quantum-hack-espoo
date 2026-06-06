"""Calibrate Ising fields ``h_i`` and couplings ``J_ij`` from systemic-risk inputs.

Three calibration surfaces, each with a literature anchor (see ``research/`` and
``research/sections/03_statistical_mechanics_ising.md``):

1. **Field <- marginal.** ``h_i = logit(p_i) = ln(p_i / (1 - p_i))`` is the ``{0, 1}``-basis
   max-entropy field reproducing marginal ``p_i`` *when couplings are off* (the doc's
   ``ln p_i`` is its small-``p`` approximation). Once ``J != 0`` the marginals drift, so the
   fields are **refit by Boltzmann learning** (``delta h_i propto p_i - <x_i>_model``) until
   the model marginals match the targets. (Filiz et al. 2012 Cor. 5; Schneidman et al. 2006.)

2. **Coupling <- correlation (inverse Ising).** Naive mean-field ``K_ij ~ -(C^-1)_ij`` (with
   optional TAP correction), where ``C`` is the *+/-1* connected-correlation matrix; the
   resulting +/-1 couplings are mapped to the ``{0, 1}`` basis via ``J_ij = 4 K_ij``.
   (Bury 2013; Nguyen, Zecchina & Berg 2017.)

3. **Coupling <- exposure (density-corrected gravity model).** Reconstruct a sparse exposure
   graph from node "fitness" (size) with ``p_ij = z chi_i psi_j / (1 + z chi_i psi_j)`` and
   gravity weights ``w_ij propto chi_i psi_j``, then set ``J_ij propto w_ij``.
   (Cimini, Squartini, Garlaschelli & Gabrielli 2015.)

Convention note. We keep ``x_i in {0, 1}`` (default indicator) everywhere, matching
:class:`systemic_risk.spec.SystemSpec`. Inverse-Ising results that are classically stated for
``sigma_i in {-1, +1}`` are derived there and converted with ``sigma_i = 2 x_i - 1`` (which
gives ``J^{01}_ij = 4 K^{+/-1}_ij`` for the off-diagonal couplings).
"""

from __future__ import annotations

import numpy as np

from systemic_risk.spec import (
    CORRELATION_SPACE_LATENT_GAUSSIAN,
    SystemSpec,
    joint_to_corr,
)

_EPS = 1e-9


def logit_fields(p: np.ndarray) -> np.ndarray:
    """Return the independent-baseline fields ``h_i = logit(p_i)``."""
    p = np.clip(np.asarray(p, dtype=float), _EPS, 1.0 - _EPS)
    return np.log(p / (1.0 - p))


def _sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40.0, 40.0)))


def mean_field_marginals(
    fields: np.ndarray,
    couplings: np.ndarray,
    *,
    max_iter: int = 2_000,
    tol: float = 1e-9,
    damping: float = 0.5,
) -> np.ndarray:
    """Naive mean-field estimate of the model marginals ``<x_i>``.

    Solves the self-consistency ``m_i = sigmoid(h_i + sum_j J_ij m_j)`` by damped fixed-point
    iteration. This is the cheap (``O(n^2)`` per sweep) proxy for the model marginals used in
    the large-``n`` Boltzmann-learning loop, where exact enumeration is impossible.
    """
    m = _sigmoid(fields)
    for _ in range(max_iter):
        new = _sigmoid(fields + couplings @ m)
        new = damping * new + (1.0 - damping) * m
        if np.max(np.abs(new - m)) < tol:
            m = new
            break
        m = new
    return np.asarray(m, dtype=float)


def fit_fields_boltzmann(
    target_p: np.ndarray,
    couplings: np.ndarray,
    *,
    initial_fields: np.ndarray | None = None,
    learning_rate: float = 1.0,
    max_iter: int = 500,
    tol: float = 1e-4,
    moment_fn=None,
):
    """Refit fields so the model marginals match ``target_p`` (Boltzmann learning).

    Update rule (Schneidman et al. 2006; the unique-MLE statement of Filiz et al. 2012,
    Cor. 5)::

        h_i <- h_i + learning_rate * (target_p_i - <x_i>_model).

    ``moment_fn(fields, couplings) -> marginals`` supplies the model marginals; it defaults to
    the naive mean-field estimate :func:`mean_field_marginals`, which is what makes the refit
    affordable at ``n = 54``. Pass an exact-enumeration callback for small ``n``.

    Returns ``(fields, info)`` where ``info`` holds the final marginal RMSE and iteration
    count.
    """
    target_p = np.clip(np.asarray(target_p, dtype=float), _EPS, 1.0 - _EPS)
    couplings = np.asarray(couplings, dtype=float)
    if moment_fn is None:
        moment_fn = mean_field_marginals

    fields = logit_fields(target_p) if initial_fields is None else np.array(initial_fields, dtype=float)
    rmse = np.inf
    iteration = 0
    for iteration in range(1, max_iter + 1):
        model_p = np.asarray(moment_fn(fields, couplings), dtype=float)
        residual = target_p - model_p
        rmse = float(np.sqrt(np.mean(residual**2)))
        if rmse < tol:
            break
        fields = fields + learning_rate * residual
    info = {"marginal_rmse": rmse, "iterations": iteration}
    return fields, info


def couplings_from_correlation(
    corr: np.ndarray,
    p: np.ndarray,
    *,
    method: str = "tap",
    coupling_cap: float = 6.0,
) -> np.ndarray:
    """Inverse-Ising couplings ``J_ij`` from a target Bernoulli correlation matrix.

    ``method`` is ``"nmf"`` (naive mean-field, ``K = -C^-1`` off-diagonal) or ``"tap"`` (adds
    the Onsager reaction correction). ``C`` is the connected-correlation (covariance) matrix
    in the ``+/-1`` convention, built from the target Pearson correlations and the ``+/-1``
    variances ``var(sigma_i) = 4 p_i (1 - p_i)``. The inferred ``+/-1`` couplings are mapped
    to the ``{0, 1}`` basis via ``J_ij = 4 K_ij`` and clipped to ``[-cap, cap]`` for
    numerical sanity. (Bury 2013; Nguyen, Zecchina & Berg 2017.)
    """
    corr = np.asarray(corr, dtype=float)
    p = np.clip(np.asarray(p, dtype=float), _EPS, 1.0 - _EPS)
    n = len(p)

    # Pearson correlation is invariant under sigma = 2x - 1, so corr(sigma) == corr(x).
    corr = (corr + corr.T) / 2.0
    np.fill_diagonal(corr, 1.0)
    # Project to PSD so the inverse is well behaved.
    eigvals, eigvecs = np.linalg.eigh(corr)
    eigvals = np.clip(eigvals, 1e-6, None)
    corr = (eigvecs * eigvals) @ eigvecs.T
    d = np.sqrt(np.diag(corr))
    corr = corr / np.outer(d, d)
    np.fill_diagonal(corr, 1.0)

    # +/-1 covariance: std(sigma_i) = 2 sqrt(p_i (1 - p_i)).
    std_pm = 2.0 * np.sqrt(p * (1.0 - p))
    cov = corr * np.outer(std_pm, std_pm)
    cov_inv = np.linalg.inv(cov + 1e-9 * np.eye(n))

    k = -cov_inv.copy()
    np.fill_diagonal(k, 0.0)

    if method == "nmf":
        pass
    elif method == "tap":
        # TAP: (C^-1)_ij ~ -K_ij - K_ij^2 m_i m_j, with m the +/-1 magnetisations.
        # Solve the quadratic K + K^2 (m_i m_j) = -C^-1 elementwise for the off-diagonal.
        m_pm = 2.0 * p - 1.0  # <sigma_i> = 2 p_i - 1
        mm = np.outer(m_pm, m_pm)
        a = mm
        b = np.ones_like(k)
        c = cov_inv  # = -(-C^-1) moved to RHS: a K^2 + b K + c = 0
        np.fill_diagonal(c, 0.0)
        disc = b**2 - 4.0 * a * c
        disc = np.clip(disc, 0.0, None)
        with np.errstate(divide="ignore", invalid="ignore"):
            root = (-b + np.sqrt(disc)) / (2.0 * a)
        # Where a ~ 0 (m_i m_j ~ 0) the quadratic degenerates to the nMF linear solution.
        small = np.abs(a) < 1e-8
        k_tap = np.where(small, k, root)
        np.fill_diagonal(k_tap, 0.0)
        # Guard against TAP blow-ups: fall back to nMF entrywise if non-finite.
        k = np.where(np.isfinite(k_tap), k_tap, k)
    else:
        raise ValueError(f"unknown inverse-Ising method {method!r}")

    j01 = 4.0 * k  # +/-1 -> {0,1} off-diagonal coupling conversion
    j01 = (j01 + j01.T) / 2.0
    np.fill_diagonal(j01, 0.0)
    return np.clip(j01, -coupling_cap, coupling_cap)


def reconstruct_exposure_graph(
    fitness: np.ndarray,
    target_density: float,
    *,
    total_weight: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Density-corrected gravity-model reconstruction of an exposure graph.

    Given per-node "fitness" (size proxy) ``chi_i`` and a target link density, returns
    ``(p_link, weights)`` where ``p_link[i, j] = z chi_i chi_j / (1 + z chi_i chi_j)`` and
    ``weights[i, j]`` are the expected gravity weights (zero off the expected support, and
    nonnegative). ``z`` is the single global parameter fixed so the expected number of links
    equals ``target_density * n (n - 1) / 2``. (Cimini et al. 2015.)
    """
    fitness = np.asarray(fitness, dtype=float)
    n = len(fitness)
    if n < 2:
        return np.zeros((n, n)), np.zeros((n, n))
    fitness = fitness / (fitness.mean() + _EPS)  # scale-free of absolute units
    target_density = float(np.clip(target_density, 1.0 / max(1, n * (n - 1) // 2), 0.999))
    target_links = target_density * n * (n - 1) / 2.0
    outer = np.outer(fitness, fitness)
    iu = np.triu_indices(n, k=1)
    chi_pairs = outer[iu]

    def expected_links(z: float) -> float:
        t = z * chi_pairs
        return float(np.sum(t / (1.0 + t)))

    # Bisection on log(z): expected_links is monotone increasing in z.
    lo, hi = 1e-8, 1e8
    for _ in range(200):
        mid = np.sqrt(lo * hi)
        if expected_links(mid) < target_links:
            lo = mid
        else:
            hi = mid
    z = np.sqrt(lo * hi)

    p_link = np.zeros((n, n))
    t = z * outer
    p_link = t / (1.0 + t)
    np.fill_diagonal(p_link, 0.0)
    p_link = (p_link + p_link.T) / 2.0

    weights = outer.copy()
    np.fill_diagonal(weights, 0.0)
    if total_weight is not None:
        scale = total_weight / (weights.sum() + _EPS)
        weights = weights * scale
    return p_link, weights


def couplings_from_exposure(
    exposure_matrix: np.ndarray,
    *,
    coupling_scale: float = 2.0,
    fitness: np.ndarray | None = None,
    target_density: float | None = None,
    coupling_cap: float = 6.0,
) -> np.ndarray:
    """Couplings ``J_ij propto w_ij`` from an exposure / adjacency matrix.

    If the exposure matrix already carries bilateral weights (our scalable specs do), the
    symmetric mutual exposure ``w_ij = W_ij + W_ji`` is used directly. If it is empty but
    node ``fitness`` and a ``target_density`` are supplied, the graph is first reconstructed
    with the density-corrected gravity model (:func:`reconstruct_exposure_graph`). Couplings
    are normalised to unit mean over realised edges, scaled by ``coupling_scale``, and
    clipped to ``[-cap, cap]``. (Cimini et al. 2015 for the reconstruction step.)
    """
    exposure_matrix = np.asarray(exposure_matrix, dtype=float)
    n = exposure_matrix.shape[0]
    w = exposure_matrix + exposure_matrix.T

    if w.sum() <= 0 and fitness is not None and target_density is not None:
        _, w = reconstruct_exposure_graph(fitness, target_density)

    np.fill_diagonal(w, 0.0)
    positive = w[w > 0]
    if positive.size == 0:
        return np.zeros((n, n))
    mean_weight = positive.mean()
    j = coupling_scale * (w / mean_weight)
    j = (j + j.T) / 2.0
    np.fill_diagonal(j, 0.0)
    return np.clip(j, -coupling_cap, coupling_cap)


def couplings_from_spec(
    spec: SystemSpec,
    *,
    route: str = "auto",
    coupling_scale: float = 2.0,
    correlation_method: str = "tap",
    coupling_cap: float = 6.0,
) -> tuple[np.ndarray, str]:
    """Derive ``J_ij`` from a :class:`SystemSpec`, choosing a route.

    ``route``:

    - ``"correlation"`` -- inverse Ising from ``spec.target_pairwise_corr``.
    - ``"exposure"`` -- gravity-weighted from ``spec.exposure_matrix``.
    - ``"auto"`` -- prefer ``"correlation"`` when a target correlation is present (it pins the
      pairwise moments directly), else fall back to ``"exposure"``.

    Returns ``(couplings, route_used)``.
    """
    has_corr = spec.target_pairwise_corr is not None
    has_exposure = float(np.sum(spec.exposure_matrix)) > 0.0

    if route == "auto":
        route = "correlation" if has_corr else "exposure"

    if route == "correlation":
        if not has_corr:
            raise ValueError("spec has no target_pairwise_corr for the correlation route")
        target_corr = spec.target_pairwise_corr
        if spec.correlation_space == CORRELATION_SPACE_LATENT_GAUSSIAN:
            target_corr = joint_to_corr(
                spec.target_pairwise_joint_probs(),
                spec.marginal_default_probs,
            )
        j = couplings_from_correlation(
            target_corr,
            spec.marginal_default_probs,
            method=correlation_method,
            coupling_cap=coupling_cap,
        )
        return j, "correlation"
    if route == "exposure":
        if not has_exposure:
            raise ValueError("spec has an all-zero exposure_matrix for the exposure route")
        fitness = spec.exposure_matrix.sum(axis=0) + spec.exposure_matrix.sum(axis=1)
        j = couplings_from_exposure(
            spec.exposure_matrix,
            coupling_scale=coupling_scale,
            fitness=fitness,
            target_density=0.1,
            coupling_cap=coupling_cap,
        )
        return j, "exposure"
    raise ValueError(f"unknown coupling route {route!r}")


def calibration_summary_joint(spec: SystemSpec) -> np.ndarray:
    """Target co-default matrix implied by the spec (for diagnostics)."""
    return spec.target_pairwise_joint_probs()
