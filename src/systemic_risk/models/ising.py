"""Pairwise Ising / Boltzmann model over binary default configurations.

The model is the project's plausibility distribution (``scenario_generation.md``)::

    Pi(x) = sum_i h_i x_i + sum_{i<j} J_ij x_i x_j,    P(x) propto exp(Pi(x)),

with ``x in {0, 1}^n`` and ``x_i = 1`` meaning institution ``i`` initially defaults
(the project convention, matching :class:`systemic_risk.spec.SystemSpec`). This is exactly
the auto-logistic / log-linear graphical model of Filiz, Guo, Morton & Sturmfels (2012,
arXiv:0809.1393, Eq. 1), with field ``h_i`` and pairwise coupling ``J_ij``.

Sampling and moments are computed by an *automatic* method choice keyed on ``n``:

============  =====================================================================
``n``         method
============  =====================================================================
``<= 20``     exact enumeration of all ``2^n`` states (exact ``Z``, marginals,
              pairwise moments, loss-count distribution, tail probabilities, CVaR)
``20`` - 40   Gibbs (single-spin-flip) MCMC
``>= 40``     Gibbs MCMC with parallel tempering (replica exchange) to cross the
              correlation-driven first-order transition (Molins & Vives 2005)
============  =====================================================================

Exact enumeration at ``n = 54`` is impossible (``2^54 ~ 1.8e16``); for the homogeneous
(uniform field / uniform coupling) case use
:class:`systemic_risk.models.mean_field_oracle.MeanFieldIsingOracle`, which is closed-form
at any ``n`` and serves as the validation oracle.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Above this many spins, full 2**n enumeration is refused (memory / time guard).
MAX_EXACT_N = 20
# Use parallel tempering at or above this many spins (strong-coupling / metastability).
PARALLEL_TEMPERING_N = 40


@dataclass
class LossDistribution:
    """Distribution of the number of initial defaults ``k = sum_i x_i``.

    ``pmf[k]`` is ``P(#defaults == k)`` for ``k = 0, ..., n``. ``exact`` records whether
    the distribution came from exact enumeration / a closed form (``True``) or from finite
    Monte-Carlo samples (``False``).
    """

    pmf: np.ndarray
    exact: bool

    @property
    def n(self) -> int:
        return len(self.pmf) - 1

    @property
    def counts(self) -> np.ndarray:
        return np.arange(len(self.pmf))

    def mean(self) -> float:
        return float(np.dot(self.counts, self.pmf))

    def tail_prob(self, k: int) -> float:
        """Return ``P(#defaults >= k)``."""
        k = max(0, min(int(k), self.n))
        return float(self.pmf[k:].sum())

    def cvar(self, alpha: float = 0.95) -> float:
        """Expected number of defaults in the worst ``(1 - alpha)`` tail (CVaR_alpha).

        Computed on the default-count distribution: ``CVaR_alpha = E[K | K >= VaR_alpha]``
        with the level chosen so the conditioning tail has mass ``>= 1 - alpha``.
        """
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must lie in (0, 1)")
        tail_target = 1.0 - alpha
        cumulative_from_top = np.cumsum(self.pmf[::-1])[::-1]
        # Smallest k whose upper tail P(K >= k) still covers the target mass.
        eligible = np.nonzero(cumulative_from_top >= tail_target - 1e-12)[0]
        var_level = int(eligible[-1]) if len(eligible) else 0
        tail_mass = cumulative_from_top[var_level]
        if tail_mass <= 0:
            return float(self.n)
        counts = self.counts[var_level:]
        weights = self.pmf[var_level:]
        return float(np.dot(counts, weights) / tail_mass)


class IsingModel:
    """Pairwise Ising / Boltzmann model ``P(x) propto exp(Pi(x))`` over ``x in {0,1}^n``."""

    def __init__(self, fields: np.ndarray, couplings: np.ndarray) -> None:
        fields = np.asarray(fields, dtype=float)
        couplings = np.asarray(couplings, dtype=float)
        if fields.ndim != 1:
            raise ValueError("fields must be a 1D array")
        n = fields.shape[0]
        if couplings.shape != (n, n):
            raise ValueError("couplings must have shape (n, n)")
        if not np.allclose(couplings, couplings.T):
            raise ValueError("couplings must be symmetric")
        if not np.all(np.isfinite(fields)) or not np.all(np.isfinite(couplings)):
            raise ValueError("fields and couplings must be finite")
        self.fields = fields
        # Store a symmetric, zero-diagonal coupling matrix (J_ii folded into nothing;
        # x_i^2 = x_i in {0,1} so any diagonal would just shift the field).
        self.couplings = couplings.copy()
        np.fill_diagonal(self.couplings, 0.0)

    @property
    def n(self) -> int:
        return self.fields.shape[0]

    # ------------------------------------------------------------------ energy
    def log_weight(self, x: np.ndarray) -> np.ndarray:
        """Return ``Pi(x)`` for one configuration or a batch of configurations."""
        x = np.asarray(x, dtype=float)
        if x.ndim == 1:
            return float(self.fields @ x + 0.5 * x @ self.couplings @ x)
        linear = x @ self.fields
        quadratic = 0.5 * np.einsum("si,ij,sj->s", x, self.couplings, x)
        return linear + quadratic

    def method_for(self, n_override: int | None = None) -> str:
        """Return the sampling method that will be used for this model's size."""
        n = self.n if n_override is None else n_override
        if n <= MAX_EXACT_N:
            return "exact"
        if n < PARALLEL_TEMPERING_N:
            return "gibbs"
        return "parallel_tempering"

    # -------------------------------------------------------------- exact path
    def _enumerate(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (states, probabilities) over all ``2^n`` configurations.

        ``states`` has shape ``(2^n, n)`` with column 0 the lowest-index institution.
        Only callable for ``n <= MAX_EXACT_N``.
        """
        n = self.n
        if n > MAX_EXACT_N:
            raise ValueError(
                f"exact enumeration refused for n={n} (> {MAX_EXACT_N}); 2^n is too large"
            )
        num_states = 1 << n
        # Bit i of the row index -> value of spin i (little-endian over institutions).
        idx = np.arange(num_states, dtype=np.uint64)
        bit_positions = np.arange(n, dtype=np.uint64)
        states = ((idx[:, None] >> bit_positions[None, :]) & np.uint64(1)).astype(np.float64)
        log_w = self.log_weight(states)
        log_w -= log_w.max()  # stabilise before exponentiating
        weights = np.exp(log_w)
        probs = weights / weights.sum()
        return states, probs

    def partition_function(self) -> float:
        """Return the exact partition function ``Z`` (only for small ``n``)."""
        n = self.n
        if n > MAX_EXACT_N:
            raise ValueError(f"exact Z refused for n={n} (> {MAX_EXACT_N})")
        num_states = 1 << n
        idx = np.arange(num_states, dtype=np.uint64)
        bit_positions = np.arange(n, dtype=np.uint64)
        states = ((idx[:, None] >> bit_positions[None, :]) & np.uint64(1)).astype(np.float64)
        log_w = self.log_weight(states)
        shift = log_w.max()
        return float(np.exp(shift) * np.exp(log_w - shift).sum())

    def exact_moments(self) -> tuple[np.ndarray, np.ndarray]:
        """Return exact ``(marginals, pairwise_joint)`` via enumeration.

        ``marginals[i] = <x_i>``; ``pairwise_joint[i, j] = <x_i x_j>`` with diagonal
        equal to the marginals.
        """
        states, probs = self._enumerate()
        marginals = probs @ states
        pairwise = np.einsum("s,si,sj->ij", probs, states, states)
        return marginals, pairwise

    def exact_loss_distribution(self) -> LossDistribution:
        """Return the exact distribution of the number of defaults (small ``n``)."""
        states, probs = self._enumerate()
        counts = states.sum(axis=1).astype(int)
        pmf = np.bincount(counts, weights=probs, minlength=self.n + 1)
        return LossDistribution(pmf=pmf, exact=True)

    # --------------------------------------------------------------- MCMC path
    def _gibbs_chain(
        self,
        n_samples: int,
        rng: np.random.Generator,
        burn_in: int,
        thin: int,
        init: np.ndarray | None = None,
    ) -> np.ndarray:
        """Run a single-spin-flip Gibbs chain and return ``n_samples`` configurations."""
        n = self.n
        if init is None:
            state = (rng.random(n) < _sigmoid(self.fields)).astype(np.int8)
        else:
            state = init.astype(np.int8).copy()
        samples = np.empty((n_samples, n), dtype=np.int8)

        for _ in range(burn_in):
            state = self._gibbs_sweep(state, rng)

        collected = 0
        while collected < n_samples:
            for _ in range(thin):
                state = self._gibbs_sweep(state, rng)
            samples[collected] = state
            collected += 1
        return samples.astype(int)

    def _gibbs_sweep(self, state: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """One sweep of random-order single-spin-flip Gibbs updates."""
        for i in rng.permutation(self.n):
            # Flipping x_i 0->1 changes Pi by h_i + sum_j J_ij x_j.
            local_field = self.fields[i] + float(self.couplings[i] @ state)
            state[i] = 1 if rng.random() < _sigmoid(local_field) else 0
        return state

    def _parallel_tempering(
        self,
        n_samples: int,
        rng: np.random.Generator,
        burn_in: int,
        thin: int,
        n_replicas: int,
        beta_min: float,
    ) -> np.ndarray:
        """Replica-exchange (parallel tempering) sampler; returns cold-chain samples.

        Replicas run at inverse temperatures ``beta`` geometrically spaced from
        ``beta_min`` (hot, fast-mixing) to ``1.0`` (cold, target distribution). Only the
        cold replica's configurations are returned. Replica exchange lets the cold chain
        escape the metastable basins that appear near the first-order transition the
        infinite-range Ising credit model exhibits (Molins & Vives 2005).
        """
        n = self.n
        betas = np.geomspace(beta_min, 1.0, n_replicas)
        replicas = (rng.random((n_replicas, n)) < _sigmoid(self.fields)).astype(np.int8)
        # Cache log-weights per replica to make swap acceptance cheap.
        log_w = np.array([self.log_weight(r) for r in replicas])

        def sweep(state: np.ndarray, beta: float) -> np.ndarray:
            for i in rng.permutation(n):
                local_field = beta * (self.fields[i] + float(self.couplings[i] @ state))
                state[i] = 1 if rng.random() < _sigmoid(local_field) else 0
            return state

        def attempt_swaps() -> None:
            # Alternate even/odd neighbour pairs for ergodic exchange.
            start = rng.integers(0, 2)
            for r in range(start, n_replicas - 1, 2):
                delta = (betas[r] - betas[r + 1]) * (log_w[r] - log_w[r + 1])
                if delta >= 0 or rng.random() < np.exp(delta):
                    replicas[[r, r + 1]] = replicas[[r + 1, r]]
                    log_w[r], log_w[r + 1] = log_w[r + 1], log_w[r]

        for _ in range(burn_in):
            for r in range(n_replicas):
                replicas[r] = sweep(replicas[r], betas[r])
                log_w[r] = self.log_weight(replicas[r])
            attempt_swaps()

        samples = np.empty((n_samples, n), dtype=np.int8)
        collected = 0
        while collected < n_samples:
            for _ in range(thin):
                for r in range(n_replicas):
                    replicas[r] = sweep(replicas[r], betas[r])
                    log_w[r] = self.log_weight(replicas[r])
                attempt_swaps()
            samples[collected] = replicas[-1]  # cold replica (beta = 1)
            collected += 1
        return samples.astype(int)

    # ------------------------------------------------------------- public API
    def sample(
        self,
        n_samples: int,
        seed: int | None = None,
        *,
        method: str | None = None,
        burn_in: int = 1_000,
        thin: int = 10,
        n_replicas: int = 8,
        beta_min: float = 0.2,
    ) -> np.ndarray:
        """Draw ``n_samples`` configurations ``x in {0,1}^n``.

        ``method`` defaults to the size-appropriate choice from :meth:`method_for`
        (``"exact"`` for small ``n``, else ``"gibbs"`` or ``"parallel_tempering"``). Exact
        sampling draws i.i.d. configurations from the enumerated distribution; the MCMC
        methods return (thinned) chain states.
        """
        if n_samples <= 0:
            raise ValueError("n_samples must be positive")
        rng = np.random.default_rng(seed)
        chosen = method or self.method_for()

        if chosen == "exact":
            states, probs = self._enumerate()
            picks = rng.choice(states.shape[0], size=n_samples, p=probs)
            return states[picks].astype(int)
        if chosen == "gibbs":
            return self._gibbs_chain(n_samples, rng, burn_in=burn_in, thin=thin)
        if chosen == "parallel_tempering":
            return self._parallel_tempering(
                n_samples,
                rng,
                burn_in=burn_in,
                thin=thin,
                n_replicas=n_replicas,
                beta_min=beta_min,
            )
        raise ValueError(f"unknown sampling method {chosen!r}")

    def moments(
        self,
        seed: int | None = None,
        *,
        method: str | None = None,
        n_samples: int = 20_000,
        **sample_kwargs: object,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(marginals, pairwise_joint)`` -- exact for small ``n``, else MC."""
        chosen = method or self.method_for()
        if chosen == "exact":
            return self.exact_moments()
        samples = self.sample(n_samples, seed=seed, method=chosen, **sample_kwargs)  # type: ignore[arg-type]
        marginals = samples.mean(axis=0)
        pairwise = (samples.T @ samples) / samples.shape[0]
        return marginals, pairwise

    def loss_distribution(
        self,
        seed: int | None = None,
        *,
        method: str | None = None,
        n_samples: int = 50_000,
        **sample_kwargs: object,
    ) -> LossDistribution:
        """Return the default-count distribution -- exact for small ``n``, else empirical."""
        chosen = method or self.method_for()
        if chosen == "exact":
            return self.exact_loss_distribution()
        samples = self.sample(n_samples, seed=seed, method=chosen, **sample_kwargs)  # type: ignore[arg-type]
        counts = samples.sum(axis=1)
        pmf = np.bincount(counts, minlength=self.n + 1).astype(float)
        pmf /= pmf.sum()
        return LossDistribution(pmf=pmf, exact=False)


def _sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40.0, 40.0)))
