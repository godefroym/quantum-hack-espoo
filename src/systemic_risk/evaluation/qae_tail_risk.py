r"""QAE tail-risk estimator: the cascade-as-oracle *calculation* surface.

Ties the two halves of the quantum-advantage plan together. The
:class:`~systemic_risk.generators.quantum_born_machine.EntangledBornMachineGenerator`
is the state-loader ``A`` that prepares the correlated default distribution
:math:`P(x)=|\langle x|A|0\rangle|^2`; the deterministic exposure cascade
(:func:`~systemic_risk.simulator.cascade.run_cascade`) is evaluated on every
computational-basis scenario to label the "severe" set; and
:mod:`~systemic_risk.generators.quantum.amplitude_estimation` (MLAE) reads off the
tail risk over that loaded distribution.

It estimates, on one shared spec / loader / severe-threshold:

* ``P(severe cascade)`` -- the amplitude ``a`` of the cascade-marked subspace, and
* ``CVaR`` of the cascade default-count -- assembled from per-level tail
  probabilities ``P(K >= k)``, each its own amplitude estimate.

Honesty (the same caveat as the AE module, restated where it is consumed):
the cascade is enumerated over all ``2^n`` scenarios and the AE operators are
simulated exactly, so this runs **only at small ``n`` and is not the speedup** --
classically simulating QAE is exponential. What is faithful is (1) the estimate
equals the classical Monte-Carlo / exact tail probability the harness computes
(equivalence) and (2) the reported **oracle-query count** is the hardware cost,
which scales like ``O(1/(eps*sqrt(a)))`` versus Monte-Carlo's ``O(1/(eps^2*a))``
and widens in the deep tail (advantage). The oracle is one qubit per institution
plus cascade-comparison ancillas and the loader is the existing QCBM, so the
*construction* extrapolates to the 54-qubit target even though this *simulation*
does not.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from systemic_risk.generators.quantum.amplitude_estimation import (
    AmplitudeEstimate,
    GroverOperator,
    mc_queries_for_relative_error,
    qae_queries_for_relative_error,
    run_mlae,
)
from systemic_risk.generators.quantum_born_machine import EntangledBornMachineGenerator
from systemic_risk.simulator.cascade import run_cascade
from systemic_risk.spec import SystemSpec


def basis_bit_matrix(n: int) -> np.ndarray:
    """Return the ``(2^n, n)`` 0/1 matrix where row ``x`` is the scenario and column ``i`` qubit ``i``.

    Bit layout matches :class:`~systemic_risk.generators.quantum.statevector.StateVector` and
    :meth:`EntangledBornMachineGenerator.exact_probabilities`: qubit ``i`` is bit ``n-1-i`` of
    the flat basis index, so this lines the cascade labeling up with the loaded amplitudes.
    """
    index = np.arange(1 << n, dtype=np.uint64)
    bit_positions = (n - 1 - np.arange(n)).astype(np.uint64)
    bits = (index[:, None] >> bit_positions[None, :]) & np.uint64(1)
    return bits.astype(int)


def cascade_sizes_over_basis(spec: SystemSpec, **cascade_kwargs) -> np.ndarray:
    """Return the deterministic final cascade size for every one of the ``2^n`` scenarios.

    This is the oracle's truth table: ``cascade(x)`` evaluated on each computational-basis
    initial-default vector ``x``. Enumeration is ``O(2^n)`` cascade calls -- the exact-simulation
    cost, paid once per spec. On hardware the cascade would instead be a reversible circuit
    applied in superposition (no enumeration).
    """
    bits = basis_bit_matrix(spec.n)
    return np.array(
        [run_cascade(bits[x], spec, **cascade_kwargs).failure_count for x in range(bits.shape[0])],
        dtype=int,
    )


def _loader_statevector(generator: EntangledBornMachineGenerator) -> np.ndarray:
    """Return the loader's exact full-system amplitudes ``A|0>`` (real for every ansatz here).

    Uses :meth:`EntangledBornMachineGenerator.exact_probabilities`; the RY/CRY, GHZ-blend and
    symmetric-shell loaders all produce *real, nonnegative* amplitudes, so the amplitude vector is
    ``sqrt(P(x))``. (Raises, via the generator, if the fit was split across cluster blocks and no
    ``2^n`` state exists.)
    """
    probs = generator.exact_probabilities()
    return np.sqrt(np.clip(probs, 0.0, None)).astype(np.complex128)


def _grover_for_mask(prepared: np.ndarray, marked: np.ndarray) -> GroverOperator:
    r"""Build ``Q = -A S_0 A^\dagger S_f`` for a given loaded state and marked mask.

    The Grover operator depends on ``A`` only through ``A|0>`` (since ``A S_0 A^\dagger`` is the
    reflection about ``A|0>``), so we realise a valid real ``A`` whose first column is the loaded
    state via one Householder reflector. This is exact and ansatz-agnostic -- a pure simulation
    convenience that reproduces the *identical* Grover dynamics any concrete loader circuit would.
    """
    dim = prepared.size
    target = np.real(prepared).astype(float)
    norm = np.linalg.norm(target)
    if norm == 0:
        raise ValueError("loaded state has zero norm")
    target = target / norm
    e0 = np.zeros(dim)
    e0[0] = 1.0
    v = e0 - target
    nv = np.linalg.norm(v)
    unitary = np.eye(dim) if nv < 1e-14 else np.eye(dim) - 2.0 * np.outer(v, v) / nv**2
    return GroverOperator(unitary=unitary.astype(np.complex128), marked=np.asarray(marked, bool))


@dataclass(frozen=True)
class TailRiskEstimate:
    """QAE estimate of one tail quantity with its exact reference and query accounting.

    ``estimate`` is the amplitude-estimation value (``P(severe)`` or a normalised CVaR amplitude);
    ``exact`` is the value computed directly from the loaded distribution and the cascade truth
    table (the equivalence target). ``oracle_queries`` is the hardware-relevant total oracle-call
    count. ``mc_queries_matched`` / ``qae_queries_matched`` are the textbook query counts each
    paradigm needs to hit ``target_relative_error`` at the exact ``a`` -- the apples-to-apples
    advantage figure.
    """

    name: str
    estimate: float
    exact: float
    amplitude_estimate: AmplitudeEstimate
    target_relative_error: float
    mc_queries_matched: int
    qae_queries_matched: int

    @property
    def abs_error(self) -> float:
        return abs(self.estimate - self.exact)

    @property
    def within_fisher_ci(self) -> bool:
        """Whether the exact value sits within ~3 Fisher one-sigma half-widths of the estimate.

        The honest equivalence check: an unbiased estimator should land near the exact answer
        up to its *own* stated statistical error bar, not to arbitrary precision.
        """
        return self.abs_error <= 3.0 * self.amplitude_estimate.fisher_ci_half_width + 1e-12

    @property
    def query_speedup(self) -> float:
        return self.mc_queries_matched / max(self.qae_queries_matched, 1)


@dataclass(frozen=True)
class QAETailRiskReport:
    """Bundle of QAE tail-risk estimates for one spec / loader / threshold."""

    severe_threshold: int
    n_qubits: int
    p_severe: TailRiskEstimate
    cvar: TailRiskEstimate
    cvar_alpha: float
    cascade_size_pmf: np.ndarray  # exact P(#defaults = k) under the loaded distribution


class QAETailRiskEstimator:
    """Estimate cascade tail risk by QAE over a Born-machine-loaded scenario distribution.

    Construct with a *fitted* :class:`EntangledBornMachineGenerator` and the spec it was fit to,
    then call :meth:`estimate_p_severe`, :meth:`estimate_cvar`, or :meth:`report`. The exact
    loaded amplitudes and the cascade truth table are computed once and reused across estimates.
    """

    def __init__(
        self,
        generator: EntangledBornMachineGenerator,
        spec: SystemSpec,
        *,
        cascade_kwargs: dict | None = None,
    ) -> None:
        self.generator = generator
        self.spec = spec
        self.cascade_kwargs = dict(cascade_kwargs or {})
        self.prepared = _loader_statevector(generator)
        self.cascade_sizes = cascade_sizes_over_basis(spec, **self.cascade_kwargs)
        self.probabilities = np.abs(self.prepared) ** 2

    # ----------------------------------------------------------------- exact references
    def exact_p_severe(self, severe_threshold: int) -> float:
        """Exact ``P(cascade size >= severe_threshold)`` under the loaded distribution."""
        mask = self.cascade_sizes >= int(severe_threshold)
        return float(self.probabilities[mask].sum())

    def cascade_size_pmf(self) -> np.ndarray:
        """Exact ``P(#final defaults = k)`` for ``k = 0..n`` under the loaded distribution."""
        return np.bincount(
            self.cascade_sizes, weights=self.probabilities, minlength=self.spec.n + 1
        )

    def exact_cvar(self, alpha: float = 0.95) -> float:
        """Exact ``CVaR_alpha`` of the cascade default-count under the loaded distribution.

        Uses the shared :class:`~systemic_risk.models.ising.LossDistribution` CVaR definition so
        the QAE route and the classical metrics agree by construction on the *quantity*.
        """
        from systemic_risk.models.ising import LossDistribution

        return LossDistribution(pmf=self.cascade_size_pmf(), exact=True).cvar(alpha=alpha)

    # ----------------------------------------------------------------- QAE estimates
    def estimate_p_severe(
        self,
        severe_threshold: int,
        *,
        num_powers: int = 7,
        shots: int = 200,
        target_relative_error: float = 0.1,
        schedule_kind: str = "lis",
        rng: np.random.Generator | None = None,
        noiseless: bool = False,
    ) -> TailRiskEstimate:
        """Estimate ``P(severe cascade)`` by MLAE over the cascade-marked subspace."""
        mask = self.cascade_sizes >= int(severe_threshold)
        return self._estimate_amplitude(
            name=f"P(cascade>={int(severe_threshold)})",
            mask=mask,
            exact=float(self.probabilities[mask].sum()),
            num_powers=num_powers,
            shots=shots,
            target_relative_error=target_relative_error,
            schedule_kind=schedule_kind,
            rng=rng,
            noiseless=noiseless,
        )

    def estimate_cvar(
        self,
        *,
        alpha: float = 0.95,
        num_powers: int = 7,
        shots: int = 200,
        target_relative_error: float = 0.1,
        schedule_kind: str = "lis",
        rng: np.random.Generator | None = None,
        noiseless: bool = False,
    ) -> TailRiskEstimate:
        r"""Estimate ``CVaR_alpha`` of the cascade default-count via threshold-level QAE.

        CVaR of a nonnegative integer count ``K`` decomposes into a sum of upper-tail
        probabilities. With ``VaR = q`` the smallest level whose upper tail covers mass
        ``>= 1-alpha``, ``CVaR_alpha = E[K | K >= q] = (sum_{k>=q} k P(K>=... ))`` -- equivalently,
        using the layer-cake identity ``E[K 1{K>=q}] = q P(K>=q) + sum_{k>q} P(K>=k)``,

            CVaR_alpha = ( q * a_q + sum_{k=q+1}^{n} a_k ) / a_q ,

        where ``a_k = P(K >= k)`` is the amplitude of the "at least ``k`` defaults" subspace.
        Each ``a_k`` is its **own** amplitude estimate (same Grover machinery, mask
        ``cascade_size >= k``); we estimate every level once and combine. The VaR level ``q`` is
        taken from the exact pmf (a cheap classical quantile), matching
        :meth:`~systemic_risk.models.ising.LossDistribution.cvar`.

        The reported query count is the **sum across all estimated levels** -- the honest total
        hardware cost of the CVaR readout, not a single-level cost.
        """
        rng = np.random.default_rng() if rng is None else rng
        pmf = self.cascade_size_pmf()
        n = self.spec.n
        # VaR level q: smallest k whose upper tail still covers mass >= (1 - alpha).
        upper_tail = np.cumsum(pmf[::-1])[::-1]  # upper_tail[k] = P(K >= k)
        eligible = np.nonzero(upper_tail >= (1.0 - alpha) - 1e-12)[0]
        q = int(eligible[-1]) if eligible.size else 0

        # Estimate a_k = P(K >= k) for every contributing level k = q..n by QAE.
        level_estimates: list[AmplitudeEstimate] = []
        a_hat: dict[int, float] = {}
        for k in range(q, n + 1):
            mask = self.cascade_sizes >= k
            a_exact = float(self.probabilities[mask].sum())
            if a_exact <= 0.0:
                a_hat[k] = 0.0
                continue
            est = self._raw_amplitude_estimate(
                mask, a_exact, num_powers, shots, schedule_kind, rng, noiseless
            )
            level_estimates.append(est)
            a_hat[k] = est.value

        a_q = a_hat.get(q, 0.0)
        if a_q <= 0.0:
            cvar_hat = float(n)
        else:
            tail_sum = q * a_q + sum(a_hat[k] for k in range(q + 1, n + 1))
            cvar_hat = tail_sum / a_q

        exact = self.exact_cvar(alpha=alpha)
        total_oracle = sum(e.oracle_queries for e in level_estimates)
        total_shots = sum(e.shot_calls for e in level_estimates)
        max_power = max((e.max_power for e in level_estimates), default=0)
        # Aggregate Fisher half-width on CVaR is dominated by the VaR-level estimate; report it
        # as a conservative stand-in (CVaR is ~ a_q-normalised, so its scale tracks a_q's error).
        fisher = (
            level_estimates[0].fisher_ci_half_width * max(n - q, 1)
            if level_estimates
            else float("inf")
        )
        combined = AmplitudeEstimate(
            value=cvar_hat,
            theta=float("nan"),
            oracle_queries=total_oracle,
            shot_calls=total_shots,
            max_power=max_power,
            fisher_ci_half_width=fisher,
            exact_value=exact,
        )
        return TailRiskEstimate(
            name=f"CVaR_{alpha:g}(cascade size)",
            estimate=cvar_hat,
            exact=exact,
            amplitude_estimate=combined,
            target_relative_error=target_relative_error,
            mc_queries_matched=mc_queries_for_relative_error(
                max(a_q, 1e-9), target_relative_error
            ),
            qae_queries_matched=qae_queries_for_relative_error(
                max(a_q, 1e-9), target_relative_error
            ),
        )

    def report(
        self,
        severe_threshold: int,
        *,
        alpha: float = 0.95,
        num_powers: int = 7,
        shots: int = 200,
        target_relative_error: float = 0.1,
        rng: np.random.Generator | None = None,
        noiseless: bool = False,
    ) -> QAETailRiskReport:
        """Estimate both ``P(severe)`` and ``CVaR`` and bundle them with exact references."""
        rng = np.random.default_rng() if rng is None else rng
        p_severe = self.estimate_p_severe(
            severe_threshold,
            num_powers=num_powers,
            shots=shots,
            target_relative_error=target_relative_error,
            rng=rng,
            noiseless=noiseless,
        )
        cvar = self.estimate_cvar(
            alpha=alpha,
            num_powers=num_powers,
            shots=shots,
            target_relative_error=target_relative_error,
            rng=rng,
            noiseless=noiseless,
        )
        return QAETailRiskReport(
            severe_threshold=int(severe_threshold),
            n_qubits=self.spec.n,
            p_severe=p_severe,
            cvar=cvar,
            cvar_alpha=alpha,
            cascade_size_pmf=self.cascade_size_pmf(),
        )

    # ----------------------------------------------------------------- internals
    def _raw_amplitude_estimate(
        self, mask, a_exact, num_powers, shots, schedule_kind, rng, noiseless
    ) -> AmplitudeEstimate:
        grover = _grover_for_mask(self.prepared, mask)
        return run_mlae(
            grover,
            num_powers=num_powers,
            shots=shots,
            schedule_kind=schedule_kind,
            rng=rng,
            exact_amplitude=a_exact,
            noiseless=noiseless,
        )

    def _estimate_amplitude(
        self,
        *,
        name,
        mask,
        exact,
        num_powers,
        shots,
        target_relative_error,
        schedule_kind,
        rng,
        noiseless,
    ) -> TailRiskEstimate:
        rng = np.random.default_rng() if rng is None else rng
        est = self._raw_amplitude_estimate(
            mask, exact, num_powers, shots, schedule_kind, rng, noiseless
        )
        safe_a = max(exact, 1e-9)
        return TailRiskEstimate(
            name=name,
            estimate=est.value,
            exact=exact,
            amplitude_estimate=est,
            target_relative_error=target_relative_error,
            mc_queries_matched=mc_queries_for_relative_error(safe_a, target_relative_error),
            qae_queries_matched=qae_queries_for_relative_error(safe_a, target_relative_error),
        )
