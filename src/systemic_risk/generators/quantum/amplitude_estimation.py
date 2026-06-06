r"""Quantum Amplitude Estimation (QAE) over a loaded scenario distribution.

This is the *calculation* surface of the project's quantum-advantage plan. The
companion :mod:`.statevector` / Born-machine modules build the state-preparation
unitary :math:`A` that loads the correlated default distribution
:math:`P(x)=|\langle x|A|0\rangle|^2`; here a deterministic cascade label
``f(x) in {0, 1}`` (``1`` = "severe") becomes a reversible oracle, and amplitude
estimation reads off the tail probability

.. math:: a = \sum_{x:\,f(x)=1} P(x) = \langle 0|A^\dagger P_1 A|0\rangle,

with a query complexity that beats classical Monte Carlo and widens in the deep
tail (see :func:`mc_queries_for_relative_error` vs :func:`mlae_queries`).

What is classical vs quantum here -- read this before quoting any number
--------------------------------------------------------------------------------
Everything in this module is an **exact classical statevector simulation** of the
quantum operators (the loader :math:`A`, the oracle reflection :math:`S_f`, and the
Grover operator :math:`Q = -A S_0 A^\dagger S_f`). It forms the full ``2^n``
amplitude vector, so it is *only* tractable for small ``n`` and is **not** itself
the speedup -- classically simulating QAE is exponential in ``n``. Two things are
honest and hardware-relevant regardless of the simulator:

* **Equivalence.** Applying :math:`Q` to :math:`A|0\rangle` rotates by exactly
  ``2*theta`` per power in the 2D span of (marked, unmarked) states, with
  ``a = sin^2(theta)``. So the estimator returns the *same* ``a`` the classical
  Monte-Carlo harness estimates and the *same* ``a`` you would get on hardware
  (up to the AE schedule's statistical error) -- the simulation just lets us
  verify that against the exact analytic value.
* **Advantage.** The figure of merit is the **oracle-query count** to reach a
  target relative error -- the number of times :math:`A`/:math:`Q` would run on a
  real device. That count scales like ``O(1/(eps*sqrt(a)))`` for QAE versus
  ``O(1/(eps^2*a))`` for Monte Carlo, and is reported by every estimator below.
  Wall-clock on this simulator is *not* the speedup and is never reported as one.

The construction extrapolates unchanged to the 54-qubit hardware target -- the
oracle is one qubit per institution plus the cascade-comparison ancillas, and
:math:`A` is the existing QCBM loader -- but the exact ``2^n`` *simulation* here
does not, by design.

Estimation scheme
-----------------
We use **Maximum-Likelihood Amplitude Estimation** (MLAE; Suzuki et al., 2020):
ancilla-free canonical AE that, instead of running quantum phase estimation,
measures the marked-probability after a schedule of Grover powers ``m_k`` and fits
``theta`` by maximum likelihood. It keeps the quadratic, deep-tail-amplified
speedup while needing only the ``n``-qubit state -- the natural fit for an exact
statevector backend and for near-term hardware.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# --------------------------------------------------------------------------- oracle / Grover
def reflection_about_marked(state: np.ndarray, marked: np.ndarray) -> np.ndarray:
    """Apply the oracle ``S_f = I - 2 P_1``: flip the sign of marked amplitudes.

    ``state`` is a flat ``(2^n,)`` complex amplitude vector and ``marked`` a boolean
    mask over the same computational-basis index. This is the reversible cascade
    oracle in its phase-kickback form (the ancilla that would hold ``f(x)`` on
    hardware is uncomputed, leaving exactly this diagonal +-1 reflection).
    """
    out = state.copy()
    out[marked] *= -1.0
    return out


@dataclass(frozen=True)
class GroverOperator:
    r"""The amplitude-estimation operator ``Q = -A S_0 A^\dagger S_f`` as a statevector map.

    ``A`` is given implicitly by ``prepared`` (``= A|0...0>``, the loaded distribution's
    amplitudes) together with the recipe to apply ``A``/``A^dagger`` exactly. Because the
    loaders used here are real, dependency-free, and small, we realise ``A`` as the dense
    ``2^n x 2^n`` unitary once; this is purely a simulation convenience and never touches
    hardware. ``S_0`` reflects about ``|0...0>`` and ``S_f`` about the marked set.

    One application of :meth:`apply` is **one oracle query** in the complexity accounting
    (``Q`` contains exactly one ``S_f``); ``Q^m`` therefore costs ``m`` queries, and a single
    state preparation (``m = 0``) costs one further ``A`` call -- mirrored in
    :func:`mlae_queries`.
    """

    unitary: np.ndarray  # dense A (2^n, 2^n); column 0 is A|0...0>
    marked: np.ndarray  # boolean mask over the 2^n basis

    @classmethod
    def from_state_loader(
        cls, apply_loader, dim: int, marked: np.ndarray
    ) -> "GroverOperator":
        """Build ``Q`` by materialising ``A`` column-by-column from a statevector loader.

        ``apply_loader(basis_state) -> amplitudes`` returns ``A|j>`` for a one-hot input;
        we call it on each basis vector to assemble the dense ``A``. Exact-simulation only
        (``dim = 2^n``).
        """
        columns = np.empty((dim, dim), dtype=np.complex128)
        for j in range(dim):
            basis = np.zeros(dim, dtype=np.complex128)
            basis[j] = 1.0
            columns[:, j] = apply_loader(basis)
        return cls(unitary=columns, marked=np.asarray(marked, dtype=bool))

    @property
    def prepared(self) -> np.ndarray:
        """The loaded state ``A|0...0>`` (first column of ``A``)."""
        return self.unitary[:, 0]

    @property
    def amplitude(self) -> float:
        """Exact marked probability ``a = sum_{x marked} |<x|A|0>|^2`` (the tail risk)."""
        psi = self.prepared
        return float(np.sum(np.abs(psi[self.marked]) ** 2))

    def _reflect_about_zero(self, state: np.ndarray) -> np.ndarray:
        out = -state
        out[0] += 2.0 * state[0]
        return out

    def apply(self, state: np.ndarray, power: int = 1) -> np.ndarray:
        r"""Return ``Q^power |state>`` with ``Q = -A S_0 A^\dagger S_f``.

        Costs ``power`` oracle queries. ``power = 0`` is the identity (the bare prepared
        state is obtained from :attr:`prepared`).
        """
        psi = state
        adj = self.unitary.conj().T
        for _ in range(int(power)):
            psi = reflection_about_marked(psi, self.marked)  # S_f
            psi = adj @ psi  # A^dagger
            psi = self._reflect_about_zero(psi)  # S_0
            psi = self.unitary @ psi  # A
            psi = -psi  # global -1
        return psi

    def marked_probability_after_powers(self, power: int) -> float:
        r"""Exact ``P(marked)`` in the state ``Q^power A|0>``.

        Equals ``sin^2((2 power + 1) theta)`` with ``a = sin^2(theta)`` -- the textbook
        AE signal MLAE fits. Computed here from the exact statevector (no sampling), so it
        is the *noiseless* likelihood mean; sampling noise is injected separately in
        :func:`run_mlae` so query/accuracy trade-offs are measured honestly.
        """
        psi = self.apply(self.prepared, power)
        return float(np.sum(np.abs(psi[self.marked]) ** 2))


# --------------------------------------------------------------------------- MLAE estimator
def likelihood_schedule(num_powers: int, *, kind: str = "lis") -> list[int]:
    """Return the Grover-power schedule ``[m_0, m_1, ...]`` for MLAE.

    ``kind="lis"`` is the Linearly Incremental Sequence ``m_k = k`` (powers ``0,1,2,...``),
    the schedule Suzuki et al. show attains the ``O(1/(eps*sqrt(a)))`` Heisenberg-like
    scaling; ``kind="eis"`` is the Exponentially Incremental Sequence
    ``m_k = 2^(k-1)`` (with ``m_0 = 0``) used in their analysis as well.
    """
    if num_powers < 1:
        raise ValueError("num_powers must be positive")
    if kind == "lis":
        return list(range(num_powers))
    if kind == "eis":
        return [0] + [2 ** (k - 1) for k in range(1, num_powers)]
    raise ValueError(f"unknown schedule kind {kind!r}")


def _negative_log_likelihood(
    theta: float, powers: np.ndarray, hits: np.ndarray, shots: np.ndarray
) -> float:
    angles = (2.0 * powers + 1.0) * theta
    p = np.sin(angles) ** 2
    p = np.clip(p, 1e-12, 1.0 - 1e-12)
    return float(-np.sum(hits * np.log(p) + (shots - hits) * np.log1p(-p)))


def _mle_theta(powers: np.ndarray, hits: np.ndarray, shots: np.ndarray, grid: int) -> float:
    """Maximise the MLAE likelihood for ``theta in [0, pi/2]`` (grid seed + local polish)."""
    thetas = np.linspace(0.0, np.pi / 2.0, grid)
    nll = np.array(
        [_negative_log_likelihood(t, powers, hits, shots) for t in thetas]
    )
    best = int(np.argmin(nll))
    lo = thetas[max(best - 1, 0)]
    hi = thetas[min(best + 1, grid - 1)]
    # Golden-section refine inside the winning cell.
    invphi = (np.sqrt(5.0) - 1.0) / 2.0
    a_, b_ = lo, hi
    c_ = b_ - invphi * (b_ - a_)
    d_ = a_ + invphi * (b_ - a_)
    for _ in range(80):
        if _negative_log_likelihood(c_, powers, hits, shots) < _negative_log_likelihood(
            d_, powers, hits, shots
        ):
            b_ = d_
        else:
            a_ = c_
        c_ = b_ - invphi * (b_ - a_)
        d_ = a_ + invphi * (b_ - a_)
    return 0.5 * (a_ + b_)


@dataclass(frozen=True)
class AmplitudeEstimate:
    """Result of an amplitude-estimation run, with hardware-relevant query accounting.

    ``value`` is the estimated amplitude ``a`` (e.g. ``P(severe)``); ``oracle_queries`` is the
    total number of oracle (``S_f``) applications across the whole schedule -- the count that
    would run on hardware and the basis for every speedup claim. ``shot_calls`` is the number
    of ``A``-preparations (one circuit execution each). ``fisher_ci_half_width`` is the
    asymptotic (Cramer-Rao) one-sigma half-width on ``a``; ``error_vs_exact`` is filled when an
    exact reference is supplied.
    """

    value: float
    theta: float
    oracle_queries: int
    shot_calls: int
    max_power: int
    fisher_ci_half_width: float
    exact_value: float | None = None
    schedule: tuple[int, ...] = ()
    shots: tuple[int, ...] = ()

    @property
    def error_vs_exact(self) -> float | None:
        if self.exact_value is None:
            return None
        return abs(self.value - self.exact_value)

    @property
    def relative_error_vs_exact(self) -> float | None:
        if self.exact_value is None or self.exact_value == 0.0:
            return None
        return abs(self.value - self.exact_value) / self.exact_value


def _fisher_half_width(theta: float, powers: np.ndarray, shots: np.ndarray) -> float:
    """One-sigma Cramer-Rao half-width on ``a = sin^2(theta)`` for an MLAE schedule.

    A Bernoulli outcome at Grover power ``m`` has mean ``sin^2((2m+1) theta)``; its Fisher
    information about ``theta`` works out to ``I_theta(m) = shots * 4 (2m+1)^2`` (the
    ``sin/cos`` factors cancel). Summed over the schedule this gives ``Var(theta) = 1/sum I``,
    and the delta method maps it to ``Var(a) = (da/dtheta)^2 Var(theta)`` with
    ``da/dtheta = sin(2theta)``.
    """
    info_theta = np.sum(shots * (2.0 * powers + 1.0) ** 2 * 4.0)
    if info_theta <= 0:
        return float("inf")
    var_theta = 1.0 / info_theta
    da_dtheta = np.sin(2.0 * theta)
    return float(abs(da_dtheta) * np.sqrt(var_theta))


def run_mlae(
    grover: GroverOperator,
    *,
    num_powers: int = 7,
    shots: int = 200,
    schedule_kind: str = "lis",
    rng: np.random.Generator | None = None,
    exact_amplitude: float | None = None,
    grid: int = 2000,
    noiseless: bool = False,
) -> AmplitudeEstimate:
    """Estimate the marked amplitude ``a`` by Maximum-Likelihood Amplitude Estimation.

    For each Grover power ``m_k`` in the schedule we take ``shots`` measurements of the
    marked indicator in ``Q^{m_k} A|0>`` (sampled from the *exact* simulated probability,
    so the only randomness is finite-shot noise -- the same noise a real device has), then
    fit ``theta`` by maximum likelihood and return ``a = sin^2(theta)``.

    ``noiseless=True`` skips sampling and feeds the exact per-power probabilities to the
    likelihood (useful for asserting the estimator is unbiased / for the analytic check).
    """
    rng = np.random.default_rng() if rng is None else rng
    powers = np.asarray(likelihood_schedule(num_powers, kind=schedule_kind), dtype=int)
    shot_vec = np.full(powers.size, int(shots), dtype=int)

    exact_p = np.array(
        [grover.marked_probability_after_powers(int(m)) for m in powers]
    )
    if noiseless:
        hits = exact_p * shot_vec
    else:
        hits = rng.binomial(shot_vec, np.clip(exact_p, 0.0, 1.0)).astype(float)

    theta = _mle_theta(powers, hits, shot_vec, grid)
    value = float(np.sin(theta) ** 2)

    # Oracle queries: the textbook Grover operator Q contains exactly one oracle reflection
    # S_f, so Q^{m_k} costs m_k oracle calls; each of the `shots` measurements re-runs it,
    # and the m=0 term is pure state-prep (zero oracle calls). This is the hardware cost.
    oracle_queries = int(np.sum(powers * shot_vec))
    shot_calls = int(np.sum(shot_vec))

    return AmplitudeEstimate(
        value=value,
        theta=float(theta),
        oracle_queries=oracle_queries,
        shot_calls=shot_calls,
        max_power=int(powers.max()),
        fisher_ci_half_width=_fisher_half_width(theta, powers, shot_vec),
        exact_value=exact_amplitude if exact_amplitude is not None else grover.amplitude,
        schedule=tuple(int(m) for m in powers),
        shots=tuple(int(s) for s in shot_vec),
    )


# --------------------------------------------------------------------------- complexity model
def mlae_queries(num_powers: int, shots: int, *, schedule_kind: str = "lis") -> int:
    """Total oracle queries an MLAE schedule would issue on hardware.

    ``sum_k m_k * shots`` -- the hardware-relevant cost reported by :func:`run_mlae`.
    """
    powers = np.asarray(likelihood_schedule(num_powers, kind=schedule_kind), dtype=int)
    return int(np.sum(powers) * int(shots))


def mlae_relative_error_model(num_powers: int, shots: int, amplitude: float) -> float:
    r"""Asymptotic relative error of LIS-MLAE at the largest power, ``~ 1/(N_q sqrt(a))``.

    With a linearly incremental schedule the dominant Fisher term is the largest power
    ``M = num_powers - 1``; the standard error on ``theta`` is ``~ 1/(M sqrt(shots))`` and the
    delta method gives a *relative* error on ``a`` of order ``1/(M sqrt(shots) sqrt(a))`` for
    small ``a`` (since ``da/dtheta = sin 2theta ~ 2 sqrt(a)``). This is the closed-form companion
    to the measured query counts; treat it as a scaling guide, not a guarantee.
    """
    theta = float(np.arcsin(np.sqrt(np.clip(amplitude, 1e-12, 1.0))))
    biggest = max(num_powers - 1, 1)
    var_theta = 1.0 / (shots * 4.0 * biggest**2)  # dominant Fisher term
    da = abs(np.sin(2.0 * theta))
    abs_err = da * np.sqrt(var_theta)
    return float(abs_err / max(amplitude, 1e-12))


def mc_queries_for_relative_error(amplitude: float, relative_error: float) -> int:
    r"""Classical Monte-Carlo oracle calls to estimate ``a`` to a target *relative* error.

    A Bernoulli(``a``) mean estimated from ``N`` draws has standard error
    ``sqrt(a(1-a)/N)``; requiring ``sqrt(a(1-a)/N) <= relative_error * a`` gives
    ``N >= (1-a)/(relative_error^2 a)`` -- the textbook ``O(1/(eps^2 a))`` that *blows up* in
    the deep tail. One Monte-Carlo draw is one oracle (cascade) evaluation, so ``N`` is the
    directly comparable query count.
    """
    if not 0.0 < amplitude < 1.0:
        raise ValueError("amplitude must lie in (0, 1)")
    if relative_error <= 0.0:
        raise ValueError("relative_error must be positive")
    return int(np.ceil((1.0 - amplitude) / (relative_error**2 * amplitude)))


def qae_queries_for_relative_error(amplitude: float, relative_error: float) -> int:
    r"""QAE oracle calls to estimate ``a`` to a target *relative* error, ``O(1/(eps sqrt(a)))``.

    Canonical amplitude estimation reaches absolute error ``~ pi/N_q`` on ``theta`` with ``N_q``
    Grover applications; the delta method (``da/dtheta = sin 2theta ~ 2 sqrt(a)`` for small ``a``)
    turns a target *relative* error ``eps`` on ``a`` into
    ``N_q ~ pi/(2 eps sqrt(a(1-a)))`` -- the deep-tail-amplified quadratic speedup. Returned as a
    query count directly comparable with :func:`mc_queries_for_relative_error`.
    """
    if not 0.0 < amplitude < 1.0:
        raise ValueError("amplitude must lie in (0, 1)")
    if relative_error <= 0.0:
        raise ValueError("relative_error must be positive")
    return int(np.ceil(np.pi / (2.0 * relative_error * np.sqrt(amplitude * (1.0 - amplitude)))))


@dataclass(frozen=True)
class QueryComplexityPoint:
    """One ``(amplitude -> required query count)`` comparison at a fixed target error."""

    amplitude: float
    relative_error: float
    mc_queries: int
    qae_queries: int

    @property
    def speedup(self) -> float:
        """MC queries divided by QAE queries (the quadratic, deep-tail-amplified gain)."""
        return self.mc_queries / max(self.qae_queries, 1)


def query_complexity_curve(
    amplitudes, relative_error: float
) -> list[QueryComplexityPoint]:
    """Tabulate MC vs QAE query counts across tail probabilities at one target error.

    The speedup ``mc/qae`` grows like ``1/sqrt(a)`` as ``a`` shrinks -- the headline
    "advantage widens in the deep tail" evidence, expressed purely in oracle-call counts.
    """
    return [
        QueryComplexityPoint(
            amplitude=float(a),
            relative_error=float(relative_error),
            mc_queries=mc_queries_for_relative_error(float(a), relative_error),
            qae_queries=qae_queries_for_relative_error(float(a), relative_error),
        )
        for a in amplitudes
    ]
