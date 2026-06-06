"""Analytic angle-setting for the entangled Born-machine scenario generator.

This is the simulator-agnostic core of the generator: it turns a
:class:`~systemic_risk.spec.SystemSpec` into the rotation/entangler angles of a
computational-basis Born machine, with the angles derived *in closed form* from
the marginals and the dependency (correlation / exposure) graph. The same angle
recipe is consumed by the numpy statevector path (default) and the Qiskit path,
and -- because it costs ``O(n + #edges)`` and never touches the ``2^n`` state -- it
extrapolates unchanged to the 54-qubit hardware target.

Encoding (matching :class:`systemic_risk.models.ising.IsingModel`): qubit ``i`` is
institution ``i``; ``|0> = survives``, ``|1> = initial default``; a measured
computational-basis bitstring is one default scenario.

Three honest facts pin every angle (all verified numerically against the exact
statevector, see ``tests/test_quantum_born_machine.py``):

* **Marginals.** ``RY(theta)|0>`` gives ``P(=1) = sin^2(theta/2)``, so
  ``theta_i = 2 arcsin(sqrt(p_i))`` reproduces marginal ``p_i`` exactly when no
  entangler touches qubit ``i``.
* **Correlations come from amplitude mixing, never from phase.** A controlled-RY
  ``CRY_{i->j}(alpha)`` on the product state ``RY(theta_i)|0> (x) RY(theta_j)|0>``
  produces the *exact* covariance

      cov_ij = p_i (1 - p_i) (sin^2((theta_j + alpha)/2) - p_j),

  which inverts in closed form for ``alpha`` given a target covariance. A
  ``Z``-diagonal gate such as ``RZZ`` would leave every ``Z``-basis marginal and
  covariance unchanged (phase only), so it cannot carry default correlations --
  the correlations here are genuine amplitude structure across basis strings.
* **Systemic tail.** A GHZ-style superposition of two homogeneous product states
  ``sqrt(w)|A>^{(x)n} + sqrt(1-w)|B>^{(x)n}`` places coherent amplitude on the
  high-Hamming-weight "everyone defaults" strings, concentrating co-default mass far
  above the independence baseline -- a rare common-shock mode. (Its *higher-order*
  structure is tunable via the benign/systemic split rather than pinned by the spec, so
  the robust beyond-second-order claim is carried by the homogeneous symmetric loader; see
  :class:`SymmetricIsingLoader` and ``tests/test_quantum_born_machine.py``.)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from systemic_risk.spec import SystemSpec

_EPS = 1e-9
_CLIP = (1e-6, 1.0 - 1e-6)


def marginal_angles(p: np.ndarray) -> np.ndarray:
    """Return ``theta_i = 2 arcsin(sqrt(p_i))`` so ``RY(theta_i)|0>`` has ``P(=1) = p_i``."""
    p = np.clip(np.asarray(p, dtype=float), *_CLIP)
    return 2.0 * np.arcsin(np.sqrt(p))


def cry_angle(p_control: float, p_target: float, target_cov: float) -> float:
    """Closed-form ``CRY_{control->target}`` angle hitting a target single-pair covariance.

    Inverts ``cov = p_c (1 - p_c) (sin^2((theta_t + alpha)/2) - p_t)`` for ``alpha`` with
    ``theta_t = 2 arcsin(sqrt(p_t))``. Exact for an isolated pair; the seed for the light
    calibration loop when edges share qubits.
    """
    p_control = float(np.clip(p_control, *_CLIP))
    p_target = float(np.clip(p_target, *_CLIP))
    var_control = p_control * (1.0 - p_control)
    sin2 = np.clip(p_target + target_cov / max(var_control, _EPS), *_CLIP)
    theta_target = 2.0 * np.arcsin(np.sqrt(p_target))
    return float(2.0 * np.arcsin(np.sqrt(sin2)) - theta_target)


def target_covariance(spec: SystemSpec) -> np.ndarray:
    """Return the spec's target co-default covariance ``cov_ij = P(i,j) - p_i p_j``."""
    p = np.clip(spec.marginal_default_probs, *_CLIP)
    joint = spec.target_pairwise_joint_probs()
    cov = joint - np.outer(p, p)
    np.fill_diagonal(cov, 0.0)
    return cov


def dependency_edges(
    spec: SystemSpec,
    *,
    threshold: float = 0.02,
    within_clusters_only: bool = False,
    max_degree: int | None = None,
) -> list[tuple[int, int]]:
    """Select entangler edges from the spec's dependency graph, strongest first.

    Pairs are scored by ``|dependency|`` (correlation, or exposure when no correlation is
    present) and kept above ``threshold``. ``within_clusters_only`` drops cross-community
    pairs so the circuit is block-separable by :attr:`SystemSpec.clusters` -- the lever that
    keeps every simulated block small at ``n = 54``. ``max_degree`` caps how many edges may
    touch a single qubit (shallower, cleaner circuits).
    """
    dep = np.abs(spec.dependency_matrix())
    if dep.max() <= 0.0 and float(np.sum(spec.exposure_matrix)) > 0.0:
        exposure = spec.exposure_matrix + spec.exposure_matrix.T
        dep = exposure / exposure.max()
    np.fill_diagonal(dep, 0.0)

    clusters = spec.clusters
    iu = zip(*np.triu_indices(spec.n, k=1))
    scored = [
        (dep[i, j], i, j)
        for i, j in iu
        if dep[i, j] > threshold
        and not (within_clusters_only and clusters is not None and clusters[i] != clusters[j])
    ]
    scored.sort(reverse=True)

    edges: list[tuple[int, int]] = []
    degree = np.zeros(spec.n, dtype=int)
    for _, i, j in scored:
        if max_degree is not None and (degree[i] >= max_degree or degree[j] >= max_degree):
            continue
        edges.append((i, j))
        degree[i] += 1
        degree[j] += 1
    return edges


@dataclass
class EntangledCircuit:
    """Analytic angles for one block of the hardware-efficient (RY + CRY) ansatz.

    ``qubits`` are the global institution indices this block spans (a whole cluster, or the
    whole system when it is small enough to simulate directly). ``ry`` holds one rotation per
    qubit; ``edges``/``cry`` hold the controlled-RY entanglers as ``(control, target)`` pairs
    on *local* (within-block) qubit indices.
    """

    qubits: list[int]
    ry: np.ndarray
    edges: list[tuple[int, int]]
    cry: np.ndarray
    target_p: np.ndarray
    target_cov: np.ndarray

    @property
    def size(self) -> int:
        return len(self.qubits)


def _block_circuit(
    qubits: list[int],
    p: np.ndarray,
    cov: np.ndarray,
    global_edges: list[tuple[int, int]],
) -> EntangledCircuit:
    """Build the analytic seed circuit for one block (no calibration yet)."""
    local = {g: k for k, g in enumerate(qubits)}
    block_p = p[qubits]
    block_cov = cov[np.ix_(qubits, qubits)]
    edges = [(local[i], local[j]) for i, j in global_edges if i in local and j in local]
    ry = marginal_angles(block_p)
    cry = np.array(
        [cry_angle(block_p[i], block_p[j], block_cov[i, j]) for i, j in edges],
        dtype=float,
    )
    return EntangledCircuit(
        qubits=list(qubits),
        ry=ry,
        edges=edges,
        cry=cry,
        target_p=block_p,
        target_cov=block_cov,
    )


def calibrate_block(
    circuit: EntangledCircuit,
    moments_fn,
    *,
    iterations: int = 30,
    ry_gain: float = 0.9,
    cry_gain: float = 0.8,
    tol: float = 1e-5,
) -> EntangledCircuit:
    """Light Newton-style refinement of a block's angles against measured moments.

    ``moments_fn(ry, edges, cry) -> (marginals, pairwise_joint)`` returns the block's exact
    statevector moments. Each step nudges every ``RY`` angle toward its marginal residual and
    every ``CRY`` angle toward its covariance residual using the closed-form local gradients
    (``dP/dtheta = sin(theta)/2`` and ``d cov_ij/d alpha = p_i (1 - p_i) sin(theta_j + alpha)/2``).
    The seed is already analytic, so this only cleans up the interference between edges that
    share a qubit; it is not a black-box training loop.
    """
    if circuit.size == 0:
        return circuit
    ry = circuit.ry.copy()
    cry = circuit.cry.copy()
    target_p = circuit.target_p
    target_cov = circuit.target_cov
    for _ in range(iterations):
        marginals, joint = moments_fn(ry, circuit.edges, cry)
        cov = joint - np.outer(marginals, marginals)
        marginal_residual = target_p - marginals
        if circuit.edges:
            cov_residual = np.array(
                [target_cov[i, j] - cov[i, j] for i, j in circuit.edges]
            )
        else:
            cov_residual = np.zeros(0)

        if np.max(np.abs(marginal_residual)) < tol and (
            cov_residual.size == 0 or np.max(np.abs(cov_residual)) < tol
        ):
            break

        ry_grad = np.clip(np.abs(0.5 * np.sin(ry)), 0.05, None)
        ry = np.clip(ry + ry_gain * marginal_residual / ry_grad, *(_EPS, np.pi - _EPS))
        for e, (i, j) in enumerate(circuit.edges):
            slope = target_p[i] * (1.0 - target_p[i]) * 0.5 * np.sin(ry[j] + cry[e])
            cry[e] += cry_gain * cov_residual[e] / np.clip(np.abs(slope), 0.02, None)

    return EntangledCircuit(
        qubits=circuit.qubits,
        ry=ry,
        edges=circuit.edges,
        cry=cry,
        target_p=target_p,
        target_cov=target_cov,
    )


def partition_blocks(
    spec: SystemSpec,
    edges: list[tuple[int, int]],
    *,
    max_block: int,
) -> list[list[int]]:
    """Group qubits into connected components of the entangler graph (capped at ``max_block``).

    Components small enough to simulate directly are kept whole; an oversize component is split
    into ``<= max_block`` pieces. The split is **edge-weight aware** (recursive Kernighan-Lin
    bisection on the entangler weights via
    :func:`systemic_risk.generators.quantum.budget_clustering.split_oversize_group`), so the cut
    falls on the *weakest* within-component links instead of an arbitrary index boundary -- the
    strongly-coupled qubits stay in the same block. Disconnected qubits become singleton blocks
    (pure ``RY``, exact).
    """
    from systemic_risk.generators.quantum.budget_clustering import split_oversize_group

    parent = list(range(spec.n))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i, j in edges:
        parent[find(i)] = find(j)

    groups: dict[int, list[int]] = {}
    for q in range(spec.n):
        groups.setdefault(find(q), []).append(q)

    # Edge weights for the weight-aware splitter: the dependency magnitude on each entangler
    # (correlation, exposure fallback) so the cut minimises severed dependency. Falls back to
    # the connectivity graph if the dependency matrix is empty.
    dependency = np.abs(spec.dependency_matrix())
    if dependency.max() <= 0.0 and float(np.sum(spec.exposure_matrix)) > 0.0:
        exposure = spec.exposure_matrix + spec.exposure_matrix.T
        dependency = exposure / exposure.max()
    weights = np.zeros((spec.n, spec.n), dtype=float)
    for i, j in edges:
        w = float(dependency[i, j]) if dependency[i, j] > 0.0 else 1.0
        weights[i, j] = weights[j, i] = w

    blocks: list[list[int]] = []
    for members in groups.values():
        if len(members) <= max_block:
            blocks.append(members)
        else:
            blocks.extend(split_oversize_group(weights, members, max_block))
    return blocks


@dataclass
class SymmetricIsingLoader:
    """Exact entangled state-loader for a *homogeneous* target (uniform marginal + coupling).

    For a homogeneous portfolio the target is exchangeable: the joint law depends only on the
    number of defaults ``k``, with weights ``w_k`` from the closed-form mean-field Ising oracle
    (``w_k propto C(n, k) exp(h k + J k(k-1)/2)``). This loader prepares the permutation-symmetric
    state that puts amplitude ``sqrt(w_k / C(n, k))`` on *every* weight-``k`` computational-basis
    string, so ``P(K = k) = w_k`` reproduces the oracle's loss-count distribution **exactly at any
    ``n``** (no ``2^n`` sum) and matches its marginal and default correlation to machine precision.

    The state is genuinely entangled -- a coherent superposition across all Hamming-weight shells
    (the systemic-risk modes), not a product -- and is the QCBM state-loader role of the project's
    quantum-advantage plan, here in its exactly-solvable homogeneous limit.
    """

    n: int
    field: float
    coupling: float

    def loss_count_pmf(self) -> np.ndarray:
        """Exact ``P(#defaults = k)`` (the homogeneous Ising weights ``w_k``)."""
        from scipy.special import gammaln

        k = np.arange(self.n + 1)
        log_w = (
            gammaln(self.n + 1)
            - gammaln(k + 1)
            - gammaln(self.n - k + 1)
            + self.field * k
            + self.coupling * k * (k - 1) / 2.0
        )
        log_w -= log_w.max()
        w = np.exp(log_w)
        return w / w.sum()

    def shell_amplitudes(self) -> np.ndarray:
        """Per-string amplitude ``a_k = sqrt(w_k / C(n, k))`` for each Hamming weight ``k``."""
        from scipy.special import gammaln

        k = np.arange(self.n + 1)
        log_binom = gammaln(self.n + 1) - gammaln(k + 1) - gammaln(self.n - k + 1)
        return np.sqrt(self.loss_count_pmf() / np.exp(log_binom))

    @classmethod
    def from_targets(
        cls, n: int, target_marginal: float, target_default_corr: float
    ) -> "SymmetricIsingLoader":
        """Solve ``(h, J)`` reproducing a target marginal and default correlation (via the oracle)."""
        from systemic_risk.models.mean_field_oracle import MeanFieldIsingOracle

        oracle = MeanFieldIsingOracle.from_targets(n, target_marginal, target_default_corr)
        return cls(n=n, field=oracle.field, coupling=oracle.coupling)


def is_homogeneous(spec: SystemSpec, *, tol: float = 1e-6) -> bool:
    """Return ``True`` when marginals are uniform and the dependency graph is equicorrelated."""
    p = spec.marginal_default_probs
    if p.size and float(p.max() - p.min()) > tol:
        return False
    dep = spec.dependency_matrix()
    iu = np.triu_indices(spec.n, k=1)
    if iu[0].size == 0:
        return True
    off = dep[iu]
    return bool(float(off.max() - off.min()) <= tol)


@dataclass
class GHZBlend:
    """A homogeneous GHZ-style systemic-shock superposition over all ``n`` qubits.

    The state ``sqrt(weight)|A>^{(x)n} + sqrt(1 - weight)|B>^{(x)n}`` superposes a benign
    product state (each institution defaults with prob ``benign``) and a systemic one (prob
    ``systemic``). The coherent superposition is genuine entanglement: it concentrates
    amplitude on the all-survive and all-default strings, so the number-of-defaults law is the
    closed form below -- exact at *any* ``n``, including 54. The all-default mode sits many
    orders of magnitude above the independence baseline; the size of the resulting
    higher-order structure is set by the benign/systemic split (``benign_fraction``), not
    fixed by the spec's marginal and correlation alone.
    """

    n: int
    weight: float
    benign: float
    systemic: float

    def loss_count_pmf(self) -> np.ndarray:
        """Exact ``P(#defaults = k)`` for ``k = 0..n`` (closed form, no ``2^n`` sum)."""
        from scipy.special import gammaln

        k = np.arange(self.n + 1)
        log_binom = gammaln(self.n + 1) - gammaln(k + 1) - gammaln(self.n - k + 1)
        binom = np.exp(log_binom)
        ca, sa = np.sqrt(1.0 - self.systemic), np.sqrt(self.systemic)
        cb, sb = np.sqrt(1.0 - self.benign), np.sqrt(self.benign)
        amplitude = np.sqrt(self.weight) * ca ** (self.n - k) * sa**k + np.sqrt(
            1.0 - self.weight
        ) * cb ** (self.n - k) * sb**k
        pmf = binom * amplitude**2
        return pmf / pmf.sum()

    def marginal(self) -> float:
        pmf = self.loss_count_pmf()
        return float(np.dot(np.arange(self.n + 1), pmf) / self.n)

    def default_correlation(self) -> float:
        pmf = self.loss_count_pmf()
        k = np.arange(self.n + 1)
        e_k = float(np.dot(k, pmf))
        e_kkm1 = float(np.dot(k * (k - 1), pmf))
        p = e_k / self.n
        q = e_kkm1 / (self.n * (self.n - 1))
        denom = p * (1.0 - p)
        return 0.0 if denom <= 0 else (q - p * p) / denom

    @classmethod
    def from_targets(
        cls,
        n: int,
        target_marginal: float,
        target_default_corr: float,
        *,
        benign_fraction: float = 0.3,
    ) -> "GHZBlend":
        """Solve ``(weight, systemic)`` for the coherent state's exact moments.

        The two product components are not orthogonal, so their coherent superposition does not
        obey the classical-mixture marginal identity. That identity is used only to seed a bounded
        two-moment solve against :meth:`marginal` and :meth:`default_correlation`.
        """
        from scipy.optimize import brentq, least_squares

        target_marginal = float(np.clip(target_marginal, *_CLIP))
        benign = max(_EPS, target_marginal * benign_fraction)

        def make(systemic: float) -> "GHZBlend":
            weight = np.clip((target_marginal - benign) / (systemic - benign), _EPS, 1.0 - _EPS)
            return cls(n=n, weight=float(weight), benign=benign, systemic=systemic)

        if abs(target_default_corr) < 1e-12:
            return cls(n=n, weight=1.0, benign=target_marginal, systemic=target_marginal)

        def corr_residual(systemic: float) -> float:
            return make(systemic).default_correlation() - target_default_corr

        lo, hi = target_marginal + 1e-3, 1.0 - 1e-6
        f_lo, f_hi = corr_residual(lo), corr_residual(hi)
        if f_lo > 0:
            systemic = lo
        elif f_hi < 0:
            systemic = hi
        else:
            systemic = brentq(corr_residual, lo, hi, xtol=1e-12, maxiter=200)
        seed = make(systemic)

        def residual(params: np.ndarray) -> np.ndarray:
            candidate = cls(
                n=n,
                weight=float(params[0]),
                benign=benign,
                systemic=float(params[1]),
            )
            return np.array(
                [
                    candidate.marginal() - target_marginal,
                    candidate.default_correlation() - target_default_corr,
                ]
            )

        solved = least_squares(
            residual,
            x0=np.array([seed.weight, seed.systemic]),
            bounds=(
                np.array([_EPS, target_marginal + 1e-6]),
                np.array([1.0 - _EPS, 1.0 - 1e-6]),
            ),
            xtol=1e-13,
            ftol=1e-13,
            gtol=1e-13,
            max_nfev=500,
        )
        if not solved.success or np.max(np.abs(residual(solved.x))) > 1e-7:
            raise RuntimeError("could not calibrate GHZ blend to the requested moments")
        return cls(
            n=n,
            weight=float(solved.x[0]),
            benign=benign,
            systemic=float(solved.x[1]),
        )
