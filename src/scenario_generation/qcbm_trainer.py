from __future__ import annotations

import numpy as np
from typing import Callable

from scipy.optimize import minimize

from systemic_risk.spec import SystemSpec
from systemic_risk.generators.quantum import ansatz as A
from systemic_risk.generators.quantum.statevector import StateVector


def _block_loss(params: np.ndarray, circuit: A.EntangledCircuit) -> float:
    # unpack params into ry and cry
    n_ry = circuit.size
    n_cry = len(circuit.edges)
    ry = params[:n_ry]
    cry = params[n_ry : n_ry + n_cry]

    # build statevector for block
    sv = StateVector(circuit.size)
    for i, theta in enumerate(ry):
        sv.ry(i, theta)
    for e, (control, target) in enumerate(circuit.edges):
        sv.cry(control, target, cry[e])

    marg = sv.marginals()
    joint = sv.pairwise_joint()
    # target moments
    tp = circuit.target_p
    tj = circuit.target_cov + np.outer(tp, tp)

    # loss: weighted sum of marginal and joint squared errors
    lm = float(np.sum((marg - tp) ** 2))
    lj = float(np.sum((joint - tj) ** 2))
    return lm + lj


class QCBMTrainer:
    """Simple trainer that refines analytic angles per block via L-BFGS-B.

    This is a light-weight optimisation harness (not a full qGAN). It accepts
    analytic seed circuits from :mod:`systemic_risk.generators.quantum.ansatz`
    and refines ``RY`` and ``CRY`` angles jointly to reduce marginal/joint
    moment error under the exact statevector simulator.
    """

    def __init__(self, maxiter: int = 50):
        self.maxiter = int(maxiter)

    def fit(self, spec: SystemSpec, *, edges: list[tuple[int, int]] | None = None) -> dict:
        if edges is None:
            edges = A.dependency_edges(spec, threshold=0.02, within_clusters_only=False)
        cov = A.target_covariance(spec)
        p = np.clip(spec.marginal_default_probs, 1e-6, 1.0 - 1e-6)

        # partition into blocks
        blocks = A.partition_blocks(spec, edges, max_block=spec.n)
        results = {}
        for qubits in blocks:
            circuit = A._block_circuit(qubits, p, cov, edges)
            # initial calibration (light)
            circuit = A.calibrate_block(circuit, lambda ry, edges, cry: self._moments_fn(ry, edges, cry), iterations=5)

            # pack params
            x0 = np.concatenate([circuit.ry, circuit.cry])
            bounds = [(1e-6, np.pi - 1e-6)] * circuit.size + [(-2.5, 2.5)] * len(circuit.edges)

            res = minimize(lambda x: _block_loss(x, circuit), x0, method="L-BFGS-B", bounds=bounds, options={"maxiter": self.maxiter})
            x_opt = res.x
            ry_opt = x_opt[: circuit.size]
            cry_opt = x_opt[circuit.size : circuit.size + len(circuit.edges)]

            circuit_opt = A.EntangledCircuit(qubits=circuit.qubits, ry=ry_opt, edges=circuit.edges, cry=cry_opt, target_p=circuit.target_p, target_cov=circuit.target_cov)
            results[tuple(qubits)] = {"circuit": circuit_opt, "opt_result": res}
        return results

    @staticmethod
    def _moments_fn(ry: np.ndarray, edges: list[tuple[int, int]], cry: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        n = ry.size
        sv = StateVector(n)
        for i, theta in enumerate(ry):
            sv.ry(i, theta)
        for e, (control, target) in enumerate(edges):
            sv.cry(control, target, cry[e])
        return sv.marginals(), sv.pairwise_joint()
