"""The n = 54 scale story — homogeneous mean-field oracle validation.

The heterogeneous entangled ansatz is block-separable and never forms the ``2^54`` statevector,
so it *runs* at 54 qubits but cannot be checked against a full-state ground truth there. The
**homogeneous** limit can: for a uniform-marginal, equicorrelated target the joint law depends
only on the number of defaults, and the closed-form mean-field Ising oracle gives the exact
loss-count distribution at any ``n`` (Molins & Vives 2005). The entangled construction's
permutation-symmetric loader reproduces that distribution analytically — also at any ``n``.

This module compares the two at n = 54 and returns the total-variation distance between their
loss-count laws together with the marginal and default-correlation each reproduces. A TV near
machine precision is the evidence that the construction scales to the hardware target exactly,
with no ``2^54`` cost on either side.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from systemic_risk.generators import EntangledBornMachineGenerator
from systemic_risk.models.mean_field_oracle import MeanFieldIsingOracle, total_variation_distance
from systemic_risk.models.ising import LossDistribution
from systemic_risk.spec import SystemSpec


@dataclass(frozen=True)
class OracleValidation:
    """Result of validating the entangled loader against the mean-field oracle at scale."""

    n: int
    target_marginal: float
    target_default_corr: float
    tv_distance: float
    generator_marginal: float
    oracle_marginal: float
    generator_default_corr: float
    oracle_default_corr: float


def validate_against_oracle(spec: SystemSpec) -> OracleValidation:
    """Validate the entangled loss-count law against the closed-form oracle for a homogeneous spec.

    ``spec`` must be homogeneous (uniform marginal, equicorrelated); the entangled generator then
    fits its exact permutation-symmetric loader and we compare its closed-form loss-count pmf to
    the oracle solved for the same marginal and default correlation.
    """
    p = spec.marginal_default_probs
    iu = np.triu_indices(spec.n, k=1)
    target_marginal = float(p.mean())
    target_corr = float(spec.dependency_matrix()[iu].mean()) if iu[0].size else 0.0

    generator = EntangledBornMachineGenerator(ansatz="entangled")
    generator.fit(spec)
    gen_pmf = generator.loss_count_pmf()

    oracle = MeanFieldIsingOracle.from_targets(spec.n, target_marginal, target_corr)
    oracle_dist = oracle.loss_distribution()

    tv = total_variation_distance(LossDistribution(pmf=gen_pmf, exact=True), oracle_dist)
    return OracleValidation(
        n=spec.n,
        target_marginal=target_marginal,
        target_default_corr=target_corr,
        tv_distance=tv,
        generator_marginal=_pmf_marginal(gen_pmf, spec.n),
        oracle_marginal=oracle.marginal_default_prob(),
        generator_default_corr=_pmf_default_corr(gen_pmf, spec.n),
        oracle_default_corr=oracle.default_correlation(),
    )


def _pmf_marginal(pmf: np.ndarray, n: int) -> float:
    return float(np.dot(np.arange(n + 1), pmf) / n)


def _pmf_default_corr(pmf: np.ndarray, n: int) -> float:
    k = np.arange(n + 1)
    e_k = float(np.dot(k, pmf))
    e_k_km1 = float(np.dot(k * (k - 1), pmf))
    p = e_k / n
    q = e_k_km1 / (n * (n - 1))
    denom = p * (1.0 - p)
    return 0.0 if denom <= 0 else (q - p * p) / denom
