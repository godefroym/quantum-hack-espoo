from __future__ import annotations

import numpy as np

from systemic_risk.models.ising import IsingModel
from systemic_risk.models.mean_field_oracle import MeanFieldIsingOracle


def test_parallel_tempering_matches_exact_homogeneous_moments() -> None:
    n = 8
    oracle = MeanFieldIsingOracle.from_targets(
        n=n,
        target_marginal=0.20,
        target_default_corr=0.15,
    )
    couplings = np.full((n, n), oracle.coupling)
    np.fill_diagonal(couplings, 0.0)
    model = IsingModel(fields=np.full(n, oracle.field), couplings=couplings)

    samples = model.sample(
        8_000,
        seed=12,
        method="parallel_tempering",
        burn_in=800,
        thin=2,
        n_replicas=6,
        beta_min=0.15,
    )
    marginal = float(samples.mean())
    corr = np.corrcoef(samples, rowvar=False)
    mean_corr = float(corr[np.triu_indices(n, k=1)].mean())

    assert abs(marginal - oracle.marginal_default_prob()) < 0.015
    assert abs(mean_corr - oracle.default_correlation()) < 0.035
