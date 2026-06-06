"""Ising / Boltzmann plausibility models for correlated-default scenarios.

The plausibility model from ``scenario_generation.md`` is a pairwise maximum-entropy
(Ising / Boltzmann) distribution over binary default configurations ``x in {0, 1}^n``::

    Pi(x) = sum_i h_i x_i + sum_{i<j} J_ij x_i x_j,    P(x) propto exp(Pi(x)).

This package provides:

- :class:`~systemic_risk.models.ising.IsingModel` -- the model itself, with exact
  enumeration (n <= ~20), Gibbs/Metropolis MCMC, and parallel tempering (n = 54).
- :mod:`~systemic_risk.models.calibration` -- field/coupling calibration: logit fields,
  Boltzmann-learning field refit, ``J`` from a correlation matrix (inverse-Ising) or from
  an exposure graph (density-corrected gravity model).
- :class:`~systemic_risk.models.mean_field_oracle.MeanFieldIsingOracle` -- the homogeneous
  mean-field Ising loss distribution, exact at *any* ``n`` (including 54).

References
----------
Filiz, Guo, Morton & Sturmfels (2012) arXiv:0809.1393; Schneidman et al. (2006) Nature 440;
Molins & Vives (2005) arXiv:cond-mat/0401378; Kitsukawa, Mori & Hisakado (2006)
arXiv:physics/0603040; Bury (2013) arXiv:1210.8380; Nguyen, Zecchina & Berg (2017)
arXiv:1702.01522; Cimini, Squartini, Garlaschelli & Gabrielli (2015) Sci. Rep. 5, 15758.
"""

from systemic_risk.models.calibration import (
    couplings_from_correlation,
    couplings_from_exposure,
    fit_fields_boltzmann,
    logit_fields,
)
from systemic_risk.models.ising import IsingModel, LossDistribution
from systemic_risk.models.mean_field_oracle import MeanFieldIsingOracle

__all__ = [
    "IsingModel",
    "LossDistribution",
    "MeanFieldIsingOracle",
    "couplings_from_correlation",
    "couplings_from_exposure",
    "fit_fields_boltzmann",
    "logit_fields",
]
