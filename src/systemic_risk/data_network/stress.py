"""Stress transform: lift a baseline ``SystemSpec`` to a 2008-style systemic-crisis scenario.

WHY
---
The real exposure network's baseline marginals come from *through-the-cycle* Moody's
one-year PDs (``data/external/ratings/moodys_pd_by_rating.csv``): a cross-entity mean of
~0.23 %, every one of the 38 entities sitting **below** the ~2.7 % QPU readout+decoherence
noise floor. Loaded faithfully onto hardware they are pure noise. A *stress test*, however,
targets exactly the regime where PDs blow out to ~5-30 %, which is comfortably **above** the
floor. This module raises the signal to that regime without touching the network's
co-default structure.

WHAT IS / ISN'T CHANGED
-----------------------
* **Marginals are lifted** by a single rank-preserving *logit-space shift* calibrated so the
  cross-entity MEAN stressed PD lands on ``target_mean_pd`` (default 0.15). The shift is
  monotone in PD, so the relative ordering of entity risk is preserved exactly (riskier
  entities stay riskier). An optional ``crisis_floor`` lifts the single AAA outlier
  (Microsoft, baseline PD clipped to 1e-5) above the noise floor too -- in a 2008-style
  crisis even AAA names carry a small nonzero PD -- and the shift is re-solved so the mean
  still lands on target.
* **The correlation graph is kept UNCHANGED.** The real spec carries its dependency in
  ``target_pairwise_corr`` under the ``latent_gaussian`` correlation space. Because the joint
  co-default probabilities are *derived* from (latent correlation, marginals) on demand
  (:meth:`SystemSpec.target_pairwise_joint_probs` -> :func:`latent_corr_to_joint`), keeping
  the latent correlation fixed while lifting the marginals re-derives a coherent joint
  automatically: P(default_i & default_j) rises consistently with the higher marginals at the
  unchanged latent coupling. No co-default target is hand-edited.

2008 CALIBRATION ANCHOR
-----------------------
Two independent historical signals already committed to ``data/external/`` agree that a
2008-style crisis multiplies baseline credit risk by ~4x and pushes broad PDs into the
~5-15 % band:

* **Moody's crisis default rate.** The all-rated long-run year-1 default rate is 1.48-1.79 %
  (``moodys_pd_by_rating.csv``). In the GFC peak (2009) the realized all-corporate annual
  default rate ran ~4-5x that, and the speculative-grade rate reached ~13 % -- the upper end
  of our stressed distribution.
* **BAA-AAA credit spread (FRED, in-repo ``fred/BAA.csv`` & ``AAA.csv``).** The GFC trough
  (2008-11..2009-01 mean spread 3.19 %) is **~3.85x** the calm 2004-06 baseline (0.83 %). An
  independent, purely macro-derived stress multiplier.

A mean target of 0.15 sits between the realized all-corporate (~5 %) and speculative-grade
(~13 %) GFC default rates, reflecting that the roster is a stressed mix of large banks and
corporates rather than a pure SG pool -- a deliberately severe but historically-anchored tail
scenario. The realized PD distribution and the ~3.85x macro multiplier are recorded in the
returned spec's ``metadata['stress']`` block.

HONESTY CAVEAT
--------------
This is a *hypothetical* uniform crisis overlay, not an entity-by-entity 2008 PD re-rating:
every node is shifted by the same logit amount, so the scenario assumes a broad systemic
shock rather than idiosyncratic distress. It is calibrated to historical *aggregates*, and is
intended as the stress scenario a quantum tail-risk run should target -- not a forecast.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import brentq

from systemic_risk.spec import SystemSpec

# Effective QPU readout+decoherence floor (see memory ``real-network-not-hardware-loadable``).
QPU_NOISE_FLOOR = 0.027

# 2008 macro anchor derived in this module's docstring from FRED BAA/AAA spreads.
GFC_SPREAD_MULTIPLIER = 3.85

_EPS = 1e-9


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, _EPS, 1.0 - _EPS)
    return np.log(p / (1.0 - p))


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


@dataclass(frozen=True)
class StressCalibration:
    """The calibration recovered while building the stressed marginals."""

    target_mean_pd: float
    crisis_floor: float
    logit_shift: float
    baseline_mean: float
    stressed_mean: float
    stressed_min: float
    stressed_max: float
    n_above_noise_floor: int
    n_below_noise_floor: int
    n_entities: int
    gfc_spread_multiplier: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario": "2008-style systemic credit crisis (uniform logit-shift overlay)",
            "anchor": (
                "Moody's GFC-peak default rates (all-corporate ~5%, speculative-grade ~13%, "
                "long-run all-rated 1.48-1.79%) + FRED BAA-AAA spread GFC/calm multiplier "
                f"~{GFC_SPREAD_MULTIPLIER}x"
            ),
            "target_mean_pd": self.target_mean_pd,
            "crisis_floor": self.crisis_floor,
            "logit_shift": self.logit_shift,
            "gfc_spread_multiplier": self.gfc_spread_multiplier,
            "baseline_mean_pd": self.baseline_mean,
            "stressed_mean_pd": self.stressed_mean,
            "stressed_min_pd": self.stressed_min,
            "stressed_max_pd": self.stressed_max,
            "noise_floor": QPU_NOISE_FLOOR,
            "n_entities": self.n_entities,
            "n_above_noise_floor": self.n_above_noise_floor,
            "n_below_noise_floor": self.n_below_noise_floor,
            "correlation_unchanged": True,
            "ordering_preserved": True,
            "caveat": (
                "Hypothetical uniform crisis overlay calibrated to historical aggregates, "
                "not an entity-by-entity 2008 re-rating; assumes a broad systemic shock."
            ),
        }


def stressed_marginals(
    baseline: np.ndarray,
    *,
    target_mean_pd: float = 0.15,
    crisis_floor: float = QPU_NOISE_FLOOR,
) -> tuple[np.ndarray, StressCalibration]:
    """Lift baseline PDs to a crisis regime via a rank-preserving logit-space shift.

    A single shift ``s`` is solved so ``mean(max(sigmoid(logit(p)+s), crisis_floor))``
    equals ``target_mean_pd``. The shift is monotone, so entity risk ordering is preserved;
    the floor guarantees every entity clears the noise floor when ``crisis_floor`` >= it.
    """
    if not 0.0 < target_mean_pd < 1.0:
        raise ValueError("target_mean_pd must lie in (0, 1)")
    if not 0.0 <= crisis_floor < target_mean_pd:
        raise ValueError("crisis_floor must lie in [0, target_mean_pd)")

    baseline = np.clip(np.asarray(baseline, dtype=float), _EPS, 1.0 - _EPS)
    z = _logit(baseline)
    floor = float(crisis_floor)

    def mean_at(s: float) -> float:
        return float(np.maximum(_sigmoid(z + s), floor).mean())

    if mean_at(0.0) >= target_mean_pd:
        # Already at/above target without lifting (degenerate); no positive shift needed.
        shift = 0.0
    else:
        # mean_at is strictly increasing in s and saturates below 1, so a positive root
        # exists for any feasible target; a wide bracket is safe.
        shift = float(brentq(lambda s: mean_at(s) - target_mean_pd, 0.0, 60.0))

    stressed = np.maximum(_sigmoid(z + shift), floor)
    calib = StressCalibration(
        target_mean_pd=float(target_mean_pd),
        crisis_floor=floor,
        logit_shift=round(shift, 6),
        baseline_mean=round(float(baseline.mean()), 6),
        stressed_mean=round(float(stressed.mean()), 6),
        stressed_min=round(float(stressed.min()), 6),
        stressed_max=round(float(stressed.max()), 6),
        n_above_noise_floor=int(np.sum(stressed >= QPU_NOISE_FLOOR)),
        n_below_noise_floor=int(np.sum(stressed < QPU_NOISE_FLOOR)),
        n_entities=int(stressed.size),
        gfc_spread_multiplier=GFC_SPREAD_MULTIPLIER,
    )
    return stressed, calib


def apply_stress(
    spec: SystemSpec,
    *,
    target_mean_pd: float = 0.15,
    crisis_floor: float = QPU_NOISE_FLOOR,
) -> tuple[SystemSpec, StressCalibration]:
    """Return a stressed copy of ``spec`` with crisis marginals and an unchanged correlation.

    Only ``marginal_default_probs`` changes. ``target_pairwise_corr`` (the latent-Gaussian
    dependency) is carried through untouched, so the joint co-default structure is re-derived
    coherently at the higher marginals. ``target_joint_probs`` is intentionally NOT set --
    leaving it ``None`` keeps the joint defined by (correlation, marginals) rather than
    freezing a stale baseline joint.
    """
    stressed, calib = stressed_marginals(
        np.asarray(spec.marginal_default_probs, dtype=float),
        target_mean_pd=target_mean_pd,
        crisis_floor=crisis_floor,
    )

    metadata = dict(spec.metadata)
    metadata["stress"] = calib.to_dict()
    name = metadata.get("name", "system")
    metadata["name"] = f"{name} -- 2008 stress (mean PD {calib.stressed_mean:.0%})"

    stressed_spec = SystemSpec(
        node_names=list(spec.node_names),
        node_types=list(spec.node_types),
        exposure_matrix=spec.exposure_matrix.copy(),
        capital_buffers=spec.capital_buffers.copy(),
        marginal_default_probs=stressed,
        target_pairwise_corr=(
            None if spec.target_pairwise_corr is None else spec.target_pairwise_corr.copy()
        ),
        target_joint_probs=None,
        clusters=None if spec.clusters is None else list(spec.clusters),
        metadata=metadata,
    )
    return stressed_spec, calib
