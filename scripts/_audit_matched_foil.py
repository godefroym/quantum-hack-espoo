"""Adversarial confound test for criteria 2 & 3.

The headline demonstration compares the entangled generator against a Gaussian-copula foil that
was calibrated to the *nominal* target correlation matrix. On the heterogeneous real community a
third of those targets are Fréchet-infeasible, so the foil collapses to a much LOWER realized
correlation (corr_mean ~0.15) than the entangled generator realizes (corr_mean ~0.43). Any
cascade-tail gap could therefore be ordinary SECOND-order (the entangled generator is simply more
correlated), not the higher-order structure the thesis claims.

This script removes the confound. It:
  1. fits the entangled generator on the real community spec and draws a large reference sample;
  2. estimates the entangled generator's OWN realized marginals p* and realized Pearson
     correlation matrix C* from that sample;
  3. builds classical foils (Gaussian copula AND Ising/Boltzmann) calibrated to p* and C*;
  4. verifies the foils actually realize p* and C* (i.e. first+second order are genuinely matched);
  5. draws equal-N samples, runs all through the SAME cascade (real exposures/buffers), and compares
     the higher-order/tail discriminators and the cascade-tail metrics.

If, at genuinely matched first+second order, the entangled generator STILL carries more
higher-order/tail structure AND still moves the cascade tail, criteria 2 & 3 are real. If the gap
vanishes once correlations are matched, the thesis as demonstrated is confounded.

Run:
    uv run python scripts/_audit_matched_foil.py
"""

from __future__ import annotations

from _demo._bootstrap import bootstrap

bootstrap()

import numpy as np  # noqa: E402

from scipy.stats import norm  # noqa: E402

from _demo._higher_order import _deep_tail_metrics  # noqa: E402
from _demo._second_order import empirical_marginals_and_corr  # noqa: E402
from _demo._specs import achievable_corr, real_community_spec  # noqa: E402
from systemic_risk.evaluation import compute_metrics  # noqa: E402
from systemic_risk.evaluation.joint_structure import _latent_correlation  # noqa: E402
from systemic_risk.generators import (  # noqa: E402
    EntangledBornMachineGenerator,
    GaussianCopulaGenerator,
    IsingBoltzmannGenerator,
)
from systemic_risk.simulator.cascade import simulate_many  # noqa: E402
from systemic_risk.spec import SystemSpec  # noqa: E402
from systemic_risk.utils.validation import nearest_psd_correlation  # noqa: E402


class LatentMatchedGaussianCopula(GaussianCopulaGenerator):
    """Gaussian copula whose LATENT correlation is solved so the binary indicators realize C*.

    The stock copula feeds the target Pearson correlation straight in as the latent correlation,
    which badly undershoots at tiny marginals (thresholding shrinks correlation). Here we invert the
    binary->latent map (the project's own ``_latent_correlation``, used by the higher-order
    reference) so the SAMPLED Pearson correlation lands on C* -- a genuine 2nd-order match. The
    copula structure (and thus its provably-zero tail dependence) is unchanged.
    """

    name = "Gaussian(latent-matched)"

    def fit(self, spec: SystemSpec) -> None:
        p = np.clip(spec.marginal_default_probs.copy(), 1e-12, 1.0 - 1e-12)
        thresholds = norm.ppf(p)
        latent = _latent_correlation(p, spec.target_pairwise_corr, thresholds)
        self.p_ = p
        self.corr_ = nearest_psd_correlation(latent)
        np.fill_diagonal(self.corr_, 1.0)
        self.thresholds_ = thresholds


def tune_ising_to_corr(cal_spec: SystemSpec, target_corr_mean: float, seed: int,
                       tune_n: int = 60_000) -> IsingBoltzmannGenerator:
    """Bisect the Ising global coupling scale so its SAMPLED mean correlation hits C*'s mean.

    The ``coupling_scale='auto'`` calibration uses a mean-field linear-response prediction that
    overshoots wildly here (it lands at corr ~0.9). We instead bisect the *empirical* sampled mean
    correlation against the entangled generator's realized mean -- a fair, like-for-like 2nd-order
    match before judging higher order.
    """
    iu_local = None

    def sampled_mean_corr(scale: float) -> float:
        nonlocal iu_local
        gen = IsingBoltzmannGenerator(route="correlation", coupling_scale=float(scale))
        gen.fit(cal_spec)
        s = gen.sample(tune_n, seed=seed)
        _, corr = empirical_marginals_and_corr(s)
        if iu_local is None:
            iu_local = np.triu_indices(s.shape[1], k=1)
        return float(corr[iu_local].mean())

    lo, hi = 0.0, 1.0
    f_hi = sampled_mean_corr(hi)
    guard = 0
    while f_hi < target_corr_mean and hi < 20.0 and guard < 40:
        hi *= 1.4
        f_hi = sampled_mean_corr(hi)
        guard += 1
    for _ in range(22):
        mid = 0.5 * (lo + hi)
        if sampled_mean_corr(mid) < target_corr_mean:
            lo = mid
        else:
            hi = mid
    scale = 0.5 * (lo + hi)
    gen = IsingBoltzmannGenerator(route="correlation", coupling_scale=float(scale))
    gen.fit(cal_spec)
    print(f"  (Ising tuned coupling_scale={scale:.4f} -> sampled corr_mean~{target_corr_mean:.3f})")
    return gen

SEED = 2026
REFERENCE_N = 400_000          # to estimate p*, C* of the entangled generator
COMPARE_N = 200_000            # equal-N head-to-head (matches the demo budget)
CONVERGENCE_SIZES = (20_000, 80_000, 320_000, 1_280_000)


def matched_calibration_spec(
    cascade_spec: SystemSpec, p_star: np.ndarray, corr_star: np.ndarray
) -> SystemSpec:
    """A spec carrying the entangled generator's realized p*/C* (zero exposure for calibration).

    The classical generators calibrate off ``marginal_default_probs`` and ``target_pairwise_corr``.
    We feed them p* and C* directly. Exposure is zeroed here so the Ising route uses the
    correlation (not the exposure graph); the cascade still uses the *real* exposure spec.
    """
    corr = nearest_psd_correlation(corr_star)
    np.fill_diagonal(corr, 1.0)
    return SystemSpec(
        node_names=list(cascade_spec.node_names),
        node_types=list(cascade_spec.node_types),
        exposure_matrix=np.zeros((cascade_spec.n, cascade_spec.n)),
        capital_buffers=cascade_spec.capital_buffers.copy(),
        marginal_default_probs=p_star.copy(),
        target_pairwise_corr=corr,
        clusters=list(cascade_spec.clusters) if cascade_spec.clusters else None,
        metadata={"kind": "matched-foil-calibration"},
    )


def second_order_report(name: str, samples: np.ndarray, p_star: np.ndarray,
                        corr_star: np.ndarray) -> dict:
    """How well a sample reproduces the entangled generator's realized p*/C*."""
    marg, corr = empirical_marginals_and_corr(samples)
    iu = np.triu_indices(len(p_star), k=1)
    return {
        "name": name,
        "corr_mean": float(corr[iu].mean()),
        "marg_rmse_vs_pstar": float(np.sqrt(np.mean((marg - p_star) ** 2))),
        "corr_rmse_vs_Cstar": float(np.sqrt(np.mean((corr[iu] - corr_star[iu]) ** 2))),
        "corr_maxerr_vs_Cstar": float(np.max(np.abs(corr[iu] - corr_star[iu]))),
    }


def full_metrics(samples: np.ndarray, cascade_spec: SystemSpec) -> dict:
    cascades = simulate_many(samples, cascade_spec)
    failures = np.array([c.failure_count for c in cascades], dtype=float)
    half = int(np.ceil(0.5 * cascade_spec.n))
    m = compute_metrics(samples, cascades, cascade_spec, severe_threshold=half)
    m.update(_deep_tail_metrics(samples, failures, cascade_spec))
    return m


def main() -> None:
    rng_seeds = np.random.SeedSequence(SEED).spawn(8)

    cascade_spec = real_community_spec().spec
    n = cascade_spec.n
    iu = np.triu_indices(n, k=1)
    print(f"Real community cascade spec: n={n}, exposure={cascade_spec.exposure_matrix.sum():.0f}")

    # 1-2. Fit entangled generator; estimate its OWN realized p*, C*.
    entangled = EntangledBornMachineGenerator(ansatz="entangled", calibrate=True)
    entangled.fit(cascade_spec)
    ref = entangled.sample(REFERENCE_N, seed=int(rng_seeds[0].generate_state(1)[0]))
    p_star, corr_star = empirical_marginals_and_corr(ref)
    print(f"\nEntangled realized: marginal mean={p_star.mean():.5f}, "
          f"corr mean={corr_star[iu].mean():.4f}, corr max={corr_star[iu].max():.4f}")

    # For context: how far is C* from the achievable Fréchet ceiling of the REAL spec?
    ach = achievable_corr(cascade_spec)
    print(f"(real achievable-ceiling corr mean={ach[iu].mean():.4f}; "
          f"the matched foils target the entangled generator's own C*, not the spec target.)")

    # 3. Build foils genuinely calibrated to p*, C*.
    #    Gaussian: invert binary->latent so the SAMPLED Pearson corr lands on C* (not undershoot).
    #    Ising:    bisect the global coupling scale so its SAMPLED mean corr hits C*'s mean.
    cal_spec = matched_calibration_spec(cascade_spec, p_star, corr_star)
    gauss = LatentMatchedGaussianCopula()
    gauss.fit(cal_spec)
    print("\nTuning Ising coupling scale to the entangled generator's realized correlation...")
    ising = tune_ising_to_corr(cal_spec, float(corr_star[iu].mean()),
                               seed=int(rng_seeds[7].generate_state(1)[0]))

    samples = {
        "Entangled": entangled.sample(COMPARE_N, seed=int(rng_seeds[1].generate_state(1)[0])),
        "Gaussian(matched)": gauss.sample(COMPARE_N, seed=int(rng_seeds[2].generate_state(1)[0])),
        "Ising(matched)": ising.sample(COMPARE_N, seed=int(rng_seeds[3].generate_state(1)[0])),
    }

    # 4. Verify the match (the crux: are 1st+2nd order genuinely equal now?).
    print("\n=== SECOND-ORDER MATCH to the entangled generator's realized p*, C* ===")
    print(f"{'generator':<20}{'corr_mean':>10}{'marg_rmse':>13}{'corr_rmse':>12}{'corr_maxerr':>13}")
    for name, s in samples.items():
        r = second_order_report(name, s, p_star, corr_star)
        print(f"{r['name']:<20}{r['corr_mean']:>10.4f}{r['marg_rmse_vs_pstar']:>13.2e}"
              f"{r['corr_rmse_vs_Cstar']:>12.4f}{r['corr_maxerr_vs_Cstar']:>13.4f}")

    # 5. Higher-order + cascade-tail at MATCHED second order.
    metrics = {name: full_metrics(s, cascade_spec) for name, s in samples.items()}
    ho_keys = ["excess_coskewness_rms", "excess_coskewness_max",
               "excess_pairwise_lower_tail_dependence", "aggregate_tail_dependence",
               "joint_tail_excess"]
    tail_keys = ["p_severe_cascade", "p_cascade_half_or_more", "p_initial_half_or_more",
                 "tail_mean_1pct", "tail_mean_5pct", "mean_cascade_size",
                 "cascade_count_cvar_999"]

    print("\n=== CRITERION 2 — higher-order/tail discriminators (matched 2nd order) ===")
    _print_metric_table(metrics, ho_keys)
    print("\n=== CRITERION 3 — cascade-tail metrics (matched 2nd order, same engine) ===")
    _print_metric_table(metrics, tail_keys)

    # 6. Excess co-skewness convergence at matched 2nd order (foil should decay; entangled persist).
    print("\n=== Excess co-skewness vs N (matched 2nd order) ===")
    from systemic_risk.evaluation.joint_structure import higher_order_structure
    print(f"{'generator':<20}" + "".join(f"N={n_:>9}" for n_ in CONVERGENCE_SIZES))
    for name, gen, sd in (
        ("Gaussian(matched)", gauss, rng_seeds[4]),
        ("Ising(matched)", ising, rng_seeds[5]),
        ("Entangled", entangled, rng_seeds[6]),
    ):
        vals = []
        for n_ in CONVERGENCE_SIZES:
            s = gen.sample(n_, seed=int(sd.generate_state(1)[0]))
            vals.append(higher_order_structure(s).excess_coskewness_rms)
        print(f"{name:<20}" + "".join(f"{v:>11.3f}" for v in vals))

    # ---- bottom line ----
    e = metrics["Entangled"]
    g = metrics["Gaussian(matched)"]
    i = metrics["Ising(matched)"]
    print("\n=== CONFOUND-TEST VERDICT ===")
    for foil_name, f in (("Gaussian(matched)", g), ("Ising(matched)", i)):
        cosk_ratio = e["excess_coskewness_rms"] / max(f["excess_coskewness_rms"], 1e-9)
        deep_ratio = e["p_cascade_half_or_more"] / max(f["p_cascade_half_or_more"], 1e-12)
        tm_ratio = e["tail_mean_1pct"] / max(f["tail_mean_1pct"], 1e-12)
        print(f"  vs {foil_name}: excess-coskew {cosk_ratio:5.1f}x | "
              f"deep p(K>=half) {deep_ratio:5.1f}x | tail-mean(1%) {tm_ratio:4.2f}x")


def _print_metric_table(metrics: dict, keys: list[str]) -> None:
    names = list(metrics.keys())
    width = max(len(k) for k in keys) + 2
    header = " " * width + "".join(f"{nm:>20}" for nm in names)
    print(header)
    for k in keys:
        row = f"{k:<{width}}" + "".join(f"{metrics[nm].get(k, float('nan')):>20.6g}" for nm in names)
        print(row)


if __name__ == "__main__":
    main()
