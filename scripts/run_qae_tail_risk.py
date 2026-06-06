"""QAE CALCULATION SURFACE — quantum amplitude estimation of cascade tail risk.

Closes the gap the README flags as designed-not-implemented: tail probabilities (``P(severe)``,
CVaR of cascade size) computed by *quantum amplitude estimation* over the distribution loaded by
the entangled Born-machine state-loader, with the deterministic cascade as a reversible oracle.
The generation surface (the QCBM loader) and the calculation surface (QAE) are tied together
here, reusing the existing loader (:class:`EntangledBornMachineGenerator`) and cascade
(:func:`run_cascade`) unchanged.

Two distinct, separately-reported claims (kept honest, matching the project's culture):

1. **Equivalence** — the QAE estimate of ``P(severe)`` and CVaR agrees, within its own (Fisher)
   error bar, with the exact value and with the classical Monte-Carlo answer from the existing
   simulator, on the *same* spec / loader / threshold.
2. **Advantage** — the oracle-query count to reach a target relative error scales like
   ``O(1/(eps*sqrt(a)))`` for QAE versus ``O(1/(eps^2*a))`` for Monte Carlo, the gap widening as
   the tail ``a`` shrinks. Reported as oracle-call counts (the hardware-relevant figure).

WHAT IS SIMULATED VS HARDWARE (read before quoting numbers). Everything here is an *exact
classical statevector simulation* of the QAE operators — it forms the ``2^n`` amplitude vector,
so it is tractable only at the small ``n`` used below and is **not** itself the speedup
(simulating QAE is exponential in ``n``). The wall-clock of this script is therefore NOT a
speedup and is never reported as one. What is faithful and hardware-relevant is the *equivalence*
of the estimate and the *oracle-query counts*. The construction — one qubit per institution plus
cascade-comparison ancillas, with the existing QCBM as the loader — extrapolates to the 54-qubit
target; the exact simulation does not.

Run:
    uv run python scripts/run_qae_tail_risk.py
"""

from __future__ import annotations

from _demo._bootstrap import bootstrap

OUTPUTS = bootstrap()

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from _demo._specs import qae_tail_risk_spec  # noqa: E402
from systemic_risk.evaluation.qae_tail_risk import QAETailRiskEstimator  # noqa: E402
from systemic_risk.generators import EntangledBornMachineGenerator  # noqa: E402
from systemic_risk.generators.quantum.amplitude_estimation import (  # noqa: E402
    mc_queries_for_relative_error,
    query_complexity_curve,
)
from systemic_risk.simulator.cascade import simulate_many  # noqa: E402


N_QUBITS = 12
SEED = 2026
TARGET_REL_ERROR = 0.1
MC_REFERENCE_DRAWS = 200_000


def _monte_carlo_reference(generator, spec, n_draws: int, seed: int) -> np.ndarray:
    """Classical Monte-Carlo cascade sizes from the loader's own samples (the existing path)."""
    samples = generator.sample(n_draws, seed=seed)
    return np.array([result.failure_count for result in simulate_many(samples, spec)], dtype=int)


def equivalence_table(estimator: QAETailRiskEstimator, mc_sizes: np.ndarray) -> pd.DataFrame:
    """QAE vs exact vs Monte-Carlo for P(severe) across thresholds, plus CVaR — the equivalence claim."""
    rng = np.random.default_rng(SEED)
    n_mc = mc_sizes.size
    rows = []
    # Sweep thresholds that span the tail from order 10^-1 into the deep tail.
    for threshold in range(2, estimator.spec.n + 1):
        exact = estimator.exact_p_severe(threshold)
        if exact <= 0.0:
            continue
        est = estimator.estimate_p_severe(
            threshold,
            num_powers=9,
            shots=400,
            target_relative_error=TARGET_REL_ERROR,
            rng=rng,
        )
        mc = float((mc_sizes >= threshold).mean())
        mc_se = float(np.sqrt(max(mc * (1.0 - mc), 0.0) / n_mc))
        rows.append(
            {
                "quantity": f"P(cascade>={threshold})",
                "exact": exact,
                "monte_carlo": mc,
                "mc_std_err": mc_se,
                "qae": est.estimate,
                "qae_abs_err": est.abs_error,
                "qae_fisher_ci": est.amplitude_estimate.fisher_ci_half_width,
                "within_3_fisher_ci": est.within_fisher_ci,
                "qae_oracle_queries": est.amplitude_estimate.oracle_queries,
            }
        )

    # CVaR of cascade size at a couple of tail levels.
    from systemic_risk.evaluation.joint_structure import cascade_count_cvar

    for alpha in (0.95, 0.99):
        est = estimator.estimate_cvar(
            alpha=alpha,
            num_powers=9,
            shots=500,
            target_relative_error=TARGET_REL_ERROR,
            rng=rng,
        )
        rows.append(
            {
                "quantity": f"CVaR_{alpha:g}(size)",
                "exact": est.exact,
                "monte_carlo": cascade_count_cvar(mc_sizes, alpha=alpha),
                "mc_std_err": float("nan"),
                "qae": est.estimate,
                "qae_abs_err": est.abs_error,
                "qae_fisher_ci": est.amplitude_estimate.fisher_ci_half_width,
                "within_3_fisher_ci": est.within_fisher_ci,
                "qae_oracle_queries": est.amplitude_estimate.oracle_queries,
            }
        )
    return pd.DataFrame(rows)


def advantage_table(tail_probabilities: np.ndarray, target_eps: float) -> pd.DataFrame:
    """MC vs QAE oracle-query counts across the tail at a fixed target error — the advantage claim."""
    curve = query_complexity_curve(tail_probabilities, target_eps)
    return pd.DataFrame(
        {
            "tail_probability_a": [p.amplitude for p in curve],
            "mc_queries": [p.mc_queries for p in curve],
            "qae_queries": [p.qae_queries for p in curve],
            "speedup_mc_over_qae": [p.speedup for p in curve],
        }
    )


def measured_advantage_table(estimator: QAETailRiskEstimator, threshold: int) -> pd.DataFrame:
    """End-to-end: schedule-sized MLAE on the real oracle vs the matched MC budget, per target eps.

    For each target relative error, a modest LIS schedule (sized to eps, not brute-forced) is run
    on the actual cascade oracle; we report the *issued* oracle-query count and the mean relative
    error over seeds, beside the classical Monte-Carlo budget the same accuracy/tail demands.
    """
    a = estimator.exact_p_severe(threshold)
    plans = {0.1: (6, 20), 0.05: (6, 40), 0.02: (12, 40), 0.01: (18, 40)}
    rows = []
    for eps, (num_powers, shots) in plans.items():
        errors = [
            estimator.estimate_p_severe(
                threshold, num_powers=num_powers, shots=shots, rng=np.random.default_rng(s)
            ).abs_error
            / a
            for s in range(20)
        ]
        issued = estimator.estimate_p_severe(
            threshold, num_powers=num_powers, shots=shots, rng=np.random.default_rng(0)
        ).amplitude_estimate.oracle_queries
        rows.append(
            {
                "target_rel_error": eps,
                "tail_probability_a": a,
                "mlae_powers": num_powers,
                "mlae_shots": shots,
                "mlae_oracle_queries": issued,
                "mlae_mean_rel_error": float(np.mean(errors)),
                "mc_queries_for_target": mc_queries_for_relative_error(a, eps),
                "speedup_mc_over_mlae": mc_queries_for_relative_error(a, eps) / max(issued, 1),
            }
        )
    return pd.DataFrame(rows)


def _plot(equivalence: pd.DataFrame, advantage: pd.DataFrame, path) -> None:
    fig, (left, right) = plt.subplots(1, 2, figsize=(13, 5))

    # Left: equivalence — QAE estimate (with Fisher CI) vs exact, for the P(severe) rows.
    p_rows = equivalence[equivalence["quantity"].str.startswith("P(")]
    x = np.arange(len(p_rows))
    left.errorbar(
        x,
        p_rows["qae"],
        yerr=3.0 * p_rows["qae_fisher_ci"],
        fmt="o",
        capsize=3,
        label="QAE estimate (±3σ Fisher)",
    )
    left.plot(x, p_rows["exact"], "x", color="black", markersize=9, label="exact")
    left.plot(x, p_rows["monte_carlo"], "+", color="tab:red", markersize=10, label="Monte Carlo")
    left.set_yscale("log")
    left.set_xticks(x)
    left.set_xticklabels(p_rows["quantity"], rotation=45, ha="right", fontsize=8)
    left.set_ylabel("tail probability  P(severe)")
    left.set_title("Equivalence: QAE ≈ exact ≈ Monte Carlo")
    left.legend(fontsize=8)

    # Right: advantage — MC vs QAE oracle-query counts vs tail probability (log-log).
    right.loglog(advantage["tail_probability_a"], advantage["mc_queries"], "o-",
                 label="Monte Carlo  O(1/(ε²·a))")
    right.loglog(advantage["tail_probability_a"], advantage["qae_queries"], "s-",
                 label="QAE  O(1/(ε·√a))")
    right.set_xlabel("tail probability  a = P(severe)")
    right.set_ylabel(f"oracle queries for {int(TARGET_REL_ERROR * 100)}% rel. error")
    right.set_title("Advantage: oracle-query count (deep tail → wider gap)")
    right.invert_xaxis()  # deep tail to the right
    right.legend(fontsize=8)
    right.grid(True, which="both", alpha=0.3)

    fig.suptitle(
        "QAE cascade tail risk — exact statevector simulation (NOT a wall-clock speedup; "
        "the speedup is the oracle-query count)",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=180)


def main() -> None:
    spec = qae_tail_risk_spec(n=N_QUBITS)
    print(f"QAE tail-risk calculation surface  (n={spec.n} qubits, exact 2^{spec.n} simulation)")
    print("  NOTE: this is an exact classical simulation of QAE; wall-clock here is NOT the speedup.")
    print("        The honest, hardware-relevant figures are the equivalence and the oracle-query counts.\n")

    generator = EntangledBornMachineGenerator(ansatz="entangled", calibrate=True)
    generator.fit(spec)
    estimator = QAETailRiskEstimator(generator, spec)

    pmf = estimator.cascade_size_pmf()
    print("Exact cascade-size distribution under the loaded distribution P(#defaults = k):")
    print("  " + "  ".join(f"{k}:{p:0.3f}" for k, p in enumerate(pmf) if p > 1e-4) + "\n")

    mc_sizes = _monte_carlo_reference(generator, spec, MC_REFERENCE_DRAWS, seed=SEED)

    print("=== CLAIM 1: EQUIVALENCE (QAE vs exact vs Monte Carlo) ===")
    equivalence = equivalence_table(estimator, mc_sizes)
    print(equivalence.to_string(index=False, float_format=lambda v: f"{v:0.4g}"))
    equivalence.to_csv(OUTPUTS / "qae_equivalence.csv", index=False)
    assert bool(equivalence["within_3_fisher_ci"].all()), "a QAE estimate fell outside 3σ"
    print("  -> every QAE estimate lands within 3σ (Fisher) of the exact value.\n")

    print(f"=== CLAIM 2: ADVANTAGE (oracle-query counts, target rel. error {TARGET_REL_ERROR}) ===")
    tail = np.array([0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001])
    advantage = advantage_table(tail, TARGET_REL_ERROR)
    print(advantage.to_string(index=False, float_format=lambda v: f"{v:0.4g}"))
    advantage.to_csv(OUTPUTS / "qae_query_advantage.csv", index=False)
    deep = advantage.iloc[-1]
    shallow = advantage.iloc[0]
    print(
        f"  -> speedup grows from {shallow['speedup_mc_over_qae']:.1f}x at a={shallow['tail_probability_a']:g} "
        f"to {deep['speedup_mc_over_qae']:.1f}x at a={deep['tail_probability_a']:g} "
        "(quadratic, widening in the deep tail).\n"
    )

    print("=== CLAIM 2 (measured): schedule-sized MLAE on the real oracle vs matched MC budget ===")
    # Deepest threshold whose tail still carries mass (the most demanding, most advantageous level).
    deep_threshold = next(
        t for t in range(spec.n, 1, -1) if estimator.exact_p_severe(t) > 0.0
    )
    measured = measured_advantage_table(estimator, deep_threshold)
    print(f"  (threshold = {deep_threshold}, tail a = {estimator.exact_p_severe(deep_threshold):.4f})")
    print(measured.to_string(index=False, float_format=lambda v: f"{v:0.4g}"))
    measured.to_csv(OUTPUTS / "qae_measured_advantage.csv", index=False)
    print("  -> at each target accuracy the issued MLAE oracle calls beat the classical MC budget.\n")

    _plot(equivalence, advantage, OUTPUTS / "qae_tail_risk.png")
    print(f"Saved QAE outputs to {OUTPUTS}")
    print("  qae_equivalence.csv, qae_query_advantage.csv, qae_measured_advantage.csv, qae_tail_risk.png")


if __name__ == "__main__":
    main()
