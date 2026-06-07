"""Error / fidelity + classical-comparison analysis of the REAL 38-entity hardware run.

Consumes the artifacts in ``outputs/real_cluster_mixture_hw/`` produced by
``run_real_cluster_mixture_hardware.py`` and answers two questions, reusing repo machinery:

1. ERROR / FIDELITY: how far did the measured per-cluster marginals + within-cluster
   correlation drift from target, and how badly did the structure collapse toward the
   maximally-mixed / independent state (decoherence). Per cluster + overall, related to
   circuit depth / 2q-gate count / qubit count and to the sub-noise-floor marginals.

2. CLASSICAL COMPARISON: run the SAME pipeline (per-cluster sampler -> common-shock
   reconciler -> real-network cascade) with the EXACT statevector classical sampler on the
   SAME partition against the SAME Gaussian-copula reference, plus the naive independent
   baseline. Lands hardware vs classical vs ground truth on marginals, correlations, and the
   downstream cascade tail risk (P(severe)/CVaR/mean cascade count).

Writes ``analysis_report.json`` + ``analysis_summary.txt`` + ``analysis_plot.png`` into the
hardware output dir. Pure post-hoc analysis -- no QPU time, no network.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from systemic_risk.data_network import build_network_spec
from systemic_risk.generators.gaussian_copula import GaussianCopulaGenerator
from systemic_risk.mixture import (
    CommonShockReconciler,
    cluster_samples_from_bitstrings,
    cross_cluster_corr_target,
    independent_global_samples,
    reconciliation_diagnostics,
    sample_cluster_statevector,
)
from systemic_risk.mixture.reconcile import _binary_corr, _mean_cross_block

HW_DIR = ROOT / "outputs" / "real_cluster_mixture_hw"
NOISE_FLOOR = 0.027
SEED = 0


def within_abs_corr(samples: np.ndarray) -> float:
    if samples.shape[1] < 2:
        return float("nan")
    corr = np.corrcoef(samples, rowvar=False)
    iu = np.triu_indices(corr.shape[0], k=1)
    return float(np.nanmean(np.abs(corr[iu])))


def total_variation_marginals(obs: np.ndarray, tgt: np.ndarray) -> float:
    """Per-bit Bernoulli total-variation averaged over the cluster (= mean |obs - tgt|)."""
    return float(np.mean(np.abs(obs - tgt)))


def main() -> None:
    report = json.loads((HW_DIR / "real_hardware_run_report.json").read_text())
    labels = np.asarray(report["labels"], dtype=int)

    spec = build_network_spec().to_system_spec()
    p_real = np.asarray(spec.marginal_default_probs, dtype=float)
    clusters_members = [hw["members"] for hw in report["hardware_per_cluster"]]

    # ---- load hardware raw samples
    hw_cluster_samples = []
    for hw in report["hardware_per_cluster"]:
        d = np.load(HW_DIR / hw["samples_file"], allow_pickle=True)
        hw_cluster_samples.append((list(hw["members"]), np.asarray(d["samples"], dtype=int)))
    n_samples = min(s.shape[0] for _, s in hw_cluster_samples)

    # ---- generate the classical (exact statevector) per-cluster samples, SAME partition
    seed_seq = np.random.SeedSequence(SEED + 1)
    child_seeds = seed_seq.spawn(len(clusters_members))
    sv_cluster_samples = []
    for members, child in zip(clusters_members, child_seeds):
        cs = sample_cluster_statevector(
            spec, members, n_samples, rng=np.random.default_rng(child)
        )
        sv_cluster_samples.append((list(cs.members), cs.samples))

    # =====================================================================
    # 1. PER-CLUSTER ERROR / FIDELITY
    # =====================================================================
    per_cluster = []
    for idx, (members, hw_s) in enumerate(hw_cluster_samples):
        meta = report["per_cluster"][idx]
        hwrep = report["hardware_per_cluster"][idx]
        tgt_marg = p_real[members]
        hw_marg = hw_s.mean(axis=0)
        sv_members, sv_s = sv_cluster_samples[idx]
        sv_marg = sv_s.mean(axis=0)

        tgt_corr = meta["target_within_abs_corr"]
        hw_corr = within_abs_corr(hw_s)
        sv_corr = within_abs_corr(sv_s)

        # decoherence signature: distance of marginals from 0.5 (max-mixed) and from target.
        # corr-recovery fraction: how much of the target |corr| survived.
        corr_recovery = hw_corr / tgt_corr if tgt_corr > 0 else float("nan")
        # how strongly the device pulled marginals toward the 0.5 max-mixed state:
        mix_pull = float(np.mean(0.5 - np.abs(hw_marg - 0.5)))  # 0 = pure (0/1), 0.5 = fully mixed
        per_cluster.append({
            "cluster": idx,
            "qubits": meta["qubits"],
            "entanglers": meta["entanglers"],
            "transpiled_depth": hwrep["circuit_depth"],
            "transpiled_2q_gates": hwrep["two_qubit_gates"],
            "target_marg_mean": round(float(tgt_marg.mean()), 6),
            "target_marg_max": round(float(tgt_marg.max()), 6),
            "all_target_below_noise_floor": bool(np.all(tgt_marg < NOISE_FLOOR)),
            "hw_marg_mean": round(float(hw_marg.mean()), 5),
            "hw_marg_min": round(float(hw_marg.min()), 5),
            "hw_marg_max": round(float(hw_marg.max()), 5),
            "hw_marginal_rmse_vs_target": round(
                float(np.sqrt(np.mean((hw_marg - tgt_marg) ** 2))), 5),
            "hw_marginal_tv_vs_target": round(total_variation_marginals(hw_marg, tgt_marg), 5),
            "sv_marginal_rmse_vs_target": round(
                float(np.sqrt(np.mean((sv_marg - tgt_marg) ** 2))), 5),
            "target_within_abs_corr": round(float(tgt_corr), 5),
            "hw_within_abs_corr": round(float(hw_corr), 5),
            "sv_within_abs_corr": round(float(sv_corr), 5),
            "hw_corr_recovery_fraction": round(float(corr_recovery), 4),
            "hw_marginal_mix_pull": round(mix_pull, 5),
        })

    hw_sq_err = np.concatenate([(s.mean(axis=0) - p_real[m]) ** 2 for m, s in hw_cluster_samples])
    overall = {
        "hw_marginal_rmse_vs_target_overall": round(float(np.sqrt(np.mean(hw_sq_err))), 5),
        "hw_mean_within_abs_corr": round(float(np.mean(
            [c["hw_within_abs_corr"] for c in per_cluster])), 5),
        "target_mean_within_abs_corr": round(float(np.mean(
            [c["target_within_abs_corr"] for c in per_cluster])), 5),
        "hw_mean_corr_recovery_fraction": round(float(np.nanmean(
            [c["hw_corr_recovery_fraction"] for c in per_cluster])), 4),
        "sv_marginal_rmse_vs_target_overall": round(float(np.sqrt(np.mean(np.concatenate([
            (s.mean(axis=0) - p_real[m]) ** 2 for m, s in sv_cluster_samples
        ])))), 5),
    }

    # =====================================================================
    # 2. CLASSICAL COMPARISON via the full reconcile + cascade pipeline
    # =====================================================================
    ref_gen = GaussianCopulaGenerator()
    ref_gen.fit(spec)
    reference = ref_gen.sample(n_samples, seed=SEED + 3)
    target_cross = cross_cluster_corr_target(spec, labels)

    def pipeline(cluster_pairs, tag):
        cs = [cluster_samples_from_bitstrings(m, s, source=tag) for m, s in cluster_pairs]
        rec = CommonShockReconciler(spec.n, labels).fit_reconcile(
            cs, target_cross, n_samples, seed=SEED + 2)
        ind = independent_global_samples(cs, spec.n)
        diag = reconciliation_diagnostics(
            reference, rec.samples, ind, labels, spec.n,
            cascade_spec=spec, cascade_max_eval=4000)
        return rec, diag

    hw_rec, hw_diag = pipeline(hw_cluster_samples, "hardware")
    sv_rec, sv_diag = pipeline(sv_cluster_samples, "statevector")

    # P(severe): fraction of scenarios whose post-cascade count exceeds reference 95th pct.
    from systemic_risk.mixture.pipeline import cascade_loss_cvar
    ref_cvar, ref_counts = cascade_loss_cvar(reference, spec, max_eval=4000)
    severe_threshold = float(np.quantile(ref_counts, 0.95))

    def p_severe(samples):
        _, counts = cascade_loss_cvar(samples, spec, max_eval=4000)
        return float(np.mean(counts > severe_threshold))

    p_sev = {
        "reference": float(np.mean(ref_counts > severe_threshold)),
        "hw_reconciled": p_severe(hw_rec.samples),
        "sv_reconciled": p_severe(sv_rec.samples),
        "independent": p_severe(independent_global_samples(
            [cluster_samples_from_bitstrings(m, s) for m, s in hw_cluster_samples], spec.n)),
    }

    comparison = {
        "reference": "full-network Gaussian-copula joint (ground truth)",
        "severe_count_threshold_p95": round(severe_threshold, 3),
        "p_severe": {k: round(v, 5) for k, v in p_sev.items()},
        "marginal_rmse_vs_ref": {
            "hw_reconciled": round(hw_diag["reconciled"]["marginal_rmse_vs_ref"], 5),
            "sv_reconciled": round(sv_diag["reconciled"]["marginal_rmse_vs_ref"], 5),
            "independent_hw": round(hw_diag["independent"]["marginal_rmse_vs_ref"], 5),
        },
        "cross_cluster_corr": {
            "reference": round(hw_diag["reference"]["cross_cluster_corr"], 5),
            "hw_reconciled": round(hw_diag["reconciled"]["cross_cluster_corr"], 5),
            "sv_reconciled": round(sv_diag["reconciled"]["cross_cluster_corr"], 5),
            "independent_hw": round(hw_diag["independent"]["cross_cluster_corr"], 5),
        },
        "count_tv_vs_ref": {
            "hw_reconciled": round(hw_diag["reconciled"]["count_tv_vs_ref"], 5),
            "sv_reconciled": round(sv_diag["reconciled"]["count_tv_vs_ref"], 5),
            "independent_hw": round(hw_diag["independent"]["count_tv_vs_ref"], 5),
        },
        "cascade_count_cvar": {
            "reference": round(hw_diag["reference"]["cascade_count_cvar"], 4),
            "hw_reconciled": round(hw_diag["reconciled"]["cascade_count_cvar"], 4),
            "sv_reconciled": round(sv_diag["reconciled"]["cascade_count_cvar"], 4),
            "independent_hw": round(hw_diag["independent"]["cascade_count_cvar"], 4),
        },
        "mean_cascade_count": {
            "reference": round(hw_diag["reference"]["mean_cascade_count"], 5),
            "hw_reconciled": round(hw_diag["reconciled"]["mean_cascade_count"], 5),
            "sv_reconciled": round(sv_diag["reconciled"]["mean_cascade_count"], 5),
            "independent_hw": round(hw_diag["independent"]["mean_cascade_count"], 5),
        },
    }

    analysis = {
        "title": "Error/fidelity + classical comparison for the REAL 38-entity hardware run",
        "backend": report["backend"],
        "n_samples_analyzed": n_samples,
        "noise_floor": NOISE_FLOOR,
        "per_cluster_fidelity": per_cluster,
        "overall_fidelity": overall,
        "classical_vs_quantum_vs_reference": comparison,
        "fitted_beta": {"hw": round(hw_rec.beta, 5), "sv": round(sv_rec.beta, 5)},
    }
    (HW_DIR / "analysis_report.json").write_text(json.dumps(analysis, indent=2))

    # ---- plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

        # (a) per-cluster observed vs target marginals + noise floor
        ax = axes[0]
        for idx, (members, hw_s) in enumerate(hw_cluster_samples):
            x = np.arange(len(members)) + idx * 0.02
            ax.scatter(x, hw_s.mean(axis=0), s=18, label=f"HW c{idx} obs")
        allt = np.concatenate([p_real[m] for m, _ in hw_cluster_samples])
        ax.axhline(NOISE_FLOOR, color="red", ls="--", lw=1, label="noise floor 2.7%")
        ax.scatter(range(len(allt)), allt, s=8, color="black", marker="x", label="target")
        ax.set_title("(a) Observed vs target marginals\n(targets all under noise floor)")
        ax.set_xlabel("qubit index (per cluster)")
        ax.set_ylabel("default probability")
        ax.legend(fontsize=7, ncol=2)

        # (b) within-cluster |corr| recovery
        ax = axes[1]
        cl = [c["cluster"] for c in per_cluster]
        tw = [c["target_within_abs_corr"] for c in per_cluster]
        hw = [c["hw_within_abs_corr"] for c in per_cluster]
        sv = [c["sv_within_abs_corr"] for c in per_cluster]
        w = 0.25
        ax.bar([c - w for c in cl], tw, w, label="target")
        ax.bar(cl, hw, w, label="hardware")
        ax.bar([c + w for c in cl], sv, w, label="statevector (classical)")
        for i, c in enumerate(per_cluster):
            ax.text(cl[i], hw[i] + 0.002, f"{c['hw_corr_recovery_fraction']:.0%}",
                    ha="center", fontsize=8)
        ax.set_title("(b) Within-cluster mean |corr|\n(% = HW recovery of target)")
        ax.set_xlabel("cluster")
        ax.set_ylabel("mean |pairwise corr|")
        ax.set_xticks(cl)
        ax.legend(fontsize=8)

        # (c) cascade tail risk: CVaR
        ax = axes[2]
        keys = ["reference", "sv_reconciled", "hw_reconciled", "independent_hw"]
        labs = ["reference\n(truth)", "classical\n(statevector)", "quantum\n(hardware)",
                "independent\n(baseline)"]
        vals = [comparison["cascade_count_cvar"][k] for k in keys]
        colors = ["black", "tab:green", "tab:blue", "tab:gray"]
        ax.bar(labs, vals, color=colors)
        ax.axhline(comparison["cascade_count_cvar"]["reference"], color="black", ls="--", lw=1)
        for i, v in enumerate(vals):
            ax.text(i, v + 0.3, f"{v:.1f}", ha="center", fontsize=9)
        ax.set_title("(c) Cascade tail risk (CVaR95 of post-contagion count)")
        ax.set_ylabel("CVaR95 cascade count")

        fig.tight_layout()
        fig.savefig(HW_DIR / "analysis_plot.png", dpi=130)
        plot_note = "analysis_plot.png written"
    except Exception as exc:  # pragma: no cover
        plot_note = f"plot skipped: {exc}"

    print(json.dumps(analysis, indent=2))
    print("\n" + plot_note)
    print(f"Saved -> {HW_DIR / 'analysis_report.json'}")


if __name__ == "__main__":
    main()
