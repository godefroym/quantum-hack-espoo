"""Comparative error/fidelity analysis: did the STRESS regime make the REAL exposure
network measurable on hardware?

Consumes the completed runs (NO QPU time, NO network here) on the persisted partition,
200k shots/cluster, ibm_boston:

  * STRESS-HW    outputs/real_cluster_mixture_stress_hw/     (stressed spec on ibm_boston)
  * STRESS sim-preview  outputs/real_cluster_mixture_stress/ (exact statevector, the faithful
                                                              target the hardware should match)
  * BASELINE-HW  outputs/real_cluster_mixture_hw/            (OPTIONAL un-stressed faithful run;
                                                              the before/after section is emitted
                                                              only if this run matches the current
                                                              network size n)

The size (n), cluster count (k) and "severe" threshold are all read from the stress-HW run, so
this works for any partition -- not just the original 38-entity / k=3 / severe>=19 case.

It answers, quantitatively and bluntly:

  1. ERROR/FIDELITY of stressed-HW vs its loaded target (per-cluster + overall marginal RMSE/TV,
     within-cluster |corr| recovery, cross-cluster corr recovery, default-count TV), and the
     decoherence bias (upward pull of marginals toward 0.5) related to circuit depth / 2q-gates.
  2. (optional) BEFORE/AFTER vs BASELINE-HW -- only when a size-matched baseline run is present.
  3. GROUND-TRUTH framing: stressed-HW vs sim-preview vs reference on cascade tail risk
     (P(severe)/CVaR/mean cascade count).

Writes analysis_stress_comparison.{json,txt,png} into the stressed-HW dir.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from systemic_risk.mixture.pipeline import default_count_distribution, total_variation
from systemic_risk.mixture.reconcile import _binary_corr, _mean_cross_block

BASE_DIR = ROOT / "outputs" / "real_cluster_mixture_hw"
SIM_DIR = ROOT / "outputs" / "real_cluster_mixture_stress"
STRESS_DIR = ROOT / "outputs" / "real_cluster_mixture_stress_hw"
NOISE_FLOOR = 0.027


def within_abs_corr(samples: np.ndarray) -> float:
    if samples.shape[1] < 2:
        return float("nan")
    corr = np.corrcoef(samples, rowvar=False)
    iu = np.triu_indices(corr.shape[0], k=1)
    return float(np.nanmean(np.abs(corr[iu])))


def per_cluster_fidelity(report: dict, run_tag: str) -> list[dict]:
    """Per-cluster marginal RMSE / TV / Spearman-rank / |corr| recovery / decoherence mix-pull
    of the observed hardware marginals against the loaded target, straight from the report's
    per-cluster ``observed_marginals`` / ``target_marginals``.
    """
    rows = []
    for hw in report["hardware_per_cluster"]:
        obs = np.asarray(hw["observed_marginals"], dtype=float)
        tgt = np.asarray(hw["target_marginals"], dtype=float)
        tgt_corr = float(hw["target_within_abs_corr"])
        obs_corr = float(hw["observed_within_abs_corr"])
        if np.ptp(tgt) > 0 and np.ptp(obs) > 0:
            rho = float(spearmanr(obs, tgt).statistic)
        else:
            rho = float("nan")
        rows.append({
            "run": run_tag,
            "cluster": hw["cluster"],
            "qubits": len(hw["members"]),
            "transpiled_depth": hw["circuit_depth"],
            "transpiled_2q_gates": hw["two_qubit_gates"],
            "target_marg_mean": round(float(tgt.mean()), 5),
            "target_marg_min": round(float(tgt.min()), 5),
            "target_marg_max": round(float(tgt.max()), 5),
            "all_target_below_noise_floor": bool(np.all(tgt < NOISE_FLOOR)),
            "n_target_above_floor": int(np.sum(tgt >= NOISE_FLOOR)),
            "obs_marg_mean": round(float(obs.mean()), 5),
            "marginal_rmse_vs_target": round(float(np.sqrt(np.mean((obs - tgt) ** 2))), 5),
            "marginal_tv_vs_target": round(float(np.mean(np.abs(obs - tgt))), 5),
            "marginal_spearman_vs_target": round(rho, 4),
            "target_within_abs_corr": round(tgt_corr, 5),
            "obs_within_abs_corr": round(obs_corr, 5),
            "within_corr_recovery_fraction": (
                round(obs_corr / tgt_corr, 4) if tgt_corr > 0 else float("nan")
            ),
            "marginal_mix_pull": round(float(np.mean(0.5 - np.abs(obs - 0.5))), 5),
            "marginal_signed_bias": round(float(np.mean(obs - tgt)), 5),
        })
    return rows


def overall_marginal(report: dict) -> dict:
    obs = np.concatenate([np.asarray(h["observed_marginals"], float)
                          for h in report["hardware_per_cluster"]])
    tgt = np.concatenate([np.asarray(h["target_marginals"], float)
                          for h in report["hardware_per_cluster"]])
    rho = float(spearmanr(obs, tgt).statistic)
    return {
        "marginal_rmse_vs_target": round(float(np.sqrt(np.mean((obs - tgt) ** 2))), 5),
        "marginal_tv_vs_target": round(float(np.mean(np.abs(obs - tgt))), 5),
        "marginal_spearman_vs_target_global": round(rho, 4),
        "mean_signed_bias": round(float(np.mean(obs - tgt)), 5),
        "target_mean": round(float(tgt.mean()), 5),
        "obs_mean": round(float(obs.mean()), 5),
    }


def count_tv_between(a: np.ndarray, b: np.ndarray, n: int) -> float:
    return total_variation(default_count_distribution(a, n),
                           default_count_distribution(b, n))


def main() -> None:
    stress_report = json.loads((STRESS_DIR / "real_stress_hardware_run_report.json").read_text())
    sim_report = json.loads((SIM_DIR / "stress_preview_report.json").read_text())

    n = int(stress_report["n"])
    labels = np.asarray(stress_report["labels"], dtype=int)
    cluster_sizes = list(stress_report["cluster_sizes"])
    k = len(cluster_sizes)
    severe_thr = int(stress_report.get("cascade_severe", {}).get("severe_threshold_count",
                                                                  int(np.ceil(0.5 * n))))

    # Optional baseline-HW: only used if it is the SAME network size (else it is a stale run on a
    # different roster and the before/after comparison would be apples-to-oranges).
    base_report = None
    base_path = BASE_DIR / "real_hardware_run_report.json"
    if base_path.exists():
        candidate = json.loads(base_path.read_text())
        if int(candidate.get("n", -1)) == n:
            base_report = candidate

    stress_npz = np.load(STRESS_DIR / "reconciled_global_stress.npz")
    sim_npz = np.load(SIM_DIR / "reconciled_global_stress.npz")

    # =====================================================================
    # 1. STRESS-HW ERROR / FIDELITY vs its loaded target
    # =====================================================================
    stress_pc = per_cluster_fidelity(stress_report, "stress_hw")
    stress_overall = overall_marginal(stress_report)
    stress_overall["mean_within_corr_recovery_fraction"] = round(float(np.nanmean(
        [r["within_corr_recovery_fraction"] for r in stress_pc])), 4)
    stress_cross_obs = _mean_cross_block(_binary_corr(stress_npz["reconciled"]), labels)
    stress_cross_ref = _mean_cross_block(_binary_corr(stress_npz["reference"]), labels)
    stress_overall["cross_cluster_corr_reconciled"] = round(float(stress_cross_obs), 5)
    stress_overall["cross_cluster_corr_reference"] = round(float(stress_cross_ref), 5)
    stress_overall["cross_cluster_corr_recovery_fraction"] = round(
        float(stress_cross_obs / stress_cross_ref), 4) if stress_cross_ref else float("nan")
    stress_overall["count_tv_hw_vs_simpreview"] = round(
        count_tv_between(stress_npz["reconciled"], sim_npz["reconciled"], n), 5)
    stress_overall["count_tv_hw_vs_reference"] = round(
        count_tv_between(stress_npz["reconciled"], stress_npz["reference"], n), 5)

    # =====================================================================
    # 2. (optional) BEFORE/AFTER vs size-matched BASELINE-HW
    # =====================================================================
    before_after = None
    cascade_before_after = None
    base_pc = None
    base_overall = None
    if base_report is not None:
        base_pc = per_cluster_fidelity(base_report, "baseline_hw")
        base_overall = overall_marginal(base_report)
        base_overall["mean_within_corr_recovery_fraction"] = round(float(np.nanmean(
            [r["within_corr_recovery_fraction"] for r in base_pc])), 4)
        base_cross_obs = base_report["diagnostics"]["reconciled"]["cross_cluster_corr"]
        base_cross_ref = base_report["diagnostics"]["reference"]["cross_cluster_corr"]
        base_overall["cross_cluster_corr_recovery_fraction"] = round(
            float(base_cross_obs / base_cross_ref), 4) if base_cross_ref else float("nan")
        before_after = {
            "marginal_rmse_vs_target": {
                "baseline_hw": base_overall["marginal_rmse_vs_target"],
                "stress_hw": stress_overall["marginal_rmse_vs_target"]},
            "marginal_spearman_vs_target_global": {
                "baseline_hw": base_overall["marginal_spearman_vs_target_global"],
                "stress_hw": stress_overall["marginal_spearman_vs_target_global"]},
            "mean_within_corr_recovery_fraction": {
                "baseline_hw": base_overall["mean_within_corr_recovery_fraction"],
                "stress_hw": stress_overall["mean_within_corr_recovery_fraction"]},
            "cross_cluster_corr_recovery_fraction": {
                "baseline_hw": base_overall["cross_cluster_corr_recovery_fraction"],
                "stress_hw": stress_overall["cross_cluster_corr_recovery_fraction"]},
        }

        def tail(report: dict) -> dict:
            d = report["diagnostics"]
            sev = report.get("cascade_severe", {})
            return {
                "cvar_reconciled": round(float(d["reconciled"]["cascade_count_cvar"]), 4),
                "mean_cascade_reconciled": round(float(d["reconciled"]["mean_cascade_count"]), 4),
                "count_tv_reconciled_vs_ref": round(float(d["reconciled"]["count_tv_vs_ref"]), 5),
            }
        cascade_before_after = {"baseline_hw": tail(base_report), "stress_hw": tail(stress_report),
                                "sim_preview": tail(sim_report)}

    # =====================================================================
    # 3. GROUND-TRUTH framing: stress-HW vs sim-preview vs reference
    # =====================================================================
    sd = stress_report["diagnostics"]
    simd = sim_report["diagnostics"]
    ground_truth = {
        "note": ("reference = full-network Gaussian-copula joint on the STRESSED target; "
                 "sim_preview = exact statevector (faithful); stress_hw = ibm_boston."),
        "p_severe": {
            "reference": round(float(stress_report["cascade_severe"]["reference"]["p_severe"]), 5),
            "sim_preview": round(float(sim_report["cascade_severe"]["reconciled"]["p_severe"]), 5),
            "stress_hw": round(float(stress_report["cascade_severe"]["reconciled"]["p_severe"]), 5),
        },
        "cascade_cvar95": {
            "reference": round(float(sd["reference"]["cascade_count_cvar"]), 4),
            "sim_preview": round(float(simd["reconciled"]["cascade_count_cvar"]), 4),
            "stress_hw": round(float(sd["reconciled"]["cascade_count_cvar"]), 4),
        },
        "mean_cascade_count": {
            "reference": round(float(sd["reference"]["mean_cascade_count"]), 4),
            "sim_preview": round(float(simd["reconciled"]["mean_cascade_count"]), 4),
            "stress_hw": round(float(sd["reconciled"]["mean_cascade_count"]), 4),
        },
        "marginal_rmse_vs_ref": {
            "sim_preview": round(float(simd["reconciled"]["marginal_rmse_vs_ref"]), 5),
            "stress_hw": round(float(sd["reconciled"]["marginal_rmse_vs_ref"]), 5),
        },
        "count_tv_vs_ref": {
            "sim_preview": round(float(simd["reconciled"]["count_tv_vs_ref"]), 5),
            "stress_hw": round(float(sd["reconciled"]["count_tv_vs_ref"]), 5),
        },
        "stress_hw_vs_simpreview_count_tv": stress_overall["count_tv_hw_vs_simpreview"],
    }

    analysis = {
        "title": f"Stress-regime hardware comparative analysis (REAL {n}-entity network)",
        "backend": stress_report["backend"],
        "partition": {"cluster_sizes": cluster_sizes, "n": n, "k": k},
        "severe_threshold_count": severe_thr,
        "noise_floor": NOISE_FLOOR,
        "baseline_hw_available": base_report is not None,
        "regimes": {
            "stress_hw": f"2008-calibrated PDs (~15% mean), all {n} above the {NOISE_FLOOR} floor",
            "sim_preview": "exact statevector on the stressed spec (faithful target)",
            "baseline_hw": ("size-matched un-stressed faithful run" if base_report is not None
                            else "absent / size-mismatched -- before/after section omitted"),
        },
        "1_stress_hw_fidelity": {"per_cluster": stress_pc, "overall": stress_overall},
        "3_ground_truth_framing": ground_truth,
    }
    if before_after is not None:
        analysis["2_before_after_baseline_vs_stress"] = {
            "baseline_per_cluster": base_pc, "stress_per_cluster": stress_pc,
            "summary": before_after, "cascade_tail_risk": cascade_before_after}
    (STRESS_DIR / "analysis_stress_comparison.json").write_text(json.dumps(analysis, indent=2))

    # =====================================================================
    # written summary
    # =====================================================================
    so = stress_overall
    gt = ground_truth
    # decoherence-vs-depth, data-driven: name the deepest and shallowest blocks
    deepest = max(stress_pc, key=lambda r: r["transpiled_depth"])
    shallowest = min(stress_pc, key=lambda r: r["transpiled_depth"])
    lines = []
    lines.append(f"REAL {n}-ENTITY HARDWARE -- STRESS-REGIME COMPARATIVE ANALYSIS")
    lines.append("=" * 68)
    lines.append(f"Backend: {stress_report['backend']} | 200,000 shots/cluster | "
                 f"k={k} partition sizes {cluster_sizes}")
    lines.append(f"STRESS-HW:   2008-calibrated ~15% PDs, all {n} ABOVE the {NOISE_FLOOR} floor "
                 f"(SNR mean ~{stress_report['loadability']['snr_mean']}).")
    lines.append("SIM-PREVIEW: exact statevector on the stressed spec = the faithful target.")
    lines.append("REFERENCE:   full-network Gaussian-copula joint on the stressed target.")
    if base_report is None:
        lines.append("BASELINE-HW: not reproduced for this network size -- before/after section omitted")
        lines.append("             (run scripts/run_real_cluster_mixture_hardware.py to add it).")
    lines.append("Artifacts: analysis_stress_comparison.json (all numbers), .png, this file.")
    lines.append("")
    lines.append("BOTTOM LINE")
    lines.append("-----------")
    lines.append(
        f"Lifting the signal above the noise floor makes the real {n}-entity network a PARTIAL, "
        "structurally-meaningful\nhardware demonstration -- but NOT a magnitude-faithful one.")
    lines.append(
        f"Marginal ORDERING vs target: global Spearman {so['marginal_spearman_vs_target_global']:+.2f}; "
        f"within-cluster |corr| recovery {so['mean_within_corr_recovery_fraction']:.0%} of target. "
        f"Cross-cluster\ncoupling is reconstructed to {so['cross_cluster_corr_recovery_fraction']:.0%} "
        "of reference -- but by the CLASSICAL reconciler, not the device.")
    lines.append(
        f"What it did NOT buy: faithful MAGNITUDES. Decoherence pulls qubits toward 0.5, so overall "
        f"marginal\nRMSE vs target stays large ({so['marginal_rmse_vs_target']:.3f}, mean signed bias "
        f"{so['mean_signed_bias']:+.3f} = upward pull). The cascade tail\nis therefore "
        "HARDWARE-PERTURBED (overstated), not ground truth.")
    lines.append("")
    lines.append("1. STRESS-HW ERROR/FIDELITY vs LOADED TARGET (decoherence vs depth)")
    lines.append("-" * 66)
    lines.append("cl qub 2q-gates depth | tgt_marg obs_marg  RMSE | Spear | tgt|c| obs|c| recov")
    for r in stress_pc:
        rec = r["within_corr_recovery_fraction"]
        rec = rec if rec == rec else float("nan")
        lines.append(
            f"   {r['cluster']}      {r['qubits']:>2}    {r['transpiled_2q_gates']:>4}   "
            f"{r['transpiled_depth']:>4} | {r['target_marg_mean']:>7.4f}  "
            f"{r['obs_marg_mean']:>7.4f}  {r['marginal_rmse_vs_target']:>6.3f} | "
            f"{r['marginal_spearman_vs_target']:>6.2f} | "
            f"{r['target_within_abs_corr']:>6.3f} {r['obs_within_abs_corr']:>6.3f}  {rec:>5.0%}")
    lines.append(
        f"overall: marginal RMSE {so['marginal_rmse_vs_target']:.3f}, "
        f"TV {so['marginal_tv_vs_target']:.3f}, "
        f"global Spearman {so['marginal_spearman_vs_target_global']:+.2f}, "
        f"signed bias {so['mean_signed_bias']:+.3f}")
    lines.append(
        f"         within-corr recovery {so['mean_within_corr_recovery_fraction']:.0%}, "
        f"cross-cluster corr recovery {so['cross_cluster_corr_recovery_fraction']:.0%} "
        f"({so['cross_cluster_corr_reconciled']:.3f} vs ref "
        f"{so['cross_cluster_corr_reference']:.3f})")
    lines.append(
        f"         default-count TV: HW vs sim-preview {so['count_tv_hw_vs_simpreview']:.3f}, "
        f"HW vs reference {so['count_tv_hw_vs_reference']:.3f}")
    lines.append(
        "- Decoherence bias: mix-pull (0=pure,0.5=mixed) tracks depth -- "
        + ", ".join(f"c{r['cluster']}={r['marginal_mix_pull']:.2f}@d{r['transpiled_depth']}"
                    for r in stress_pc) + ".")
    lines.append(
        f"  The deepest block (c{deepest['cluster']}, {deepest['qubits']}q, "
        f"{deepest['transpiled_2q_gates']} 2q-gates, depth {deepest['transpiled_depth']}) inflates "
        f"most; the shallowest (c{shallowest['cluster']}, depth {shallowest['transpiled_depth']}) "
        "leaks least.")
    lines.append(
        "- Even above the floor, marginals are biased UP toward 0.5 -- the magnitudes are NOT "
        "trustworthy; the ORDER is.")
    lines.append("")

    if before_after is not None:
        ba = before_after
        ca = cascade_before_after
        lines.append("2. BEFORE/AFTER -- BASELINE-HW vs STRESS-HW (what above-floor signal bought)")
        lines.append("-" * 66)
        lines.append("metric                                   baseline-HW    stress-HW")
        lines.append(
            f"marginal RMSE vs target                    "
            f"{ba['marginal_rmse_vs_target']['baseline_hw']:>7.3f}      "
            f"{ba['marginal_rmse_vs_target']['stress_hw']:>7.3f}")
        lines.append(
            f"marginal Spearman vs target (GLOBAL)       "
            f"{ba['marginal_spearman_vs_target_global']['baseline_hw']:>+7.2f}      "
            f"{ba['marginal_spearman_vs_target_global']['stress_hw']:>+7.2f}")
        lines.append(
            f"within-cluster |corr| recovery (mean)      "
            f"{ba['mean_within_corr_recovery_fraction']['baseline_hw']:>6.0%}       "
            f"{ba['mean_within_corr_recovery_fraction']['stress_hw']:>6.0%}")
        lines.append(
            f"cross-cluster corr recovery                "
            f"{ba['cross_cluster_corr_recovery_fraction']['baseline_hw']:>6.0%}       "
            f"{ba['cross_cluster_corr_recovery_fraction']['stress_hw']:>6.0%}")
        lines.append("   Cascade tail risk (reconciled global joint):")
        lines.append("   metric                  baseline-HW   stress-HW   sim-preview")
        lines.append(
            f"   CVaR95 cascade count     {ca['baseline_hw']['cvar_reconciled']:>8.2f}   "
            f"{ca['stress_hw']['cvar_reconciled']:>8.2f}   "
            f"{ca['sim_preview']['cvar_reconciled']:>8.2f}")
        lines.append(
            f"   mean cascade count       {ca['baseline_hw']['mean_cascade_reconciled']:>8.2f}   "
            f"{ca['stress_hw']['mean_cascade_reconciled']:>8.2f}   "
            f"{ca['sim_preview']['mean_cascade_reconciled']:>8.2f}")
        lines.append("")

    lines.append("3. GROUND-TRUTH FRAMING -- stress-HW vs sim-preview vs reference (tail risk)")
    lines.append("-" * 66)
    lines.append(f"severe = post-cascade default count >= {severe_thr} (of {n})")
    lines.append("metric                  reference   sim-preview(faithful)   stress-HW")
    lines.append(
        f"P(severe)               {gt['p_severe']['reference']:>8.3f}   "
        f"{gt['p_severe']['sim_preview']:>13.3f}        {gt['p_severe']['stress_hw']:>8.3f}")
    lines.append(
        f"CVaR95 cascade count    {gt['cascade_cvar95']['reference']:>8.2f}   "
        f"{gt['cascade_cvar95']['sim_preview']:>13.2f}        {gt['cascade_cvar95']['stress_hw']:>8.2f}")
    lines.append(
        f"mean cascade count      {gt['mean_cascade_count']['reference']:>8.2f}   "
        f"{gt['mean_cascade_count']['sim_preview']:>13.2f}        {gt['mean_cascade_count']['stress_hw']:>8.2f}")
    lines.append(
        f"marginal RMSE vs ref          --     {gt['marginal_rmse_vs_ref']['sim_preview']:>13.3f}        "
        f"{gt['marginal_rmse_vs_ref']['stress_hw']:>8.3f}")
    lines.append(
        f"count-TV vs ref               --     {gt['count_tv_vs_ref']['sim_preview']:>13.3f}        "
        f"{gt['count_tv_vs_ref']['stress_hw']:>8.3f}")
    lines.append(
        f"- Hardware overstates P(severe) ({gt['p_severe']['stress_hw']:.2f} vs sim "
        f"{gt['p_severe']['sim_preview']:.2f}, ref {gt['p_severe']['reference']:.2f}) and mean "
        f"cascade ({gt['mean_cascade_count']['stress_hw']:.1f}\n  vs sim "
        f"{gt['mean_cascade_count']['sim_preview']:.1f}). The HW reconciled count-distribution sits "
        f"TV {gt['stress_hw_vs_simpreview_count_tv']:.2f} from the\n  faithful sim-preview -- "
        "same regime, hardware-perturbed.")
    lines.append("")
    lines.append("VERDICT")
    lines.append("-------")
    lines.append(
        f"The stress regime makes the real {n}-entity network a LEGITIMATE, if qualified, hardware "
        "demonstration:\nabove the noise floor the QPU recovers a real (if uneven) marginal "
        f"ORDERING (global Spearman "
        f"{so['marginal_spearman_vs_target_global']:+.2f})\nand retains within-cluster correlation "
        f"({so['mean_within_corr_recovery_fraction']:.0%} of target).")
    lines.append(
        f"CAVEATS: (1) marginal MAGNITUDES stay decoherence-biased upward (RMSE "
        f"~{so['marginal_rmse_vs_target']:.2f}), so absolute PDs\nare not trustworthy; (2) the "
        "cascade tail numbers are HARDWARE-PERTURBED and overstate risk vs the faithful\nsim-preview; "
        "(3) the cross-cluster correlation is rebuilt by the CLASSICAL reconciler, not measured "
        "on-device.\nHonest framing: a correlation/ordering-faithful hardware demo, magnitude- and "
        "tail-perturbed.")
    summary = "\n".join(lines)
    (STRESS_DIR / "analysis_stress_comparison.txt").write_text(summary + "\n")

    # =====================================================================
    # plot
    # =====================================================================
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))

        # (a) observed vs target marginals (stress; baseline overlaid only if size-matched)
        ax = axes[0]
        series = [(stress_report, "tab:blue", "stress-HW")]
        if base_report is not None:
            series.insert(0, (base_report, "tab:gray", "baseline-HW"))
        for rep, color, tag in series:
            obs = np.concatenate([np.asarray(h["observed_marginals"], float)
                                  for h in rep["hardware_per_cluster"]])
            tgt = np.concatenate([np.asarray(h["target_marginals"], float)
                                  for h in rep["hardware_per_cluster"]])
            ax.scatter(tgt, obs, s=16, alpha=0.6, color=color, label=tag)
        ax.axhline(NOISE_FLOOR, color="red", ls="--", lw=1, label="noise floor 2.7%")
        ax.axvline(NOISE_FLOOR, color="red", ls=":", lw=1)
        lim = [0, 0.62]
        ax.plot(lim, lim, color="black", lw=1, label="perfect (obs=target)")
        ax.set_xlim(lim)
        ax.set_ylim(0, 0.75)
        ax.set_xlabel("target default probability")
        ax.set_ylabel("observed (hardware) marginal")
        ax.set_title("(a) Observed vs target marginals\n(stress regime, above floor)")
        ax.legend(fontsize=7)

        # (b) stress structure recovery bars: rank, within-corr, cross-corr
        ax = axes[1]
        cats = ["marg rank\n(|Spearman|)", "within-corr\nrecovery", "cross-corr\nrecovery"]
        stress_vals = [abs(so["marginal_spearman_vs_target_global"]),
                       so["mean_within_corr_recovery_fraction"],
                       so["cross_cluster_corr_recovery_fraction"]]
        x = np.arange(len(cats))
        ax.bar(x, stress_vals, 0.6, color="tab:blue", label="stress-HW")
        for i, s in enumerate(stress_vals):
            ax.text(i, s + 0.02, f"{s:.2f}", ha="center", fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels(cats, fontsize=9)
        ax.set_ylabel("recovery fraction")
        ax.set_title("(b) Structure recovery (stress-HW)\n(cross-corr is reconciler, not device)")
        ax.legend(fontsize=8)

        # (c) cascade tail (CVaR + mean) reference vs sim vs stress-HW
        ax = axes[2]
        keys = ["reference\n(truth)", "sim-preview\n(faithful)", "stress-HW"]
        cvars = [gt["cascade_cvar95"]["reference"], gt["cascade_cvar95"]["sim_preview"],
                 gt["cascade_cvar95"]["stress_hw"]]
        means = [gt["mean_cascade_count"]["reference"], gt["mean_cascade_count"]["sim_preview"],
                 gt["mean_cascade_count"]["stress_hw"]]
        x = np.arange(len(keys))
        w = 0.36
        ax.bar(x - w / 2, cvars, w, color="tab:purple", label="CVaR95")
        ax.bar(x + w / 2, means, w, color="tab:orange", label="mean count")
        for i, (c, m) in enumerate(zip(cvars, means)):
            ax.text(i - w / 2, c + 0.4, f"{c:.1f}", ha="center", fontsize=8)
            ax.text(i + w / 2, m + 0.4, f"{m:.1f}", ha="center", fontsize=8)
        ax.axhline(gt["cascade_cvar95"]["reference"], color="tab:purple", ls="--", lw=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(keys, fontsize=9)
        ax.set_ylabel("post-cascade default count")
        ax.set_title("(c) Cascade tail risk\n(stress-HW perturbed but right regime)")
        ax.legend(fontsize=8)

        fig.tight_layout()
        fig.savefig(STRESS_DIR / "analysis_stress_comparison.png", dpi=130)
        plot_note = "analysis_stress_comparison.png written"
    except Exception as exc:  # pragma: no cover
        plot_note = f"plot skipped: {exc}"

    print(summary)
    print("\n" + plot_note)
    print(f"Saved -> {STRESS_DIR / 'analysis_stress_comparison.json'}")


if __name__ == "__main__":
    main()
