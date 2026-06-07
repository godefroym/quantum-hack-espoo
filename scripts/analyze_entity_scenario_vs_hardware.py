"""Per-named-entity comparison: committed scenario dataset vs the hardware joint counts.

Joins three things on a single named-entity axis (the 38 real G-SIBs/corporates):

  1. ``data/scenario_dataset/real_gsib/`` -- the committed, pre-sampled scenarios
     from every generator (bernoulli, gaussian_copula, student_t_copula,
     entangled_born_machine) on the BASELINE real network spec, plus that spec's
     target marginals/correlation.
  2. ``outputs/real_cluster_mixture_stress_hw/joint_counts.csv`` -- the reconciled
     global joint distribution actually measured on ibm_boston (STRESSED spec).
  3. ``.../qubit_legend.csv`` and the run report -- entity names per qubit and the
     stressed loaded target marginal each qubit was asked to reproduce.

It is deliberately independent of ``analyze_real_cluster_mixture_stress_hw.py``:
that script does a 4-way cascade/fidelity comparison; this one produces a flat
per-entity table (marginals + correlation connectivity, named) and the structural
similarity between each dataset generator and the hardware joint.

Important: the committed dataset is on the BASELINE spec (~0.2% PDs) while the
hardware run is on the STRESSED spec (~15% target, decohered up toward 0.5). The
magnitudes are therefore NOT comparable -- what is comparable is the cross-entity
STRUCTURE (rank ordering of who is riskiest / most connected). The script reports
both, and labels the magnitude columns by their regime so nothing is conflated.

Run:
    uv run python scripts/analyze_entity_scenario_vs_hardware.py

Writes, into outputs/real_cluster_mixture_stress_hw/:
    entity_scenario_vs_hardware.csv     # the per-entity table
    entity_scenario_vs_hardware.json    # table + summary similarity metrics
    entity_scenario_vs_hardware.txt     # human-readable summary
    entity_scenario_vs_hardware.png     # marginals + connectivity figure
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
DATASET_DIR = REPO / "data" / "scenario_dataset" / "real_gsib"
HW_DIR = REPO / "outputs" / "real_cluster_mixture_stress_hw"
LEGEND_CSV = HW_DIR / "qubit_legend.csv"
JOINT_CSV = HW_DIR / "joint_counts.csv"
REPORT_JSON = HW_DIR / "real_stress_hardware_run_report.json"

GENERATORS = ["bernoulli", "gaussian_copula", "student_t_copula", "entangled_born_machine"]
GEN_SHORT = {
    "bernoulli": "ber",
    "gaussian_copula": "gcop",
    "student_t_copula": "tcop",
    "entangled_born_machine": "ebm",
}


# --------------------------------------------------------------------------- io

def load_legend() -> list[dict]:
    """qubit -> entity metadata, ordered by qubit index 0..37."""
    rows = []
    with LEGEND_CSV.open() as fh:
        for r in csv.DictReader(fh):
            r["qubit"] = int(r["qubit"])
            r["cluster"] = int(r["cluster"])
            rows.append(r)
    rows.sort(key=lambda r: r["qubit"])
    return rows


def safe_corr(samples: np.ndarray) -> np.ndarray:
    """corrcoef with constant (never-default) columns mapped to 0 instead of NaN.

    At baseline ~0.2% PDs many entities never default in 4000 samples, so their
    empirical column is constant -> correlation undefined. This is the rare-event
    wall; we report it as zero measurable correlation rather than propagate NaN.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        c = np.corrcoef(samples, rowvar=False)
    c = np.nan_to_num(c, nan=0.0)
    np.fill_diagonal(c, 1.0)
    return c


def load_dataset_generator(gen: str) -> dict:
    npz = np.load(DATASET_DIR / f"real_gsib__{gen}.npz", allow_pickle=True)
    samples = npz["samples"].astype(float)  # (n_samples, 38)
    # fraction of entities with an undefined empirical correlation (constant column)
    n_constant = int(np.sum(samples.std(axis=0) == 0))
    return {
        "samples": samples,
        "marginals": samples.mean(axis=0),
        "corr": safe_corr(samples),
        "n_constant_cols": n_constant,
        "target_marginals": npz["target_marginals"].astype(float),
        "target_corr": npz["target_corr"].astype(float),
        "node_names": [str(x) for x in npz["node_names"]],
    }


def load_hardware_joint(n_qubits: int) -> dict:
    """Weighted marginals + correlation from the reconciled global joint counts.

    Bit order is left->right = qubit 0..37 (per the file's header comment); we
    verify that against the n_defaults column on the first rows.
    """
    bits_rows: list[np.ndarray] = []
    weights: list[float] = []
    checked = 0
    with JOINT_CSV.open() as fh:
        for line in fh:
            if line.startswith("#") or line.startswith("bitstring"):
                continue
            bitstring, count, prob, n_def = line.rstrip("\n").split(",")
            if len(bitstring) != n_qubits:
                raise ValueError(f"bitstring width {len(bitstring)} != {n_qubits}")
            arr = np.frombuffer(bitstring.encode(), dtype=np.uint8) - ord("0")
            if checked < 50:
                assert int(arr.sum()) == int(n_def), "bit order / n_defaults mismatch"
                checked += 1
            bits_rows.append(arr.astype(float))
            weights.append(float(prob))

    X = np.vstack(bits_rows)              # (n_unique, 38)
    w = np.asarray(weights)
    w = w / w.sum()                       # renormalise (counts -> probabilities)

    ex = w @ X                            # E[x_i]
    exx = X.T @ (w[:, None] * X)          # E[x_i x_j]
    cov = exx - np.outer(ex, ex)
    var = ex - ex * ex                    # binary variance E[x]-E[x]^2
    denom = np.sqrt(np.outer(var, var))
    with np.errstate(divide="ignore", invalid="ignore"):
        corr = np.where(denom > 0, cov / denom, 0.0)
    np.fill_diagonal(corr, 1.0)
    return {"marginals": ex, "corr": corr, "n_unique": X.shape[0]}


def load_stressed_target(n_qubits: int) -> np.ndarray:
    """Loaded stressed target marginal per global qubit, from the run report."""
    report = json.loads(REPORT_JSON.read_text())
    tgt = np.full(n_qubits, np.nan)
    for c in report["hardware_per_cluster"]:
        for q, m in zip(c["members"], c["target_marginals"]):
            tgt[q] = float(m)
    return tgt


# ------------------------------------------------------------------ structure

def off_diagonal(m: np.ndarray) -> np.ndarray:
    iu = np.triu_indices_from(m, k=1)
    return m[iu]


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 2:
        return float("nan")
    a, b = a[mask], b[mask]
    if np.ptp(a) == 0 or np.ptp(b) == 0:
        return float("nan")
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def mean_abs_offdiag_corr(corr: np.ndarray) -> np.ndarray:
    """Per-entity connectivity: mean |corr| to all other entities."""
    a = np.abs(corr).copy()
    np.fill_diagonal(a, 0.0)
    return a.sum(axis=1) / (a.shape[1] - 1)


# ----------------------------------------------------------------------- main

def main() -> None:
    legend = load_legend()
    n = len(legend)
    names = [r["name"] for r in legend]

    data = {gen: load_dataset_generator(gen) for gen in GENERATORS}
    # the baseline target is identical across generators (same spec)
    baseline_target = data[GENERATORS[0]]["target_marginals"]
    baseline_target_corr = data[GENERATORS[0]]["target_corr"]

    # sanity: dataset node order must match the legend's qubit order
    if data[GENERATORS[0]]["node_names"] != names:
        raise ValueError("dataset node_names do not match qubit_legend order")

    hw = load_hardware_joint(n)
    stressed_target = load_stressed_target(n)

    # ---- per-entity connectivity (mean |corr|) for each source
    conn = {gen: mean_abs_offdiag_corr(data[gen]["corr"]) for gen in GENERATORS}
    conn["hw"] = mean_abs_offdiag_corr(hw["corr"])
    # the spec's intended correlation structure -- well-defined even where the
    # baseline empirical correlations collapse to noise (rare-event wall)
    conn["baseline_target"] = mean_abs_offdiag_corr(baseline_target_corr)

    # ---- per-entity table
    table = []
    for i, r in enumerate(legend):
        row = {
            "qubit": r["qubit"],
            "ticker": r["ticker"],
            "name": r["name"],
            "node_type": r["node_type"],
            "business_type": r["business_type"],
            "region": r["region"],
            "cluster": r["cluster"],
            "bloc": r["bloc"],
            # BASELINE regime (committed dataset)
            "baseline_target_pd": round(float(baseline_target[i]), 6),
        }
        for gen in GENERATORS:
            row[f"{GEN_SHORT[gen]}_pd"] = round(float(data[gen]["marginals"][i]), 6)
        # STRESSED / hardware regime
        row["hw_stressed_target_pd"] = round(float(stressed_target[i]), 6)
        row["hw_observed_pd"] = round(float(hw["marginals"][i]), 6)
        # connectivity (structure)
        row["ebm_mean_abs_corr"] = round(float(conn["entangled_born_machine"][i]), 5)
        row["hw_mean_abs_corr"] = round(float(conn["hw"][i]), 5)
        table.append(row)

    # ---- structural similarity: each generator's joint vs the hardware joint
    hw_off = off_diagonal(hw["corr"])
    similarity = {}
    for gen in GENERATORS:
        gen_off = off_diagonal(data[gen]["corr"])
        similarity[gen] = {
            # marginal rank agreement with hardware (magnitude is regime-mismatched)
            "marginal_spearman_vs_hw": round(
                spearman(data[gen]["marginals"], hw["marginals"]), 4),
            # correlation-structure agreement with hardware
            "corr_offdiag_spearman_vs_hw": round(spearman(gen_off, hw_off), 4),
            "corr_offdiag_rmse_vs_hw": round(
                float(np.sqrt(np.mean((gen_off - hw_off) ** 2))), 4),
            # connectivity-ranking agreement with hardware
            "connectivity_spearman_vs_hw": round(
                spearman(conn[gen], conn["hw"]), 4),
            # how many entities had no measurable empirical correlation (rare-event wall)
            "n_constant_cols": data[gen]["n_constant_cols"],
        }
    # the SPEC's intended structure (not its noisy empirical sample) vs hardware --
    # the meaningful structural reference, since baseline empirical corr is rare-event noise
    target_off = off_diagonal(baseline_target_corr)
    similarity["baseline_target_spec"] = {
        "corr_offdiag_spearman_vs_hw": round(spearman(target_off, hw_off), 4),
        "corr_offdiag_rmse_vs_hw": round(
            float(np.sqrt(np.mean((target_off - hw_off) ** 2))), 4),
        "connectivity_spearman_vs_hw": round(
            spearman(conn["baseline_target"], conn["hw"]), 4),
    }

    # marginal rank agreement: dataset generators vs their own baseline target,
    # and hardware-observed vs its own stressed target (within-regime fidelity)
    within_regime = {
        gen: round(spearman(data[gen]["marginals"], baseline_target), 4)
        for gen in GENERATORS
    }
    within_regime["hw_vs_stressed_target"] = round(
        spearman(hw["marginals"], stressed_target), 4)

    summary = {
        "n_entities": n,
        "n_samples_dataset": int(data[GENERATORS[0]]["samples"].shape[0]),
        "hw_n_unique_bitstrings": int(hw["n_unique"]),
        "regimes": {
            "baseline_mean_pd": round(float(baseline_target.mean()), 5),
            "stressed_target_mean_pd": round(float(np.nanmean(stressed_target)), 5),
            "hw_observed_mean_pd": round(float(hw["marginals"].mean()), 5),
        },
        "generator_vs_hardware_structure": similarity,
        "within_regime_marginal_rank_fidelity": within_regime,
    }

    out_json = {"summary": summary, "entities": table}
    (HW_DIR / "entity_scenario_vs_hardware.json").write_text(json.dumps(out_json, indent=2))

    # ---- CSV
    fields = list(table[0].keys())
    with (HW_DIR / "entity_scenario_vs_hardware.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(table)

    # ---- human-readable text
    write_text(summary, table)

    # ---- figure
    try:
        write_figure(legend, data, hw, baseline_target, stressed_target, conn)
    except Exception as exc:  # pragma: no cover - plotting is best-effort
        print(f"[warn] figure skipped: {exc}")

    print(f"Wrote entity_scenario_vs_hardware.{{csv,json,txt,png}} to {HW_DIR}")
    print(f"  dataset: {summary['n_samples_dataset']} samples x {n} entities, "
          f"{len(GENERATORS)} generators (BASELINE spec)")
    print(f"  hardware: {summary['hw_n_unique_bitstrings']} unique bitstrings "
          f"(STRESSED spec, ibm_boston)")
    ebm = similarity["entangled_born_machine"]
    print(f"  EBM vs HW structure: corr-Spearman {ebm['corr_offdiag_spearman_vs_hw']:+.2f}, "
          f"connectivity-Spearman {ebm['connectivity_spearman_vs_hw']:+.2f}")


def write_text(summary: dict, table: list[dict]) -> None:
    L = []
    L.append("PER-ENTITY: COMMITTED SCENARIO DATASET vs HARDWARE JOINT (ibm_boston)")
    L.append("=" * 70)
    reg = summary["regimes"]
    L.append(f"Entities: {summary['n_entities']} named real G-SIBs/corporates "
             f"(one per qubit).")
    L.append(f"Dataset : {summary['n_samples_dataset']} samples/generator on the "
             f"BASELINE spec (mean PD {reg['baseline_mean_pd']:.4f}).")
    L.append(f"Hardware: {summary['hw_n_unique_bitstrings']} unique bitstrings on the "
             f"STRESSED spec")
    L.append(f"          (stressed target mean PD {reg['stressed_target_mean_pd']:.3f}, "
             f"observed {reg['hw_observed_mean_pd']:.3f} -- decohered up toward 0.5).")
    L.append("")
    L.append("MAGNITUDES ARE REGIME-MISMATCHED (baseline vs stressed); compare STRUCTURE.")
    L.append("")
    L.append("GENERATOR vs HARDWARE -- cross-entity structure agreement")
    L.append("-" * 70)
    L.append(f"{'source':<22} {'marg-Spear':>11} {'corr-Spear':>11} "
             f"{'corr-RMSE':>10} {'conn-Spear':>11}")
    for gen, s in summary["generator_vs_hardware_structure"].items():
        marg = s.get("marginal_spearman_vs_hw")
        marg_s = f"{marg:>+11.3f}" if marg is not None else f"{'--':>11}"
        L.append(f"{gen:<22} {marg_s} "
                 f"{s['corr_offdiag_spearman_vs_hw']:>+11.3f} "
                 f"{s['corr_offdiag_rmse_vs_hw']:>10.3f} "
                 f"{s['connectivity_spearman_vs_hw']:>+11.3f}")
    n_const = summary["generator_vs_hardware_structure"]["entangled_born_machine"].get(
        "n_constant_cols")
    L.append("")
    L.append(f"Rare-event wall: at ~0.2% PDs a co-default is essentially never observed in "
             f"{summary['n_samples_dataset']} samples,")
    L.append(f"so empirical pairwise correlations are noise (and {n_const}/"
             f"{summary['n_entities']} entities never default at all -> undefined, set 0).")
    L.append("The 'baseline_target_spec' row uses the spec's INTENDED correlation instead "
             "-- the honest")
    L.append("structural reference at baseline PDs. This is exactly what motivates stressing "
             "the spec.")
    L.append("")
    L.append("WITHIN-REGIME marginal rank fidelity (each source vs its OWN target)")
    L.append("-" * 70)
    for k, v in summary["within_regime_marginal_rank_fidelity"].items():
        L.append(f"  {k:<32} Spearman {v:>+.3f}")
    L.append("")
    L.append("TOP-10 ENTITIES BY HARDWARE-OBSERVED CONNECTIVITY (mean |corr|)")
    L.append("-" * 70)
    L.append(f"{'name':<34}{'clu':>4}{'hw|corr|':>10}{'ebm|corr|':>11}"
             f"{'hw_pd':>8}{'ebm_pd':>9}")
    for r in sorted(table, key=lambda x: -x["hw_mean_abs_corr"])[:10]:
        L.append(f"{r['name'][:33]:<34}{r['cluster']:>4}"
                 f"{r['hw_mean_abs_corr']:>10.3f}{r['ebm_mean_abs_corr']:>11.3f}"
                 f"{r['hw_observed_pd']:>8.3f}{r['ebm_pd']:>9.4f}")
    L.append("")
    (HW_DIR / "entity_scenario_vs_hardware.txt").write_text("\n".join(L) + "\n")


def write_figure(legend, data, hw, baseline_target, stressed_target, conn) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(legend)
    names = [r["ticker"] for r in legend]
    clusters = np.array([r["cluster"] for r in legend])
    cluster_colors = np.array(["#1f77b4", "#ff7f0e", "#2ca02c"])
    x = np.arange(n)

    fig, axes = plt.subplots(3, 1, figsize=(16, 13))

    # (1) baseline-regime marginals: dataset generators + baseline target
    ax = axes[0]
    ax.plot(x, baseline_target, "k--", lw=1.5, label="baseline target", zorder=5)
    for gen, mark in zip(["bernoulli", "gaussian_copula", "student_t_copula",
                          "entangled_born_machine"], ["o", "s", "^", "D"]):
        ax.plot(x, data[gen]["marginals"], mark, ms=4, alpha=0.7, label=gen)
    ax.set_title("Baseline regime -- per-entity marginal PD (committed dataset, "
                 "4000 samples/generator)")
    ax.set_ylabel("default prob")
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=90, fontsize=6)
    ax.legend(fontsize=7, ncol=5); ax.grid(alpha=0.3)

    # (2) stressed/hardware-regime marginals
    ax = axes[1]
    ax.plot(x, stressed_target, "k--", lw=1.5, label="stressed loaded target")
    ax.bar(x, hw["marginals"], color=cluster_colors[clusters], alpha=0.8,
           label="hardware observed (ibm_boston)")
    ax.axhline(0.5, color="grey", ls=":", lw=1, label="0.5 (max decoherence)")
    ax.set_title("Stressed regime -- per-entity marginal PD (hardware joint vs loaded "
                 "target); bar colour = cluster")
    ax.set_ylabel("default prob")
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=90, fontsize=6)
    ax.legend(fontsize=7); ax.grid(alpha=0.3)

    # (3) structure: per-entity connectivity, EBM (baseline) vs hardware (stressed)
    ax = axes[2]
    ax.plot(x, conn["entangled_born_machine"], "D-", ms=4, color="#9467bd",
            label="EBM mean |corr| (baseline)")
    ax.plot(x, conn["hw"], "o-", ms=4, color="#d62728",
            label="hardware mean |corr| (stressed)")
    sp = spearman(conn["entangled_born_machine"], conn["hw"])
    ax.set_title(f"Structure -- per-entity connectivity (mean |corr| to others); "
                 f"EBM vs hardware rank Spearman {sp:+.2f}")
    ax.set_ylabel("mean |corr|")
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=90, fontsize=6)
    ax.legend(fontsize=7); ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(HW_DIR / "entity_scenario_vs_hardware.png", dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    main()
