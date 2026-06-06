"""Interpretable diagrams for an IBM hardware Born-machine run.

Reads a ``*_report.json`` + ``*_samples.npz`` pair produced by
``run_ibm_quantum_large.py`` and renders:

* moments figure  -- per-bank default rates, hardware-vs-ideal calibration scatter,
                     and the systemic loss distribution P(#defaults = k);
* correlation figure -- ideal vs hardware co-default correlation heatmaps and their error.

When the run was beyond the exact-simulation limit, the ideal panels fall back to the
analytic moment targets and the loss-distribution panel shows hardware only.
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from run_ibm_quantum_large import EXACT_LIMIT, banded_spec

from systemic_risk.generators import EntangledBornMachineGenerator
from systemic_risk.generators.moments import empirical_moments, targets_from_spec
from systemic_risk.spec import joint_to_corr


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report",
        help="Path to a *_report.json. Defaults to the most recent large-run report.",
    )
    return parser.parse_args()


def latest_report() -> str:
    reports = sorted(glob.glob(str(ROOT / "outputs" / "ibm_quantum" / "hardware_*_report.json")))
    if not reports:
        raise SystemExit("no hardware reports found in outputs/ibm_quantum/")
    return reports[-1]


def corr_from(marginals: np.ndarray, joint: np.ndarray) -> np.ndarray:
    c = joint_to_corr(joint, marginals)
    np.fill_diagonal(c, 1.0)
    return c


def main() -> None:
    args = parse_args()
    report_path = Path(args.report or latest_report())
    report = json.loads(report_path.read_text())
    samples = np.load(report_path.with_name(report_path.name.replace("_report.json", "_samples.npz")))[
        "samples"
    ]
    n = report["n_qubits"]
    banks = np.arange(1, n + 1)

    observed = empirical_moments(samples)
    hw_corr = observed.pairwise_corr

    # Reconstruct the reference distribution exactly as the run did.
    spec = banded_spec(n)
    can_exact = report.get("exact_ground_truth", n <= EXACT_LIMIT)
    if can_exact:
        gen = EntangledBornMachineGenerator(
            ansatz="entangled", backend="statevector", max_degree=report["max_degree"],
            max_block_qubits=max(n, 22), calibrate=True,
        )
        gen.fit(spec)
        ref_marg, ref_joint = gen.exact_moments()
        ref_loss = gen.loss_count_pmf()
        ref_label = "ideal (exact statevector)"
    else:
        t = targets_from_spec(spec)
        ref_marg, ref_joint = t.marginals, t.pairwise_joint
        ref_loss = None
        ref_label = "analytic target"
    ref_corr = corr_from(ref_marg, ref_joint)

    figdir = ROOT / "outputs" / "ibm_quantum" / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    stem = report_path.name.replace("_report.json", "")
    title = f"{report['backend']}  -  {n} qubits, {report['two_qubit_gates']} 2Q gates, depth {report['circuit_depth']}"

    # ---- Figure 1: moments -------------------------------------------------
    fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(16, 4.6))
    fig.suptitle(f"Quantum default-scenario loader on hardware\n{title}", fontsize=12)

    w = 0.4
    ax0.bar(banks - w / 2, ref_marg, w, label=ref_label, color="#4C72B0")
    ax0.bar(
        banks + w / 2,
        observed.marginals,
        w,
        label=f"hardware ({report['shots']:,} shots)",
        color="#DD8452",
    )
    ax0.set_xlabel("bank"); ax0.set_ylabel("P(default)")
    ax0.set_title("Per-bank default probability"); ax0.legend(fontsize=8)

    lim = max(ref_marg.max(), observed.marginals.max()) * 1.1
    ax1.plot([0, lim], [0, lim], "k--", lw=1, label="perfect")
    ax1.scatter(ref_marg, observed.marginals, c=banks, cmap="viridis", s=45)
    ax1.set_xlim(0, lim); ax1.set_ylim(0, lim)
    ax1.set_xlabel(f"{ref_label}  P(default)"); ax1.set_ylabel("hardware  P(default)")
    ax1.set_title("Calibration: hardware vs reference"); ax1.legend(fontsize=8)

    hw_counts = samples.sum(axis=1)
    bins = np.arange(0, n + 2) - 0.5
    ax2.hist(hw_counts, bins=bins, density=True, color="#DD8452", alpha=0.8,
             label="hardware", align="mid")
    if ref_loss is not None:
        ax2.plot(np.arange(n + 1), ref_loss, "o-", color="#4C72B0", lw=1.5, ms=4, label="ideal")
    ax2.set_xlabel("number of simultaneous defaults  k")
    ax2.set_ylabel("P(k defaults)")
    ax2.set_title("Systemic loss distribution"); ax2.legend(fontsize=8)
    ax2.set_xlim(-0.5, max(hw_counts.max() + 2, 8))

    fig.tight_layout(rect=(0, 0, 1, 0.93))
    moments_path = figdir / f"{stem}_moments.png"
    fig.savefig(moments_path, dpi=140); plt.close(fig)

    # ---- Figure 2: correlations -------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    fig.suptitle(f"Co-default correlation structure\n{title}", fontsize=12)
    vmax = max(np.abs(ref_corr - np.eye(n)).max(), np.abs(hw_corr - np.eye(n)).max())
    for ax, mat, lab in (
        (axes[0], ref_corr, ref_label), (axes[1], hw_corr, "hardware"),
    ):
        m = mat.copy(); np.fill_diagonal(m, np.nan)
        im = ax.imshow(m, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set_title(f"{lab} correlation"); ax.set_xlabel("bank"); ax.set_ylabel("bank")
        fig.colorbar(im, ax=ax, fraction=0.046)
    diff = hw_corr - ref_corr; np.fill_diagonal(diff, np.nan)
    im = axes[2].imshow(diff, cmap="PuOr", vmin=-vmax, vmax=vmax)
    axes[2].set_title("hardware - reference (error)"); axes[2].set_xlabel("bank"); axes[2].set_ylabel("bank")
    fig.colorbar(im, ax=axes[2], fraction=0.046)

    fig.tight_layout(rect=(0, 0, 1, 0.92))
    corr_path = figdir / f"{stem}_correlations.png"
    fig.savefig(corr_path, dpi=140); plt.close(fig)

    print(json.dumps({"moments_figure": str(moments_path), "correlation_figure": str(corr_path),
                       "reference": ref_label}, indent=2))


if __name__ == "__main__":
    main()
