from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "outputs"
MPL_CACHE = OUTPUT_ROOT / ".matplotlib"
XDG_CACHE = OUTPUT_ROOT / ".cache"
MPL_CACHE.mkdir(parents=True, exist_ok=True)
XDG_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE))
os.environ.setdefault("XDG_CACHE_HOME", str(XDG_CACHE))
sys.path.insert(0, str(ROOT / "src"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from systemic_risk.data import (
    bank_asset_to_system_spec,
    make_huang_2008_style_system,
)
from systemic_risk.generators import EntangledPQCGenerator, GaussianCopulaGenerator
from systemic_risk.generators.moments import (
    empirical_moments,
    moment_errors,
    targets_from_spec,
)
from systemic_risk.simulator import simulate_huang_scenarios


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare Gaussian and entangled scenario generators under Huang fire-sale contagion."
    )
    parser.add_argument("--samples", type=int, default=5_000)
    parser.add_argument("--banks", type=int, default=16)
    parser.add_argument("--seed", type=int, default=2008)
    parser.add_argument("--alpha", type=float, default=0.08)
    parser.add_argument("--mean-pd", type=float, default=0.04)
    parser.add_argument("--severe-fraction", type=float, default=0.5)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_ROOT / "huang_generator_comparison",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.samples <= 0:
        raise ValueError("--samples must be positive")
    if not 0 < args.severe_fraction <= 1:
        raise ValueError("--severe-fraction must lie in (0, 1]")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    bank_asset_spec = make_huang_2008_style_system(
        n_banks=args.banks,
        seed=args.seed,
    )
    generator_spec = bank_asset_to_system_spec(
        bank_asset_spec,
        alpha=args.alpha,
        mean_default_probability=args.mean_pd,
    )
    targets = targets_from_spec(generator_spec)

    gaussian = GaussianCopulaGenerator()
    entangled = EntangledPQCGenerator(
        layers=2,
        coupling_scale=1.0,
        gibbs_sweeps=18,
        burn_in=50,
    )
    gaussian.fit(generator_spec)
    entangled.fit(generator_spec)
    training_history = entangled.train(
        n_steps=12,
        n_samples=min(max(1_000, args.samples // 2), 3_000),
        seed=args.seed + 10,
    )

    generators = [
        ("Gaussian copula (B)", gaussian),
        ("Entangled generator (C)", entangled),
    ]
    common_asset_shocks = {
        "construction_and_land_development": 0.90,
        "nonfarm_nonresidential": 0.95,
        "residential_1_to_4_family": 0.97,
    }
    severe_threshold = int(np.ceil(args.severe_fraction * args.banks))

    sample_sets: dict[str, np.ndarray] = {}
    failure_counts: dict[str, np.ndarray] = {}
    rows: list[dict[str, float | str]] = []

    for index, (label, generator) in enumerate(generators):
        samples = generator.sample(args.samples, seed=args.seed + 100 + index)
        results = simulate_huang_scenarios(
            samples,
            bank_asset_spec,
            asset_price_shocks=common_asset_shocks,
            alpha=args.alpha,
            eta=0.0,
            seed=args.seed + 200,
        )
        failures = np.array([result.failure_count for result in results])
        errors = moment_errors(samples, targets)
        sample_sets[label] = samples
        failure_counts[label] = failures
        rows.append(
            {
                "generator": label,
                "backend": getattr(generator, "backend_", "classical"),
                "n_samples": float(args.samples),
                "marginal_rmse": errors.marginal_rmse,
                "pairwise_joint_rmse": errors.pairwise_joint_rmse,
                "pairwise_corr_rmse": errors.pairwise_corr_rmse,
                "mean_initial_defaults": float(samples.sum(axis=1).mean()),
                "mean_final_failures": float(failures.mean()),
                "max_final_failures": float(failures.max()),
                "p_severe": float(np.mean(failures >= severe_threshold)),
                "tail_mean_5pct": _tail_mean(failures, 0.05),
                "scenario_diversity": float(
                    np.unique(samples, axis=0).shape[0] / len(samples)
                ),
            }
        )

    summary = pd.DataFrame(rows)
    summary.to_csv(args.output_dir / "comparison_summary.csv", index=False)
    np.savez_compressed(
        args.output_dir / "samples_and_failures.npz",
        gaussian_samples=sample_sets["Gaussian copula (B)"],
        gaussian_failures=failure_counts["Gaussian copula (B)"],
        entangled_samples=sample_sets["Entangled generator (C)"],
        entangled_failures=failure_counts["Entangled generator (C)"],
    )
    pd.DataFrame(training_history).to_csv(
        args.output_dir / "entangled_training_history.csv",
        index=False,
    )
    _plot_comparison(
        args.output_dir / "generator_huang_comparison.png",
        targets,
        sample_sets,
        failure_counts,
        args.banks,
        severe_threshold,
        entangled.backend_,
    )

    print(summary.to_string(index=False, float_format=lambda value: f"{value:.5f}"))
    print()
    print(f"C backend: {entangled.backend_}")
    if entangled.backend_ != "qiskit":
        print("Important: C is currently a classical fallback, not a quantum result.")
    print(f"Outputs: {args.output_dir}")


def _plot_comparison(
    path: Path,
    targets,
    sample_sets: dict[str, np.ndarray],
    failure_counts: dict[str, np.ndarray],
    n_banks: int,
    severe_threshold: int,
    c_backend: str,
) -> None:
    colors = {
        "Gaussian copula (B)": "#2f6f9f",
        "Entangled generator (C)": "#b24a3c",
    }
    observed = {
        label: empirical_moments(samples)
        for label, samples in sample_sets.items()
    }
    mask = targets.off_diagonal_mask

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    ax = axes[0, 0]
    for label, moments in observed.items():
        ax.scatter(
            targets.marginals,
            moments.marginals,
            s=34,
            alpha=0.75,
            label=label,
            color=colors[label],
        )
    marginal_limit = max(
        float(targets.marginals.max()),
        *(float(item.marginals.max()) for item in observed.values()),
    )
    ax.plot([0, marginal_limit], [0, marginal_limit], color="#333333", linestyle="--")
    ax.set_xlabel("Target default probability")
    ax.set_ylabel("Sampled default probability")
    ax.set_title("Marginal calibration")
    ax.legend()

    ax = axes[0, 1]
    target_joint = targets.pairwise_joint[mask]
    for label, moments in observed.items():
        ax.scatter(
            target_joint,
            moments.pairwise_joint[mask],
            s=18,
            alpha=0.4,
            label=label,
            color=colors[label],
        )
    joint_min = min(
        float(target_joint.min()),
        *(float(item.pairwise_joint[mask].min()) for item in observed.values()),
    )
    joint_max = max(
        float(target_joint.max()),
        *(float(item.pairwise_joint[mask].max()) for item in observed.values()),
    )
    ax.plot(
        [joint_min, joint_max],
        [joint_min, joint_max],
        color="#333333",
        linestyle="--",
    )
    ax.set_xlabel("Target co-default probability")
    ax.set_ylabel("Sampled co-default probability")
    ax.set_title("Pairwise co-default calibration")

    ax = axes[1, 0]
    bins = np.arange(-0.5, n_banks + 1.5, 1)
    for label, failures in failure_counts.items():
        ax.hist(
            failures,
            bins=bins,
            density=True,
            histtype="step",
            linewidth=2.2,
            label=label,
            color=colors[label],
        )
    ax.axvline(
        severe_threshold,
        color="#333333",
        linestyle="--",
        label=f"Severe threshold = {severe_threshold}",
    )
    ax.set_xlabel("Final failed banks")
    ax.set_ylabel("Probability mass")
    ax.set_title("Huang cascade-size distribution")
    ax.legend()

    ax = axes[1, 1]
    thresholds = np.arange(0, n_banks + 1)
    for label, failures in failure_counts.items():
        tail = np.array([np.mean(failures >= threshold) for threshold in thresholds])
        ax.step(
            thresholds,
            tail,
            where="post",
            linewidth=2.2,
            label=label,
            color=colors[label],
        )
    ax.axvline(severe_threshold, color="#333333", linestyle="--")
    ax.set_yscale("log")
    ax.set_ylim(1e-4, 1.05)
    ax.set_xlabel("Failure threshold k")
    ax.set_ylabel("P(final failures >= k)")
    ax.set_title("Cascade tail")
    ax.legend()

    fig.suptitle(
        "Shared moments, shared Huang engine, different scenario generators\n"
        f"C backend: {c_backend}",
        fontsize=15,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _tail_mean(values: np.ndarray, fraction: float) -> float:
    k = max(1, int(np.ceil(fraction * len(values))))
    return float(np.sort(values)[-k:].mean())


if __name__ == "__main__":
    main()
