from __future__ import annotations

import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
MPL_CACHE = ROOT / "outputs" / ".matplotlib"
XDG_CACHE = ROOT / "outputs" / ".cache"
MPL_CACHE.mkdir(parents=True, exist_ok=True)
XDG_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE))
os.environ.setdefault("XDG_CACHE_HOME", str(XDG_CACHE))
sys.path.insert(0, str(ROOT / "src"))

import matplotlib.pyplot as plt
import pandas as pd

from systemic_risk.data import make_synthetic_system
from systemic_risk.evaluation import EvaluationHarness
from systemic_risk.generators import (
    BernoulliGenerator,
    EntangledPQCGenerator,
    GaussianCopulaGenerator,
    StudentTCopulaGenerator,
)


def main() -> None:
    output_dir = ROOT / "outputs"
    output_dir.mkdir(exist_ok=True)

    rows = []
    for n in [8, 12, 16, 20]:
        spec = make_synthetic_system(n=n, seed=100 + n)
        generators = [
            BernoulliGenerator(),
            GaussianCopulaGenerator(),
            StudentTCopulaGenerator(df=4.0),
            EntangledPQCGenerator(layers=2),
        ]
        harness = EvaluationHarness(spec, n_samples=1_000, seed=900 + n)
        frame = harness.to_frame(harness.run(generators))
        frame["n"] = n
        rows.append(frame)
        print(f"\nN={n}")
        print(frame[["generator", "p_severe_cascade", "tail_mean_5pct"]].to_string(index=False))

    combined = pd.concat(rows, ignore_index=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    for generator, group in combined.groupby("generator"):
        ax.plot(group["n"], group["p_severe_cascade"], marker="o", label=generator)
    ax.set_xlabel("Number of institutions")
    ax.set_ylabel("P(severe cascade)")
    ax.set_title("Severe cascade frequency by generator")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "scaling_severe_frequency.png", dpi=180)
    combined.to_csv(output_dir / "scaling_experiment.csv", index=False)
    print(f"\nSaved scaling outputs to {output_dir}")


if __name__ == "__main__":
    main()
