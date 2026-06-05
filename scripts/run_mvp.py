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

from systemic_risk.data import make_synthetic_system
from systemic_risk.evaluation import EvaluationHarness
from systemic_risk.generators import (
    BernoulliGenerator,
    EntangledPQCGenerator,
    GaussianCopulaGenerator,
    StudentTCopulaGenerator,
)
from systemic_risk.visualization import plot_financial_network, save_crisis_card


def main() -> None:
    output_dir = ROOT / "outputs"
    output_dir.mkdir(exist_ok=True)

    spec = make_synthetic_system(n=16, seed=7)
    spec.save_json(output_dir / "synthetic_system.json")
    plot_financial_network(spec, output_dir / "network.png")

    generators = [
        BernoulliGenerator(),
        GaussianCopulaGenerator(),
        StudentTCopulaGenerator(df=4.0),
        EntangledPQCGenerator(layers=2),
    ]
    harness = EvaluationHarness(spec, n_samples=2_000, seed=2026)
    results = harness.run(generators)
    frame = harness.to_frame(results)
    frame.to_csv(output_dir / "comparison.csv", index=False)

    print("\nCascade comparison")
    print(frame.to_string(index=False, float_format=lambda value: f"{value:0.4f}"))

    for result in results:
        worst_idx = max(
            range(len(result.cascade_results)),
            key=lambda idx: result.cascade_results[idx].failure_count,
        )
        safe_name = result.generator_name.lower().replace(" ", "_").replace("-", "_")
        save_crisis_card(
            output_dir / f"crisis_card_{safe_name}.md",
            spec,
            result.samples[worst_idx],
            result.cascade_results[worst_idx],
            result.generator_name,
            worst_idx,
        )

    print(f"\nSaved outputs to {output_dir}")


if __name__ == "__main__":
    main()
