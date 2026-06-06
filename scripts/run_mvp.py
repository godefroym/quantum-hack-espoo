"""SMOKE TEST — fast MVP benchmark on the real-data foundation.

A lightweight, CI-time entry point that exercises the whole pipeline on the real 28-bank network:
build the spec, render the community network, run every generator (the classical baselines and the
entangled Born machine) through the shared cascade engine, and write a comparison CSV plus a
worst-case crisis card per generator. This is also the dashboard feed.

For the CANONICAL end-to-end run — the rigorous per-criterion head-to-head with the higher-order
and tail discriminators plus the n=54 scale story — use::

    uv run python scripts/run_demonstration.py
"""

from __future__ import annotations

from _demo._bootstrap import bootstrap

OUTPUTS = bootstrap()

import numpy as np  # noqa: E402

from _demo._specs import real_full_spec  # noqa: E402
from systemic_risk.evaluation import EvaluationHarness  # noqa: E402
from systemic_risk.generators import (  # noqa: E402
    BernoulliGenerator,
    EntangledBornMachineGenerator,
    GaussianCopulaGenerator,
    StudentTCopulaGenerator,
)
from systemic_risk.visualization import plot_community_network, save_crisis_card  # noqa: E402


def main() -> None:
    bundle = real_full_spec()
    spec = bundle.spec
    print(f"Spec: {bundle.label}  (n={spec.n})")
    if bundle.used_fallback:
        print("  NOTE: real offline build unavailable; using the synthetic fallback.")

    spec.save_json(OUTPUTS / "real_system.json")
    plot_community_network(
        spec, OUTPUTS / "network.png",
        title="Real 28-bank G-SIB exposure network — detected communities")

    generators = [
        BernoulliGenerator(),
        GaussianCopulaGenerator(),
        StudentTCopulaGenerator(df=4.0),
        EntangledBornMachineGenerator(ansatz="entangled", calibrate=True),
    ]
    harness = EvaluationHarness(
        spec,
        n_samples=20_000,
        seed=2026,
        include_joint_structure=False,
    )
    results = harness.run(generators)
    frame = harness.to_frame(results)
    frame.to_csv(OUTPUTS / "comparison.csv", index=False)

    print("\nCascade comparison (real network)")
    print(frame.to_string(index=False, float_format=lambda value: f"{value:0.4f}"))

    for result in results:
        worst_idx = int(np.argmax([c.failure_count for c in result.cascade_results]))
        safe = result.generator_name.lower().replace(" ", "_").replace("-", "_")
        save_crisis_card(
            OUTPUTS / f"crisis_card_{safe}.md",
            spec,
            result.samples[worst_idx],
            result.cascade_results[worst_idx],
            result.generator_name,
            worst_idx,
        )

    print(f"\nSaved outputs to {OUTPUTS}")
    print("For the full per-criterion verdict, run: uv run python scripts/run_demonstration.py")


if __name__ == "__main__":
    main()
