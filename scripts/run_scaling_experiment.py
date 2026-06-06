"""Scaling experiment — how the entangled generator behaves as the system grows.

Two complementary scaling stories, written to ``outputs/``:

1. **Sampling/fidelity scaling** on calibrated-synthetic specs from a handful of qubits up to the
   n=54 hardware target: for each size, run every generator through the shared cascade engine and
   record the severe-cascade frequency and tail-mean, plus the entangled generator's block
   structure (it stays block-separable, so per-block cost is bounded as n grows).
2. **Exact-at-scale validation**: the homogeneous mean-field oracle check at increasing n,
   including 54, showing the entangled construction reproduces the exact loss-count law to machine
   precision with no ``2^n`` cost — the evidence that generation scales to real hardware.

Run:
    uv run python scripts/run_scaling_experiment.py
"""

from __future__ import annotations

from _demo._bootstrap import bootstrap

OUTPUTS = bootstrap()

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from _demo._scale import validate_against_oracle  # noqa: E402
from _demo._specs import homogeneous_oracle_spec, synthetic_scale_spec  # noqa: E402
from systemic_risk.evaluation import EvaluationHarness  # noqa: E402
from systemic_risk.generators import (  # noqa: E402
    BernoulliGenerator,
    EntangledBornMachineGenerator,
    GaussianCopulaGenerator,
    StudentTCopulaGenerator,
)


SIZES = (8, 16, 24, 32, 54)


def _generators() -> list:
    return [
        BernoulliGenerator(),
        GaussianCopulaGenerator(),
        StudentTCopulaGenerator(df=4.0),
        EntangledBornMachineGenerator(ansatz="entangled", calibrate=True),
    ]


def sampling_scaling() -> pd.DataFrame:
    """Run the harness on calibrated-synthetic specs across sizes; return the combined frame."""
    rows = []
    for n in SIZES:
        spec = synthetic_scale_spec(n=n).spec
        harness = EvaluationHarness(
            spec,
            n_samples=4_000,
            seed=900 + n,
            include_joint_structure=False,
        )
        frame = harness.to_frame(harness.run(_generators()))
        frame["n"] = n
        rows.append(frame)
        entangled = EntangledBornMachineGenerator(ansatz="entangled", calibrate=True)
        entangled.fit(spec)
        diag = entangled.diagnostics_summary()
        print(f"n={n:>2}: entangled fit -> {diag.n_blocks} blocks, max block {diag.max_block_size} "
              f"qubits  (never forms 2^{n})")
    return pd.concat(rows, ignore_index=True)


def oracle_scaling() -> pd.DataFrame:
    """Validate the entangled loss-count law against the mean-field oracle across sizes."""
    rows = []
    for n in SIZES:
        result = validate_against_oracle(homogeneous_oracle_spec(n=n))
        rows.append(
            {
                "n": n,
                "tv_distance": result.tv_distance,
                "marginal_err": abs(result.generator_marginal - result.oracle_marginal),
                "default_corr_err": abs(
                    result.generator_default_corr - result.oracle_default_corr
                ),
            }
        )
    frame = pd.DataFrame(rows)
    print("\nHomogeneous mean-field oracle validation (exact ground truth, no 2^n state):")
    print(frame.to_string(index=False, float_format=lambda v: f"{v:0.2e}"))
    return frame


def main() -> None:
    print("=== Sampling / cascade-tail scaling on calibrated-synthetic specs ===")
    combined = sampling_scaling()

    fig, ax = plt.subplots(figsize=(9, 5))
    for generator, group in combined.groupby("generator"):
        ax.plot(group["n"], group["p_severe_cascade"], marker="o", label=generator)
    ax.set_xlabel("Number of institutions (qubits)")
    ax.set_ylabel("P(severe cascade)")
    ax.set_title("Severe-cascade frequency by generator vs system size")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUTS / "scaling_severe_frequency.png", dpi=180)
    combined.to_csv(OUTPUTS / "scaling_experiment.csv", index=False)

    oracle = oracle_scaling()
    oracle.to_csv(OUTPUTS / "scaling_oracle_validation.csv", index=False)

    print(f"\nSaved scaling outputs to {OUTPUTS}")


if __name__ == "__main__":
    main()
