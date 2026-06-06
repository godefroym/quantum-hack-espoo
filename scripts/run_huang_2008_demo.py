"""SECONDARY entry point — single Huang fire-sale cascade on a 2008-style bank-asset network.

A minimal illustration of the optional overlapping-portfolio / price-impact contagion channel
(see ``docs/huang_simulation.md``): shock three real-estate asset classes and trace the resulting
bank-failure rounds and asset write-downs. For the generator comparison under this channel use
``scripts/compare_generators_huang.py``; for the canonical end-to-end run use
``scripts/run_demonstration.py``.

Run:
    uv run python scripts/run_huang_2008_demo.py
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from systemic_risk.data import make_huang_2008_style_system
from systemic_risk.simulator import run_huang_cascade


def main() -> None:
    spec = make_huang_2008_style_system(n_banks=16, seed=2007)
    shocks = {
        "construction_and_land_development": 0.65,
        "nonfarm_nonresidential": 0.82,
        "residential_1_to_4_family": 0.90,
    }
    result = run_huang_cascade(
        spec,
        asset_price_shocks=shocks,
        alpha=0.08,
        eta=0.10,
        seed=2008,
    )

    print("Huang 2008-style bank-asset cascade")
    print(f"Banks failed: {result.failure_count}/{spec.n_banks}")
    print(f"Failure rounds: {result.rounds_to_convergence}")
    print()

    for round_idx, new_failures in enumerate(result.new_failures_by_round, start=1):
        names = [
            name
            for name, failed in zip(spec.bank_names, new_failures)
            if failed == 1
        ]
        print(f"Round {round_idx}: {', '.join(names)}")

    price_changes = 1 - result.final_asset_price_factors
    largest_moves = np.argsort(price_changes)[::-1][:5]
    print("\nLargest final asset value reductions:")
    for asset_idx in largest_moves:
        print(
            f"- {spec.asset_names[asset_idx]}: "
            f"{100 * price_changes[asset_idx]:.1f}%"
        )


if __name__ == "__main__":
    main()
