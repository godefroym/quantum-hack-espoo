from __future__ import annotations

from pathlib import Path

import numpy as np

from systemic_risk.simulator.cascade import CascadeResult
from systemic_risk.spec import SystemSpec


def make_crisis_card(
    spec: SystemSpec,
    initial_defaults: np.ndarray,
    result: CascadeResult,
    generator_name: str,
    scenario_id: int,
) -> str:
    initial_names = _names_for_mask(spec, initial_defaults)
    lines = [
        f"# Scenario #{scenario_id} - {generator_name}",
        "",
        "## Initial shocks",
        _bullet_list(initial_names),
        "",
        "## Cascade",
    ]
    previous = np.zeros(spec.n, dtype=int)
    for round_idx, state in enumerate(result.states_by_round):
        new_failures = state.astype(int) - previous
        names = _names_for_mask(spec, new_failures)
        if round_idx == 0:
            lines.append(f"- Round 0: {len(names)} initial defaults")
        else:
            lines.append(f"- Round {round_idx}: {', '.join(names) if names else 'no new failures'}")
        previous = state.astype(int)
    lines.extend(
        [
            "",
            "## Final impact",
            f"{result.failure_count} / {spec.n} institutions failed.",
            "",
            "## Interpretation",
            _interpretation(spec, initial_defaults, result),
        ]
    )
    return "\n".join(lines)


def save_crisis_card(
    path: str | Path,
    spec: SystemSpec,
    initial_defaults: np.ndarray,
    result: CascadeResult,
    generator_name: str,
    scenario_id: int,
) -> None:
    Path(path).write_text(
        make_crisis_card(spec, initial_defaults, result, generator_name, scenario_id),
        encoding="utf-8",
    )


def _names_for_mask(spec: SystemSpec, mask: np.ndarray) -> list[str]:
    mask = np.asarray(mask, dtype=int)
    return [name for name, value in zip(spec.node_names, mask) if value == 1]


def _bullet_list(items: list[str]) -> str:
    if not items:
        return "- None"
    return "\n".join(f"- {item}" for item in items)


def _interpretation(spec: SystemSpec, initial: np.ndarray, result: CascadeResult) -> str:
    clusters = spec.clusters or ["unknown"] * spec.n
    shocked_clusters = sorted({clusters[i] for i, value in enumerate(initial) if value == 1})
    failed_clusters = sorted({clusters[i] for i, value in enumerate(result.final_defaults) if value == 1})
    if len(failed_clusters) > len(shocked_clusters):
        return (
            "The initial shock crosses capital thresholds beyond its starting cluster, "
            f"moving from {', '.join(shocked_clusters) or 'no initial cluster'} into "
            f"{', '.join(failed_clusters)}."
        )
    return (
        "The scenario remains concentrated in its initial dependency cluster, "
        "but still provides a reproducible stress case for the shared cascade engine."
    )
