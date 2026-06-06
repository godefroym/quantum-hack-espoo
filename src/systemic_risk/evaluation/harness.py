from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from systemic_risk.evaluation.metrics import compute_metrics
from systemic_risk.generators.base import ScenarioGenerator
from systemic_risk.simulator.cascade import CascadeResult, simulate_many
from systemic_risk.spec import SystemSpec


@dataclass
class GeneratorRunResult:
    generator_name: str
    samples: np.ndarray
    cascade_results: list[CascadeResult]
    metrics: dict[str, float]


class EvaluationHarness:
    """Fit generators, sample scenarios, run the shared cascade engine, and compare."""

    def __init__(
        self,
        spec: SystemSpec,
        n_samples: int = 2_000,
        severe_threshold: int | None = None,
        collapse_threshold: float = 0.5,
        seed: int = 123,
        max_rounds: int | None = None,
        lgd: float | np.ndarray = 1.0,
        fail_on_equal: bool = False,
        include_joint_structure: bool = True,
    ) -> None:
        if n_samples <= 0:
            raise ValueError("n_samples must be positive")
        self.spec = spec
        self.n_samples = n_samples
        self.severe_threshold = (
            int(np.ceil(0.5 * spec.n))
            if severe_threshold is None
            else int(severe_threshold)
        )
        if not 0 <= self.severe_threshold <= spec.n:
            raise ValueError("severe_threshold must lie between 0 and spec.n")
        self.collapse_threshold = collapse_threshold
        self.seed = seed
        self.max_rounds = max_rounds
        self.lgd = lgd
        self.fail_on_equal = fail_on_equal
        self.include_joint_structure = include_joint_structure

    def run(self, generators: list[ScenarioGenerator]) -> list[GeneratorRunResult]:
        results: list[GeneratorRunResult] = []
        seed_sequence = np.random.SeedSequence(self.seed)
        child_seeds = seed_sequence.spawn(len(generators))
        for generator, child_seed in zip(generators, child_seeds):
            generator.fit(self.spec)
            if hasattr(generator, "train"):
                generator.train(seed=int(child_seed.generate_state(1)[0]))
            sample_seed = int(child_seed.generate_state(1)[0])
            samples = generator.sample(self.n_samples, seed=sample_seed)
            cascades = simulate_many(
                samples,
                self.spec,
                max_rounds=self.max_rounds,
                collapse_threshold=self.collapse_threshold,
                lgd=self.lgd,
                fail_on_equal=self.fail_on_equal,
            )
            metrics = compute_metrics(
                samples,
                cascades,
                self.spec,
                severe_threshold=self.severe_threshold,
                include_joint_structure=self.include_joint_structure,
            )
            results.append(
                GeneratorRunResult(
                    generator_name=generator.name,
                    samples=samples,
                    cascade_results=cascades,
                    metrics=metrics,
                )
            )
        return results

    @staticmethod
    def to_frame(results: list[GeneratorRunResult]) -> pd.DataFrame:
        rows = [{"generator": result.generator_name, **result.metrics} for result in results]
        return pd.DataFrame(rows).sort_values(
            ["p_severe_cascade", "tail_mean_5pct", "max_cascade_size"],
            ascending=False,
        )
