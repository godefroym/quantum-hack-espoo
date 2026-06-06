from __future__ import annotations

import numpy as np
from typing import Optional

from systemic_risk.generators.base import ScenarioGenerator
from systemic_risk.spec import SystemSpec

from .gam_generator import GAMGenerator


class HybridGAMGenerator(ScenarioGenerator):
    """Hybrid generator that mixes GAM (classical) with an entangled Born machine.

    Parameters
    - quantum_fraction: fraction of samples drawn from the quantum generator (0..1).
    - entangled_kwargs: passed to EntangledBornMachineGenerator constructor.
    """

    name = "Hybrid GAM (quantum-classical mix)"

    def __init__(self, quantum_fraction: float = 0.1, *, entangled_kwargs: dict | None = None, augmentation_strength: float = 0.05) -> None:
        if not (0.0 <= quantum_fraction <= 1.0):
            raise ValueError("quantum_fraction must lie in [0,1]")
        self.quantum_fraction = float(quantum_fraction)
        self.entangled_kwargs = dict(entangled_kwargs or {})
        self.augmentation_strength = float(augmentation_strength)

        self._quantum: Optional[object] = None
        self._gam: Optional[GAMGenerator] = None
        self._spec: Optional[SystemSpec] = None

    def fit(self, spec: SystemSpec) -> None:
        # Lazy import to avoid heavy quantum deps unless used
        from systemic_risk.generators import EntangledBornMachineGenerator

        self._spec = spec
        self._gam = GAMGenerator(augmentation_strength=self.augmentation_strength)
        self._gam.fit(spec)

        # instantiate and fit the entangled (quantum) generator
        qargs = dict(self.entangled_kwargs)
        q = EntangledBornMachineGenerator(**qargs)
        q.fit(spec)
        self._quantum = q

    def sample(self, n_samples: int, seed: int | None = None) -> np.ndarray:
        if self._spec is None or self._gam is None or self._quantum is None:
            raise RuntimeError("HybridGAMGenerator must be fit before sampling")
        rng = np.random.default_rng(seed)
        n_q = int(round(self.quantum_fraction * n_samples))
        n_c = n_samples - n_q

        parts = []
        if n_q > 0:
            q_samples = self._quantum.sample(n_q, seed=rng.integers(0, 2 ** 31))
            parts.append(q_samples.astype(int))
        if n_c > 0:
            c_samples = self._gam.sample(n_c, seed=rng.integers(0, 2 ** 31))
            parts.append(c_samples.astype(int))

        if len(parts) == 1:
            combined = parts[0]
        else:
            combined = np.vstack(parts)
            # shuffle rows to mix quantum and classical draws
            idx = rng.permutation(combined.shape[0])
            combined = combined[idx]

        return combined.astype(int)
