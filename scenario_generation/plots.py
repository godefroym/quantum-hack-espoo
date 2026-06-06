from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def plot_marginals(target: np.ndarray, sampled: np.ndarray, out: str | Path) -> None:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    idx = np.arange(len(target))
    plt.figure(figsize=(8, 4))
    plt.bar(idx - 0.2, target, width=0.4, label='target')
    plt.bar(idx + 0.2, sampled, width=0.4, label='sampled')
    plt.xlabel('node')
    plt.ylabel('marginal P(default)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(str(out))
    plt.close()


def plot_corrs(target: np.ndarray, sampled: np.ndarray, out: str | Path) -> None:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(target, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('target correlation')
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.subplot(1, 2, 2)
    plt.imshow(sampled, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('sampled correlation')
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(str(out))
    plt.close()
