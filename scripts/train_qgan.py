"""Train the adversarial Born-machine generator (qGAN) on a synthetic spec.

The project's canonical generator uses analytic angles (no training). This
script exercises the *experimental* adversarial path: a parameterised Born
machine trained against a small NumPy discriminator, with generator gradients
estimated by finite differences on the exact statevector.

Run:
    uv run python scripts/train_qgan.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from systemic_risk.data.synthetic import make_synthetic_system
from systemic_risk.generators.gaussian_copula import GaussianCopulaGenerator
from systemic_risk.generators.quantum import ansatz as A
from systemic_risk.generators.quantum.statevector import StateVector, sample_bitstrings
from scenario_generation.qgan_trainer import QGANTrainer


def _generate(circuit: A.EntangledCircuit, n_samples: int, rng) -> np.ndarray:
    """Sample binary default vectors from a trained block circuit."""
    sv = StateVector(circuit.size)
    for i, th in enumerate(circuit.ry):
        sv.ry(i, float(th))
    for e, (c, t) in enumerate(circuit.edges):
        sv.cry(int(c), int(t), float(circuit.cry[e]))
    bits = sample_bitstrings(sv.probabilities(), circuit.size, n_samples, rng)
    out = np.zeros((n_samples, max(circuit.qubits) + 1), dtype=int)
    for col, q in enumerate(circuit.qubits):
        out[:, q] = bits[:, col]
    return out


def main() -> None:
    n = 12
    spec = make_synthetic_system(n=n, seed=7)
    print(f"Spec: n={spec.n}  marginals(min/mean/max)="
          f"{spec.marginal_default_probs.min():.3f}/"
          f"{spec.marginal_default_probs.mean():.3f}/"
          f"{spec.marginal_default_probs.max():.3f}")

    trainer = QGANTrainer(
        n_epochs=40,
        batch_size=256,
        disc_steps=3,
        gen_steps=1,
        lr_disc=5e-2,
        lr_gen=8e-2,
        fd_eps=1e-2,
        seed=0,
    )
    print("Training qGAN (finite-difference generator gradients)...")
    result = trainer.fit(spec, n_real=4000)
    circuit = result["circuit"]
    hist = result["history"]

    block = list(circuit.qubits)
    print(f"Trained block qubits={block}  #params={circuit.size + len(circuit.edges)}")
    print(f"disc_loss: {hist['disc_loss'][0]:.4f} -> {hist['disc_loss'][-1]:.4f}")
    print(f"gen_loss:  {hist['gen_loss'][0]:.4f} -> {hist['gen_loss'][-1]:.4f}")

    # --- validation: compare generated vs. copula-target moments on the block ---
    rng = np.random.default_rng(123)
    gen = _generate(circuit, 8000, rng)[:, block]

    gauss = GaussianCopulaGenerator()
    gauss.fit(spec)
    real = gauss.sample(8000, seed=999)[:, block]

    gen_marg = gen.mean(axis=0)
    real_marg = real.mean(axis=0)
    marg_mae = float(np.abs(gen_marg - real_marg).mean())

    gen_corr = np.corrcoef(gen, rowvar=False)
    real_corr = np.corrcoef(real, rowvar=False)
    iu = np.triu_indices(len(block), k=1)
    corr_mae = float(np.abs(gen_corr[iu] - real_corr[iu]).mean())

    print(f"\nValidation on block (vs. Gaussian-copula target):")
    print(f"  marginal MAE:    {marg_mae:.4f}")
    print(f"  pairwise corr MAE: {corr_mae:.4f}")

    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)
    metrics = {
        "n": spec.n,
        "block": block,
        "n_params": int(circuit.size + len(circuit.edges)),
        "disc_loss_first": hist["disc_loss"][0],
        "disc_loss_last": hist["disc_loss"][-1],
        "gen_loss_first": hist["gen_loss"][0],
        "gen_loss_last": hist["gen_loss"][-1],
        "marginal_mae": marg_mae,
        "corr_mae": corr_mae,
    }
    (out_dir / "qgan_training.json").write_text(json.dumps(metrics, indent=2))
    print(f"\nWrote {out_dir / 'qgan_training.json'}")


if __name__ == "__main__":
    main()
