"""Train the adversarial Born-machine generator (qGAN) and save the result.

The project's canonical generator uses analytic angles (no training). This
script exercises the *experimental* adversarial path: a parameterised Born
machine trained against a NumPy discriminator, with the generator objective
evaluated as the **exact expectation** over the Born distribution and its
gradient taken by central differences on that exact expectation (the noise-free
analogue of the hardware parameter-shift rule).

It trains two configurations:

  --mode dense  (default) the strongest n=12 dependency block, full graph
  --mode hw     a hardware-friendly chain: inflated marginals (above the QPU
                noise floor) + a max-degree-2 (near-neighbour) entangler graph,
                whose trained circuit is saved to outputs/qgan_hw_circuit.npz for
                submission by scripts/run_qgan_hardware.py.

Run:
    uv run python scripts/train_qgan.py --mode dense
    uv run python scripts/train_qgan.py --mode hw
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from systemic_risk.data.synthetic import make_synthetic_system
from systemic_risk.generators.gaussian_copula import GaussianCopulaGenerator
from systemic_risk.generators.quantum import ansatz as A
from systemic_risk.generators.quantum.statevector import StateVector, sample_bitstrings
from scenario_generation.qgan_trainer import QGANTrainer


def _generate(circuit: A.EntangledCircuit, n_samples: int, rng) -> np.ndarray:
    """Sample binary default vectors from a trained block circuit (block columns only)."""
    sv = StateVector(circuit.size)
    for i, th in enumerate(circuit.ry):
        sv.ry(i, float(th))
    for e, (c, t) in enumerate(circuit.edges):
        sv.cry(int(c), int(t), float(circuit.cry[e]))
    bits = sample_bitstrings(sv.probabilities(), circuit.size, n_samples, rng)
    return bits  # (n_samples, circuit.size) in block-local order


def _chain_order(spec, block):
    """Order banks so the most strongly-correlated are adjacent (heavy-hex line, 0 SWAPs)."""
    dep = np.abs(spec.dependency_matrix())
    np.fill_diagonal(dep, 0.0)
    remaining = set(block)
    start = max(block, key=lambda q: dep[q, block].sum())
    order = [start]
    remaining.discard(start)
    while remaining:
        last = order[-1]
        nxt = max(remaining, key=lambda q: dep[last, q])
        order.append(nxt)
        remaining.discard(nxt)
    return order


def _inflate_marginals(spec, inflate_to=0.10):
    """Lift PDs above the ~2-3% QPU noise floor (preserving order) for a loadable demo."""
    p = np.clip(spec.marginal_default_probs, 1e-6, 1 - 1e-6)
    return np.clip(p / p.max() * inflate_to + 0.03, 0.03, 0.45)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["dense", "hw"], default="dense")
    ap.add_argument("--epochs", type=int, default=150)
    args = ap.parse_args()

    n = 12
    spec = make_synthetic_system(n=n, seed=7)
    print(f"Spec: n={spec.n}  marginals(min/mean/max)="
          f"{spec.marginal_default_probs.min():.3f}/"
          f"{spec.marginal_default_probs.mean():.3f}/"
          f"{spec.marginal_default_probs.max():.3f}")

    circuit_seed = None
    if args.mode == "dense":
        block = list(range(n))
        tag = "dense"
        gauss = GaussianCopulaGenerator()
        gauss.fit(spec)
        real = gauss.sample(8000, seed=4242)
    else:
        tag = "hw"
        # Inflate PDs above the noise floor, then draw the target from the SAME
        # inflated spec so generator target and discriminator data are consistent.
        order = _chain_order(spec, list(range(n)))
        block = order
        p_hw = _inflate_marginals(spec, inflate_to=0.10)
        spec.marginal_default_probs[:] = p_hw
        gauss = GaussianCopulaGenerator()
        gauss.fit(spec)
        real = gauss.sample(8000, seed=4242)
        global_edges = [(order[i], order[i + 1]) for i in range(len(order) - 1)]
        circuit_seed = A._block_circuit(order, p_hw, A.target_covariance(spec), global_edges)
        print(f"HW chain order: {order}")
        print(f"HW marginals(min/mean/max)={p_hw.min():.3f}/{p_hw.mean():.3f}/{p_hw.max():.3f}  "
              f"edges={len(circuit_seed.edges)} (max-degree-2 chain)")

    trainer = QGANTrainer(
        n_epochs=args.epochs,
        batch_size=512,
        disc_steps=2,
        gen_steps=1,
        lr_disc=2e-2,
        lr_gen=1.2e-1,
        fd_eps=1e-2,
        seed=0,
    )

    # analytic-seed baseline moment error (before any training)
    if circuit_seed is None:
        edges = A.dependency_edges(spec, threshold=0.02, within_clusters_only=False)
        circuit_seed = A._block_circuit(
            block, np.clip(spec.marginal_default_probs, 1e-6, 1 - 1e-6),
            A.target_covariance(spec), edges,
        )
    bt = QGANTrainer._bit_table(circuit_seed.size)
    seed_err = trainer._moment_error(
        circuit_seed, trainer._probabilities(circuit_seed, trainer._build_circuit_params(circuit_seed)), bt,
    )

    print(f"\nTraining qGAN [{tag}] for {args.epochs} epochs "
          f"({circuit_seed.size} RY + {len(circuit_seed.edges)} CRY params)...")
    result = trainer.fit(spec, circuit=circuit_seed, real=real)
    circuit = result["circuit"]
    hist = result["history"]

    print(f"disc_loss: {hist['disc_loss'][0]:.4f} -> {hist['disc_loss'][-1]:.4f}")
    print(f"gen_obj:   {hist['gen_loss'][0]:.4f} -> {hist['gen_loss'][-1]:.4f}")
    print(f"moment_err: seed {seed_err:.5f} -> best {hist['best_moment_err']:.5f}  "
          f"({100*(1-hist['best_moment_err']/max(seed_err,1e-12)):.1f}% reduction)")

    # --- validation: generated vs. copula-target moments on the block ---
    rng = np.random.default_rng(123)
    gen = _generate(circuit, 16000, rng)           # block-local order
    real_block = real[:, block]                    # same order as circuit.qubits

    gen_marg = gen.mean(axis=0)
    real_marg = real_block.mean(axis=0)
    marg_mae = float(np.abs(gen_marg - real_marg).mean())

    gen_corr = np.corrcoef(gen, rowvar=False)
    real_corr = np.corrcoef(real_block, rowvar=False)
    iu = np.triu_indices(circuit.size, k=1)
    corr_mae = float(np.abs(gen_corr[iu] - real_corr[iu]).mean())

    # adjacent (directly-entangled) pairs -- what a chain circuit can represent
    adj = [(e[0], e[1]) for e in circuit.edges]
    adj_mae = float(np.mean([abs(gen_corr[i, j] - real_corr[i, j]) for i, j in adj])) if adj else 0.0

    print(f"\nValidation on block (vs. Gaussian-copula target):")
    print(f"  marginal MAE:           {marg_mae:.4f}")
    print(f"  pairwise corr MAE (all):     {corr_mae:.4f}")
    print(f"  pairwise corr MAE (adjacent): {adj_mae:.4f}  ({len(adj)} directly-entangled pairs)")

    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)
    metrics = {
        "mode": tag,
        "n": spec.n,
        "block": list(map(int, circuit.qubits)),
        "n_params": int(circuit.size + len(circuit.edges)),
        "epochs": args.epochs,
        "seed_moment_err": seed_err,
        "best_moment_err": hist["best_moment_err"],
        "disc_loss_last": hist["disc_loss"][-1],
        "gen_obj_last": hist["gen_loss"][-1],
        "marginal_mae": marg_mae,
        "corr_mae": corr_mae,
        "corr_mae_adjacent": adj_mae,
    }
    (out_dir / f"qgan_training_{tag}.json").write_text(json.dumps(metrics, indent=2))
    print(f"Wrote {out_dir / f'qgan_training_{tag}.json'}")

    if tag == "hw":
        np.savez(
            out_dir / "qgan_hw_circuit.npz",
            qubits=np.array(circuit.qubits, dtype=int),
            ry=circuit.ry,
            edges=np.array(circuit.edges, dtype=int),
            cry=circuit.cry,
            target_p=circuit.target_p,
            target_cov=circuit.target_cov,
        )
        print(f"Wrote {out_dir / 'qgan_hw_circuit.npz'} (for run_qgan_hardware.py)")


if __name__ == "__main__":
    main()
