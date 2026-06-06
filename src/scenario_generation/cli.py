from __future__ import annotations

import argparse
import sys
import numpy as np

from scenario_generation.io import load_system_spec, save_scenarios
from scenario_generation.gam_generator import GAMGenerator


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate synthetic scenarios with GAM")
    parser.add_argument("--system-spec", required=True)
    parser.add_argument("--n-scenarios", type=int, default=1000)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--augmentation-strength", type=float, default=0.10,
                        help="probability that an exposure-triggered buffer breach becomes a default (0..1)")
    parser.add_argument("--corr-threshold", type=float, default=0.10,
                        help="maximum allowed mean absolute off-diagonal correlation drift before warning/failing")
    parser.add_argument("--fail-on-drift", action="store_true",
                        help="exit nonzero if correlation drift exceeds --corr-threshold")
    parser.add_argument("--hybrid", action="store_true",
                        help="mix a fraction of quantum samples into the generated scenarios")
    parser.add_argument("--quantum-fraction", type=float, default=0.05,
                        help="fraction of samples drawn from the entangled quantum generator (0..1)")
    parser.add_argument("--quantum-ansatz", type=str, default="entangled",
                        help="ansatz passed to EntangledBornMachineGenerator (entangled|ghz_systemic)")
    parser.add_argument("--quantum-calibrate", action="store_true",
                        help="ask the entangled generator to calibrate against exact statevector moments when fitting")
    parser.add_argument("--quantum-calibration-iterations", type=int, default=30,
                        help="number of calibration iterations for the entangled generator")
    parser.add_argument("--train-qcbm", action="store_true",
                        help="train a QCBM (light) against the system spec and produce comparison plots")
    parser.add_argument("--qcbm-maxiter", type=int, default=20,
                        help="max optimizer iterations per block for the QCBM trainer (keeps training light)")
    parser.add_argument("--report-n-samples", type=int, dest="report_n_samples", default=2000,
                        help="how many samples to draw for the comparison report")
    parser.add_argument("--report-out", type=str, default="outputs/reports",
                        help="directory to write report plots")
    args = parser.parse_args(argv)

    spec = load_system_spec(args.system_spec)
    if args.hybrid:
        from scenario_generation.quantum_hybrid import HybridGAMGenerator

        ent_args = {
            "ansatz": args.quantum_ansatz,
            "calibrate": args.quantum_calibrate,
            "calibration_iterations": args.quantum_calibration_iterations,
        }
        gen = HybridGAMGenerator(quantum_fraction=args.quantum_fraction, entangled_kwargs=ent_args, augmentation_strength=args.augmentation_strength)
        gen.fit(spec)
        samples = gen.sample(args.n_scenarios, seed=args.seed)
    else:
        gen = GAMGenerator(augmentation_strength=args.augmentation_strength)
        gen.fit(spec)
        samples = gen.sample(args.n_scenarios, seed=args.seed)
    save_scenarios(args.out, samples, spec.node_names)
    print(f"Wrote {samples.shape[0]} scenarios to {args.out}")

    # Validate correlation drift using validation helper
    from scenario_generation.validation import validate_pairwise_corr

    sampled_corr, target_corr = validate_pairwise_corr(samples, spec)
    off = np.triu_indices(sampled_corr.shape[0], k=1)
    mean_abs_drift = float(np.mean(np.abs(sampled_corr[off] - target_corr[off])))
    print(f"mean_abs_correlation_drift={mean_abs_drift:.6f}")
    if mean_abs_drift > args.corr_threshold:
        msg = f"Correlation drift {mean_abs_drift:.6f} exceeds threshold {args.corr_threshold:.6f}"
        if args.fail_on_drift:
            print("ERROR:", msg)
            return 2
        else:
            print("WARNING:", msg)
    # optional training & comparison report
    if args.train_qcbm:
        from scenario_generation.qcbm_trainer import QCBMTrainer
        from scenario_generation.plots import plot_marginals, plot_corrs
        from systemic_risk.generators.quantum import ansatz as A

        trainer = QCBMTrainer(maxiter=args.qcbm_maxiter)
        print("Training QCBM (light) ...")
        results = trainer.fit(spec)
        # build statevector samples from trained circuits (small blocks only)
        # For simplicity, sample from each trained block exact probabilities and combine
        built_probs = []
        for block_key, info in results.items():
            circ = info['circuit']
            # exact statevector for block
            from systemic_risk.generators.quantum.statevector import StateVector

            sv = StateVector(circ.size)
            for i, th in enumerate(circ.ry):
                sv.ry(i, th)
            for e, (cidx, tidx) in enumerate(circ.edges):
                sv.cry(cidx, tidx, circ.cry[e])
            built_probs.append((circ.qubits, sv.probabilities()))
        # For reporting, compare marginals/corrs on full spec via entangled generator seed
        # assemble a full sampled distribution by drawing 2000 samples from hybrid gen
        import numpy as _np

        sample_for_report = samples[: args.report_n_samples]
        sampled_marg = sample_for_report.mean(axis=0)
        # compute sampled corr
        pairwise_joint = (sample_for_report.T @ sample_for_report) / max(sample_for_report.shape[0], 1)
        sampled_corr = _np.eye(spec.n)
        for i in range(spec.n):
            for j in range(i + 1, spec.n):
                denom = _np.sqrt(sampled_marg[i] * (1 - sampled_marg[i]) * sampled_marg[j] * (1 - sampled_marg[j]))
                corr = 0.0 if denom == 0 else (pairwise_joint[i, j] - sampled_marg[i] * sampled_marg[j]) / denom
                sampled_corr[i, j] = sampled_corr[j, i] = float(_np.clip(corr, -1.0, 1.0))

        # target corr
        from systemic_risk.spec import joint_to_corr

        target_corr = joint_to_corr(spec.target_pairwise_joint_probs(), spec.marginal_default_probs)
        # save plots
        plot_marginals(spec.marginal_default_probs, sampled_marg, args.report_out / 'marginals.png')
        plot_corrs(target_corr, sampled_corr, args.report_out / 'corrs.png')
        print(f"Saved report plots to {args.report_out}")
    # print diagnostics if available
    if hasattr(gen, "_last_diagnostics"):
        pre = gen._last_diagnostics.get("pre")
        post = gen._last_diagnostics.get("post")
        if pre is not None and post is not None:
            print(f"pre_marginal_mean={float(pre.sampled_marginals.mean()):.6f}")
            print(f"post_marginal_mean={float(post.sampled_marginals.mean()):.6f}")
            # show small summary of pairwise corr change
            if pre.sampled_pairwise_corr is not None and post.sampled_pairwise_corr is not None:
                pre_corr = pre.sampled_pairwise_corr
                post_corr = post.sampled_pairwise_corr
                off = np.triu_indices(pre_corr.shape[0], k=1)
                print(f"pre_post_corr_mean_abs_diff={float(abs(post_corr[off]-pre_corr[off]).mean()):.6f}")
    return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
