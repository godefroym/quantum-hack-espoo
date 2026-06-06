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
