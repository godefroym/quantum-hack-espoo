from __future__ import annotations

import csv
import numpy as np
from systemic_risk.data.loaders import load_system_spec
from systemic_risk.simulator.cascade import simulate_many
from systemic_risk.spec import joint_to_corr


def main():
    spec = load_system_spec('outputs/data_network/system_spec.json')
    path = 'outputs/scenarios/gam_scenarios.csv'
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)
        data = np.array([list(map(int, row)) for row in reader], dtype=int)

    n_samples, n = data.shape
    print('csv_path:', path)
    print('shape:', data.shape)
    print('header_count:', len(header), 'spec.n:', spec.n)

    sampled_marginals = data.mean(axis=0)
    target_marginals = np.asarray(spec.marginal_default_probs)
    max_abs_diff = float(np.max(np.abs(sampled_marginals - target_marginals)))
    mean_abs_diff = float(np.mean(np.abs(sampled_marginals - target_marginals)))
    print(f'marginals: target mean={float(target_marginals.mean()):.6f}, sampled mean={float(sampled_marginals.mean()):.6f}')
    print(f'marginals: max_abs_diff={max_abs_diff:.6f}, mean_abs_diff={mean_abs_diff:.6f}')

    # sample correlation
    m = sampled_marginals
    pairwise_joint = (data.T @ data) / max(n_samples, 1)
    sampled_corr = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            denom = np.sqrt(m[i] * (1 - m[i]) * m[j] * (1 - m[j]))
            corr = 0.0 if denom == 0 else (pairwise_joint[i, j] - m[i] * m[j]) / denom
            sampled_corr[i, j] = sampled_corr[j, i] = float(np.clip(corr, -1.0, 1.0))

    if spec.target_pairwise_corr is not None:
        target_corr = np.asarray(spec.target_pairwise_corr)
    else:
        target_corr = joint_to_corr(spec.target_pairwise_joint_probs(), np.asarray(spec.marginal_default_probs))

    off = np.triu_indices(n, k=1)
    abs_diff_corr = np.abs(sampled_corr[off] - target_corr[off])
    print(f'corr_offdiag: mean_abs_diff={float(abs_diff_corr.mean()):.6f}, max_abs_diff={float(abs_diff_corr.max()):.6f}')

    # run simulator on first 100 scenarios to confirm acceptance
    sample_subset = data[:100]
    results = simulate_many(sample_subset, spec)
    print('simulate_many accepted sample subset:', len(results), 'examples: first failure_count=', results[0].failure_count)


if __name__ == '__main__':
    main()
