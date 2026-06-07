"""Generate a labelled scenario dataset across data sources and generators.

For each data source (the real G-SIB network plus the synthetic network specs)
this samples correlated binary default scenarios from every generator and writes
them side-by-side, so the files can be used to train/benchmark generators or to
compare classical vs quantum samplers on a common footing.

Layout (under --out-dir, default data/scenario_dataset/):

    <source>/
        spec.json                      # the SystemSpec (marginals, correlation, ...)
        <source>__<generator>.csv      # one row per scenario, header = institutions
        <source>__<generator>.npz      # samples + node_names + target marginals/corr
        moments.json                   # per-generator fit quality (RMSE vs targets)
    manifest.json                      # full index: shapes, seeds, moment errors

Run:
    uv run python scripts/build_dataset.py                 # defaults
    uv run python scripts/build_dataset.py --n-samples 8000
    uv run python scripts/build_dataset.py --sources real synthetic_16
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from systemic_risk.data.synthetic import make_synthetic_system, make_scalable_system
from systemic_risk.data_network.assemble import build_system_spec

from systemic_risk.generators.bernoulli import BernoulliGenerator
from systemic_risk.generators.gaussian_copula import GaussianCopulaGenerator
from systemic_risk.generators.student_t_copula import StudentTCopulaGenerator
from systemic_risk.generators.quantum_born_machine import EntangledBornMachineGenerator

from scenario_generation.io import save_scenarios


# data sources: real network + the genuinely-synthetic network specs
# (clustered / Huang are excluded -- they are synthetic stylised models)
SOURCES: dict[str, tuple[str, callable]] = {
    "real_gsib": ("real", lambda: build_system_spec()),
    "synthetic_16": ("synthetic", lambda: make_synthetic_system(n=16, seed=7)),
    "synthetic_54": ("synthetic", lambda: make_scalable_system(n=54, seed=11)),
}

# generators sampled side-by-side; each is a fresh instance per source.
# (IsingBoltzmannGenerator is excluded: it freezes into a degenerate all-default
# state on sparse, low-PD networks like the real G-SIB net.)
GENERATORS: dict[str, callable] = {
    "bernoulli": lambda: BernoulliGenerator(),
    "gaussian_copula": lambda: GaussianCopulaGenerator(),
    "student_t_copula": lambda: StudentTCopulaGenerator(df=4.0),
    "entangled_born_machine": lambda: EntangledBornMachineGenerator(),
}


def moment_errors(samples: np.ndarray, spec) -> dict:
    """Fit quality + validity flags for one generator's samples.

    ``degenerate`` marks samples that have collapsed (e.g. an Ising model frozen
    into its ordered all-default phase): the marginals are wildly off target or
    the support has collapsed to a handful of distinct scenarios.
    """
    emp_marg = samples.mean(axis=0)
    tgt_marg = spec.marginal_default_probs
    marg_rmse = float(np.sqrt(np.mean((emp_marg - tgt_marg) ** 2)))
    n_unique = int(np.unique(samples, axis=0).shape[0])

    out = {
        "marginal_rmse": marg_rmse,
        "mean_default_rate": float(emp_marg.mean()),
        "n_unique_scenarios": n_unique,
    }
    if spec.target_pairwise_corr is not None and samples.shape[0] > 1:
        with np.errstate(invalid="ignore"):
            emp_corr = np.corrcoef(samples, rowvar=False)
        emp_corr = np.nan_to_num(emp_corr)
        iu = np.triu_indices(spec.n, k=1)
        corr_rmse = float(np.sqrt(np.mean((emp_corr[iu] - spec.target_pairwise_corr[iu]) ** 2)))
        out["corr_rmse"] = corr_rmse
    # collapsed support, or marginals an order of magnitude off the target scale
    out["degenerate"] = bool(marg_rmse > 0.10 or n_unique < max(10, samples.shape[0] // 100))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-samples", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--sources", nargs="*", default=list(SOURCES), choices=list(SOURCES))
    ap.add_argument("--generators", nargs="*", default=list(GENERATORS), choices=list(GENERATORS))
    ap.add_argument("--out-dir", type=Path, default=Path("data/scenario_dataset"))
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "n_samples": args.n_samples,
        "seed": args.seed,
        "sources": {},
    }

    for src_name in args.sources:
        kind, builder = SOURCES[src_name]
        print(f"\n=== {src_name} ({kind}) ===")
        spec = builder()
        src_dir = args.out_dir / src_name
        src_dir.mkdir(parents=True, exist_ok=True)
        spec.save_json(src_dir / "spec.json")

        src_entry = {
            "kind": kind,
            "n_institutions": int(spec.n),
            "node_names": list(spec.node_names),
            "marginals_mean": float(spec.marginal_default_probs.mean()),
            "generators": {},
        }
        moments = {}

        for gen_name in args.generators:
            try:
                gen = GENERATORS[gen_name]()
                t0 = time.perf_counter()
                gen.fit(spec)
                samples = np.asarray(gen.sample(args.n_samples, seed=args.seed), dtype=int)
                dt = time.perf_counter() - t0

                stem = f"{src_name}__{gen_name}"
                save_scenarios(src_dir / f"{stem}.csv", samples, spec.node_names)
                np.savez_compressed(
                    src_dir / f"{stem}.npz",
                    samples=samples,
                    node_names=np.array(spec.node_names),
                    target_marginals=spec.marginal_default_probs,
                    target_corr=(spec.target_pairwise_corr
                                 if spec.target_pairwise_corr is not None
                                 else np.empty((0, 0))),
                    source=src_name,
                    generator=gen_name,
                    seed=args.seed,
                )
                err = moment_errors(samples, spec)
                moments[gen_name] = err
                src_entry["generators"][gen_name] = {
                    "shape": list(samples.shape),
                    "fit_sample_seconds": round(dt, 2),
                    **err,
                }
                flag = "  [DEGENERATE]" if err.get("degenerate") else ""
                msg = (f"marg_rmse={err['marginal_rmse']:.4f} "
                       f"corr_rmse={err.get('corr_rmse', float('nan')):.4f} "
                       f"uniq={err['n_unique_scenarios']}")
                print(f"  {gen_name:24s} {samples.shape}  {msg}  ({dt:.1f}s){flag}")
            except Exception as e:  # noqa: BLE001 -- record and continue
                src_entry["generators"][gen_name] = {"error": repr(e)}
                print(f"  {gen_name:24s} FAILED: {e!r}")

        (src_dir / "moments.json").write_text(json.dumps(moments, indent=2))
        manifest["sources"][src_name] = src_entry

    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nWrote dataset to {args.out_dir}/  (manifest.json indexes everything)")


if __name__ == "__main__":
    main()
