"""Build a SystemSpec for parts B/C/D, end-to-end (Part A deliverable + entrypoint).

WHAT THIS PRODUCES
------------------
A flat ``systemic_risk.spec.SystemSpec`` — the exact object the generators (B/C), the
cascade simulator (D), and the evaluation harness already consume. It is written to
``outputs/data_network/system_spec.json`` (and ``.npz``), so B/C/D can load a frozen spec
instead of rebuilding it every run::

    from systemic_risk.spec import SystemSpec
    spec = SystemSpec.load_json("outputs/data_network/system_spec.json")

Or skip the file entirely and call the in-process entrypoints::

    from systemic_risk.data_network import build_system_spec, build_synthetic_system_spec
    spec = build_system_spec()                 # the REAL 28-bank exposure network
    spec = build_synthetic_system_spec(n=54)   # calibrated-synthetic, scales to 54 qubits

PIPELINE (real path)
--------------------
    load raw roster + equity correlation  ->  estimate empirical layer (p_i, corr, totals)
 -> reconstruct bilateral exposures  ->  detect communities  ->  assemble a frozen
    NetworkSpec  ->  validate (round-trip, cluster stability, B/C/D contract)
 -> render the community plot  ->  save the flat SystemSpec for B/C/D.

RUN
---
    uv run python scripts/build_system_spec.py                    # real network (default)
    uv run python scripts/build_system_spec.py --method min_density   # sparse exposures
    uv run python scripts/build_system_spec.py --refresh-equity   # re-fetch from Yahoo
    uv run python scripts/build_system_spec.py --synthetic 54     # synthetic, n=54
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

# --- make the script runnable from a bare checkout (no install needed) ------- #
# Point matplotlib / XDG caches inside outputs/ (writable, git-ignored) and put the
# package on sys.path so `uv run python scripts/...` works without `uv sync` first.
ROOT = Path(__file__).resolve().parents[1]
MPL_CACHE = ROOT / "outputs" / ".matplotlib"
XDG_CACHE = ROOT / "outputs" / ".cache"
MPL_CACHE.mkdir(parents=True, exist_ok=True)
XDG_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE))
os.environ.setdefault("XDG_CACHE_HOME", str(XDG_CACHE))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from systemic_risk.data_network.assemble import build_network_spec
from systemic_risk.data_network.sources.synthetic import synthetic_network_spec
from systemic_risk.data_network.validate import validate_spec
from systemic_risk.visualization import plot_community_network


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a SystemSpec for B/C/D.")
    parser.add_argument("--method", default="max_entropy",
                        choices=["max_entropy", "min_density"],
                        help="bilateral-exposure reconstruction method (real path)")
    parser.add_argument("--refresh-equity", action="store_true",
                        help="re-fetch equity prices from Yahoo and rewrite the snapshot")
    parser.add_argument("--synthetic", type=int, metavar="N", default=None,
                        help="build the calibrated-synthetic network at n=N instead of the "
                             "real roster (use to scale toward the 54-qubit target)")
    parser.add_argument("--seed", type=int, default=11,
                        help="seed for the synthetic build")
    parser.add_argument("--out", default=str(ROOT / "outputs" / "data_network"),
                        help="output directory")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- 1. BUILD the layered NetworkSpec (real roster, or synthetic at n=N) --- #
    if args.synthetic is not None:
        print(f"Building SYNTHETIC network spec (n={args.synthetic}, seed={args.seed}) ...")
        spec = synthetic_network_spec(n=args.synthetic, seed=args.seed)
    else:
        print(f"Building REAL network spec (method={args.method}, "
              f"equity={'live refresh' if args.refresh_equity else 'snapshot'}) ...")
        spec = build_network_spec(
            reconstruction_method=args.method,
            prefer_snapshot=not args.refresh_equity,  # snapshot keeps the build reproducible
            write_snapshot=args.refresh_equity,       # only overwrite when explicitly asked
        )

    # Provenance.fit_params carries the build's headline numbers for a quick readout.
    fp = spec.provenance.fit_params
    print(f"\n  nodes ............... {spec.n}")
    print(f"  communities ......... {fp.get('n_communities')}  "
          f"(modularity {fp.get('modularity')})")
    print(f"  cluster stability ... mean ARI {fp.get('cluster_mean_ari')}")
    if "equity_window" in fp:
        print(f"  equity window ....... {fp['equity_window']}  ({fp['equity_n_obs']} obs)")
    print(f"  reconstruction ...... {spec.reconstructed.method} "
          f"({int((spec.reconstructed.edge_matrix > 0).sum())} directed edges)")
    print(f"  content hash ........ {spec.provenance.content_hash[:16]}")

    # --- 2. VALIDATE (the Part A end-to-end test) ----------------------------- #
    # Round-trip and the B/C/D contract are hard requirements; cluster stability is a
    # property of the REAL equity signal (the synthetic toy correlation is near-zero, so
    # its communities are legitimately unstable — warn, don't fail, in that case).
    print("\nValidating ...")
    report = validate_spec(spec)
    print(f"  round-trip lossless . {report.roundtrip_ok}")
    print(f"  clusters stable ..... {report.clusters_stable} "
          f"(mean ARI {report.cluster_mean_ari:.3f})")
    print(f"  B/C/D contract ...... {report.contract_ok}  {report.details}")
    if not (report.roundtrip_ok and report.contract_ok):
        raise SystemExit("VALIDATION FAILED (round-trip or B/C/D contract)")
    if not report.clusters_stable and args.synthetic is None:
        raise SystemExit("VALIDATION FAILED (real-network communities are not stable)")

    # --- 3. SAVE the artifacts B/C/D load ------------------------------------- #
    # network_spec.json = the full layered NetworkSpec (audit / provenance / reload).
    # system_spec.json / .npz = the flat SystemSpec that B/C/D consume directly.
    network_json = out_dir / "network_spec.json"
    system_json = out_dir / "system_spec.json"
    system_npz = out_dir / "system_spec.npz"
    spec.save_json(network_json)
    system = spec.to_system_spec()             # <-- the B/C/D-facing object
    system.save_json(system_json)
    system.save_npz(system_npz)

    # --- 4. RENDER the community plot (the legibility deliverable) ------------- #
    plot_path = out_dir / "community_network.png"
    plot_community_network(system, plot_path)

    # Community composition by region — the human-readable cluster story.
    print("\nCommunity composition (region counts):")
    regions = system.metadata.get("regions", [""] * system.n)
    clusters = system.clusters or []
    for lab in sorted(set(clusters)):
        members = [regions[i] for i in range(system.n) if clusters[i] == lab]
        counts = {r: members.count(r) for r in sorted(set(members))}
        print(f"  {lab}: {counts}")

    print(f"\nSaved:\n  {network_json}\n  {system_json}\n  {system_npz}\n  {plot_path}")
    print("\nB/C/D can now load it with:\n"
          f"  SystemSpec.load_json(r\"{system_json}\")")


if __name__ == "__main__":
    main()
