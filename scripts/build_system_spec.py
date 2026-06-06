"""Build the real bank-network SystemSpec end-to-end (part A deliverable).

Pipeline:  load raw roster + equity correlation  ->  estimate empirical layer
        ->  reconstruct bilateral exposures  ->  detect communities
        ->  assemble a frozen NetworkSpec  ->  validate (round-trip, cluster
            stability, B/C/D contract)  ->  render the community plot  ->  save specs.

Run:

    uv run python scripts/build_system_spec.py
    uv run python scripts/build_system_spec.py --refresh-equity   # re-fetch from Yahoo
    uv run python scripts/build_system_spec.py --method min_density
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

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
from systemic_risk.data_network.validate import validate_spec
from systemic_risk.visualization import plot_community_network


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the real bank-network SystemSpec.")
    parser.add_argument("--method", default="max_entropy",
                        choices=["max_entropy", "min_density"],
                        help="bilateral-exposure reconstruction method")
    parser.add_argument("--refresh-equity", action="store_true",
                        help="re-fetch equity prices from Yahoo and rewrite the snapshot")
    parser.add_argument("--out", default=str(ROOT / "outputs" / "data_network"),
                        help="output directory")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Building network spec (method={args.method}, "
          f"equity={'live refresh' if args.refresh_equity else 'snapshot'}) ...")
    spec = build_network_spec(
        reconstruction_method=args.method,
        prefer_snapshot=not args.refresh_equity,
        write_snapshot=args.refresh_equity,
    )

    fp = spec.provenance.fit_params
    print(f"\n  nodes ............... {spec.n}")
    print(f"  communities ......... {fp['n_communities']}  (modularity {fp['modularity']})")
    print(f"  cluster stability ... mean ARI {fp['cluster_mean_ari']} "
          f"(stable={fp['cluster_stable']})")
    print(f"  equity window ....... {fp['equity_window']}  ({fp['equity_n_obs']} obs)")
    print(f"  reconstruction ...... {spec.reconstructed.method} "
          f"({int((spec.reconstructed.edge_matrix > 0).sum())} directed edges)")
    print(f"  content hash ........ {spec.provenance.content_hash[:16]}")

    # --- validate (the A end-to-end test) --------------------------------- #
    print("\nValidating ...")
    report = validate_spec(spec)
    print(f"  round-trip lossless . {report.roundtrip_ok}")
    print(f"  clusters stable ..... {report.clusters_stable} "
          f"(mean ARI {report.cluster_mean_ari:.3f})")
    print(f"  B/C/D contract ...... {report.contract_ok}  {report.details}")
    if not report.ok:
        raise SystemExit("VALIDATION FAILED")

    # --- save artifacts ---------------------------------------------------- #
    network_json = out_dir / "network_spec.json"
    system_json = out_dir / "system_spec.json"
    system_npz = out_dir / "system_spec.npz"
    spec.save_json(network_json)
    system = spec.to_system_spec()
    system.save_json(system_json)
    system.save_npz(system_npz)

    plot_path = out_dir / "community_network.png"
    plot_community_network(system, plot_path)

    # Community composition by region, for the legibility story.
    print("\nCommunity composition (region counts):")
    regions = system.metadata["regions"]
    clusters = system.clusters or []
    for lab in sorted(set(clusters)):
        members = [regions[i] for i in range(system.n) if clusters[i] == lab]
        counts = {r: members.count(r) for r in sorted(set(members))}
        print(f"  {lab}: {counts}")

    print(f"\nSaved:\n  {network_json}\n  {system_json}\n  {system_npz}\n  {plot_path}")


if __name__ == "__main__":
    main()
