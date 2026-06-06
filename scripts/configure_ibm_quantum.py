"""Interactively save IBM Quantum Platform credentials outside the repository."""

from __future__ import annotations

import argparse
from getpass import getpass

from qiskit_ibm_runtime import QiskitRuntimeService


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instance", help="Optional IBM service instance name or CRN")
    parser.add_argument("--name", default="quantum-systemic-risk")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    token = getpass("IBM Quantum API key: ").strip()
    if not token:
        raise SystemExit("No API key provided.")
    QiskitRuntimeService.save_account(
        channel="ibm_quantum_platform",
        token=token,
        instance=args.instance,
        name=args.name,
        overwrite=args.overwrite,
        set_as_default=True,
    )
    print(f"Saved IBM Quantum account {args.name!r} in Qiskit's user configuration.")


if __name__ == "__main__":
    main()
