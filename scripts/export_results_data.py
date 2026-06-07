"""Export the real IBM quantum-hardware run into a JSON the web demo can read.

Reads the committed report + sample bitstrings from ``outputs/results/`` and
derives the views the results page presents (marginals, correlation heatmap,
default-count distribution, top sampled scenarios). The .npz is binary, so this
bakes everything into ``frontend/public/results/hardware.json``.

Usage:
    uv run python scripts/export_results_data.py
"""

from __future__ import annotations

import csv
import json
from itertools import combinations
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "outputs" / "results"
ROSTER = ROOT / "data" / "external" / "banks" / "gsib_roster.csv"
OUT_DIR = ROOT / "frontend" / "public" / "results"


def load_institutions(n: int) -> list[dict]:
    """Label the n qubits with the project's real G-SIB roster (first n banks).

    The hardware smoke test ran on a synthetic n-institution spec, so these are
    the roster names the network is built around, used as the institution labels.
    """
    rows: list[dict] = []
    if ROSTER.exists():
        with ROSTER.open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    region_name = {
        "US": "North America",
        "CA": "North America",
        "UK": "UK & Europe",
        "EU": "UK & Europe",
        "JP": "Japan",
        "LatAm": "Energy & LatAm",
    }
    out: list[dict] = []
    for i in range(n):
        if i < len(rows):
            region = rows[i].get("region", "")
            out.append(
                {
                    "ticker": rows[i]["ticker"],
                    "name": rows[i]["name"],
                    "region": region,
                    "regionName": region_name.get(region, region or "Other"),
                }
            )
        else:
            out.append(
                {
                    "ticker": f"Q{i + 1}",
                    "name": f"Institution {i + 1}",
                    "region": "",
                    "regionName": "Other",
                }
            )
    return out


# headquarters coordinates [lat, lon] for the first 20 roster institutions
HQ_COORDS: dict[str, tuple[float, float]] = {
    "JPM": (40.755, -73.976),  # New York
    "BAC": (35.227, -80.843),  # Charlotte
    "C": (40.721, -74.009),  # New York
    "WFC": (37.791, -122.402),  # San Francisco
    "GS": (40.715, -74.013),  # New York
    "MS": (40.764, -73.979),  # New York
    "USB": (44.977, -93.271),  # Minneapolis
    "PNC": (40.441, -79.994),  # Pittsburgh
    "TFC": (35.224, -80.840),  # Charlotte
    "BK": (40.711, -74.011),  # New York
    "STT": (42.353, -71.055),  # Boston
    "HSBC": (51.505, -0.019),  # London (Canary Wharf)
    "BCS": (51.504, -0.017),  # London
    "LYG": (51.513, -0.091),  # London
    "DB": (50.114, 8.671),  # Frankfurt
    "SAN": (40.404, -3.873),  # Boadilla del Monte, Madrid
    "BBVA": (43.263, -2.935),  # Bilbao
    "ING": (52.372, 4.896),  # Amsterdam
    "UBS": (47.370, 8.541),  # Zurich
    "MUFG": (35.680, 139.764),  # Tokyo
}


def write_prototype_shots(samples: np.ndarray, report: dict) -> None:
    """Emit the raw hardware shots + node metadata for the embedded prototype.

    The prototype reduces these to its own sufficient statistics. Each shot is a
    bitmask (bit i set = institution i defaulted). Only consumed at build time.
    """
    proto_src = (
        ROOT / "prototyping-for-quantum" / "prototyping-for-quantum" / "src"
    )
    if not proto_src.exists():
        return
    shots, n = samples.shape
    masks = [int(sum(int(b) << i for i, b in enumerate(row))) for row in samples]
    inst = load_institutions(n)
    target = report.get("target_marginals", [0.0] * n)
    nodes = []
    for i in range(n):
        ticker = inst[i]["ticker"]
        coord = HQ_COORDS.get(ticker)
        nodes.append(
            {
                "label": ticker,
                "sector": inst[i].get("region") or "NA",
                "sectorName": inst[i].get("regionName") or "Other",
                "pd": float(target[i]) if i < len(target) else 0.0,
                "lat": coord[0] if coord else None,
                "lon": coord[1] if coord else None,
            }
        )
    payload = {
        "backend": report.get("backend", ""),
        "jobId": report.get("job_id", ""),
        "shots": masks,
        "nodes": nodes,
    }
    (proto_src / "quantum-shots.json").write_text(json.dumps(payload))
    print(f"  wrote {proto_src / 'quantum-shots.json'} ({shots} shots)")


def find_run() -> tuple[Path, Path]:
    reports = sorted(RESULTS_DIR.glob("*_report.json"))
    if not reports:
        raise SystemExit(f"no *_report.json in {RESULTS_DIR}")
    report = reports[-1]
    samples = report.with_name(report.name.replace("_report.json", "_samples.npz"))
    if not samples.exists():
        raise SystemExit(f"missing samples file {samples}")
    return report, samples


def compute_posteriors(
    samples: np.ndarray,
    min_support: int = 100,
    max_b: int = 3,
    pool: int = 40,
) -> list[dict]:
    """Mine conditional default rules P(A=1 | all of B = 1) from the shots.

    A is one institution failing; B is a collection of other institutions
    failing together. We require B to occur in at least ``min_support`` shots so
    the conditional estimate is reliable. Returns a pooled candidate set that
    supports both rankings the UI offers (by posterior and by lift): the union
    of the top ``pool`` rules by P(A|B) and the top ``pool`` by lift. The
    frontend sorts, de-duplicates per target A, and slices.
    """
    shots, n = samples.shape
    base = samples.mean(axis=0)  # marginal P(A=1)
    cols = [samples[:, i].astype(bool) for i in range(n)]

    rules: list[dict] = []
    for size in range(1, max_b + 1):
        for B in combinations(range(n), size):
            mask = cols[B[0]].copy()
            for b in B[1:]:
                mask &= cols[b]
            support = int(mask.sum())
            if support < min_support:
                continue
            csum = samples[mask].sum(axis=0)
            for a in range(n):
                if a in B:
                    continue
                joint = int(csum[a])
                p = joint / support
                lift = p / base[a] if base[a] > 0 else 0.0
                rules.append(
                    {
                        "a": a,
                        "b": list(B),
                        "p": round(p, 4),
                        "support": support,
                        "joint": joint,
                        "baseline": round(float(base[a]), 4),
                        "lift": round(float(lift), 2),
                    }
                )

    by_p = sorted(rules, key=lambda r: (r["p"], r["support"]), reverse=True)[:pool]
    by_lift = sorted(rules, key=lambda r: (r["lift"], r["support"]), reverse=True)[:pool]

    seen: set = set()
    out: list[dict] = []
    for r in [*by_p, *by_lift]:
        key = (r["a"], tuple(r["b"]))
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


def compute_graph(
    samples: np.ndarray,
    top_corr: int = 8,
    edge_min_strength: float = 0.25,
    max_edges: int = 60,
) -> dict:
    """Build the systemic-influence graph from the joint distribution.

    * node ``influence`` = total extra default probability injected into the
      rest of the system when that institution fails:
      sum_j max(0, P(j | i) - P(j)). Drives node size.
    * an ``edge`` joins two institutions when one failing makes the other
      likely to fail: strength = max(P(j | i), P(i | j)). Drives edge presence
      and thickness.
    * ``top_corr_pairs`` are the most strongly correlated pairs, for colour
      coding.
    """
    shots, n = samples.shape
    base = samples.mean(axis=0)
    joint = (samples.T @ samples) / shots  # P(i & j)
    cond = np.zeros((n, n))  # cond[i, j] = P(j | i)
    for i in range(n):
        if base[i] > 0:
            cond[i] = joint[i] / base[i]
    corr = np.nan_to_num(np.corrcoef(samples.T))

    nodes = []
    for i in range(n):
        infl = sum(max(0.0, cond[i][j] - base[j]) for j in range(n) if j != i)
        nodes.append(
            {
                "i": i,
                "influence": round(float(infl), 4),
                "baseline": round(float(base[i]), 4),
            }
        )

    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            strength = max(cond[i][j], cond[j][i])
            pairs.append((strength, i, j))
    pairs.sort(reverse=True)
    edges = []
    for strength, i, j in pairs:
        if strength < edge_min_strength or len(edges) >= max_edges:
            break
        edges.append(
            {
                "i": i,
                "j": j,
                "strength": round(float(strength), 4),
                "corr": round(float(corr[i][j]), 4),
                "p_j_given_i": round(float(cond[i][j]), 4),
                "p_i_given_j": round(float(cond[j][i]), 4),
            }
        )

    cpairs = sorted(
        ((float(corr[i][j]), i, j) for i in range(n) for j in range(i + 1, n)),
        reverse=True,
    )
    top_corr_pairs = [
        {"i": i, "j": j, "corr": round(c, 4)} for c, i, j in cpairs[:top_corr]
    ]

    return {"nodes": nodes, "edges": edges, "top_corr_pairs": top_corr_pairs}


def wilson(k: np.ndarray, n: int, z: float = 1.96) -> tuple[np.ndarray, np.ndarray]:
    """Wilson score interval for a binomial proportion k/n (vectorised)."""
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return np.clip(center - half, 0, 1), np.clip(center + half, 0, 1)


def survival_with_ci(counts: np.ndarray, n_nodes: int, shots: int) -> dict:
    s_vals = np.arange(n_nodes + 1)
    k = np.array([(counts >= s).sum() for s in s_vals])  # exceedance counts
    surv = k / shots
    lo, hi = wilson(k, shots)
    return {
        "survival": [round(float(x), 6) for x in surv],
        "lo": [round(float(x), 6) for x in lo],
        "hi": [round(float(x), 6) for x in hi],
    }


def node_tail_joint(X: np.ndarray, counts: np.ndarray, n_nodes: int) -> list[list[float]]:
    """P(node_i = 1 AND total >= s) for each severity s (rows) and node (cols)."""
    shots = X.shape[0]
    out = []
    for s in range(n_nodes + 1):
        mask = counts >= s
        out.append([round(float(v), 6) for v in (X[mask].sum(axis=0) / shots)])
    return out


def compute_tail(samples: np.ndarray, copula_shots: int = 100000) -> dict:
    """Quantum (hardware) vs copula baselines on the default-count survival
    function, with binomial confidence intervals so the tail divergence can be
    judged against sampling noise. Copulas are matched to the hardware's exact
    marginals and pairwise co-default probabilities.
    """
    from systemic_risk.generators import (
        GaussianCopulaGenerator,
        StudentTCopulaGenerator,
    )
    from systemic_risk.spec import SystemSpec

    n_q, n = samples.shape
    marg = samples.mean(axis=0)
    joint = (samples.T @ samples) / n_q

    spec = SystemSpec(
        node_names=[f"I{i}" for i in range(n)],
        node_types=["bank"] * n,
        exposure_matrix=np.zeros((n, n)),
        capital_buffers=np.ones(n),
        marginal_default_probs=marg,
        target_joint_probs=joint,
    )

    cq = samples.sum(axis=1)
    series: dict[str, dict] = {
        "quantum": {
            **survival_with_ci(cq, n, n_q),
            "node_joint": node_tail_joint(samples, cq, n),
        }
    }
    for key, Gen in [
        ("gaussian", GaussianCopulaGenerator),
        ("student_t", StudentTCopulaGenerator),
    ]:
        gen = Gen()
        gen.fit(spec)
        Y = gen.sample(copula_shots, seed=1)
        cc = Y.sum(axis=1)
        entry = survival_with_ci(cc, n, copula_shots)
        if key == "gaussian":
            entry["node_joint"] = node_tail_joint(Y, cc, n)
        series[key] = entry

    return {
        "s_values": list(range(n + 1)),
        "shots": {"quantum": int(n_q), "copula": int(copula_shots)},
        "series": series,
    }


def main() -> None:
    report_path, samples_path = find_run()
    report = json.loads(report_path.read_text())
    samples = np.load(samples_path)["samples"].astype(int)  # (shots, n_qubits)
    shots, n = samples.shape

    # default-count distribution (how many institutions default per shot)
    counts = samples.sum(axis=1)
    count_hist = np.bincount(counts, minlength=n + 1).astype(int).tolist()

    # pairwise correlation + joint default probability from the hardware samples
    corr = np.corrcoef(samples.T)
    corr = np.nan_to_num(corr)
    joint = (samples.T @ samples) / shots  # P(i=1 & j=1)

    # most frequent joint-default scenarios (bitstrings)
    rows = [tuple(int(x) for x in r) for r in samples]
    uniq: dict[tuple, int] = {}
    for r in rows:
        uniq[r] = uniq.get(r, 0) + 1
    top = sorted(uniq.items(), key=lambda kv: kv[1], reverse=True)[:16]
    top_patterns = [
        {
            "indices": [i for i, b in enumerate(pat) if b],
            "count": c,
            "freq": c / shots,
        }
        for pat, c in top
    ]

    out = {
        # provenance / circuit
        "backend": report["backend"],
        "job_id": report["job_id"],
        "shots": shots,
        "n_qubits": n,
        "institutions": load_institutions(n),
        "max_degree": report.get("max_degree"),
        "entanglers": report.get("entanglers"),
        "entanglement_depth": report.get("entanglement_depth"),
        "circuit_depth": report.get("circuit_depth"),
        "two_qubit_gates": report.get("two_qubit_gates"),
        "circuit_operations": report.get("circuit_operations", {}),
        "exact_ground_truth": report.get("exact_ground_truth", False),
        # fidelity
        "marginal_rmse_vs_target": report["marginal_rmse_vs_target"],
        "pairwise_joint_rmse_vs_target": report["pairwise_joint_rmse_vs_target"],
        "marginal_rmse_vs_ideal": report["marginal_rmse_vs_ideal"],
        "pairwise_joint_rmse_vs_ideal": report["pairwise_joint_rmse_vs_ideal"],
        # marginals (per institution)
        "target_marginals": report["target_marginals"],
        "hardware_marginals": report["hardware_marginals"],
        "ideal_marginals": report["ideal_marginals"],
        # derived distributions
        "default_count_hist": count_hist,
        "mean_defaults": float(counts.mean()),
        "expected_defaults_target": float(np.sum(report["target_marginals"])),
        "pairwise_corr": [[round(float(x), 4) for x in row] for row in corr],
        "pairwise_joint": [[round(float(x), 5) for x in row] for row in joint],
        "top_patterns": top_patterns,
        "n_unique_patterns": len(uniq),
        "posteriors": compute_posteriors(samples),
        "graph": compute_graph(samples),
        "tail": compute_tail(samples),
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "hardware.json").write_text(json.dumps(out))
    print(f"wrote {OUT_DIR / 'hardware.json'}")
    print(
        f"  backend={out['backend']} n={n} shots={shots} "
        f"unique_patterns={out['n_unique_patterns']} mean_defaults={out['mean_defaults']:.2f}"
    )
    write_prototype_shots(samples, report)


if __name__ == "__main__":
    main()
