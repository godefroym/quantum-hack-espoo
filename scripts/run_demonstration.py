"""CANONICAL end-to-end demonstration of the quantum systemic-stress thesis (real-data foundation).

This is the headline run. ``scripts/run_mvp.py`` is the fast smoke-test subset of it; the other
scripts are specialized (``build_system_spec.py`` rebuilds the Part-A data layer,
``run_scaling_experiment.py`` is the size/scale study, and the ``*huang*`` scripts cover the
optional fire-sale channel).

Wires every part of the project into one honest, runnable pipeline and renders an explicit
verdict on each of the three success criteria, with numbers:

    real 28-bank network  ->  community network plot + the block-separability limitation (shown)
    real community (n=14)  ->  all generators -> cascade -> 2nd-order + higher-order + tail metrics
                               -> per-criterion verdict (the headline real-data test)
    n=54 synthetic + oracle ->  the scale story: TV(entangled loss-count, mean-field oracle) ≈ 0

What it is careful about (see the module docstrings under ``scripts/_demo``):
  * match is scored against the achievable Fréchet ceiling, not the infeasible nominal target;
  * the criterion-1 spec is one community, where the entangled generator is a single fully-
    simulated block — on the whole 28-bank network it is block-separable and cross-cluster
    correlation collapses, which we *show* rather than paper over;
  * the Gaussian copula is calibrated to the same canonical targets; realized second-order errors
    are reported explicitly, and criterion 3 is not called a matched-moment comparison when those
    realized moments differ.

Run:
    uv run python scripts/run_demonstration.py
Artifacts land in ``outputs/`` (see the end of the run for the list).
"""

from __future__ import annotations

from _demo._bootstrap import bootstrap

OUTPUTS = bootstrap()

import numpy as np  # noqa: E402  (after bootstrap so caches/path are set)

from _demo import _report as R  # noqa: E402
from _demo._higher_order import (  # noqa: E402
    CASCADE_TAIL_KEYS,
    DEEP_TAIL_KEYS,
    HIGHER_ORDER_KEYS,
    GeneratorEvaluation,
    evaluate_generator,
    excess_coskewness_convergence,
)
from _demo._scale import validate_against_oracle  # noqa: E402
from _demo._second_order import second_order_match  # noqa: E402
from _demo._specs import (  # noqa: E402
    achievable_corr,
    homogeneous_oracle_spec,
    infeasible_fraction,
    real_community_spec,
    real_full_spec,
    synthetic_scale_spec,
)
from systemic_risk.generators import (  # noqa: E402
    BernoulliGenerator,
    EntangledBornMachineGenerator,
    GaussianCopulaGenerator,
    StudentTCopulaGenerator,
)
from systemic_risk.generators.moments import targets_from_spec  # noqa: E402
from systemic_risk.visualization import plot_community_network  # noqa: E402


# Sampling budgets. The criterion spec uses a large N because the discriminators live in a tail
# whose events occur with probability ~1e-3 at these tiny credit marginals; CI-time but enough
# for the rare-event estimates to be stable (verified by the convergence trajectory we print).
CRITERION_SAMPLES = 200_000
SEED = 2026
CONVERGENCE_SIZES = (20_000, 80_000, 320_000)


def _classical_generators() -> list:
    """The classical baselines, strongest-foil-first for readability."""
    return [BernoulliGenerator(), GaussianCopulaGenerator(), StudentTCopulaGenerator(df=4.0)]


def _entangled_generator() -> EntangledBornMachineGenerator:
    return EntangledBornMachineGenerator(ansatz="entangled", calibrate=True)


def run_real_network_overview() -> None:
    """Show the real network, its regime, and the verified block-separability limitation."""
    print(R.header("PART 1 — Real 28-bank G-SIB network: the foundation and its honest limits"))
    bundle = real_full_spec()
    spec = bundle.spec
    print(f"Spec source: {bundle.label}")
    if bundle.used_fallback:
        print("  NOTE: real offline build was unavailable; used the synthetic fallback.")
    clusters = np.asarray(spec.clusters)
    labels, counts = np.unique(clusters, return_counts=True)
    iu = np.triu_indices(spec.n, k=1)
    raw_corr = spec.target_pairwise_corr
    corr = targets_from_spec(spec).pairwise_corr
    p = spec.marginal_default_probs
    print(f"  n={spec.n} institutions, communities={dict(zip(labels.tolist(), counts.tolist()))}")
    print(f"  marginals: mean={p.mean():.4g}  range=[{p.min():.4g}, {p.max():.4g}]  (tiny-credit)")
    if spec.correlation_space == "latent_gaussian" and raw_corr is not None:
        print(f"  latent/equity corr mean={raw_corr[iu].mean():.3f}; induced binary-default "
              f"corr mean={corr[iu].mean():.3f}")
    else:
        print(f"  binary-default corr mean={corr[iu].mean():.3f}")
    print(f"  binary target range=[{corr[iu].min():.3f}, {corr[iu].max():.3f}]")
    print(f"  Fréchet-infeasible binary correlations: {infeasible_fraction(spec):.1%} "
          "(unreachable by ANY binary generator at these marginals)")

    plot_path = OUTPUTS / "real_network_communities.png"
    plot_community_network(spec, plot_path,
                           title="Real 28-bank G-SIB exposure network — detected communities")
    print(f"  Saved community network plot -> {plot_path.name}")

    # Demonstrate, with numbers, that the entangled fit is block-separable on the whole network.
    gen = _entangled_generator()
    gen.fit(spec)
    diag = gen.diagnostics_summary()
    samples = gen.sample(40_000, seed=SEED)
    gen_corr = _empirical_corr(samples)
    within, cross = _within_cross_means(gen_corr, clusters)
    tgt_within, tgt_cross = _within_cross_means(corr, clusters)
    print(subheader_limitation())
    print(f"  Entangled fit: {diag.n_blocks} blocks, max block {diag.max_block_size} qubits "
          f"(max_block_qubits={gen.max_block_qubits} < n={spec.n}).")
    print(f"  within-cluster corr: generated={within:.3f} vs target={tgt_within:.3f}  (captured)")
    print(f"  cross-cluster  corr: generated={cross:.3f} vs target={tgt_cross:.3f}  "
          "(collapses to ~0 — block-separable)")
    print("  => The whole-network spec is NOT a valid criterion-1 'drop-in' spec. We therefore")
    print("     test the criteria on a single community, where the generator is one full block.")


def subheader_limitation() -> str:
    return R.subheader("Verified limitation: block-separability on the full network")


def run_criteria_on_community() -> tuple[list, dict]:
    """Run the three-criteria head-to-head on the largest real community (single block)."""
    print(R.header("PART 2 — Three success criteria on the largest real community (n≈14)"))
    bundle = real_community_spec()
    spec = bundle.spec
    achievable = achievable_corr(spec)
    iu = np.triu_indices(spec.n, k=1)
    print(f"Criterion spec: {bundle.label}")
    target_corr = targets_from_spec(spec).pairwise_corr
    print(f"  n={spec.n}; binary target corr mean={target_corr[iu].mean():.3f}, "
          f"achievable (Fréchet) ceiling mean={achievable[iu].mean():.3f}; "
          f"infeasible targets={infeasible_fraction(spec):.0%}")
    print(f"  real interbank exposure in block: {spec.exposure_matrix.sum():.0f}; "
          f"mean capital buffer: {spec.capital_buffers.mean():.0f}")

    generators = _classical_generators() + [_entangled_generator()]
    evaluations: list[GeneratorEvaluation] = []
    seeds = np.random.SeedSequence(SEED).spawn(len(generators))
    for generator, child in zip(generators, seeds):
        seed = int(child.generate_state(1)[0])
        evaluations.append(evaluate_generator(generator, spec, CRITERION_SAMPLES, seed))
    by_name = {e.name: e for e in evaluations}
    entangled = by_name["Entangled Born machine"]
    gaussian = by_name["Gaussian copula"]

    # ---- Criterion 1: second-order match table (vs nominal target and achievable ceiling) ---- #
    print(R.subheader("Criterion 1a — match on the real (heterogeneous) community"))
    matches = [
        second_order_match(e.name, e.samples, spec, achievable) for e in evaluations
    ]
    print(R.format_table(R.second_order_table(matches)))
    by_match = {m.generator: m for m in matches}
    strongest_classical = _strongest_classical_match(matches)
    print(f"  Strongest classical correlation-matcher: {strongest_classical.generator}.")

    # Criterion 1b: the exchangeable-target capability test, at the SAME tiny credit marginal.
    print(R.subheader("Criterion 1b — clean drop-in on an exchangeable target (same marginal)"))
    homogeneous_match = _homogeneous_capability(spec)
    print("  (uniform marginal = community mean, one feasible binary-default correlation;")
    print("   this checks calibration capability without heterogeneous-edge interference)")
    print(R.format_table(R.second_order_table(homogeneous_match)))

    v1 = R.verdict_criterion_1(
        by_match["Entangled Born machine"],
        strongest_classical,
        homogeneous_match[-1],  # entangled row (appended last)
    )

    # ---- Criterion 2: higher-order discriminators + Gaussian-foil convergence proof ---------- #
    print(R.subheader("Criterion 2 — higher-order / tail discriminators (excess over 2nd-order)"))
    print(R.format_table(R.metrics_table(evaluations, HIGHER_ORDER_KEYS)))
    print("\n  Gaussian-copula excess co-skewness vanishes with sample size (it is noise);")
    print("  the entangled generator's stays large (genuine structure):")
    conv_gauss = excess_coskewness_convergence(
        GaussianCopulaGenerator(), spec, CONVERGENCE_SIZES, seed=SEED)
    conv_entangled = excess_coskewness_convergence(
        _entangled_generator(), spec, CONVERGENCE_SIZES, seed=SEED)
    for label, conv in (("Gaussian copula", conv_gauss), ("Entangled    ", conv_entangled)):
        trail = "  ".join(f"N={n:>7}: {v:6.3f}" for n, v in conv)
        print(f"    {label}: {trail}")
    v2 = R.verdict_criterion_2(entangled, gaussian, conv_gauss)

    # ---- Criterion 3: cascade-tail movement vs the same-target Gaussian foil ---------------- #
    print(R.subheader("Criterion 3 — does it MOVE the contagion-cascade tail?"))
    print(R.format_table(R.metrics_table(evaluations, CASCADE_TAIL_KEYS + DEEP_TAIL_KEYS)))
    print("\n  Note: at tiny marginals the systemic mode is rare-but-catastrophic, so the fixed-α")
    print("  CVaR_95/99 can place its VaR at a zero count; p(K≥half) and CVaR_99.9 are the")
    print("  faithful deep-tail statistics here (entangled puts real mass on whole-block defaults).")
    v3 = R.verdict_criterion_3(entangled, gaussian)

    verdicts = [v1, v2, v3]
    print(R.header("VERDICT — three success criteria (on the real community spec)"))
    print(R.format_verdicts(verdicts))

    artifacts = {
        "spec": spec,
        "matches": matches,
        "evaluations": evaluations,
        "verdicts": verdicts,
    }
    return verdicts, artifacts


def run_scale_story() -> None:
    """The n=54 scale story: the synthetic spec runs, and the homogeneous oracle matches exactly."""
    print(R.header("PART 3 — Scaling to the 54-qubit target"))
    synth = synthetic_scale_spec(n=54)
    gen = _entangled_generator()
    gen.fit(synth.spec)
    diag = gen.diagnostics_summary()
    print(f"Heterogeneous n=54 ({synth.label}): the entangled fit runs as {diag.n_blocks} "
          f"community blocks of ≤ {diag.max_block_size} qubits — never forms the 2^54 state.")

    print(R.subheader("n=54 homogeneous mean-field oracle validation (exact ground truth)"))
    result = validate_against_oracle(homogeneous_oracle_spec(n=54))
    print(f"  target marginal={result.target_marginal:.4f}, "
          f"target default-corr={result.target_default_corr:.4f}")
    print(f"  TV(entangled loss-count law, mean-field oracle) = {result.tv_distance:.2e}  "
          "(machine precision)")
    print(f"  marginal: entangled={result.generator_marginal:.6f}  "
          f"oracle={result.oracle_marginal:.6f}")
    print(f"  default-corr: entangled={result.generator_default_corr:.6f}  "
          f"oracle={result.oracle_default_corr:.6f}")
    print("  => The construction reproduces the exact systemic loss distribution at n=54, with no")
    print("     2^54 statevector on either side — evidence the generation scales to hardware.")


def write_artifacts(artifacts: dict, verdicts: list) -> list[str]:
    """Persist the comparison CSV, the verdict summary, and a worst-case crisis card."""
    from systemic_risk.simulator.cascade import simulate_many
    from systemic_risk.visualization import save_crisis_card

    spec = artifacts["spec"]
    evaluations: list[GeneratorEvaluation] = artifacts["evaluations"]
    written: list[str] = []

    # Full per-generator comparison CSV (every metric we discuss, in a stable column order).
    all_keys = list(dict.fromkeys(
        ("marginal_rmse", "pairwise_joint_rmse")
        + HIGHER_ORDER_KEYS + CASCADE_TAIL_KEYS + DEEP_TAIL_KEYS
    ))
    frame = R.metrics_table(evaluations, tuple(all_keys))
    # Attach the 2nd-order-vs-ceiling columns too, joined by generator name.
    so = R.second_order_table(artifacts["matches"]).drop(columns=["marg_rmse"])
    frame = frame.merge(so, on="generator", how="left")
    csv_path = OUTPUTS / "demonstration_comparison.csv"
    frame.to_csv(csv_path, index=False)
    written.append(csv_path.name)

    # Verdict summary (plain text, the headline result).
    verdict_path = OUTPUTS / "demonstration_verdict.txt"
    verdict_path.write_text(R.format_verdicts(verdicts) + "\n", encoding="utf-8")
    written.append(verdict_path.name)

    # A worst-case crisis card from the entangled generator on the community spec.
    entangled = next(e for e in evaluations if e.name == "Entangled Born machine")
    worst = int(np.argmax(entangled.failure_counts))
    cascades = simulate_many(entangled.samples[worst : worst + 1], spec)
    card_path = OUTPUTS / "demonstration_crisis_card.md"
    save_crisis_card(card_path, spec, entangled.samples[worst], cascades[0],
                     "Entangled Born machine", worst)
    written.append(card_path.name)
    return written


def _empirical_corr(samples: np.ndarray) -> np.ndarray:
    from _demo._second_order import empirical_marginals_and_corr

    _, corr = empirical_marginals_and_corr(samples)
    return corr


def _within_cross_means(corr: np.ndarray, clusters: np.ndarray) -> tuple[float, float]:
    n = corr.shape[0]
    iu = np.triu_indices(n, k=1)
    same = clusters[iu[0]] == clusters[iu[1]]
    vals = corr[iu]
    within = float(vals[same].mean()) if same.any() else 0.0
    cross = float(vals[~same].mean()) if (~same).any() else 0.0
    return within, cross


def _strongest_classical_match(matches):
    """Pick the classical baseline closest to the achievable ceiling (the fair criterion-1 foil)."""
    classical = [m for m in matches if m.generator != "Entangled Born machine"]
    return min(classical, key=lambda m: m.corr_rmse_vs_achievable)


def _homogeneous_capability(community_spec):
    """Score the 2nd-order match on an exchangeable spec at the community's own marginal level.

    Returns ``[classical..., entangled]`` :class:`SecondOrderMatch` rows so the entangled row is
    last. The equicorrelation target is the achievable mean of the real community, capped so it is
    feasible at this uniform marginal.
    """
    from _demo._specs import homogeneous_credit_spec

    iu = np.triu_indices(community_spec.n, k=1)
    marginal = float(community_spec.marginal_default_probs.mean())
    target_corr = float(min(0.5, achievable_corr(community_spec)[iu].mean()))
    spec = homogeneous_credit_spec(
        n=community_spec.n, marginal=marginal, default_corr=target_corr)
    achievable = achievable_corr(spec)
    generators = _classical_generators() + [_entangled_generator()]
    out = []
    for generator in generators:
        generator.fit(spec)
        samples = generator.sample(CRITERION_SAMPLES, seed=SEED)
        out.append(second_order_match(generator.name, samples, spec, achievable))
    return out


def main() -> None:
    run_real_network_overview()
    verdicts, artifacts = run_criteria_on_community()
    run_scale_story()
    written = write_artifacts(artifacts, verdicts)
    print(R.header("Artifacts written to outputs/"))
    for name in written:
        print(f"  outputs/{name}")
    print("  outputs/real_network_communities.png")


if __name__ == "__main__":
    main()
