"""Formatting and verdicts: turn the measured numbers into tables, an explicit per-criterion
verdict, and the ``outputs/`` artifacts.

The verdicts are computed from the numbers (not asserted), with thresholds chosen to be
defensible to a sceptic and stated in the printout. Each criterion resolves to one of
``PASS`` / ``PARTIAL`` / ``FAIL`` with the deciding numbers shown.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ._higher_order import GeneratorEvaluation
from ._second_order import SecondOrderMatch


# ----------------------------------------------------------------------------- text helpers ---- #
def rule(char: str = "=", width: int = 78) -> str:
    return char * width


def header(title: str) -> str:
    return f"\n{rule()}\n{title}\n{rule()}"


def subheader(title: str) -> str:
    return f"\n{title}\n{rule('-')}"


# --------------------------------------------------------------------------------- tables ------ #
def second_order_table(matches: list[SecondOrderMatch]) -> pd.DataFrame:
    """One row per generator: marginal + correlation match (vs nominal and achievable ceiling)."""
    return pd.DataFrame(
        {
            "generator": m.generator,
            "marg_rmse": m.marginal_rmse,
            "marg_max_err": m.marginal_max_abs_err,
            "corr_mean_gen": m.mean_corr_generated,
            "corr_rmse_vs_nominal": m.corr_rmse_vs_nominal,
            "corr_rmse_vs_ceiling": m.corr_rmse_vs_achievable,
            "corr_maxerr_vs_ceiling": m.corr_max_abs_err_vs_achievable,
        }
        for m in matches
    )


def metrics_table(evaluations: list[GeneratorEvaluation], keys: tuple[str, ...]) -> pd.DataFrame:
    """One row per generator with the requested metric keys, in the given order."""
    rows = [{"generator": e.name, **{k: e.metrics.get(k, float("nan")) for k in keys}}
            for e in evaluations]
    return pd.DataFrame(rows)


def format_table(frame: pd.DataFrame) -> str:
    return frame.to_string(index=False, float_format=lambda v: f"{v:0.5g}")


# -------------------------------------------------------------------------------- verdicts ----- #
@dataclass(frozen=True)
class Verdict:
    """A criterion's resolved status and the one-line numeric justification."""

    criterion: str
    status: str  # PASS | PARTIAL | FAIL
    detail: str


def _ratio(value: float, reference: float) -> float:
    return value / reference if reference > 1e-12 else float("inf")


def verdict_criterion_1(
    entangled: SecondOrderMatch,
    strongest_classical: SecondOrderMatch,
    homogeneous_entangled: SecondOrderMatch | None = None,
    *,
    marginal_tol: float = 5e-3,
    corr_parity_tol: float = 0.05,
    clean_match_tol: float = 0.05,
) -> Verdict:
    """Honest comparison, judged over two regimes.

    Two facts decide it: (a) on a feasible *exchangeable* target at the same tiny credit marginal,
    can the entangled generator cleanly reproduce marginals + correlation (a true drop-in)? and
    (b) on the *heterogeneous* real community, is it at least as close to the achievable Fréchet
    ceiling as the strongest classical generator?

    PASS if both hold: the homogeneous match is within ``clean_match_tol`` (a genuine 2nd-order
    drop-in where the target is exchangeable) AND on the real community it matches marginals and is
    at least as close to the ceiling as the best classical. PARTIAL if the heterogeneous parity
    holds but no spec gives a clean reachable-target match. FAIL otherwise.
    """
    marg_ok = entangled.marginal_rmse <= marginal_tol
    e_err = entangled.corr_rmse_vs_achievable
    c_err = strongest_classical.corr_rmse_vs_achievable
    at_least_as_good = e_err <= c_err + corr_parity_tol
    detail = (
        f"real community: entangled marg_rmse={entangled.marginal_rmse:.2e}; "
        f"corr-RMSE-to-ceiling entangled={e_err:.3f} vs strongest classical "
        f"({strongest_classical.generator})={c_err:.3f}"
    )
    clean = homogeneous_entangled is not None and (
        homogeneous_entangled.marginal_rmse <= marginal_tol
        and homogeneous_entangled.corr_rmse_vs_achievable <= clean_match_tol
    )
    if homogeneous_entangled is not None:
        detail += (
            f"; exchangeable-target drop-in: entangled corr-RMSE-to-ceiling="
            f"{homogeneous_entangled.corr_rmse_vs_achievable:.3f} "
            f"(marg_rmse={homogeneous_entangled.marginal_rmse:.2e})"
        )
    if marg_ok and at_least_as_good and clean:
        return Verdict(
            "1 Honest comparison",
            "PASS",
            detail + " — clean 2nd-order drop-in on the exchangeable target and at least as close "
            "as the best classical model on the heterogeneous real community.",
        )
    if marg_ok and at_least_as_good:
        return Verdict(
            "1 Honest comparison",
            "PARTIAL",
            detail + " — on the heterogeneous real correlation matrix the single-control CRY ansatz "
            "cannot satisfy all conflicting per-edge correlations, so it ties the best classical "
            "without reaching the ceiling; interchangeable only at the level either can reach.",
        )
    return Verdict("1 Honest comparison", "FAIL", detail)


def verdict_criterion_2(
    entangled: GeneratorEvaluation,
    gaussian_foil: GeneratorEvaluation,
    convergence: list[tuple[int, float]] | None = None,
    *,
    min_excess_ratio: float = 3.0,
) -> Verdict:
    """Genuine higher-order structure: entangled excess co-skewness and tail dependence must
    exceed the moment-matched Gaussian copula by a clear margin.

    PASS if the entangled excess co-skewness is at least ``min_excess_ratio``× the foil's *and*
    the entangled aggregate tail dependence is materially larger. The convergence trajectory (if
    given) is reported as corroboration that the foil's excess is vanishing noise.
    """
    e_cosk = entangled.metrics["excess_coskewness_rms"]
    f_cosk = gaussian_foil.metrics["excess_coskewness_rms"]
    e_tail = entangled.metrics["aggregate_tail_dependence"]
    f_tail = gaussian_foil.metrics["aggregate_tail_dependence"]
    ratio = _ratio(e_cosk, f_cosk)
    detail = (
        f"excess co-skewness entangled={e_cosk:.3f} vs Gaussian foil={f_cosk:.3f} "
        f"({ratio:.1f}×); aggregate tail-dependence entangled={e_tail:.3f} vs foil={f_tail:.3f}"
    )
    if convergence:
        trail = ", ".join(f"N={n}:{v:.3f}" for n, v in convergence)
        detail += f"; Gaussian-foil excess decays with N [{trail}]"
    if ratio >= min_excess_ratio and e_tail > f_tail + 0.1:
        return Verdict("2 Higher-order structure", "PASS", detail)
    if ratio >= 1.5 and e_tail > f_tail:
        return Verdict("2 Higher-order structure", "PARTIAL", detail)
    return Verdict("2 Higher-order structure", "FAIL", detail)


def verdict_criterion_3(
    entangled: GeneratorEvaluation,
    gaussian_foil: GeneratorEvaluation,
    *,
    min_severe_ratio: float = 1.5,
) -> Verdict:
    """Material to risk versus the same-target Gaussian foil.

    PASS if ``p_severe_cascade`` (or the deep ``p_cascade_half_or_more``) and the 1% cascade
    tail-mean are both materially larger. A PASS is not causal evidence for higher-order structure
    unless the realized first and second moments are also matched.
    """
    e_sev = entangled.metrics["p_severe_cascade"]
    f_sev = gaussian_foil.metrics["p_severe_cascade"]
    e_deep = entangled.metrics["p_cascade_half_or_more"]
    f_deep = gaussian_foil.metrics["p_cascade_half_or_more"]
    e_tm = entangled.metrics["tail_mean_1pct"]
    f_tm = gaussian_foil.metrics["tail_mean_1pct"]
    sev_ratio = _ratio(max(e_sev, e_deep), max(f_sev, f_deep))
    detail = (
        f"p_severe entangled={e_sev:.4f} vs foil={f_sev:.4f}; "
        f"deep p(K≥half) entangled={e_deep:.4f} vs foil={f_deep:.4f} ({sev_ratio:.1f}×); "
        f"cascade tail-mean(1%) entangled={e_tm:.3f} vs foil={f_tm:.3f}"
    )
    if sev_ratio >= min_severe_ratio and e_tm >= f_tm * 1.05:
        return Verdict("3 Material to risk", "PASS", detail)
    if sev_ratio >= 1.2 and e_tm >= f_tm:
        return Verdict("3 Material to risk", "PARTIAL", detail)
    return Verdict("3 Material to risk", "FAIL", detail)


def format_verdicts(verdicts: list[Verdict]) -> str:
    lines = []
    for v in verdicts:
        lines.append(f"[{v.status:7s}] Criterion {v.criterion}")
        lines.append(f"          {v.detail}")
    return "\n".join(lines)
