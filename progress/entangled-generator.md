# Progress Log — Entangled Generator

**Date:** 2026-06-06
**Branch:** `alex/entanglement-generator`
**Scope:** The project's core IP — the entangled, quantum-native scenario generator — built on top of the classical data foundation (see `data-foundation.md`).

## The problem this stage solves

Build an entanglement-structured generator of correlated default scenarios that is an **honest drop-in replacement** for the classical baseline: it reproduces the same marginals and pairwise correlations, so that any difference in downstream results comes from one place only — the **higher-order joint structure** that entanglement can carry and second-order classical models cannot.

What counts as solving it:

1. **Honest comparison** — its first- and second-order statistics match the *strongest* classical generator we have, within tolerance, so the two are genuinely interchangeable at that level. (Beating a weak baseline proves nothing.)
2. **Genuine higher-order structure** — it carries joint-tail dependence that the best classical model cannot reproduce even when calibrated to those same marginals and correlations.
3. **Material to risk** — that structure must measurably move the systemic-risk outcome (the contagion tail), not merely show up as a static distributional difference. A richer joint that leaves the tail unchanged is not an advantage.
