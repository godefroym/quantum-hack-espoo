import { Check, X } from "lucide-react"

const does = [
  "Quantum advantage over the scenario distribution (estimation) and the search space (discovery).",
  "An exact statevector QAE that agrees with the classical Monte-Carlo answer.",
  "A quadratic oracle-query advantage that grows in the deep tail.",
  "A hardware-ready construction: one qubit per institution + cascade-comparison ancillas.",
]

const doesNot = [
  "No quantum speedup for a single cascade; the advantage is distributional.",
  "The cascade is not quantum-simulated except as an oracle.",
  "Reverse-stress-test optimization (QAOA) is heuristic, not proven optimal.",
  "No quantum-linear-algebra (HHL) advantage is claimed, and no wall-clock speedup.",
]

export function ClaimsSection() {
  return (
    <section
      id="claims"
      className="relative flex min-h-screen flex-col justify-center border-t border-border px-6 py-24"
    >
      <div className="mx-auto w-full max-w-5xl">
        <span className="text-sm font-semibold uppercase tracking-widest text-amber-400">
          Honest Scope
        </span>
        <h2 className="mt-4 max-w-3xl text-4xl font-bold tracking-tight sm:text-5xl">
          What we claim, and what we don&rsquo;t
        </h2>
        <p className="mt-6 max-w-3xl text-lg leading-relaxed text-muted-foreground">
          The advantage is in oracle-query count, verified against the classical
          answer. We are precise about the boundary so the result stands up to
          scrutiny.
        </p>

        <div className="mt-12 grid gap-6 md:grid-cols-2">
          <div className="rounded-xl border border-emerald-400/30 bg-emerald-500/5 p-7 backdrop-blur">
            <h3 className="flex items-center gap-2 text-lg font-semibold text-emerald-300">
              <Check className="size-5" /> What this does
            </h3>
            <ul className="mt-4 space-y-3">
              {does.map((d) => (
                <li
                  key={d}
                  className="flex gap-3 text-sm leading-relaxed text-muted-foreground"
                >
                  <Check className="mt-0.5 size-4 shrink-0 text-emerald-400" />
                  {d}
                </li>
              ))}
            </ul>
          </div>

          <div className="rounded-xl border border-rose-400/30 bg-rose-500/5 p-7 backdrop-blur">
            <h3 className="flex items-center gap-2 text-lg font-semibold text-rose-300">
              <X className="size-5" /> What it does not claim
            </h3>
            <ul className="mt-4 space-y-3">
              {doesNot.map((d) => (
                <li
                  key={d}
                  className="flex gap-3 text-sm leading-relaxed text-muted-foreground"
                >
                  <X className="mt-0.5 size-4 shrink-0 text-rose-400" />
                  {d}
                </li>
              ))}
            </ul>
          </div>
        </div>
      </div>
    </section>
  )
}
