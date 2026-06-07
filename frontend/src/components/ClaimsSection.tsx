import { Check, X } from "lucide-react"

const does = [
  "An entangled generator that loads a correlated default distribution, run on real quantum hardware (20 qubits, 100k shots).",
  "Hardware output matching the target distribution and the exact simulator within sampling error.",
  "A quadratic oracle-query reduction for P(severe) / CVaR estimation, in exact statevector simulation.",
  "A hardware-ready oracle: one qubit per institution plus cascade-comparison ancillas.",
]

const doesNot = [
  "No quantum speedup for a single cascade; the benefit is over the distribution and the search space.",
  "The cascade is classical; it enters the quantum circuit only as an oracle.",
  "No wall-clock speedup is claimed; the advantage is in oracle-query count.",
  "No quantum-linear-algebra (HHL) speedup; reverse-stress-test optimisation (QAOA) is heuristic.",
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
          answer. The precise boundary is below.
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
