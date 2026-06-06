import { Check } from "lucide-react"

const generators = [
  {
    name: "Independent Bernoulli",
    kind: "Baseline",
    body: "Each institution defaults independently at its marginal rate, the no-correlation reference point.",
    quantum: false,
  },
  {
    name: "Gaussian copula",
    kind: "Baseline",
    body: "Thresholds a multivariate-normal latent to induce correlated defaults. A strong, standard baseline.",
    quantum: false,
  },
  {
    name: "Student-t copula",
    kind: "Baseline",
    body: "Heavier tails than Gaussian, so more joint extreme defaults and a tougher tail baseline.",
    quantum: false,
  },
  {
    name: "Entangled Born machine",
    kind: "Quantum",
    body: "A quantum circuit Born machine whose entangling layers sample a non-factorized joint default distribution.",
    quantum: true,
  },
]

const matched = [
  "Same real 28-bank G-SIB exposure network",
  "Same marginal default probabilities",
  "Same pairwise dependency targets",
  "Same deterministic cascade simulator",
]

export function ApproachSection() {
  return (
    <section
      id="approach"
      className="relative flex min-h-screen flex-col justify-center border-t border-border px-6 py-24"
    >
      <div className="mx-auto w-full max-w-5xl">
        <span className="text-sm font-semibold uppercase tracking-widest text-cyan-400">
          The Approach
        </span>
        <h2 className="mt-4 max-w-3xl text-4xl font-bold tracking-tight sm:text-5xl">
          A generator-agnostic benchmark that keeps the comparison honest
        </h2>
        <p className="mt-6 max-w-3xl text-lg leading-relaxed text-muted-foreground">
          Hold everything fixed except the scenario generator. Whatever
          differences show up in the contagion tail are attributable to the
          generator alone, not to a different network, different
          marginals, or a different simulator.
        </p>

        <div className="mt-10 flex flex-wrap gap-3">
          {matched.map((m) => (
            <span
              key={m}
              className="inline-flex items-center gap-2 rounded-full border border-border bg-card/40 px-4 py-2 text-sm text-muted-foreground backdrop-blur"
            >
              <Check className="size-4 text-cyan-400" />
              {m}
            </span>
          ))}
        </div>

        <div className="mt-12 grid gap-4 sm:grid-cols-2">
          {generators.map((g) => (
            <div
              key={g.name}
              className={`rounded-xl border p-6 backdrop-blur transition-colors ${
                g.quantum
                  ? "border-cyan-400/40 bg-cyan-500/5 hover:border-cyan-400/70"
                  : "border-border bg-card/40 hover:border-foreground/20"
              }`}
            >
              <div className="flex items-center justify-between">
                <h3 className="text-lg font-semibold">{g.name}</h3>
                <span
                  className={`rounded-full px-2.5 py-0.5 text-xs font-medium ${
                    g.quantum
                      ? "bg-cyan-400/15 text-cyan-300"
                      : "bg-muted text-muted-foreground"
                  }`}
                >
                  {g.kind}
                </span>
              </div>
              <p className="mt-2 text-sm leading-relaxed text-muted-foreground">
                {g.body}
              </p>
            </div>
          ))}
        </div>
      </div>
    </section>
  )
}
