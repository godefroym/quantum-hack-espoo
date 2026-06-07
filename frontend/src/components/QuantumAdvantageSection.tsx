import { Atom, Calculator, Search } from "lucide-react"

export function QuantumAdvantageSection() {
  return (
    <section
      id="quantum"
      className="relative flex min-h-screen flex-col justify-center border-t border-border px-6 py-24"
    >
      {/* glow */}
      <div className="pointer-events-none absolute inset-0 -z-10">
        <div className="absolute left-1/3 top-1/2 h-[32rem] w-[32rem] -translate-y-1/2 rounded-full bg-fuchsia-500/10 blur-[140px]" />
      </div>

      <div className="mx-auto w-full max-w-5xl">
        <span className="text-sm font-semibold uppercase tracking-widest text-fuchsia-400">
          The two quantum surfaces
        </span>
        <h2 className="mt-4 max-w-3xl text-4xl font-bold tracking-tight sm:text-5xl">
          One qubit per institution.
        </h2>
        <p className="mt-6 max-w-3xl text-lg leading-relaxed text-muted-foreground">
          Each qubit is one entity:{" "}
          <code className="rounded bg-muted px-1.5 py-0.5 text-sm">|0&rangle;</code>{" "}
          survives,{" "}
          <code className="rounded bg-muted px-1.5 py-0.5 text-sm">|1&rangle;</code>{" "}
          defaults.{" "}
          <code className="rounded bg-muted px-1.5 py-0.5 text-sm">Ry</code>{" "}
          rotations encode default tendencies; entangling gates on the exposure
          graph make linked institutions sample from a non-factorized joint
          distribution.
        </p>

        <div className="mt-12 grid gap-6 md:grid-cols-2">
          <div className="rounded-xl border border-indigo-400/40 bg-indigo-500/5 p-7 backdrop-blur">
            <div className="mb-4 inline-flex rounded-lg bg-indigo-500/15 p-3 text-indigo-300">
              <Atom className="size-6" />
            </div>
            <h3 className="text-xl font-semibold">Generation</h3>
            <p className="mt-3 text-sm leading-relaxed text-muted-foreground">
              A quantum circuit Born machine samples{" "}
              <code className="rounded bg-muted px-1 py-0.5">x</code> with
              probability{" "}
              <code className="rounded bg-muted px-1 py-0.5">
                |&langle;x|U(&theta;)|0&rangle;|&sup2;
              </code>
              . Its entangling layers encode a correlated, non-factorized
              distribution, and the <em>same</em> circuit doubles as the
              state-loader for the calculation step.
            </p>
          </div>

          <div className="rounded-xl border border-fuchsia-400/40 bg-fuchsia-500/5 p-7 backdrop-blur">
            <div className="mb-4 inline-flex rounded-lg bg-fuchsia-500/15 p-3 text-fuchsia-300">
              <Calculator className="size-6" />
            </div>
            <h3 className="text-xl font-semibold">Calculation</h3>
            <p className="mt-3 text-sm leading-relaxed text-muted-foreground">
              The cascade becomes a reversible oracle{" "}
              <code className="rounded bg-muted px-1 py-0.5">
                |x&rangle;|0&rangle; &rarr; |x&rangle;|severity &ge; s&rangle;
              </code>
              . Evaluated over the loaded distribution in superposition, it
              unlocks amplitude estimation and amplitude amplification.
            </p>
          </div>
        </div>

        <div className="mt-10 grid gap-6 md:grid-cols-2">
          <div className="rounded-xl border border-border bg-card/40 p-6 backdrop-blur">
            <div className="mb-3 inline-flex rounded-lg bg-muted p-2.5 text-foreground">
              <Calculator className="size-5" />
            </div>
            <h4 className="font-semibold">Quantum Amplitude Estimation</h4>
            <p className="mt-2 text-sm leading-relaxed text-muted-foreground">
              Estimates{" "}
              <code className="rounded bg-muted px-1 py-0.5">P(severe)</code>{" "}
              and CVaR using{" "}
              <code className="rounded bg-muted px-1 py-0.5">
                O(1/(&epsilon;&middot;&radic;a))
              </code>{" "}
              oracle queries vs{" "}
              <code className="rounded bg-muted px-1 py-0.5">
                O(1/(&epsilon;&sup2;&middot;a))
              </code>{" "}
              for Monte Carlo: a quadratic reduction in queries, larger for
              rarer events. Implemented as an exact statevector simulation; the
              advantage is in query count, not wall-clock time.
            </p>
          </div>

          <div className="rounded-xl border border-border bg-card/40 p-6 backdrop-blur">
            <div className="mb-3 inline-flex rounded-lg bg-muted p-2.5 text-foreground">
              <Search className="size-5" />
            </div>
            <h4 className="font-semibold">Amplitude Amplification</h4>
            <p className="mt-2 text-sm leading-relaxed text-muted-foreground">
              Surfaces rare, severe scenarios in about{" "}
              <code className="rounded bg-muted px-1 py-0.5">O(1/&radic;a)</code>{" "}
              iterations versus{" "}
              <code className="rounded bg-muted px-1 py-0.5">O(1/a)</code>{" "}
              classical draws, for worst-case scenario search.
            </p>
          </div>
        </div>

        <pre className="mt-10 overflow-x-auto rounded-xl border border-border bg-[#0b0f1a] p-6 text-sm leading-relaxed text-cyan-200/90">
{`A = U_QCBM (load P(x)) -> U_severity (cascade oracle) -> mark severe
  => QAE : quantify P(severe), CVaR        (quadratic, deep-tail amplified)
  => AA  : discover the worst plausible scenario`}
        </pre>
      </div>
    </section>
  )
}
