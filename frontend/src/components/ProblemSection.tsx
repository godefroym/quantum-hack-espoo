import { Dices, PencilRuler, Spline } from "lucide-react"

const tools = [
  {
    icon: PencilRuler,
    title: "Hand-designed scenarios",
    body: "Expert-crafted shocks, like a major bank failing. Intuitive and legible, but few in number and blind to the combinations nobody thought to write down.",
  },
  {
    icon: Dices,
    title: "Independent sampling",
    body: "Draw each institution's default at its own rate. Simple and fast, but it treats failures as unrelated and misses the correlated shocks that define real crises.",
  },
  {
    icon: Spline,
    title: "Copulas",
    body: "Gaussian and Student-t models impose a dependency structure across defaults. Strong, standard baselines that capture pairwise dependence, but not higher-order joint structure.",
  },
]

export function ProblemSection() {
  return (
    <section
      id="problem"
      className="relative flex min-h-screen flex-col justify-center border-t border-border px-6 py-24"
    >
      <div className="mx-auto w-full max-w-5xl">
        <span className="text-sm font-semibold uppercase tracking-widest text-indigo-400">
          The Problem
        </span>
        <h2 className="mt-4 max-w-3xl text-4xl font-bold tracking-tight sm:text-5xl">
          Institutions fail. How does this cascade to other institutions?
        </h2>
        <p className="mt-6 max-w-3xl text-lg leading-relaxed text-muted-foreground">
          Banks and corporates are linked by directed financial exposures, so
          one institution&rsquo;s failure becomes another&rsquo;s loss. Stress
          testing asks which initial defaults set off the most severe downstream
          cascades. Three families of tools are used to generate those
          scenarios.
        </p>

        <div className="mt-14 grid gap-6 md:grid-cols-3">
          {tools.map(({ icon: Icon, title, body }) => (
            <div
              key={title}
              className="rounded-xl border border-border bg-card/40 p-6 backdrop-blur transition-colors hover:border-indigo-400/40"
            >
              <div className="mb-4 inline-flex rounded-lg bg-indigo-500/10 p-3 text-indigo-300">
                <Icon className="size-6" />
              </div>
              <h3 className="text-lg font-semibold">{title}</h3>
              <p className="mt-2 text-sm leading-relaxed text-muted-foreground">
                {body}
              </p>
            </div>
          ))}
        </div>

        <p className="mt-14 max-w-3xl rounded-xl border-l-2 border-indigo-400 bg-card/30 px-6 py-5 text-lg italic leading-relaxed text-foreground/90">
          &ldquo;Under matched marginals and pairwise dependencies, which
          generator reaches the most severe plausible contagion tails?&rdquo;
        </p>
      </div>
    </section>
  )
}
