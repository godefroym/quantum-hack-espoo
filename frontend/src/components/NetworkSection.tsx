const facts = [
  {
    label: "Nodes",
    value: "28 G-SIBs",
    body: "A curated roster of real, publicly listed G-SIB / large banks.",
  },
  {
    label: "Marginals",
    value: "S&P → PD",
    body: "Each bank's public S&P rating mapped to a 1-year PD via the committed Moody's Exhibit-17 table.",
  },
  {
    label: "Correlation",
    value: "755 obs",
    body: "Real daily equity-return correlation (2021-2024), the genuine network signal driving community detection.",
  },
  {
    label: "Edges",
    value: "Reconstructed",
    body: "Bilateral exposures reconstructed from interbank totals (max-entropy or min-density). Confidential matrices are never used.",
  },
  {
    label: "Communities",
    value: "3 stable",
    body: "Greedy-modularity detection yields North America, Europe/UK/LatAm, and Japan, with a mean ARI of 0.96 under perturbation.",
  },
  {
    label: "Pipeline",
    value: "Frozen spec",
    body: "One real dataset → a frozen canonical spec → a legible network, consumed without loss by every generator.",
  },
]

export function NetworkSection() {
  return (
    <section
      id="network"
      className="relative flex min-h-screen flex-col justify-center border-t border-border px-6 py-24"
    >
      <div className="mx-auto w-full max-w-5xl">
        <span className="text-sm font-semibold uppercase tracking-widest text-emerald-400">
          The Real Exposure Network · Part A
        </span>
        <h2 className="mt-4 max-w-3xl text-4xl font-bold tracking-tight sm:text-5xl">
          Grounded in a real 28-bank G-SIB network
        </h2>
        <p className="mt-6 max-w-3xl text-lg leading-relaxed text-muted-foreground">
          The benchmark runs on a real anchor, not a toy. Real banks, real
          ratings, and a real equity-return correlation are reconciled into a
          single frozen, canonical spec that every generator and the cascade
          simulator consume identically.
        </p>

        <div className="mt-12 grid gap-5 sm:grid-cols-2 lg:grid-cols-3">
          {facts.map((f) => (
            <div
              key={f.label}
              className="rounded-xl border border-border bg-card/40 p-6 backdrop-blur transition-colors hover:border-emerald-400/40"
            >
              <div className="text-xs font-semibold uppercase tracking-widest text-emerald-400/80">
                {f.label}
              </div>
              <div className="mt-1 text-2xl font-bold">{f.value}</div>
              <p className="mt-3 text-sm leading-relaxed text-muted-foreground">
                {f.body}
              </p>
            </div>
          ))}
        </div>
      </div>
    </section>
  )
}
