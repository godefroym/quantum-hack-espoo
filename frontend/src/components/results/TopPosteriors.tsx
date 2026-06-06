import { useMemo, useState } from "react"
import { ArrowRight, TrendingUp } from "lucide-react"
import { nameOf, ticker, type HardwareData, type Posterior } from "@/lib/results"

type SortKey = "p" | "lift"

/**
 * Conditional failure posteriors mined from the joint distribution: when a
 * collection of institutions B fails, how likely is institution A to fail too?
 * P(A | B), with its lift over the unconditional baseline. Sortable by either
 * metric.
 */
export function TopPosteriors({
  data,
  limit = 5,
}: {
  data: HardwareData
  limit?: number
}) {
  const [sortKey, setSortKey] = useState<SortKey>("p")

  const rows = useMemo(() => {
    const sorted = [...data.posteriors].sort((x, y) =>
      y[sortKey] !== x[sortKey] ? y[sortKey] - x[sortKey] : y.support - x.support
    )
    // strongest rule per target institution, for variety
    const seen = new Set<number>()
    const out: Posterior[] = []
    for (const r of sorted) {
      if (seen.has(r.a)) continue
      seen.add(r.a)
      out.push(r)
      if (out.length >= limit) break
    }
    return out
  }, [data.posteriors, sortKey, limit])

  return (
    <div>
      {/* sort toggle */}
      <div className="mb-4 flex items-center gap-2 text-xs">
        <span className="text-muted-foreground">Sort by</span>
        <div className="flex rounded-lg border border-border bg-background/40 p-0.5">
          <SortButton active={sortKey === "p"} onClick={() => setSortKey("p")}>
            P(A | B)
          </SortButton>
          <SortButton active={sortKey === "lift"} onClick={() => setSortKey("lift")}>
            Lift
          </SortButton>
        </div>
      </div>

      <div className="space-y-3">
        {rows.map((r, idx) => (
          <div
            key={idx}
            className="flex flex-col gap-4 rounded-lg border border-border bg-background/40 p-4 lg:flex-row lg:items-center"
          >
            {/* rule: B -> A */}
            <div className="flex flex-1 flex-wrap items-center gap-2">
              <div className="flex flex-wrap gap-1">
                {r.b.map((i) => (
                  <span
                    key={i}
                    title={nameOf(data, i)}
                    className="rounded-md bg-slate-600/40 px-2 py-1 text-xs font-medium text-slate-200"
                  >
                    {ticker(data, i)}
                  </span>
                ))}
              </div>
              <ArrowRight className="size-4 shrink-0 text-muted-foreground" />
              <span
                title={nameOf(data, r.a)}
                className="rounded-md bg-rose-500/20 px-2 py-1 text-xs font-semibold text-rose-200"
              >
                {ticker(data, r.a)}
              </span>
            </div>

            {/* metrics */}
            <div className="flex items-center gap-4">
              {/* baseline -> conditional bar */}
              <div className="hidden w-36 sm:block">
                <div className="relative h-2 w-full overflow-hidden rounded-full bg-muted">
                  <div
                    className="absolute inset-y-0 left-0 bg-slate-500/70"
                    style={{ width: `${r.baseline * 100}%` }}
                  />
                  <div
                    className="absolute inset-y-0 left-0 rounded-full bg-cyan-400/80"
                    style={{ width: `${r.p * 100}%`, mixBlendMode: "screen" }}
                  />
                </div>
                <div className="mt-1 flex justify-between text-[10px] text-muted-foreground">
                  <span>base {(r.baseline * 100).toFixed(1)}%</span>
                  <span>n={r.support.toLocaleString()}</span>
                </div>
              </div>

              {/* P(A|B) */}
              <div
                className={`min-w-[4.5rem] text-right ${
                  sortKey === "p" ? "" : "opacity-80"
                }`}
              >
                <div className="text-2xl font-bold tabular-nums text-cyan-300">
                  {(r.p * 100).toFixed(1)}%
                </div>
                <div className="text-[10px] uppercase tracking-wide text-muted-foreground">
                  P(A | B)
                </div>
              </div>

              {/* lift, surfaced prominently */}
              <div
                className={`flex flex-col items-center rounded-lg border px-3 py-1.5 ${
                  sortKey === "lift"
                    ? "border-emerald-400/50 bg-emerald-500/15"
                    : "border-emerald-400/25 bg-emerald-500/10"
                }`}
              >
                <div className="inline-flex items-center gap-1 text-2xl font-bold tabular-nums text-emerald-300">
                  <TrendingUp className="size-4" />
                  {r.lift.toFixed(1)}×
                </div>
                <div className="text-[10px] uppercase tracking-wide text-emerald-400/80">
                  lift
                </div>
              </div>
            </div>
          </div>
        ))}
      </div>

      <p className="pt-3 text-xs text-muted-foreground">
        When the institutions on the left fail together, the institution on the
        right fails with probability P(A | B), a{" "}
        <span className="font-medium text-emerald-300">lift</span> over its
        unconditional rate P(A). Estimated from the 100k hardware shots. Each
        rule is conditioned on at least 100 co-failure events.
      </p>
    </div>
  )
}

function SortButton({
  active,
  onClick,
  children,
}: {
  active: boolean
  onClick: () => void
  children: React.ReactNode
}) {
  return (
    <button
      onClick={onClick}
      className={`rounded-md px-2.5 py-1 transition-colors ${
        active
          ? "bg-primary text-primary-foreground"
          : "text-muted-foreground hover:text-foreground"
      }`}
    >
      {children}
    </button>
  )
}
