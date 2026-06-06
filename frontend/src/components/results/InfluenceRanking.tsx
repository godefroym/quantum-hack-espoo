import { nameOf, ticker, type HardwareData } from "@/lib/results"

/**
 * Institutions ranked by systemic impact: how much extra default probability
 * their failure injects into the rest of the system. A compact companion to the
 * influence map.
 */
export function InfluenceRanking({ data }: { data: HardwareData }) {
  const nodes = [...data.graph.nodes].sort((a, b) => b.influence - a.influence)
  const max = Math.max(...nodes.map((n) => n.influence), 1e-9)

  return (
    <div className="space-y-1.5">
      {nodes.map((n) => (
        <div key={n.i} className="flex items-center gap-3">
          <span
            className="w-12 shrink-0 text-xs font-medium"
            title={nameOf(data, n.i)}
          >
            {ticker(data, n.i)}
          </span>
          <div className="h-3 flex-1 overflow-hidden rounded bg-muted">
            <div
              className="h-full rounded bg-gradient-to-r from-indigo-400 to-cyan-300"
              style={{ width: `${(n.influence / max) * 100}%` }}
            />
          </div>
          <span className="w-12 shrink-0 text-right text-xs tabular-nums text-muted-foreground">
            +{n.influence.toFixed(2)}
          </span>
        </div>
      ))}
      <p className="pt-2 text-xs text-muted-foreground">
        Value is the total extra default probability spread across the other
        institutions when this one fails, summed from the hardware shots.
      </p>
    </div>
  )
}
