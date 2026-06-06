import { nameOf, ticker, type HardwareData } from "@/lib/results"

/**
 * Most frequently sampled joint-default scenarios. Each row is a bitstring the
 * QPU returned; lit cells are the institutions defaulting together.
 */
export function TopScenarios({ data }: { data: HardwareData }) {
  const n = data.n_qubits
  const max = Math.max(...data.top_patterns.map((p) => p.freq))

  return (
    <div className="space-y-1.5">
      <div className="flex items-center gap-2 pl-1 text-[10px] text-muted-foreground">
        <span className="w-[44%]">{n} institutions</span>
        <span>frequency</span>
      </div>
      {data.top_patterns.map((p, idx) => (
        <div key={idx} className="flex items-center gap-3">
          {/* bitstring cells */}
          <div
            className="grid flex-1 gap-px"
            style={{ gridTemplateColumns: `repeat(${n}, 1fr)` }}
          >
            {Array.from({ length: n }, (_, i) => {
              const on = p.indices.includes(i)
              return (
                <div
                  key={i}
                  className={`h-4 rounded-[2px] ${
                    on ? "bg-rose-500" : "bg-slate-700/40"
                  }`}
                  title={`${ticker(data, i)}: ${nameOf(data, i)}`}
                />
              )
            })}
          </div>
          {/* freq bar */}
          <div className="flex w-28 items-center gap-2">
            <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-muted">
              <div
                className="h-full rounded-full bg-cyan-400"
                style={{ width: `${(p.freq / max) * 100}%` }}
              />
            </div>
            <span className="w-12 shrink-0 text-right text-[11px] tabular-nums text-muted-foreground">
              {(p.freq * 100).toFixed(2)}%
            </span>
          </div>
        </div>
      ))}
      <p className="pt-2 text-xs text-muted-foreground">
        The all-survive scenario dominates; the long tail of{" "}
        {data.n_unique_patterns.toLocaleString()} distinct correlated patterns is
        the distribution the generator loads for tail-risk estimation.
      </p>
    </div>
  )
}
