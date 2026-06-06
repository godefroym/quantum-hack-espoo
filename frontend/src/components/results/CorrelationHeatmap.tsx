import { useState } from "react"
import { ticker, type HardwareData } from "@/lib/results"

/**
 * Pairwise correlation heatmap from the 100k hardware shots. Off-diagonal
 * structure is the entanglement-induced correlated defaults.
 */
export function CorrelationHeatmap({ data }: { data: HardwareData }) {
  const m = data.pairwise_corr
  const n = data.n_qubits
  const [hover, setHover] = useState<{ i: number; j: number } | null>(null)

  // diverging blue-white-red around 0
  const color = (v: number) => {
    const t = Math.max(-1, Math.min(1, v))
    if (t >= 0) {
      const a = t // 0..1 -> white to cyan
      return `rgb(${Math.round(255 - a * 221)}, ${Math.round(
        255 - a * 22
      )}, ${Math.round(255 - a * 6)})`
    }
    const a = -t // white to rose
    return `rgb(255, ${Math.round(255 - a * 192)}, ${Math.round(255 - a * 161)})`
  }

  const cell = 16
  const grid = n * cell

  return (
    <div className="flex flex-col items-center">
      <svg
        viewBox={`0 0 ${grid} ${grid}`}
        className="h-auto w-full max-w-md"
        onMouseLeave={() => setHover(null)}
      >
        {m.map((row, i) =>
          row.map((v, j) => (
            <rect
              key={`${i}-${j}`}
              x={j * cell}
              y={i * cell}
              width={cell}
              height={cell}
              fill={i === j ? "#1e293b" : color(v)}
              onMouseEnter={() => setHover({ i, j })}
            />
          ))
        )}
        {hover && hover.i !== hover.j && (
          <rect
            x={hover.j * cell}
            y={hover.i * cell}
            width={cell}
            height={cell}
            fill="none"
            stroke="#fff"
            strokeWidth={1.5}
          />
        )}
      </svg>
      <div className="mt-3 flex items-center gap-3 text-xs text-muted-foreground">
        <span>−1</span>
        <span
          className="h-2.5 w-40 rounded"
          style={{
            background:
              "linear-gradient(to right, rgb(255,63,94), rgb(255,255,255), rgb(34,233,249))",
          }}
        />
        <span>+1</span>
        <span className="ml-2">
          {hover && hover.i !== hover.j
            ? `corr(${ticker(data, hover.i)}, ${ticker(data, hover.j)}) = ${m[
                hover.i
              ][hover.j].toFixed(3)}`
            : "pairwise default correlation"}
        </span>
      </div>
    </div>
  )
}
