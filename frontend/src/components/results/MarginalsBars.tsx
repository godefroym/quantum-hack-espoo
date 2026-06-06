import { ticker, type HardwareData } from "@/lib/results"
import { SERIES } from "@/lib/results"

const W = 760
const H = 300
const PAD = { l: 44, r: 12, t: 12, b: 48 }

/** Grouped bars: target, ideal, and hardware marginal per institution. */
export function MarginalsBars({ data }: { data: HardwareData }) {
  const n = data.n_qubits
  const series = [
    { key: "target", vals: data.target_marginals },
    { key: "ideal", vals: data.ideal_marginals },
    { key: "hardware", vals: data.hardware_marginals },
  ] as const
  const ymax = Math.max(
    ...data.target_marginals,
    ...data.ideal_marginals,
    ...data.hardware_marginals
  )
  const plotW = W - PAD.l - PAD.r
  const plotH = H - PAD.t - PAD.b
  const group = plotW / n
  const barW = (group * 0.8) / series.length
  const sy = (v: number) => PAD.t + plotH - (v / ymax) * plotH

  return (
    <div>
      <svg viewBox={`0 0 ${W} ${H}`} className="h-auto w-full">
        {[0, 0.25, 0.5, 0.75, 1].map((f) => {
          const v = ymax * f
          return (
            <g key={f}>
              <line
                x1={PAD.l}
                x2={W - PAD.r}
                y1={sy(v)}
                y2={sy(v)}
                stroke="rgba(148,163,184,0.12)"
              />
              <text
                x={PAD.l - 6}
                y={sy(v) + 3}
                textAnchor="end"
                className="fill-slate-500 text-[9px]"
              >
                {(v * 100).toFixed(0)}%
              </text>
            </g>
          )
        })}
        {Array.from({ length: n }, (_, i) => (
          <g key={i}>
            {series.map((s, k) => {
              const x = PAD.l + i * group + group * 0.1 + k * barW
              const y = sy(s.vals[i])
              return (
                <rect
                  key={s.key}
                  x={x}
                  y={y}
                  width={barW - 0.5}
                  height={PAD.t + plotH - y}
                  fill={SERIES[s.key].color}
                  rx={0.5}
                >
                  <animate
                    attributeName="height"
                    from="0"
                    to={PAD.t + plotH - y}
                    dur="0.5s"
                    fill="freeze"
                  />
                  <animate
                    attributeName="y"
                    from={PAD.t + plotH}
                    to={y}
                    dur="0.5s"
                    fill="freeze"
                  />
                </rect>
              )
            })}
            <text
              x={PAD.l + i * group + group / 2}
              y={H - PAD.b + 12}
              textAnchor="end"
              transform={`rotate(-45 ${PAD.l + i * group + group / 2} ${
                H - PAD.b + 12
              })`}
              className="fill-slate-500 text-[8px]"
            >
              {ticker(data, i)}
            </text>
          </g>
        ))}
      </svg>
      <Legend />
    </div>
  )
}

export function Legend() {
  return (
    <div className="mt-2 flex flex-wrap gap-x-4 gap-y-1 text-xs">
      {(["target", "ideal", "hardware"] as const).map((k) => (
        <span key={k} className="inline-flex items-center gap-1.5">
          <span
            className="inline-block h-2.5 w-2.5 rounded-sm"
            style={{ background: SERIES[k].color }}
          />
          {SERIES[k].label}
        </span>
      ))}
    </div>
  )
}
