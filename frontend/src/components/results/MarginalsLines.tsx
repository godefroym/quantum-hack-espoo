import type { HardwareData } from "@/lib/results"
import { SERIES } from "@/lib/results"
import { Legend } from "@/components/results/MarginalsBars"

const W = 760
const H = 280
const PAD = { l: 44, r: 12, t: 12, b: 40 }

/** Line overlay: the three marginal curves tracking each other. */
export function MarginalsLines({ data }: { data: HardwareData }) {
  const n = data.n_qubits
  const ymax = Math.max(
    ...data.target_marginals,
    ...data.ideal_marginals,
    ...data.hardware_marginals
  )
  const plotW = W - PAD.l - PAD.r
  const plotH = H - PAD.t - PAD.b
  const sx = (i: number) => PAD.l + (i / (n - 1)) * plotW
  const sy = (v: number) => PAD.t + plotH - (v / ymax) * plotH

  const line = (vals: number[]) =>
    vals.map((v, i) => `${sx(i)},${sy(v)}`).join(" ")

  const series = [
    { key: "target", vals: data.target_marginals },
    { key: "ideal", vals: data.ideal_marginals },
    { key: "hardware", vals: data.hardware_marginals },
  ] as const

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
        {series.map((s) => (
          <g key={s.key}>
            <polyline
              points={line(s.vals)}
              fill="none"
              stroke={SERIES[s.key].color}
              strokeWidth={s.key === "hardware" ? 2.4 : 1.6}
              strokeLinejoin="round"
              opacity={s.key === "target" ? 0.7 : 0.95}
            />
            {s.key === "hardware" &&
              s.vals.map((v, i) => (
                <circle
                  key={i}
                  cx={sx(i)}
                  cy={sy(v)}
                  r={2.5}
                  fill={SERIES.hardware.color}
                />
              ))}
          </g>
        ))}
        <text
          x={PAD.l + plotW / 2}
          y={H - 6}
          textAnchor="middle"
          className="fill-slate-500 text-[10px]"
        >
          institution (qubit)
        </text>
      </svg>
      <Legend />
    </div>
  )
}
