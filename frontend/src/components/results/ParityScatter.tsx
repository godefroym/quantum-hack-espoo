import type { HardwareData } from "@/lib/results"
import { SERIES } from "@/lib/results"

const S = 340
const PAD = 40

/**
 * Parity scatter: hardware (and ideal) marginal vs analytic target. Points on
 * the diagonal mean the hardware reproduced the target exactly.
 */
export function ParityScatter({ data }: { data: HardwareData }) {
  const max =
    Math.max(
      ...data.target_marginals,
      ...data.hardware_marginals,
      ...data.ideal_marginals
    ) * 1.05
  const plot = S - PAD * 2
  const sx = (v: number) => PAD + (v / max) * plot
  const sy = (v: number) => S - PAD - (v / max) * plot

  const points = (vals: number[], color: string, r: number) =>
    vals.map((v, i) => (
      <circle
        key={i}
        cx={sx(data.target_marginals[i])}
        cy={sy(v)}
        r={r}
        fill={color}
        opacity={0.85}
      />
    ))

  const ticks = [0, 0.25, 0.5, 0.75, 1].map((f) => f * max)

  return (
    <div>
      <svg viewBox={`0 0 ${S} ${S}`} className="mx-auto h-auto w-full max-w-sm">
        {/* axes box */}
        <rect
          x={PAD}
          y={PAD}
          width={plot}
          height={plot}
          fill="none"
          stroke="rgba(148,163,184,0.18)"
        />
        {/* y = x reference */}
        <line
          x1={sx(0)}
          y1={sy(0)}
          x2={sx(max)}
          y2={sy(max)}
          stroke="rgba(148,163,184,0.35)"
          strokeDasharray="4 4"
        />
        {ticks.map((t) => (
          <g key={t}>
            <text
              x={sx(t)}
              y={S - PAD + 14}
              textAnchor="middle"
              className="fill-slate-500 text-[9px]"
            >
              {(t * 100).toFixed(0)}%
            </text>
            <text
              x={PAD - 6}
              y={sy(t) + 3}
              textAnchor="end"
              className="fill-slate-500 text-[9px]"
            >
              {(t * 100).toFixed(0)}%
            </text>
          </g>
        ))}
        {points(data.ideal_marginals, SERIES.ideal.color, 4)}
        {points(data.hardware_marginals, SERIES.hardware.color, 3)}
        <text
          x={S / 2}
          y={S - 6}
          textAnchor="middle"
          className="fill-slate-500 text-[10px]"
        >
          analytic target marginal
        </text>
        <text
          x={12}
          y={S / 2}
          textAnchor="middle"
          transform={`rotate(-90 12 ${S / 2})`}
          className="fill-slate-500 text-[10px]"
        >
          sampled marginal
        </text>
      </svg>
      <div className="mt-2 flex flex-wrap justify-center gap-x-4 gap-y-1 text-xs">
        {(["ideal", "hardware"] as const).map((k) => (
          <span key={k} className="inline-flex items-center gap-1.5">
            <span
              className="inline-block h-2.5 w-2.5 rounded-full"
              style={{ background: SERIES[k].color }}
            />
            {SERIES[k].label} vs target
          </span>
        ))}
      </div>
    </div>
  )
}
