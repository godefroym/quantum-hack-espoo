import { useMemo } from "react"
import { TAIL_COLORS, TAIL_LABELS, type HardwareData } from "@/lib/results"

const W = 760
const H = 340
const PAD = { l: 56, r: 16, t: 16, b: 40 }
const YMIN = 1e-5 // log-scale floor

const KEYS = ["quantum", "gaussian", "student_t"] as const

/**
 * Survival function P(at least s institutions default) on a log axis, with
 * Wilson confidence bands from the finite shot counts. The bands answer the
 * skeptic: a tail gap is only real if the intervals separate.
 */
export function SurvivalTail({ data }: { data: HardwareData }) {
  const t = data.tail

  // trim to where any series still has meaningful mass
  const sMax = useMemo(() => {
    let m = 4
    for (const k of KEYS) {
      const surv = t.series[k].survival
      for (let s = surv.length - 1; s >= 0; s--) {
        if (surv[s] >= YMIN) {
          m = Math.max(m, s)
          break
        }
      }
    }
    return Math.min(m, t.s_values.length - 1)
  }, [t])

  const plotW = W - PAD.l - PAD.r
  const plotH = H - PAD.t - PAD.b
  const sx = (s: number) => PAD.l + (s / sMax) * plotW
  const logMin = Math.log10(YMIN)
  const sy = (v: number) => {
    const c = Math.log10(Math.max(v, YMIN))
    return PAD.t + plotH * (1 - (c - logMin) / (0 - logMin))
  }

  const yTicks = [0, -1, -2, -3, -4, -5].map((e) => Math.pow(10, e))

  const bandPath = (lo: number[], hi: number[]) => {
    const top = []
    const bot = []
    for (let s = 0; s <= sMax; s++) {
      top.push(`${sx(s)},${sy(hi[s])}`)
      bot.push(`${sx(s)},${sy(lo[s])}`)
    }
    return `M${top.join(" L")} L${bot.reverse().join(" L")} Z`
  }
  const line = (v: number[]) =>
    Array.from({ length: sMax + 1 }, (_, s) => `${sx(s)},${sy(v[s])}`).join(" ")

  return (
    <div>
      <svg viewBox={`0 0 ${W} ${H}`} className="h-auto w-full">
        {yTicks.map((v) => (
          <g key={v}>
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
              {v >= 0.01 ? `${v * 100}%` : `${v * 100}%`}
            </text>
          </g>
        ))}
        {Array.from({ length: sMax + 1 }, (_, s) => (
          <text
            key={s}
            x={sx(s)}
            y={H - PAD.b + 14}
            textAnchor="middle"
            className="fill-slate-500 text-[9px]"
          >
            {s}
          </text>
        ))}

        {/* confidence bands */}
        {KEYS.map((k) => (
          <path
            key={`band-${k}`}
            d={bandPath(t.series[k].lo, t.series[k].hi)}
            fill={TAIL_COLORS[k]}
            opacity={0.16}
          />
        ))}
        {/* survival lines */}
        {KEYS.map((k) => (
          <polyline
            key={`line-${k}`}
            points={line(t.series[k].survival)}
            fill="none"
            stroke={TAIL_COLORS[k]}
            strokeWidth={k === "quantum" ? 2.4 : 1.8}
            strokeLinejoin="round"
          />
        ))}

        <text
          x={PAD.l + plotW / 2}
          y={H - 4}
          textAnchor="middle"
          className="fill-slate-500 text-[10px]"
        >
          severity threshold s (institutions defaulting)
        </text>
      </svg>

      <div className="mt-2 flex flex-wrap gap-x-4 gap-y-1 text-xs">
        {KEYS.map((k) => (
          <span key={k} className="inline-flex items-center gap-1.5">
            <span
              className="inline-block h-2.5 w-4 rounded"
              style={{ background: TAIL_COLORS[k] }}
            />
            {TAIL_LABELS[k]}
          </span>
        ))}
      </div>
      <p className="mt-3 text-xs text-muted-foreground">
        Shaded bands are 95% Wilson intervals from{" "}
        {t.shots.quantum.toLocaleString()} hardware shots and{" "}
        {t.shots.copula.toLocaleString()} copula samples. The quantum and
        Gaussian-copula bands overlap throughout, so their tails are equal within
        sampling error. The Student-t band separates above both in the deep tail,
        a genuine difference rather than noise.
      </p>
    </div>
  )
}
