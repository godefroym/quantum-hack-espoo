import { useMemo, useState } from "react"
import {
  PAIR_PALETTE,
  pairKey,
  ticker,
  TAIL_COLORS,
  type HardwareData,
} from "@/lib/results"

const YMIN = 1e-5

/**
 * Interactive severity-threshold explorer. Drag the threshold into the tail and
 * watch the survival readout (with confidence intervals) and the two networks:
 * institutions light up by their probability of being in a collapse of at least
 * s defaults. As s grows the tail mass shrinks for both generators.
 */
export function TailExplorer({ data }: { data: HardwareData }) {
  const t = data.tail
  const q = t.series.quantum
  const g = t.series.gaussian
  const n = data.n_qubits

  const sMax = useMemo(() => {
    let m = 4
    for (let s = q.survival.length - 1; s >= 0; s--) {
      if (q.survival[s] >= YMIN || g.survival[s] >= YMIN) {
        m = s
        break
      }
    }
    return Math.min(m, t.s_values.length - 1)
  }, [q, g, t])

  const [s, setS] = useState(Math.min(7, sMax))

  // reference brightness: a node's mass at s=1 equals its marginal
  const ref = useMemo(() => {
    const qn = q.node_joint?.[1] ?? []
    const gn = g.node_joint?.[1] ?? []
    return Math.max(1e-9, ...qn, ...gn)
  }, [q, g])

  const cmp = useMemo(() => {
    const qlo = q.lo[s]
    const qhi = q.hi[s]
    const glo = g.lo[s]
    const ghi = g.hi[s]
    if (qlo > ghi)
      return {
        tone: "emerald",
        text: "Quantum tail clears the copula interval — significantly heavier.",
      }
    if (glo > qhi)
      return {
        tone: "amber",
        text: "Copula tail is significantly heavier than quantum at this threshold.",
      }
    return {
      tone: "slate",
      text: "Intervals overlap: equal within sampling error (no significant gap).",
    }
  }, [q, g, s])

  return (
    <div>
      {/* slider */}
      <div className="mb-5">
        <div className="flex items-center justify-between text-sm">
          <span className="font-medium">
            Severity threshold: at least{" "}
            <span className="text-cyan-300">{s}</span> of {n} default
          </span>
          <span className="text-xs text-muted-foreground">
            drag into the tail →
          </span>
        </div>
        <input
          type="range"
          min={1}
          max={sMax}
          value={s}
          onChange={(e) => setS(Number(e.target.value))}
          className="mt-2 w-full accent-cyan-400"
        />
      </div>

      {/* survival readout */}
      <div className="grid gap-3 sm:grid-cols-2">
        <Readout
          label="Quantum hardware"
          color={TAIL_COLORS.quantum}
          p={q.survival[s]}
          lo={q.lo[s]}
          hi={q.hi[s]}
          shots={t.shots.quantum}
        />
        <Readout
          label="Gaussian copula"
          color={TAIL_COLORS.gaussian}
          p={g.survival[s]}
          lo={g.lo[s]}
          hi={g.hi[s]}
          shots={t.shots.copula}
        />
      </div>

      <div
        className={`mt-3 rounded-lg px-3 py-2 text-sm font-medium ${
          cmp.tone === "emerald"
            ? "bg-emerald-500/15 text-emerald-300"
            : cmp.tone === "amber"
              ? "bg-amber-500/15 text-amber-300"
              : "bg-muted text-muted-foreground"
        }`}
      >
        {cmp.text}
      </div>

      {/* networks */}
      <div className="mt-5 grid gap-4 sm:grid-cols-2">
        <NetworkPanel
          title="Quantum hardware"
          data={data}
          joint={q.node_joint?.[s] ?? []}
          refMax={ref}
        />
        <NetworkPanel
          title="Gaussian copula"
          data={data}
          joint={g.node_joint?.[s] ?? []}
          refMax={ref}
        />
      </div>
      <p className="mt-3 text-xs text-muted-foreground">
        A node's brightness is its probability of defaulting inside a collapse of
        at least {s} institutions. Both networks dim together as the threshold
        moves into the tail, which is the honest reading of this run: the
        entangled hardware sample and the matched Gaussian copula carry the same
        tail mass.
      </p>
    </div>
  )
}

function Readout({
  label,
  color,
  p,
  lo,
  hi,
  shots,
}: {
  label: string
  color: string
  p: number
  lo: number
  hi: number
  shots: number
}) {
  return (
    <div className="rounded-lg border border-border bg-background/40 p-3">
      <div className="flex items-center gap-2 text-xs text-muted-foreground">
        <span
          className="inline-block size-2.5 rounded-full"
          style={{ background: color }}
        />
        {label}
      </div>
      <div className="mt-1 text-2xl font-bold tabular-nums">{fmt(p)}</div>
      <div className="text-[11px] text-muted-foreground">
        95% CI [{fmt(lo)}, {fmt(hi)}] · ≈{Math.round(p * shots).toLocaleString()}{" "}
        in {shots.toLocaleString()}
      </div>
    </div>
  )
}

function NetworkPanel({
  title,
  data,
  joint,
  refMax,
}: {
  title: string
  data: HardwareData
  joint: number[]
  refMax: number
}) {
  const n = data.n_qubits
  const size = 260
  const cx = size / 2
  const cy = size / 2
  const R = size / 2 - 34
  const pos = (i: number) => {
    const a = -Math.PI / 2 + (i / n) * Math.PI * 2
    return { x: cx + Math.cos(a) * R, y: cy + Math.sin(a) * R }
  }
  const bright = (i: number) => Math.min(1, (joint[i] ?? 0) / refMax)
  const lit = joint.filter((v) => v / refMax > 0.08).length

  const pairColor = new Map<string, string>()
  data.graph.top_corr_pairs.forEach((p, idx) =>
    pairColor.set(pairKey(p.i, p.j), PAIR_PALETTE[idx % PAIR_PALETTE.length])
  )

  return (
    <div className="rounded-xl border border-border bg-[#0b0f1a] p-2">
      <div className="flex items-center justify-between px-2 pt-1 text-xs">
        <span className="font-medium">{title}</span>
        <span className="text-muted-foreground">{lit} lit</span>
      </div>
      <svg viewBox={`0 0 ${size} ${size}`} className="h-auto w-full">
        {/* faint top-correlation edges for context */}
        {data.graph.top_corr_pairs.map((p, idx) => {
          const a = pos(p.i)
          const b = pos(p.j)
          return (
            <line
              key={idx}
              x1={a.x}
              y1={a.y}
              x2={b.x}
              y2={b.y}
              stroke={PAIR_PALETTE[idx % PAIR_PALETTE.length]}
              strokeOpacity={0.18}
              strokeWidth={1}
            />
          )
        })}
        {Array.from({ length: n }, (_, i) => {
          const pt = pos(i)
          const b = bright(i)
          return (
            <g key={i}>
              {b > 0.05 && (
                <circle cx={pt.x} cy={pt.y} r={11} fill="#22d3ee" opacity={b * 0.4} />
              )}
              <circle
                cx={pt.x}
                cy={pt.y}
                r={6}
                fill="#22d3ee"
                opacity={0.12 + b * 0.88}
              />
              <text
                x={pt.x}
                y={pt.y - 12}
                textAnchor="middle"
                className="fill-slate-400 text-[7px]"
              >
                {ticker(data, i)}
              </text>
            </g>
          )
        })}
      </svg>
    </div>
  )
}

function fmt(x: number): string {
  if (x <= 0) return "0%"
  if (x < 0.001) return `${(x * 100).toFixed(3)}%`
  if (x < 0.01) return `${(x * 100).toFixed(2)}%`
  return `${(x * 100).toFixed(1)}%`
}
