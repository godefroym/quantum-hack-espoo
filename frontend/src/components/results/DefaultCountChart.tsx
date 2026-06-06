import { useEffect, useRef, useState } from "react"
import { Play, RotateCcw } from "lucide-react"
import { Button } from "@/components/ui/button"
import type { HardwareData } from "@/lib/results"

const W = 760
const H = 300
const PAD = { l: 52, r: 16, t: 16, b: 40 }

/**
 * Default-count distribution: how many institutions jointly default per shot,
 * across the 100k hardware samples. Optionally animate it filling in
 * shot-by-shot.
 */
export function DefaultCountChart({ data }: { data: HardwareData }) {
  const hist = data.default_count_hist
  const total = data.shots
  // trim trailing empty bins for a tighter axis
  let xmax = hist.length - 1
  while (xmax > 4 && hist[xmax] === 0) xmax--

  const [revealed, setRevealed] = useState(total) // shots shown so far
  const [animating, setAnimating] = useState(false)
  const raf = useRef(0)

  const play = () => {
    setAnimating(true)
    setRevealed(0)
    const start = performance.now()
    const dur = 1800
    const tick = (t: number) => {
      const p = Math.min(1, (t - start) / dur)
      // ease-out
      const eased = 1 - Math.pow(1 - p, 3)
      setRevealed(Math.round(eased * total))
      if (p < 1) raf.current = requestAnimationFrame(tick)
      else setAnimating(false)
    }
    raf.current = requestAnimationFrame(tick)
  }
  useEffect(() => () => cancelAnimationFrame(raf.current), [])

  const frac = revealed / total
  const ymax = Math.max(...hist) * 1.05
  const plotW = W - PAD.l - PAD.r
  const plotH = H - PAD.t - PAD.b
  const barW = (plotW / (xmax + 1)) * 0.8
  const sx = (c: number) => PAD.l + (c / (xmax + 1)) * plotW
  const sy = (v: number) => PAD.t + plotH - (v / ymax) * plotH

  const meanX = sx(data.mean_defaults) + barW / 2

  return (
    <div>
      <div className="mb-2 flex items-center gap-2">
        <Button size="sm" variant="secondary" onClick={play} disabled={animating}>
          <Play /> Replay sampling
        </Button>
        <Button
          size="sm"
          variant="ghost"
          onClick={() => setRevealed(total)}
          disabled={animating}
        >
          <RotateCcw /> Full
        </Button>
        <span className="ml-auto text-xs text-muted-foreground">
          {revealed.toLocaleString()} / {total.toLocaleString()} shots
        </span>
      </div>
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
                {Math.round(v).toLocaleString()}
              </text>
            </g>
          )
        })}
        {/* mean marker */}
        <line
          x1={meanX}
          x2={meanX}
          y1={PAD.t}
          y2={PAD.t + plotH}
          stroke="rgba(34,233,249,0.6)"
          strokeDasharray="4 4"
        />
        <text x={meanX + 4} y={PAD.t + 10} className="fill-cyan-300 text-[9px]">
          mean {data.mean_defaults.toFixed(2)}
        </text>

        {Array.from({ length: xmax + 1 }, (_, c) => {
          const v = hist[c] * frac
          const y = sy(v)
          return (
            <g key={c}>
              <rect
                x={sx(c) + (plotW / (xmax + 1)) * 0.1}
                y={y}
                width={barW}
                height={PAD.t + plotH - y}
                fill="#22d3ee"
                opacity={0.85}
                rx={1}
              />
              <text
                x={sx(c) + (plotW / (xmax + 1)) * 0.1 + barW / 2}
                y={H - PAD.b + 14}
                textAnchor="middle"
                className="fill-slate-500 text-[9px]"
              >
                {c}
              </text>
            </g>
          )
        })}
        <text
          x={PAD.l + plotW / 2}
          y={H - 6}
          textAnchor="middle"
          className="fill-slate-500 text-[10px]"
        >
          institutions defaulting per scenario
        </text>
      </svg>
    </div>
  )
}
