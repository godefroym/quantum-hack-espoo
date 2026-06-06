import { useEffect, useMemo, useRef } from "react"
import {
  forceCenter,
  forceCollide,
  forceLink,
  forceManyBody,
  forceSimulation,
  type Simulation,
} from "d3-force"
import {
  PAIR_PALETTE,
  pairKey,
  ticker,
  nameOf,
  type HardwareData,
} from "@/lib/results"

type SimNode = {
  i: number
  r: number
  x: number
  y: number
  vx: number
  vy: number
}

const R_MIN = 10
const R_MAX = 30

/**
 * Systemic influence map. Node size is how much an institution failing raises
 * everyone else's default probability; an edge means one failing makes the
 * other likely to fail (thickness = that likelihood); the most correlated
 * pairs are colour coded.
 */
export function ContagionGraph({ data }: { data: HardwareData }) {
  const g = data.graph
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const wrapRef = useRef<HTMLDivElement>(null)
  const nodesRef = useRef<SimNode[]>([])
  const hoverRef = useRef<number>(-1)

  // colour per top-correlated pair
  const pairColor = useMemo(() => {
    const m = new Map<string, string>()
    g.top_corr_pairs.forEach((p, idx) =>
      m.set(pairKey(p.i, p.j), PAIR_PALETTE[idx % PAIR_PALETTE.length])
    )
    return m
  }, [g])

  // adjacency for hover highlighting
  const neighbors = useMemo(() => {
    const adj = new Map<number, Set<number>>()
    for (const e of g.edges) {
      if (!adj.has(e.i)) adj.set(e.i, new Set())
      if (!adj.has(e.j)) adj.set(e.j, new Set())
      adj.get(e.i)!.add(e.j)
      adj.get(e.j)!.add(e.i)
    }
    return adj
  }, [g])

  useEffect(() => {
    const canvas = canvasRef.current
    const wrap = wrapRef.current
    if (!canvas || !wrap) return
    const ctx = canvas.getContext("2d")
    if (!ctx) return

    const infl = g.nodes.map((n) => n.influence)
    const lo = Math.min(...infl)
    const hi = Math.max(...infl)
    const radius = (v: number) =>
      R_MIN + (hi > lo ? (v - lo) / (hi - lo) : 0.5) * (R_MAX - R_MIN)
    const maxStrength = Math.max(...g.edges.map((e) => e.strength), 0.001)

    const nodes: SimNode[] = g.nodes.map((n) => ({
      i: n.i,
      r: radius(n.influence),
      x: 0,
      y: 0,
      vx: 0,
      vy: 0,
    }))
    nodesRef.current = nodes

    const links = g.edges.map((e) => ({ source: e.i, target: e.j, e }))

    let width = 0
    let height = 0
    let dpr = Math.min(window.devicePixelRatio || 1, 2)

    const sim: Simulation<SimNode, undefined> = forceSimulation(nodes)
      .force("charge", forceManyBody().strength(-220))
      .force(
        "link",
        forceLink<SimNode, (typeof links)[number]>(links)
          .id((d) => d.i)
          .distance((l) => 70 + (1 - l.e.strength / maxStrength) * 120)
          .strength((l) => 0.15 + (l.e.strength / maxStrength) * 0.35)
      )
      .force("collide", forceCollide<SimNode>().radius((d) => d.r + 12))
      .stop()

    const resize = () => {
      width = wrap.clientWidth
      height = wrap.clientHeight
      dpr = Math.min(window.devicePixelRatio || 1, 2)
      canvas.width = Math.round(width * dpr)
      canvas.height = Math.round(height * dpr)
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
      sim.force("center", forceCenter(width / 2, height / 2))
      sim.alpha(0.9).restart().stop()
    }

    const draw = () => {
      ctx.clearRect(0, 0, width, height)
      const hover = hoverRef.current
      const hl = hover >= 0
      const isLit = (i: number) =>
        !hl || i === hover || neighbors.get(hover)?.has(i)

      // edges
      for (const e of g.edges) {
        const a = nodes[e.i]
        const b = nodes[e.j]
        const col = pairColor.get(pairKey(e.i, e.j))
        const lit = !hl || e.i === hover || e.j === hover
        const w = col ? 3 : 1 + (e.strength / maxStrength) * 2.5
        ctx.strokeStyle = col
          ? hexA(col, lit ? 0.95 : 0.12)
          : `rgba(148,163,184,${lit ? 0.28 : 0.06})`
        ctx.lineWidth = w
        ctx.beginPath()
        ctx.moveTo(a.x, a.y)
        ctx.lineTo(b.x, b.y)
        ctx.stroke()
      }

      // nodes
      for (const node of nodes) {
        const lit = isLit(node.i)
        ctx.globalAlpha = lit ? 1 : 0.22
        // body
        ctx.beginPath()
        ctx.arc(node.x, node.y, node.r, 0, Math.PI * 2)
        ctx.fillStyle = "#6366f1" // indigo-500
        ctx.fill()
        ctx.lineWidth = node.i === hover ? 2.5 : 1
        ctx.strokeStyle =
          node.i === hover ? "#fff" : "rgba(199,210,254,0.5)"
        ctx.stroke()

        // ticker label, with a dark halo for legibility
        const label = ticker(data, node.i)
        ctx.font = "bold 10px ui-sans-serif, system-ui, sans-serif"
        ctx.textAlign = "center"
        ctx.textBaseline = "middle"
        ctx.lineWidth = 3
        ctx.strokeStyle = "rgba(2,6,23,0.85)"
        ctx.strokeText(label, node.x, node.y)
        ctx.fillStyle = "#fff"
        ctx.fillText(label, node.x, node.y)
        ctx.globalAlpha = 1
      }

      // tooltip
      if (hl) {
        const node = nodes.find((nn) => nn.i === hover)!
        const gn = g.nodes[hover]
        const lines = [
          `${ticker(data, hover)} · ${nameOf(data, hover)}`,
          `if it fails: +${gn.influence.toFixed(2)} system default prob`,
          `fails on its own ${(gn.baseline * 100).toFixed(1)}% of the time`,
        ]
        ctx.font = "11px ui-sans-serif, system-ui, sans-serif"
        const tw = Math.max(...lines.map((l) => ctx.measureText(l).width))
        const bx = Math.min(Math.max(node.x - tw / 2 - 8, 4), width - tw - 20)
        const by = node.y - node.r - 52
        ctx.fillStyle = "rgba(2,6,23,0.92)"
        ctx.fillRect(bx, by, tw + 16, 46)
        ctx.textAlign = "left"
        ctx.fillStyle = "#e2e8f0"
        lines.forEach((l, k) => {
          ctx.fillStyle = k === 0 ? "#fff" : "#94a3b8"
          ctx.fillText(l, bx + 8, by + 13 + k * 14)
        })
      }
    }

    let raf = 0
    const loop = () => {
      if (sim.alpha() > 0.005) sim.tick()
      draw()
      raf = requestAnimationFrame(loop)
    }

    const pick = (cx: number, cy: number) => {
      const rect = canvas.getBoundingClientRect()
      const x = cx - rect.left
      const y = cy - rect.top
      let best = -1
      let bestD = Infinity
      for (const node of nodes) {
        const dx = node.x - x
        const dy = node.y - y
        const d = dx * dx + dy * dy
        if (d < node.r * node.r && d < bestD) {
          best = node.i
          bestD = d
        }
      }
      return best
    }
    const onMove = (e: MouseEvent) => {
      hoverRef.current = pick(e.clientX, e.clientY)
      canvas.style.cursor = hoverRef.current >= 0 ? "pointer" : "default"
    }
    const onLeave = () => {
      hoverRef.current = -1
    }

    resize()
    nodes.forEach((nd) => {
      nd.x = width / 2 + (Math.random() - 0.5) * 120
      nd.y = height / 2 + (Math.random() - 0.5) * 120
    })
    sim.alpha(0.95).restart().stop()
    window.addEventListener("resize", resize)
    canvas.addEventListener("mousemove", onMove)
    canvas.addEventListener("mouseleave", onLeave)
    raf = requestAnimationFrame(loop)

    return () => {
      cancelAnimationFrame(raf)
      sim.stop()
      window.removeEventListener("resize", resize)
      canvas.removeEventListener("mousemove", onMove)
      canvas.removeEventListener("mouseleave", onLeave)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [g, pairColor, neighbors, data])

  return (
    <div className="grid gap-4 lg:grid-cols-[1fr_16rem]">
      <div
        ref={wrapRef}
        className="relative h-[60vh] min-h-[460px] overflow-hidden rounded-xl border border-border bg-[#0b0f1a]"
      >
        <canvas ref={canvasRef} className="h-full w-full" />
      </div>

      {/* legend */}
      <div className="flex flex-col gap-4">
        <div className="rounded-xl border border-border bg-card/40 p-4 text-xs">
          <div className="mb-2 font-semibold uppercase tracking-widest text-muted-foreground">
            How to read it
          </div>
          <ul className="space-y-2 text-muted-foreground">
            <li className="flex items-center gap-2">
              <span className="flex items-end gap-1">
                <span className="inline-block size-2 rounded-full bg-indigo-500" />
                <span className="inline-block size-3.5 rounded-full bg-indigo-500" />
              </span>
              Node size = systemic impact if it fails
            </li>
            <li className="flex items-center gap-2">
              <span className="inline-block h-[3px] w-6 rounded bg-slate-400/60" />
              Edge = one failing makes the other likely to fail
            </li>
            <li className="flex items-center gap-2">
              <span className="inline-block h-[3px] w-6 rounded bg-amber-400" />
              Coloured edge = most-correlated pair
            </li>
          </ul>
        </div>

        <div className="rounded-xl border border-border bg-card/40 p-4 text-xs">
          <div className="mb-2 font-semibold uppercase tracking-widest text-muted-foreground">
            Top correlated pairs
          </div>
          <ul className="space-y-1.5">
            {g.top_corr_pairs.map((p, idx) => (
              <li key={idx} className="flex items-center gap-2">
                <span
                  className="inline-block size-2.5 rounded-sm"
                  style={{ background: PAIR_PALETTE[idx % PAIR_PALETTE.length] }}
                />
                <span className="font-medium text-foreground">
                  {ticker(data, p.i)} ↔ {ticker(data, p.j)}
                </span>
                <span className="ml-auto tabular-nums text-muted-foreground">
                  {p.corr.toFixed(3)}
                </span>
              </li>
            ))}
          </ul>
        </div>
      </div>
    </div>
  )
}

function hexA(hex: string, a: number): string {
  const n = parseInt(hex.slice(1), 16)
  const r = (n >> 16) & 255
  const g = (n >> 8) & 255
  const b = n & 255
  return `rgba(${r},${g},${b},${a})`
}
