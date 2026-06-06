import { useEffect, useRef } from "react"

type Node = {
  x: number
  y: number
  vx: number
  vy: number
  cluster: number
  /** offset from the cluster centre, so the cluster keeps its shape */
  ox: number
  oy: number
  r: number
  /** 0..1 glow level, eased toward a target each frame */
  glow: number
  /** is the node currently present (false while it has failed) */
  alive: boolean
  /** 0..1 visibility, eased toward alive ? 1 : 0 */
  vis: number
}

type Cluster = {
  x: number
  y: number
  vx: number
  vy: number
}

const CLUSTER_COUNT = 9
const NODES_PER_CLUSTER = 12
const LINK_DIST = 130 // px: edge between two same-cluster nodes within this
const MOUSE_RADIUS = 180 // px: proximity that lights nodes up
const MIN_SEPARATION = 46 // px: nodes from different clusters repel within this
const WALL_MARGIN = 70 // px: nodes get pushed back inside this far from an edge
const WALL_FORCE = 0.35 // strength of the push away from the walls

// ambient contagion cascade
const P_SPREAD = 0.5 // chance a failure propagates across each intra-cluster link
const STEP_MS = 240 // gap between successive failures (one node disappears at a time)
const HOLD_MS = 1300 // pause once the cascade has run its course
const REGEN_MS = 1300 // time given for failed nodes to fade back in
const GAP_MS = 1100 // pause before the next cascade is seeded
const DIE_EASE = 0.2 // how fast a failed node disappears
const REVIVE_EASE = 0.05 // how gently a node fades back in

type Phase = "idle" | "cascading" | "hold" | "regen"

export function NodeField() {
  const canvasRef = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext("2d")
    if (!ctx) return

    const reduceMotion = window.matchMedia(
      "(prefers-reduced-motion: reduce)"
    ).matches

    let width = 0
    let height = 0
    let dpr = Math.min(window.devicePixelRatio || 1, 2)

    const clusters: Cluster[] = []
    const nodes: Node[] = []
    const mouse = { x: -9999, y: -9999, active: false }

    const rand = (min: number, max: number) => min + Math.random() * (max - min)
    const lerp = (a: number, b: number, t: number) => a + (b - a) * t

    const build = () => {
      clusters.length = 0
      nodes.length = 0
      for (let c = 0; c < CLUSTER_COUNT; c++) {
        const cx = rand(width * 0.08, width * 0.92)
        const cy = rand(height * 0.08, height * 0.92)
        clusters.push({
          x: cx,
          y: cy,
          vx: rand(-0.12, 0.12),
          vy: rand(-0.12, 0.12),
        })
        const spread = rand(60, 110)
        for (let i = 0; i < NODES_PER_CLUSTER; i++) {
          const ox = rand(-spread, spread)
          const oy = rand(-spread, spread)
          nodes.push({
            x: cx + ox,
            y: cy + oy,
            vx: 0,
            vy: 0,
            cluster: c,
            ox,
            oy,
            r: rand(1.4, 2.8),
            glow: 0,
            alive: true,
            vis: 1,
          })
        }
      }
    }

    const resize = () => {
      width = canvas.clientWidth
      height = canvas.clientHeight
      dpr = Math.min(window.devicePixelRatio || 1, 2)
      canvas.width = Math.round(width * dpr)
      canvas.height = Math.round(height * dpr)
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
      build()
    }

    // build one stochastic percolation cascade: a seed fails, then the failure
    // spreads to same-cluster neighbours with probability P_SPREAD. Because the
    // spread is probabilistic it usually dies out, so only part of the cluster
    // fails. Returns the failures in the order they should disappear.
    const generateCascade = (): number[] => {
      const aliveIdx: number[] = []
      for (let i = 0; i < nodes.length; i++) {
        if (nodes[i].alive && nodes[i].vis > 0.6) aliveIdx.push(i)
      }
      if (!aliveIdx.length) return []
      const seed = aliveIdx[Math.floor(Math.random() * aliveIdx.length)]
      const failed = new Set<number>([seed])
      const order = [seed]
      const queue = [seed]
      while (queue.length) {
        const u = queue.shift()!
        const nu = nodes[u]
        for (let v = 0; v < nodes.length; v++) {
          if (v === u) continue
          const nv = nodes[v]
          if (nv.cluster !== nu.cluster || !nv.alive || failed.has(v)) continue
          const dx = nu.x - nv.x
          const dy = nu.y - nv.y
          if (dx * dx + dy * dy > LINK_DIST * LINK_DIST) continue
          if (Math.random() < P_SPREAD) {
            failed.add(v)
            order.push(v)
            queue.push(v)
          }
        }
      }
      return order
    }

    let phase: Phase = "idle"
    let order: number[] = []
    let idx = 0
    let tNext = performance.now() + 700

    const stepCascade = (now: number) => {
      if (phase === "idle") {
        if (now < tNext) return
        order = generateCascade()
        if (order.length === 0) {
          tNext = now + GAP_MS
        } else {
          phase = "cascading"
          idx = 0
          tNext = now // first failure right away
        }
      } else if (phase === "cascading") {
        if (now < tNext) return
        if (idx < order.length) {
          nodes[order[idx]].alive = false
          idx++
          tNext = now + STEP_MS
        }
        if (idx >= order.length) {
          phase = "hold"
          tNext = now + HOLD_MS
        }
      } else if (phase === "hold") {
        if (now < tNext) return
        for (const i of order) nodes[i].alive = true // start regenerating
        phase = "regen"
        tNext = now + REGEN_MS
      } else if (phase === "regen") {
        if (now < tNext) return
        phase = "idle"
        tNext = now + GAP_MS
      }
    }

    const draw = () => {
      const now = performance.now()
      stepCascade(now)
      ctx.clearRect(0, 0, width, height)

      // drift cluster centres, gently bouncing off the edges
      for (const cl of clusters) {
        cl.x += cl.vx
        cl.y += cl.vy
        if (cl.x < width * 0.05 || cl.x > width * 0.95) cl.vx *= -1
        if (cl.y < height * 0.05 || cl.y > height * 0.95) cl.vy *= -1
      }

      // pass 1: spring toward anchor, wall repulsion, glow
      for (const n of nodes) {
        const cl = clusters[n.cluster]
        n.vx += (cl.x + n.ox - n.x) * 0.002
        n.vy += (cl.y + n.oy - n.y) * 0.002

        if (n.x < WALL_MARGIN) n.vx += (1 - n.x / WALL_MARGIN) * WALL_FORCE
        else if (n.x > width - WALL_MARGIN)
          n.vx -= (1 - (width - n.x) / WALL_MARGIN) * WALL_FORCE
        if (n.y < WALL_MARGIN) n.vy += (1 - n.y / WALL_MARGIN) * WALL_FORCE
        else if (n.y > height - WALL_MARGIN)
          n.vy -= (1 - (height - n.y) / WALL_MARGIN) * WALL_FORCE

        let glowTarget = 0
        if (mouse.active && n.vis > 0.3) {
          const dx = n.x - mouse.x
          const dy = n.y - mouse.y
          const d2 = dx * dx + dy * dy
          if (d2 < MOUSE_RADIUS * MOUSE_RADIUS) {
            glowTarget = 1 - Math.sqrt(d2) / MOUSE_RADIUS
          }
        }
        n.glow += (glowTarget - n.glow) * 0.12
      }

      // pass 2: inter-cluster repulsion (present nodes only)
      for (let i = 0; i < nodes.length; i++) {
        const a = nodes[i]
        for (let j = i + 1; j < nodes.length; j++) {
          const b = nodes[j]
          if (a.cluster === b.cluster) continue
          const dx = a.x - b.x
          const dy = a.y - b.y
          const d2 = dx * dx + dy * dy
          if (d2 >= MIN_SEPARATION * MIN_SEPARATION) continue
          const d = Math.sqrt(d2) || 1
          const live = a.vis * b.vis
          const f = (1 - d / MIN_SEPARATION) * 0.5 * live
          a.vx += (dx / d) * f
          a.vy += (dy / d) * f
          b.vx -= (dx / d) * f
          b.vy -= (dy / d) * f
        }
      }

      // pass 3: ease visibility, damp, integrate
      for (const n of nodes) {
        const target = n.alive ? 1 : 0
        n.vis += (target - n.vis) * (n.alive ? REVIVE_EASE : DIE_EASE)
        n.vx = n.vx * 0.92 + rand(-0.01, 0.01)
        n.vy = n.vy * 0.92 + rand(-0.01, 0.01)
        n.x += n.vx
        n.y += n.vy
      }

      // edges within a cluster, present only while both endpoints are visible
      for (let i = 0; i < nodes.length; i++) {
        const a = nodes[i]
        if (a.vis <= 0.12) continue
        for (let j = i + 1; j < nodes.length; j++) {
          const b = nodes[j]
          if (a.cluster !== b.cluster || b.vis <= 0.12) continue
          const dx = a.x - b.x
          const dy = a.y - b.y
          const d = Math.hypot(dx, dy)
          if (d > LINK_DIST) continue
          const live = Math.min(a.vis, b.vis)
          const proximity = Math.max(a.glow, b.glow)
          const alpha = ((1 - d / LINK_DIST) * 0.12 + proximity * 0.22) * live
          ctx.strokeStyle = `rgba(129, 140, 248, ${alpha})` // indigo-400
          ctx.lineWidth = 1
          ctx.beginPath()
          ctx.moveTo(a.x, a.y)
          ctx.lineTo(b.x, b.y)
          ctx.stroke()
        }
      }

      // nodes
      for (const n of nodes) {
        if (n.vis <= 0.02) continue
        const radius = n.r * n.vis
        const baseAlpha = (0.45 + n.glow * 0.5) * n.vis
        const dying = n.alive ? 0 : 1 - n.vis // red flash as it fails

        if (n.glow > 0.02 && dying < 0.3) {
          const halo = n.glow * n.vis
          const g = ctx.createRadialGradient(n.x, n.y, 0, n.x, n.y, n.r * 6)
          g.addColorStop(0, `rgba(103, 232, 249, ${halo * 0.35})`)
          g.addColorStop(1, "rgba(103, 232, 249, 0)")
          ctx.fillStyle = g
          ctx.beginPath()
          ctx.arc(n.x, n.y, n.r * 6, 0, Math.PI * 2)
          ctx.fill()
        }

        // slate -> cyan when lit, -> rose while failing
        const lit = n.glow
        let cr = lerp(148, 103, lit)
        let cg = lerp(163, 232, lit)
        let cb = lerp(184, 249, lit)
        cr = lerp(cr, 244, dying)
        cg = lerp(cg, 63, dying)
        cb = lerp(cb, 94, dying)
        ctx.fillStyle = `rgba(${Math.round(cr)}, ${Math.round(cg)}, ${Math.round(
          cb
        )}, ${baseAlpha})`
        ctx.beginPath()
        ctx.arc(n.x, n.y, Math.max(radius, 0.3), 0, Math.PI * 2)
        ctx.fill()
      }

      if (!reduceMotion) raf = requestAnimationFrame(draw)
    }

    const onMove = (e: MouseEvent) => {
      const rect = canvas.getBoundingClientRect()
      mouse.x = e.clientX - rect.left
      mouse.y = e.clientY - rect.top
      mouse.active = mouse.y >= 0 && mouse.y <= rect.height
    }
    const onLeave = () => {
      mouse.active = false
    }

    let raf = 0
    resize()
    window.addEventListener("resize", resize)
    window.addEventListener("mousemove", onMove)
    window.addEventListener("mouseout", onLeave)

    if (reduceMotion) {
      draw() // single static frame
    } else {
      raf = requestAnimationFrame(draw)
    }

    return () => {
      cancelAnimationFrame(raf)
      window.removeEventListener("resize", resize)
      window.removeEventListener("mousemove", onMove)
      window.removeEventListener("mouseout", onLeave)
    }
  }, [])

  return (
    <canvas
      ref={canvasRef}
      aria-hidden="true"
      className="absolute inset-0 h-full w-full"
    />
  )
}
