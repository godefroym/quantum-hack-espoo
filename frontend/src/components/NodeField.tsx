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
  /** 0..1 redness, rises fast on failure so the node flashes red before it goes */
  red: number
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

// contagion cascade, triggered when the cursor reaches a cluster
const TRIGGER_RADIUS = 125 // px: cursor within this of a cluster centre seeds it
const STEP_MS = 120 // gap between successive failures (one node at a time)
const HOLD_MS = 1100 // pause once the cascade has run its course
const COOLDOWN_MS = 1500 // extra pause (after regen) before a cluster can re-trigger
const DIE_EASE = 0.12 // how fast a failed node fades out, after it has reddened
const REVIVE_EASE = 0.05 // how gently a node fades back in
const RED_RISE = 0.25 // how fast a failing node turns red
const RED_FADE = 0.06 // how fast red clears on revival

type SchedEvent = { t: number; node: number; revive: boolean }

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
            red: 0,
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
      busyUntil.length = 0
      for (let c = 0; c < CLUSTER_COUNT; c++) busyUntil.push(0)
      scheduled.length = 0
    }

    const scheduled: SchedEvent[] = []
    const busyUntil: number[] = []

    // build one cascade within a cluster, seeded at the node nearest the cursor.
    // It spreads deterministically outward hop by hop: every same-cluster
    // neighbour within LINK_DIST collapses. Returns failures in the order they
    // should disappear (breadth-first from the seed).
    const cascadeOrder = (c: number, mx: number, my: number): number[] => {
      const alive: number[] = []
      for (let i = 0; i < nodes.length; i++) {
        if (nodes[i].cluster === c && nodes[i].alive && nodes[i].vis > 0.6)
          alive.push(i)
      }
      if (!alive.length) return []

      // seed = alive node in this cluster closest to the cursor
      let seed = alive[0]
      let best = Infinity
      for (const i of alive) {
        const dx = nodes[i].x - mx
        const dy = nodes[i].y - my
        const d = dx * dx + dy * dy
        if (d < best) {
          best = d
          seed = i
        }
      }

      const failed = new Set<number>([seed])
      const order = [seed]
      let frontier = [seed]
      while (frontier.length) {
        const next: number[] = []
        for (const u of frontier) {
          const nu = nodes[u]
          for (const v of alive) {
            if (failed.has(v)) continue
            const dx = nu.x - nodes[v].x
            const dy = nu.y - nodes[v].y
            if (dx * dx + dy * dy <= LINK_DIST * LINK_DIST) {
              failed.add(v)
              order.push(v)
              next.push(v)
            }
          }
        }
        frontier = next
      }
      return order
    }

    const triggerCluster = (c: number, now: number) => {
      const order = cascadeOrder(c, mouse.x, mouse.y)
      if (!order.length) return
      order.forEach((node, k) =>
        scheduled.push({ t: now + k * STEP_MS, node, revive: false })
      )
      const reviveT = now + order.length * STEP_MS + HOLD_MS
      for (const node of order) scheduled.push({ t: reviveT, node, revive: true })
      busyUntil[c] = reviveT + COOLDOWN_MS
    }

    const draw = () => {
      const now = performance.now()

      // run due schedule events
      for (let i = scheduled.length - 1; i >= 0; i--) {
        if (scheduled[i].t <= now) {
          nodes[scheduled[i].node].alive = scheduled[i].revive
          scheduled.splice(i, 1)
        }
      }

      // seed a cascade the instant the cursor reaches a fresh cluster
      if (mouse.active) {
        for (let c = 0; c < CLUSTER_COUNT; c++) {
          if (now < busyUntil[c]) continue
          const dx = clusters[c].x - mouse.x
          const dy = clusters[c].y - mouse.y
          if (dx * dx + dy * dy < TRIGGER_RADIUS * TRIGGER_RADIUS) {
            triggerCluster(c, now)
          }
        }
      }

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

      // pass 3: ease redness + visibility, damp, integrate. A failing node
      // reddens first (at full size) and only starts shrinking once it is red,
      // so it visibly glows red before it disappears.
      for (const n of nodes) {
        if (n.alive) {
          n.red += (0 - n.red) * RED_FADE
          n.vis += (1 - n.vis) * REVIVE_EASE
        } else {
          n.red += (1 - n.red) * RED_RISE
          if (n.red > 0.8) n.vis += (0 - n.vis) * DIE_EASE
        }
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

        if (n.red > 0.12) {
          // red glow while the node is failing
          const rg = n.red * n.vis
          const g = ctx.createRadialGradient(n.x, n.y, 0, n.x, n.y, n.r * 7)
          g.addColorStop(0, `rgba(244, 63, 94, ${rg * 0.55})`)
          g.addColorStop(1, "rgba(244, 63, 94, 0)")
          ctx.fillStyle = g
          ctx.beginPath()
          ctx.arc(n.x, n.y, n.r * 7, 0, Math.PI * 2)
          ctx.fill()
        } else if (n.glow > 0.02) {
          // cyan glow when lit by the cursor
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
        cr = lerp(cr, 244, n.red)
        cg = lerp(cg, 63, n.red)
        cb = lerp(cb, 94, n.red)
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
