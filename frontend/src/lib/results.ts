import { useEffect, useState } from "react"

export type TopPattern = {
  indices: number[]
  count: number
  freq: number
}

export type Posterior = {
  /** the institution that fails (target A) */
  a: number
  /** the collection of institutions failing together (condition B) */
  b: number[]
  /** P(A=1 | all of B = 1) */
  p: number
  /** shots in which all of B failed (the conditioning support) */
  support: number
  joint: number
  /** unconditional P(A=1) */
  baseline: number
  /** p / baseline */
  lift: number
}

export type Institution = { ticker: string; name: string }

export type GraphNode = { i: number; influence: number; baseline: number }
export type GraphEdge = {
  i: number
  j: number
  /** max(P(j|i), P(i|j)) — how likely one is to fail given the other does */
  strength: number
  corr: number
  p_j_given_i: number
  p_i_given_j: number
}
export type CorrPair = { i: number; j: number; corr: number }
export type SystemicGraph = {
  nodes: GraphNode[]
  edges: GraphEdge[]
  top_corr_pairs: CorrPair[]
}

export type HardwareData = {
  backend: string
  job_id: string
  shots: number
  n_qubits: number
  institutions: Institution[]
  max_degree: number
  entanglers: number
  entanglement_depth: number
  circuit_depth: number
  two_qubit_gates: number
  circuit_operations: Record<string, number>
  exact_ground_truth: boolean
  marginal_rmse_vs_target: number
  pairwise_joint_rmse_vs_target: number
  marginal_rmse_vs_ideal: number
  pairwise_joint_rmse_vs_ideal: number
  target_marginals: number[]
  hardware_marginals: number[]
  ideal_marginals: number[]
  default_count_hist: number[]
  mean_defaults: number
  expected_defaults_target: number
  pairwise_corr: number[][]
  pairwise_joint: number[][]
  top_patterns: TopPattern[]
  n_unique_patterns: number
  posteriors: Posterior[]
  graph: SystemicGraph
  tail: TailData
}

export type TailSeries = {
  survival: number[]
  lo: number[]
  hi: number[]
  /** P(node_i = 1 AND total >= s), rows = severity s, cols = node */
  node_joint?: number[][]
}

export type TailData = {
  s_values: number[]
  shots: { quantum: number; copula: number }
  series: {
    quantum: TailSeries
    gaussian: TailSeries
    student_t: TailSeries
  }
}

export const TAIL_COLORS: Record<string, string> = {
  quantum: "#22d3ee", // cyan
  gaussian: "#94a3b8", // slate
  student_t: "#f59e0b", // amber
}

export const TAIL_LABELS: Record<string, string> = {
  quantum: "Quantum hardware",
  gaussian: "Gaussian copula",
  student_t: "Student-t copula",
}

/** distinct colours for the most-correlated pairs (edges + legend). */
export const PAIR_PALETTE = [
  "#f59e0b", // amber
  "#ec4899", // pink
  "#34d399", // emerald
  "#a78bfa", // violet
  "#f43f5e", // rose
  "#38bdf8", // sky
  "#fb923c", // orange
  "#4ade80", // green
]

export function pairKey(i: number, j: number): string {
  return i < j ? `${i}-${j}` : `${j}-${i}`
}

type Load<T> = { data: T | null; error: string | null; loading: boolean }

export function useHardware(): Load<HardwareData> {
  const [state, setState] = useState<Load<HardwareData>>({
    data: null,
    error: null,
    loading: true,
  })
  useEffect(() => {
    let alive = true
    fetch(`${import.meta.env.BASE_URL}results/hardware.json`)
      .then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`)
        return r.json()
      })
      .then((data) => alive && setState({ data, error: null, loading: false }))
      .catch(
        (e) => alive && setState({ data: null, error: String(e), loading: false })
      )
    return () => {
      alive = false
    }
  }, [])
  return state
}

/** ticker label for institution i (falls back to a qubit label). */
export function ticker(data: HardwareData, i: number): string {
  return data.institutions[i]?.ticker ?? `Q${i + 1}`
}

/** full name for institution i (for tooltips). */
export function nameOf(data: HardwareData, i: number): string {
  return data.institutions[i]?.name ?? `Institution ${i + 1}`
}

export const SERIES = {
  target: { label: "Analytic target", color: "#94a3b8" }, // slate
  // For the 48-qubit stress run there is no exact 2^48 simulator; the "ideal"
  // reference series is the full-network Gaussian-copula fit to the target.
  ideal: { label: "Gaussian reference", color: "#818cf8" }, // indigo
  hardware: { label: "IBM hardware", color: "#22d3ee" }, // cyan
}
