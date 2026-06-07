import { Link } from "react-router-dom"
import { Atom, ArrowLeft, Cpu, CheckCircle2, Loader2 } from "lucide-react"
import { useHardware, type HardwareData } from "@/lib/results"
import { TopPosteriors } from "@/components/results/TopPosteriors"
import { InfluenceRanking } from "@/components/results/InfluenceRanking"
import { SurvivalTail } from "@/components/results/SurvivalTail"

export function ResultsPage() {
  const { data, loading, error } = useHardware()

  return (
    <main className="min-h-screen bg-background text-foreground antialiased">
      <header className="sticky top-0 z-40 border-b border-border/60 bg-background/80 backdrop-blur-md">
        <div className="mx-auto flex h-14 max-w-6xl items-center gap-4 px-6">
          <Link
            to="/"
            className="inline-flex items-center gap-2 text-sm text-muted-foreground transition-colors hover:text-foreground"
          >
            <ArrowLeft className="size-4" /> Home
          </Link>
          <div className="flex items-center gap-2 font-semibold tracking-tight">
            <Atom className="size-5 text-indigo-400" />
            Quantum hardware results
          </div>
        </div>
      </header>

      <div className="mx-auto max-w-6xl px-6 py-10">
        {loading && (
          <div className="flex h-64 items-center justify-center text-muted-foreground">
            <Loader2 className="mr-2 size-5 animate-spin" /> Loading results…
          </div>
        )}
        {error && (
          <div className="rounded-xl border border-rose-400/30 bg-rose-500/5 p-6 text-sm text-rose-300">
            Failed to load results: {error}. Run{" "}
            <code className="rounded bg-muted px-1.5 py-0.5">
              uv run python scripts/export_results_data.py
            </code>
            .
          </div>
        )}
        {data && <Results data={data} />}
      </div>
    </main>
  )
}

function Results({ data }: { data: HardwareData }) {
  return (
    <>
      {/* hero */}
      <section>
        <h1 className="max-w-3xl text-4xl font-bold tracking-tight sm:text-5xl">
          An entangled Born machine ran on{" "}
          <span className="bg-gradient-to-r from-indigo-400 to-cyan-300 bg-clip-text text-transparent">
            real quantum hardware
          </span>
        </h1>
        <p className="mt-4 max-w-2xl text-lg leading-relaxed text-muted-foreground">
          {data.n_qubits} qubits, {data.shots.toLocaleString()} shots. The QPU
          reproduced the target correlated default distribution, matching both
          the analytic target and the exact statevector simulator.
        </p>

        <div className="mt-8 grid grid-cols-2 gap-4 sm:grid-cols-4">
          <Stat big value={String(data.n_qubits)} label="qubits / institutions" />
          <Stat big value={data.shots.toLocaleString()} label="hardware shots" />
          <Stat
            big
            value={pct(data.marginal_rmse_vs_target)}
            label="marginal RMSE vs target"
            accent
          />
          <Stat
            big
            value={pct(data.pairwise_joint_rmse_vs_target)}
            label="pairwise-joint RMSE"
            accent
          />
        </div>
      </section>

      {/* featured: systemic failure network (embedded prototype) */}
      <section className="mt-12">
        <h2 className="text-2xl font-bold tracking-tight">
          Systemic failure network
        </h2>
        <p className="mt-1 max-w-3xl text-sm text-muted-foreground">
          The co-failure association graph from the 100k hardware shots, with each
          institution placed at its headquarters on the world map. Node size is
          standalone failure probability, glow is eigenvector centrality, colour
          is the detected community, and edges show the sign and strength of the
          co-failure correlation. Use the threshold slider and filters; scroll to
          zoom.
        </p>
        <div className="mt-5 overflow-hidden rounded-xl border border-border bg-[#0b0f1a]">
          <iframe
            src="/proto/index.html"
            title="Systemic failure network"
            className="block h-[78vh] min-h-[560px] w-full"
            loading="lazy"
          />
        </div>

        {/* explanation of the simulation */}
        <div className="mt-4 grid gap-4 md:grid-cols-2">
          <div className="rounded-xl border border-border bg-card/40 p-5 text-sm leading-relaxed text-muted-foreground">
            <div className="mb-2 font-semibold text-foreground">
              What the simulation is
            </div>
            The entangled Born-machine circuit was measured{" "}
            {data.shots.toLocaleString()} times on {data.backend}. Each shot is a{" "}
            {data.n_qubits}-bit string; bit i set means institution i defaulted in
            that scenario. Together the shots are a sampled probability
            distribution over which institutions fail together. From them we
            measure pairwise co-failure correlations (φ), conditional
            probabilities, eigenvector centrality, communities (Louvain), and
            frequent co-failure baskets (Apriori). The page is built from these
            reduced statistics, not the raw shots.
          </div>
          <div className="rounded-xl border border-border bg-card/40 p-5 text-sm leading-relaxed text-muted-foreground">
            <div className="mb-2 font-semibold text-foreground">
              How to read it
            </div>
            <ul className="space-y-1.5">
              <li>
                <span className="text-foreground">Position</span>: each node is at
                its company headquarters on the world map.
              </li>
              <li>
                <span className="text-foreground">Size</span>: P(this institution
                fails). <span className="text-foreground">Glow</span>: systemic
                centrality.
              </li>
              <li>
                <span className="text-foreground">Colour</span>: detected
                community. <span className="text-foreground">Edges</span>: sign
                and strength of the co-failure correlation.
              </li>
              <li>
                <span className="text-foreground">Threshold slider</span>: hide
                weaker links; filter to contagion or hedges. Drag nodes, scroll
                to zoom.
              </li>
            </ul>
            <p className="mt-2">
              A graph is a pairwise view of the full joint distribution; the
              higher-order structure it drops is listed separately as co-failure
              baskets.
            </p>
          </div>
        </div>
      </section>

      {/* influence ranking, directly below the network */}
      <section className="mt-4">
        <div className="rounded-xl border border-border bg-card/40 p-5">
          <h3 className="text-lg font-semibold">Systemic influence ranking</h3>
          <p className="mb-4 mt-1 text-sm text-muted-foreground">
            Institutions ordered by how much their failure raises everyone
            else's default probability, the same metric that sizes the nodes
            above.
          </p>
          <InfluenceRanking data={data} />
        </div>
      </section>

      {/* circuit + fidelity panels */}
      <section className="mt-12 grid gap-4 md:grid-cols-2">
        <div className="rounded-xl border border-border bg-card/40 p-5">
          <div className="mb-3 flex items-center gap-2 text-sm font-semibold">
            <Cpu className="size-4 text-indigo-300" /> What ran on hardware
          </div>
          <div className="grid grid-cols-2 gap-3 text-sm">
            <KV k="Circuit depth" v={data.circuit_depth} />
            <KV k="Two-qubit gates" v={data.two_qubit_gates} />
            <KV k="Entanglers" v={data.entanglers} />
            <KV k="Max degree" v={data.max_degree} />
          </div>
          <div className="mt-3 flex flex-wrap gap-1.5">
            {Object.entries(data.circuit_operations).map(([op, c]) => (
              <span
                key={op}
                className="rounded-md bg-muted px-2 py-0.5 text-[11px] text-muted-foreground"
              >
                {op} ×{c}
              </span>
            ))}
          </div>
        </div>

        <div className="rounded-xl border border-border bg-card/40 p-5">
          <div className="mb-3 flex items-center gap-2 text-sm font-semibold">
            <CheckCircle2 className="size-4 text-emerald-400" /> Fidelity
          </div>
          <FidelityBar
            label="Marginals: hardware vs target"
            rmse={data.marginal_rmse_vs_target}
          />
          <FidelityBar
            label="Marginals: hardware vs simulator"
            rmse={data.marginal_rmse_vs_ideal}
          />
          <FidelityBar
            label="Pairwise joint: hardware vs target"
            rmse={data.pairwise_joint_rmse_vs_target}
          />
          {data.exact_ground_truth && (
            <p className="mt-3 text-xs text-muted-foreground">
              Verified against exact ground truth (full statevector).
            </p>
          )}
        </div>
      </section>

      <Section title="Conditional failure posteriors" subtitle="When a collection of institutions fails, which other institution is most likely to fail with them? Mined from the joint distribution: P(A fails | collection B fails), with the lift over A's unconditional rate.">
        <div className="rounded-xl border border-border bg-card/40 p-5">
          <TopPosteriors data={data} limit={5} />
        </div>
      </Section>

      <Section title="Is the tail real, or sampling noise?" subtitle="The deep tail is where the rare, severe collapses live, and where finite-shot sampling error is largest. The survival function with confidence bands is the check.">
        <div className="rounded-xl border border-border bg-card/40 p-5">
          <div className="mb-4 max-w-3xl space-y-2 text-sm leading-relaxed text-muted-foreground">
            <p>
              The survival function S(s) = P(at least s institutions default in a
              scenario) is the upper tail of the loss distribution. Read left to
              right, it shows how fast probability drains as scenarios get more
              severe. The axis is logarithmic so the rare deep-tail events stay
              visible. Quantum is the hardware sample; the copulas are matched to
              its exact marginals and pairwise dependencies.
            </p>
            <p>
              Each point is estimated from a finite number of shots, so it carries
              sampling error. The shaded band is the 95% confidence interval for
              that estimate. Two generators differ for real only where their bands
              stop overlapping: overlapping bands mean the gap is within sampling
              noise; separated bands mean the difference is not noise. Here the
              quantum and Gaussian-copula bands overlap at every threshold (equal
              within noise), while the Student-t band separates above them in the
              deep tail (a genuine difference, not an artefact of too few shots).
            </p>
          </div>
          <SurvivalTail data={data} />
        </div>
      </Section>
    </>
  )
}

function Section({
  title,
  subtitle,
  children,
}: {
  title: string
  subtitle: string
  children: React.ReactNode
}) {
  return (
    <section className="mt-12">
      <h2 className="text-2xl font-bold tracking-tight">{title}</h2>
      <p className="mt-1 max-w-2xl text-sm text-muted-foreground">{subtitle}</p>
      <div className="mt-5 grid gap-4">{children}</div>
    </section>
  )
}

function Stat({
  value,
  label,
  big,
  accent,
}: {
  value: string
  label: string
  big?: boolean
  accent?: boolean
}) {
  return (
    <div className="rounded-xl border border-border bg-card/40 p-4">
      <div
        className={`font-bold ${big ? "text-3xl" : "text-2xl"} ${
          accent ? "text-cyan-300" : ""
        }`}
      >
        {value}
      </div>
      <div className="mt-1 text-xs text-muted-foreground">{label}</div>
    </div>
  )
}

function KV({ k, v }: { k: string; v: number | string }) {
  return (
    <div>
      <div className="text-lg font-semibold">{v}</div>
      <div className="text-xs text-muted-foreground">{k}</div>
    </div>
  )
}

function FidelityBar({ label, rmse }: { label: string; rmse: number }) {
  // map RMSE (small is good) to a fill; 5% RMSE -> empty, 0 -> full
  const fill = Math.max(0, Math.min(1, 1 - rmse / 0.05))
  return (
    <div className="mb-3">
      <div className="flex items-center justify-between text-xs">
        <span className="text-muted-foreground">{label}</span>
        <span className="tabular-nums">{pct(rmse)} RMSE</span>
      </div>
      <div className="mt-1 h-1.5 w-full overflow-hidden rounded-full bg-muted">
        <div
          className="h-full rounded-full bg-gradient-to-r from-indigo-400 to-emerald-400"
          style={{ width: `${fill * 100}%` }}
        />
      </div>
    </div>
  )
}

function pct(x: number): string {
  return `${(x * 100).toFixed(2)}%`
}
