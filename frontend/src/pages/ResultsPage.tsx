import { Link } from "react-router-dom"
import { Atom, ArrowLeft, Cpu, CheckCircle2, Loader2 } from "lucide-react"
import { useHardware, type HardwareData } from "@/lib/results"
import { MarginalsBars } from "@/components/results/MarginalsBars"
import { ParityScatter } from "@/components/results/ParityScatter"
import { MarginalsLines } from "@/components/results/MarginalsLines"
import { CorrelationHeatmap } from "@/components/results/CorrelationHeatmap"
import { DefaultCountChart } from "@/components/results/DefaultCountChart"
import { TopScenarios } from "@/components/results/TopScenarios"
import { TopPosteriors } from "@/components/results/TopPosteriors"
import { ContagionGraph } from "@/components/results/ContagionGraph"
import { InfluenceRanking } from "@/components/results/InfluenceRanking"
import { SurvivalTail } from "@/components/results/SurvivalTail"
import { TailExplorer } from "@/components/results/TailExplorer"

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

      {/* featured: systemic influence map */}
      <section className="mt-12">
        <h2 className="text-2xl font-bold tracking-tight">
          Systemic influence map
        </h2>
        <p className="mt-1 max-w-3xl text-sm text-muted-foreground">
          Every institution at a glance. Each node is sized by how much its
          failure raises everyone else's default probability; an edge links two
          institutions when one failing makes the other likely to fail; the
          most correlated pairs are colour coded. Hover a node to isolate it.
        </p>
        <div className="mt-5">
          <ContagionGraph data={data} />
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

      {/* presentation chooser note */}
      <div className="mt-12 rounded-lg border border-indigo-400/30 bg-indigo-500/5 px-4 py-3 text-sm text-muted-foreground">
        The same hardware run is shown several ways below, each labelled an{" "}
        <span className="font-medium text-foreground">Option</span>. Pick the
        ones to keep.
      </div>

      <Section title="Marginals fidelity" subtitle="Per-institution default probability: target vs exact simulator vs hardware.">
        <OptionCard letter="A" name="Grouped bars" caption="Three bars per institution; easy direct comparison, but dense at 20 institutions.">
          <MarginalsBars data={data} />
        </OptionCard>
        <OptionCard letter="B" name="Parity scatter" caption="Sampled vs target with a y=x line. Points on the diagonal mean the hardware matched the target. Compact and precise.">
          <ParityScatter data={data} />
        </OptionCard>
        <OptionCard letter="C" name="Line overlay" caption="Three curves tracking each other across institutions; reads as 'the lines sit on top of one another.'">
          <MarginalsLines data={data} />
        </OptionCard>
      </Section>

      <Section title="Joint / correlation structure" subtitle="The entanglement-induced correlations between institutions, measured from the 100k shots.">
        <OptionCard letter="D" name="Correlation heatmap" caption="20×20 pairwise default correlation. Off-diagonal colour shows the correlations between institutions induced by the entanglement.">
          <CorrelationHeatmap data={data} />
        </OptionCard>
      </Section>

      <Section title="Default-count distribution" subtitle="How many institutions jointly default per scenario, across all shots.">
        <OptionCard letter="E" name="Distribution (with replay)" caption="The correlated loss distribution the QPU produced. 'Replay sampling' animates it filling in shot-by-shot.">
          <DefaultCountChart data={data} />
        </OptionCard>
      </Section>

      <Section title="Sampled scenarios" subtitle="The most frequent joint-default bitstrings the hardware returned.">
        <OptionCard letter="F" name="Top scenarios grid" caption="Each row is a sampled scenario; lit cells are institutions defaulting together.">
          <TopScenarios data={data} />
        </OptionCard>
      </Section>

      <Section title="Conditional failure posteriors" subtitle="Mined from the joint distribution: when a collection of institutions fails, which other institution is most likely to fail with them?">
        <OptionCard letter="G" name="Highest P(A | B) rules" caption="The strongest contagion conditionals: P(A fails | collection B fails). Toggle the ranking between raw posterior and lift over the unconditional baseline.">
          <TopPosteriors data={data} limit={5} />
        </OptionCard>
      </Section>

      <Section title="Systemic influence ranking" subtitle="A compact companion to the influence map: institutions ordered by how much their failure spreads to the rest of the system.">
        <OptionCard letter="H" name="Influence ranking bars" caption="The same systemic-impact metric that sizes the map nodes, as a ranked bar list.">
          <InfluenceRanking data={data} />
        </OptionCard>
      </Section>

      <Section title="Is the tail real, or sampling noise?" subtitle="Tail counts come from finite shots and are small, so a divergence only counts if the confidence intervals separate. Quantum is the hardware sample; the copulas are matched to its exact marginals and pairwise dependencies.">
        <OptionCard letter="I" name="Survival function with confidence bands" caption="P(at least s default) on a log axis with 95% Wilson bands. The quantum and Gaussian-copula bands overlap; the Student-t band separates above, a real difference.">
          <SurvivalTail data={data} />
        </OptionCard>
        <OptionCard letter="J" name="Interactive severity-threshold explorer" caption="Drag the threshold into the tail. The survival readout shows each interval and whether quantum clears the copula's; the two networks light up the institutions in collapses of that size.">
          <TailExplorer data={data} />
        </OptionCard>
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

function OptionCard({
  letter,
  name,
  caption,
  children,
}: {
  letter: string
  name: string
  caption: string
  children: React.ReactNode
}) {
  return (
    <div className="rounded-xl border border-border bg-card/40 p-5">
      <div className="mb-1 flex items-center gap-2">
        <span className="inline-flex size-6 items-center justify-center rounded-md bg-indigo-500/15 text-xs font-bold text-indigo-300">
          {letter}
        </span>
        <span className="font-semibold">{name}</span>
      </div>
      <p className="mb-4 text-xs text-muted-foreground">{caption}</p>
      {children}
    </div>
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
