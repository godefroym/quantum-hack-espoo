import { ChevronDown, PlayCircle } from "lucide-react"
import { Link } from "react-router-dom"
import { Button, buttonVariants } from "@/components/ui/button"
import { NodeField } from "@/components/NodeField"

export function Hero() {
  const scrollToProblem = () => {
    document
      .getElementById("problem")
      ?.scrollIntoView({ behavior: "smooth", block: "start" })
  }

  return (
    <section className="relative flex min-h-screen flex-col items-center justify-center overflow-hidden px-6 text-center">
      {/* Background layers: node simulation under an ambient quantum glow */}
      <div className="pointer-events-none absolute inset-0 z-0">
        <div className="absolute left-1/2 top-1/3 h-[40rem] w-[40rem] -translate-x-1/2 -translate-y-1/2 rounded-full bg-indigo-500/20 blur-[120px]" />
        <div className="absolute right-1/4 bottom-1/4 h-[30rem] w-[30rem] rounded-full bg-cyan-400/10 blur-[120px]" />
        <div className="absolute left-1/4 top-1/4 h-[24rem] w-[24rem] rounded-full bg-fuchsia-500/10 blur-[120px]" />
        <NodeField />
      </div>

      {/* Foreground content sits above the simulation */}
      <div className="relative z-10 flex flex-col items-center">
        <span className="mb-6 inline-flex items-center rounded-full border border-border bg-card/50 px-4 py-1.5 text-xs font-medium tracking-wide text-muted-foreground backdrop-blur">
          Quantum-native scenario generation and tail-risk calculation
        </span>

        <h1 className="max-w-5xl bg-gradient-to-br from-white via-slate-200 to-slate-400 bg-clip-text text-5xl font-bold leading-tight tracking-tight text-transparent sm:text-6xl md:text-7xl">
          Quantum Systemic Stress
          <br />
          <span className="bg-gradient-to-r from-indigo-400 via-cyan-300 to-fuchsia-400 bg-clip-text text-transparent">
            Scenario Discovery
          </span>
        </h1>

        <p className="mt-8 max-w-2xl text-lg leading-relaxed text-muted-foreground sm:text-xl">
          Finance is risky. Let's minimise that risk. We propose an entangled
          Born-machine generator for correlated default scenarios, paired with a
          deterministic cascade simulator that evaluates contagion. Quantum
          advantage on two surfaces: generation and calculation.
        </p>

        <div className="mt-12 flex flex-col items-center gap-4 sm:flex-row">
          <Link to="/results" className={buttonVariants({ size: "xl" })}>
            <PlayCircle />
            See the results
          </Link>
          <Button
            size="xl"
            variant="outline"
            onClick={scrollToProblem}
            className="group"
          >
            View the pitch
            <ChevronDown className="transition-transform group-hover:translate-y-0.5" />
          </Button>
        </div>
      </div>

      <button
        onClick={scrollToProblem}
        aria-label="Scroll to problem statement"
        className="absolute bottom-10 left-1/2 z-10 -translate-x-1/2 animate-bounce text-muted-foreground transition-colors hover:text-foreground"
      >
        <ChevronDown className="size-7" />
      </button>
    </section>
  )
}
