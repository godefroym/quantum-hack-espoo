export function Footer() {
  return (
    <footer className="border-t border-border px-6 py-12">
      <div className="mx-auto flex max-w-5xl flex-col items-center gap-3 text-center">
        <p className="text-lg font-semibold tracking-tight">
          Quantum Systemic Stress Scenario Discovery
        </p>
        <p className="max-w-2xl text-sm leading-relaxed text-muted-foreground">
          Classical baselines, the entangled QCBM generator, both contagion
          channels, and the QAE calculation surface are implemented and
          validated end-to-end. What remains is running the calculation on the
          54-qubit machine, access incoming.
        </p>
        <p className="mt-2 text-xs text-muted-foreground/70">
          Built for the Espoo quantum hackathon.
        </p>
      </div>
    </footer>
  )
}
