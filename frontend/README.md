# Quantum Stress — Demo Site

A judge-facing site for **Quantum Systemic Stress Scenario Discovery**.

Built with **Vite + React + TypeScript**, **Tailwind CSS v4**, **shadcn/ui**, and
**react-router**.

Two routes:

- **`/` — landing / pitch.** Title, tagline, a live node-cluster background
  (clusters collapse via probabilistic percolation when the cursor is near), and
  the pitch sections: Problem → Approach → Quantum advantage → Network → Scope.
- **`/results` — quantum hardware results.** Presents the real IBM `ibm_fez`
  20-qubit, 100k-shot run of the entangled Born-machine generator. The results
  are determined (not interactive); the page shows the same run several ways,
  each labelled an **Option**, so the set can be narrowed down:
  - **A** grouped marginals bars · **B** parity scatter · **C** line overlay
  - **D** pairwise correlation heatmap
  - **E** default-count distribution (with a "replay sampling" build-up visual)
  - **F** top sampled scenarios grid

## Run locally

```bash
npm install      # first run only
npm run dev      # http://localhost:5173
npm run build    # type-check + production build into dist/
npm run preview  # serve the production build
```

## Results data

The page reads `public/results/hardware.json`, baked from the committed hardware
run in `outputs/results/` (the `.npz` samples + report). Regenerate it from the
repo root with:

```bash
uv run python scripts/export_results_data.py
```

It recomputes the derived views (correlation matrix, default-count histogram,
top scenarios) from the raw 100k shots.

## Adding more shadcn components

Pre-configured for the shadcn CLI (`components.json`):

```bash
npx shadcn@latest add card badge   # components land in src/components/ui/
```
