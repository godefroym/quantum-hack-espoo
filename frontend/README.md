# Failure Network — Demo Site

A judge-facing site for **Quantum Systemic Stress Scenario Discovery**.

Built with **Vite + React + TypeScript + Tailwind CSS v4**. The page is a single,
full-screen view that embeds the interactive **failure-network** visualization.

## Structure

- `frontend/` (this app) is a thin shell: it renders one full-screen `<iframe>`
  pointing at `/proto/index.html`.
- The actual visualization lives in `prototyping-for-quantum/prototyping-for-quantum/`
  (a separate Vite + d3 app — the 3D globe / flat map, clusters, threshold
  slider, etc.). Its production build is copied into `frontend/public/proto/`.

To rebuild the embedded visualization after changing it:

```bash
cd prototyping-for-quantum/prototyping-for-quantum
npm install            # first run only
npm run build          # -> dist/
cp -r dist/* ../../frontend/public/proto/
```

## Run locally

```bash
npm install      # first run only
npm run dev      # http://localhost:5173
npm run build    # type-check + production build into dist/
npm run preview  # serve the production build
```

## Demo data

The visualization reads its own `data.json` (bundled into the prototype). It is
generated from the latest hardware run — the **48-entity 2008-stress, 4-cluster
mixture** on IBM `ibm_boston` (200k reconciled shots,
`outputs/real_cluster_mixture_stress_hw/`):

```bash
uv run python scripts/export_failure_network.py
```

This computes the sufficient statistics (marginals, pairwise φ, conditionals,
eigenvector centrality, the run's clusters, co-failure baskets) and writes
`prototyping-for-quantum/prototyping-for-quantum/src/data.json`. Rebuild + copy
the prototype afterwards (see above).

## Deprecated (kept, not used by the page)

The page no longer renders a multi-panel results page, so the following are
retained only for reference / a possible future results page and are **not read
by the current demo**:

- `frontend/public/results/hardware.json`
- `scripts/export_results_data.py` (old 20-qubit `ibm_fez` run, `outputs/results/`)
- `scripts/export_stress_results_data.py` (48-entity run → `hardware.json`)

Each is marked deprecated in its header / `_deprecated` field. The live demo's
data path is `scripts/export_failure_network.py` only.
