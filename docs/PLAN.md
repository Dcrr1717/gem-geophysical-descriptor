# Plan — GEM article (geophysical fluids)

Adapt the Topological–Spectral Multiview Descriptor (persistent homology +
graph-spectral + multiview), originally built for neural-architecture
discrimination (PanAm 2026, EPN Quito), to **flow-regime / coherent-structure
classification in geophysical fluids**.

- **Venue:** Springer *GEM – International Journal on Geomathematics*, collection
  "Computational Methods and Techniques Applicable to Geophysical Fluids".
  Q2 Scopus, hybrid (publishable without APC via the subscription route).
- **Deadline:** 15 Dec 2026.
- **Go/No-Go checkpoint:** 1 Nov 2026 — if the descriptor does not separate
  regimes better than baselines, fall back to Plan B/C.
- **Fallback:** extended abstract (3–6 pp.) in *Selecciones Matemáticas* (SPMAC),
  open to all PanAm speakers, no topic restriction.
- **Data:** own 2D simulations (Rayleigh–Bénard / Kolmogorov) with a
  pseudo-spectral solver.
- **Collaborators / reviewers:** Jaime Lincovil, Enrique Cayún.

## Roadmap
1. Go/No-Go experiment (this repo) — descriptor vs baselines on 2D regimes.
2. If GO: scale up (larger grids, more regimes, Rayleigh–Bénard), add real
   reanalysis snapshots, ablate the two views, write the paper.
3. If NO-GO: redesign features or switch to the SPMAC extended abstract.
