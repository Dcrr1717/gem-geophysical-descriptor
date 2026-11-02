# GEM — Topological–Spectral Descriptor for Geophysical-Flow Regimes

Adapting the **Topological–Spectral Multiview Descriptor** (persistent homology +
graph-spectral features + multiview fusion) from neural-architecture
discrimination to the **classification of flow regimes / coherent structures in
geophysical fluids**.

Target: Springer **GEM – International Journal on Geomathematics**, collection
*"Computational Methods and Techniques Applicable to Geophysical Fluids"*
(PanAm 2026). Submission deadline: **15 Dec 2026**.

## Go / No-Go question
> Does the multiview descriptor separate flow regimes **better than physical /
> statistical baselines**?

`src/eval/gonogo.py` answers it: it generates labelled 2D vorticity snapshots
across regimes, computes the descriptor and the baselines, runs stratified
cross-validation, and emits a **GO** / **NO-GO** verdict (descriptor must beat
the best baseline by `go_margin`, default +0.05 accuracy).

## Method
- **Data** (`src/sims`): forced-dissipative 2D turbulence in vorticity form,
  pseudo-spectral (FFT) solver with 2/3 dealiasing and integrating-factor time
  stepping. Regimes = laminar / transitional / turbulent, set by the viscosity
  (Reynolds number).
- **Descriptor** (`src/descriptor`):
  - *spectral view* — radially averaged energy spectrum + inertial-range slope,
    and the top normalized graph-Laplacian eigenvalues of a coherence graph;
  - *topological view* — sublevel-set persistent homology (H0/H1) of the
    vorticity field as a grayscale image (GUDHI; light fallback without it).
- **Baselines** (`src/baselines`): vorticity moments/enstrophy, spectral slope
  only, and raw down-sampled pixels.
- **Evaluation** (`src/eval`): LogReg + RandomForest under StratifiedKFold,
  accuracy / macro-F1 / silhouette; Markdown + JSON report in `results/`.

## Run
```bash
python -m pip install -r requirements.txt
python -m src.eval.gonogo --config config/experiment.yaml
# outputs: results/gonogo.md and results/gonogo.json
```
The default config is a fast local smoke run. For publication-scale runs raise
`grid`, `n_steps`, `spinup` and `n_snapshots` (see comments in the config) — this
is the configuration intended for the cloud session.

## Hard structure-only benchmark (1 Nov 2026 Go/No-Go) — result: **NO-GO**
The toy-regime run above is trivially separable, so a fair, hard benchmark was
added: coherent 256² turbulence vs **IAAFT surrogates** (`src/sims/surrogate.py`)
with the same power spectrum and the same value histogram, so only spatial
(phase) structure differs. Evaluation uses grouped CV by trajectory, ablations of
both views, a 6-class variant, leak detectors and structure-aware physical
baselines (Okubo–Weiss, gradient statistics, increment flatness).

- Descriptor 0.989 vs spectrum / moment baselines 0.50 (blind by construction):
  GO against the baselines named in the brief (+0.207).
- But gradient statistics reach 1.000 and Okubo–Weiss 0.974; adding topology
  to all baselines gives +0.000: **NO-GO** against strong baselines.
  Recommendation: SPMAC extended abstract.
- Follow-up: persistence images, Betti curves and a multi-scale filtration
  (`src/descriptor/topology_rich.py`) reach parity (0.99–1.00) but never +0.05
  over all baselines combined; the multi-scale filtration makes topology robust
  to 0 dB noise (0.60 → 1.00), an edge that disappears when the baselines get
  the same low-pass. Still **NO-GO**.
- Second follow-up: the authentic functional-graph view X1 (|Pearson| > 0.40
  between 8×8 mesh regions, ported verbatim in `src/descriptor/authentic.py`)
  scores 0.69–0.75. That beats the old graph-Laplacian view but is far below
  the physical baselines. The full authentic descriptor X1+X2 and
  X1+X2+richer tie `baselines_union` (best margin over any configuration and
  task +0.011; headroom ≤ 0.011). Still **NO-GO**.

Details: [`docs/RESULTS.md`](docs/RESULTS.md), [`results/hard_benchmark.md`](results/hard_benchmark.md),
figures in `results/figures/`.

```bash
python -m pip install -r requirements.txt gudhi
python -m src.eval.hard_benchmark --config config/hard_benchmark.yaml        # ~1 h, 4 cores
python -m src.eval.noise_sweep   --config config/hard_benchmark.yaml        # exploratory
python -m src.eval.hard_benchmark --config config/hard_benchmark_smoke.yaml  # ~3 min check
```

## Status
Scaffold + runnable first pass. Known limitations of this first version:
snapshots within a regime come from one trajectory (temporally correlated);
regimes are viscosity-controlled Kolmogorov forcing (Rayleigh–Bénard and real
reanalysis fields are planned extensions).

Collaborators / reviewers: Jaime Lincovil, Enrique Cayún.

## License
MIT — see `LICENSE`.
