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

## Status
Scaffold + runnable first pass. Known limitations of this first version:
snapshots within a regime come from one trajectory (temporally correlated);
regimes are viscosity-controlled Kolmogorov forcing (Rayleigh–Bénard and real
reanalysis fields are planned extensions).

Collaborators / reviewers: Jaime Lincovil, Enrique Cayún.

## License
MIT — see `LICENSE`.
