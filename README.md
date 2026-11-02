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
- Third follow-up, pre-registered: robustness on degraded fields
  ([`docs/NOISE_ROBUSTNESS.md`](docs/NOISE_ROBUSTNESS.md)). Five realistic
  degradations were tested at 4 levels each: white noise, random masking,
  swath / block gaps, quantization and resolution loss. The multi-scale lead
  **does not generalize**.
  - A CI-backed ≥ +0.05 advantage over `baselines_union` appears only at
    −5 dB SNR (+0.287).
  - It vanishes against the same baselines computed on the same Gaussian
    pyramid (−0.006): it is smoothing, not topology.
  - Masking, gaps and quantization leave no headroom (`baselines_union`
    ≥ 0.995), and resolution loss hurts topology more (−0.139 at ×16).
- Fourth follow-up, pre-registered: richer topology and a genuinely
  topological task ([`docs/ECT_RESULTS.md`](docs/ECT_RESULTS.md)).
  - Two new views: the Euler Characteristic Transform (ECT,
    `src/descriptor/ect.py`) and a differentiable ECT (DECT,
    `src/descriptor/dect.py`, optional torch).
  - A new task: merged vs separate co-rotating vortex pairs
    (`src/sims/vortex_events.py`).
  - **NO-GO on both tasks.** On the IAAFT task, ECT scores 0.967 and DECT
    0.888, against 1.000 for the baselines.
  - On the vortex-merger task, H0/H1 counts are perfect (1.000) but tie
    `baselines_union` and the power spectrum alone (1.000). A merger also
    doubles the core size, which the spectrum sees.
  - Topology does **not** reach the +0.05 margin.

Details: [`docs/RESULTS.md`](docs/RESULTS.md), [`results/hard_benchmark.md`](results/hard_benchmark.md),
figures in `results/figures/`.

```bash
python -m pip install -r requirements.txt gudhi
python -m src.eval.hard_benchmark --config config/hard_benchmark.yaml        # ~1 h, 4 cores
python -m src.eval.noise_sweep   --config config/hard_benchmark.yaml        # exploratory
python -m src.eval.robustness_sweep --config config/noise_robustness.yaml  # degradations, ~2 h
python -m src.eval.ect_benchmark  --config config/ect_vortex.yaml       # ECT/DECT + vortex merger, ~2.5 h (torch optional)
python -m src.eval.hard_benchmark --config config/hard_benchmark_smoke.yaml  # ~3 min check
```

## Real-data benchmark: ERA5 winter blocking vs zonal flow (to be run locally)
A real-data test of the same question, in a setting chosen to favour topology:
large-scale flow regimes rather than small-scale structure. ERA5 500 hPa
winter days (DJF 1979/80–2023/24, Euro-Atlantic) are labelled BLOCKING or
ZONAL with the Tibaldi–Molteni (1990) gradient-reversal index computed from
geopotential height. The relative-vorticity fields then go through the
unchanged descriptor, the baselines, the grouped CV and the verdict rules of
the hard benchmark. CV groups are whole winters, so consecutive days never
straddle train and test.

The CDS download needs a free account and runs on your machine; no ERA5 data
is in the repository and no result exists yet. Steps, credentials and how to
read the verdict: [`docs/REALDATA.md`](docs/REALDATA.md).

```bash
python -m pip install -r requirements.txt -r requirements-realdata.txt
python -m src.realdata.download_era5          --config config/realdata.yaml   # needs ~/.cdsapirc
python -m src.realdata.label_blocking         --config config/realdata.yaml
python -m src.realdata.run_realdata_benchmark --config config/realdata.yaml
```

## Status
Scaffold + runnable first pass. Known limitations of this first version:
snapshots within a regime come from one trajectory (temporally correlated);
regimes are viscosity-controlled Kolmogorov forcing (Rayleigh–Bénard and real
reanalysis fields are planned extensions).

Collaborators / reviewers: Jaime Lincovil, Enrique Cayún.

## License
MIT — see `LICENSE`.
