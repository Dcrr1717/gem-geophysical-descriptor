# SPMAC extended abstract — outline (draft)

- **Status:** outline for the authors to review. It is not the paper text.
- **Target:** extended abstract, 3–6 pp., *Selecciones Matemáticas* (SPMAC),
  Plan B of `docs/PLAN.md`.
- **Depends on the ERA5 test.** `docs/PLAN.md` schedules a pre-registered
  ERA5 blocking-vs-zonal test before switching to Plan B. That test has not
  been run. A NO-GO (or no run) keeps this outline as is. A GO would move the
  work back to the GEM paper, with this benchmark as its negative control.
- **Sources:**
  - `docs/RESULTS.md`: hard benchmark, richer topology, authentic X1/X2.
  - `docs/NOISE_ROBUSTNESS.md`: pre-registered degraded-field study.
  - `results/*.md`: full tables.
- **Numbers:** every number below is copied from those files. The last
  section lists the file each one comes from.

## Working title

1. *Topology sees coherent vortices, but no better than classical
   diagnostics: a structure-only benchmark for topological flow descriptors*
   (recommended).
2. *A structure-only benchmark for persistent-homology descriptors of 2D
   turbulence.*
3. *Persistent homology vs Okubo–Weiss on phase-randomized turbulence.*

## Message in one sentence

On a task where spectrum and one-point statistics are blind by construction,
persistent homology detects coherent structure almost perfectly. So do
classical coherent-structure diagnostics, and no topological configuration
beats them. Getting that comparison right takes three controls that are easy
to skip.

## Draft abstract (~200 words)

> Persistent homology is increasingly used to describe coherent structures in
> geophysical flows. We ask whether it adds value over classical diagnostics
> on a task where simpler statistics are blind by construction. We build a
> structure-only benchmark: 240 snapshots of forced two-dimensional
> turbulence (256², three forcing scales) and their IAAFT surrogates, which
> share the power spectrum and the one-point distribution, so that only
> spatial phase structure separates the classes. Spectral and one-point
> baselines score at chance (0.50), as intended. A sublevel-set persistence
> descriptor detects coherent structure at 98–100% accuracy under
> trajectory-grouped cross-validation. However, seven vorticity-gradient and
> Laplacian statistics reach 100%, Okubo–Weiss and increment-flatness
> statistics 97%,
> and the union of non-topological baselines 99.9%. No topological
> configuration exceeds that union by more than 0.011: not persistence
> images, Betti curves, a multi-scale filtration, or a functional-graph
> multiview descriptor. Under white observation noise, single-resolution
> persistence summaries degrade (1.000 to 0.802 at 0 dB). A multi-scale
> filtration restores them, but the same smoothing gives the classical
> diagnostics the same robustness. We report these parity results with
> three methodological lessons: saturated benchmarks leave no headroom,
> standardization can leak amplitude at round-off level, and robustness
> claims need matched preprocessing.

## Page budget (≈5 pp.)

| Section | Pages |
|---|---|
| Abstract + 1. Introduction | 0.6 |
| 2. Benchmark | 1.0 |
| 3. Descriptors and baselines | 0.7 |
| 4. Results | 1.7 |
| 5. Methodological lessons | 0.5 |
| 6. Conclusion and outlook | 0.3 |
| References | 0.4 |

## 1. Introduction

- **Context.**
  - Topological data analysis (sublevel-set persistent homology) is a
    natural language for vortices and their connectivity. Kramár et al.
    (2016) applied it to Kolmogorov flow and Rayleigh–Bénard convection.
  - Our multiview descriptor (functional graph + persistent homology) was
    first built for neural-network activations (Reinoso & Escudero, PanAm
    2026). This work ports it to fluid fields.
- **Problem.**
  - Regime-classification tasks are often easy. Our first toy
    regime-classification run was trivially separable, so success there says
    nothing about structure.
  - Weak baselines inflate results. Against the baselines first planned
    (moments, spectral slope, raw pixels) the descriptor "wins" by +0.207,
    but those baselines are blind on this task. This is a one-sentence
    motivating example.
- **Contributions.**
  1. A structure-only benchmark (coherent turbulence vs IAAFT surrogates)
     with built-in leak detectors and trajectory-grouped CV.
  2. A parity result. Topology matches but does not beat Okubo–Weiss,
     gradient and intermittency diagnostics. This holds over 21
     configurations, including richer vectorizations and the authentic
     multiview descriptor.
  3. A pre-registered degraded-field study with a matched-preprocessing
     control.
  4. Three methodological lessons for evaluating topological descriptors of
     fields.

## 2. Benchmark

- **2.1 Flows.**
  - Forced 2D turbulence, vorticity form.
  - Pseudo-spectral solver: 2/3 dealiasing, integrating-factor RK4, ν =
    10⁻³, linear Ekman drag 0.1, grid 256².
  - Kolmogorov forcing at k_f ∈ {4, 8, 16}.
  - 8 independent trajectories per k_f, 10 snapshots each, 3 time units
    apart after 25 time units of spin-up: 240 fields.
  - Fields standardized to zero mean and unit variance, without ε (see §5).
- **2.2 Surrogates.**
  - One 2D IAAFT surrogate per field (Schreiber & Schmitz 1996), 100
    iterations, Hermitian symmetry enforced.
  - Primary variant ends on the spectral step: Fourier amplitudes match to
    3×10⁻¹⁶, histogram to 0.023 σ.
  - Robustness variant ends on the rank-order step: exact histogram,
    spectrum to 4×10⁻⁴. It gives the same conclusions.
- **2.3 Tasks.**
  - Coherent vs surrogate, pooled over k_f (primary, 480 fields).
  - The same task per k_f.
  - 6 classes (k_f × coherent / surrogate).
  - The IAAFT robustness variant.
- **2.4 Protocol.**
  - Repeated StratifiedGroupKFold, 4 folds × 5 repeats. The group is the
    trajectory, so a field, its surrogate and all correlated snapshots of a
    run share a fold.
  - Logistic regression and random forest; the better of the two is
    reported for every set.
  - Paired group-bootstrap 95% CIs (2000 resamples of trajectories).
  - Decision rule fixed before the run: a margin ≥ +0.05 accuracy over the
    best baseline.
  - We define **headroom** = 1 − accuracy of the reference baseline.

## 3. Descriptors and baselines

Table 1 lists the feature sets with their dimensions (see `results/hard_benchmark.md`).

- **3.1 Topological descriptors.**
  - All use sublevel-set cubical persistence (GUDHI), H0 and H1.
  - (a) 8 summaries (count, total, max, entropy) on a 64² image
    (pre-registered) and at 256².
  - (b) X2: six persistence summaries per dimension (`sig6`), periodic
    complex, 256².
  - (c) Persistence images (Adams et al. 2017), 2 × 10×10.
  - (d) Betti curves, 33 thresholds.
  - (e) Multi-scale filtration: Gaussian σ = 1, 2, 4, 8 px.
  - Every hyper-parameter is fixed a priori.
- **3.2 Graph views.**
  - X1, the authentic functional graph: 8×8 = 64 mesh regions are the
    nodes, joined when |Pearson| > 0.40; the view is 10 graph invariants.
    Ported verbatim from the reference implementation, snapshot mode.
  - The older graph-Laplacian spectrum, kept as a baseline for X1.
  - X1+X2, and the fusion X3 = [X1/‖X1‖, X2/‖X2‖].
- **3.3 Baselines (no topology).**
  - Leak detectors: full log power spectrum, one-point quantiles. They must
    score at chance.
  - Structure-aware diagnostics:
    - Okubo–Weiss statistics (Okubo 1970; Weiss 1991), with the core
      threshold of Isern-Fontanet et al. (2006);
    - 7 statistics of vorticity gradients and the Laplacian;
    - flatness of vorticity increments at r = 1…32.
  - `baselines_union`: every non-topological baseline concatenated (136
    dimensions). This is the reference for every margin.
  - For degraded fields only: a **matched-scale control**, the same
    diagnostics computed on the same Gaussian pyramid as the multi-scale
    filtration.

## 4. Results

- **4.1 Validity.** The power spectrum scores 0.500 and the quantiles 0.503,
  so the task is structure-only. The spectral view of the old descriptor
  scores 0.580.
- **4.2 Topology detects coherent structure.** The 8-summary view scores
  0.983 at 64² and 1.000 at 256². This is the positive finding.
- **4.3 Parity on clean fields** (Table 2, primary and 6-class tasks).
  - Gradient statistics score 1.000, Okubo–Weiss 0.974, increment flatness
    0.976, and `baselines_union` 0.999.
  - The descriptor scores 0.989, a margin of −0.011 [−0.018, −0.005] vs the
    best baseline.
  - Adding topology to `baselines_union` changes accuracy by +0.000.
  - Caveat to state: in the 6-class task, topology beats the physical
    diagnostics alone (0.982 vs 0.926, +0.056). That is forcing-scale
    information (extrema counts grow with k_f). It vanishes against
    `baselines_union` (0.998).
- **4.4 Richer vectorizations and the authentic multiview descriptor.**
  - Persistence images score 0.993, Betti curves 0.988, and the multi-scale
    filtration 1.000.
  - X2, X1+X2, X3 and X1+X2+richer score 0.998–1.000.
  - X1 alone scores 0.69–0.75. It beats the Laplacian view (+0.117 [+0.063,
    +0.171]) but is far from the physical diagnostics.
  - Over 21 configurations × 6 tasks the largest margin over
    `baselines_union` is +0.011. No CI lower bound is above 0. Headroom is
    ≤ 0.011 on every clean task.
  - Gains over the old 64² view come from resolution, not from the richer
    vectorization: the 8 summaries at 256² already score 1.000.
- **4.5 Degraded fields** (pre-registered, Fig. 3 / Table 3).
  - Design: 5 degradation models × 4 levels, applied identically to both
    classes: white noise, random masking, swath / block gaps, quantization,
    resolution loss. 552 comparisons, with Benjamini–Hochberg correction.
  - Noise:
    - Single-resolution persistence summaries are fragile despite the
      stability theorem: X2 drops from 1.000 to 0.802 at 0 dB, because short
      noise pairs dominate the summaries.
    - The multi-scale filtration restores accuracy, reaching +0.287 [+0.253,
      +0.322] over `baselines_union` at −5 dB.
    - The matched-scale control erases that win: −0.006 [−0.017, +0.004].
      It is smoothing, not topology.
  - Masking, gaps up to 40% and quantization down to 2 bits:
    `baselines_union` stays ≥ 0.995, so there is no headroom.
  - Resolution loss hurts topology more than the baselines: −0.139 [−0.176,
    −0.101] at ×16.

## 5. Methodological lessons

1. **Report headroom.** If the reference baseline scores ≥ 1 − margin, a
   "better than baselines" claim cannot be tested, and a NO-GO is guaranteed
   whatever the descriptor. This benchmark is saturated (headroom ≤ 0.011).
2. **Standardize without ε.** (w − m)/(σ + ε) has standard deviation
   1 − ε/σ, so it encodes the original amplitude at round-off level.
   StandardScaler turns that into a feature: one-point moments scored 1.000
   on exactly matched histograms. Fix: no ε, and round features to 10⁻⁸.
3. **Match preprocessing before claiming robustness.** A filtration with
   built-in smoothing must be compared with baselines given the same
   smoothing. Without it, the −5 dB result would read as a strong
   topological win.
4. **Two shorter points, optional if space is tight.**
   - Group the CV by trajectory.
   - Rank-final IAAFT puts ~10⁻⁷ of its variance above the dealiasing
     cutoff, where the coherent field has none. That could leak through
     spectral features; in practice both IAAFT variants gave identical
     accuracies.

## 6. Conclusion and outlook

- **Conclusion.** Persistent homology captures coherent structure, but on
  this benchmark not better than the diagnostics geophysicists already use.
  Its robustness to noise is a smoothing effect.
- **Outlook.**
  - Where topology could still matter: large-scale connectivity, e.g.
    atmospheric blocking vs zonal flow, where local statistics may struggle.
  - An ERA5 pipeline with a Tibaldi–Molteni blocking index is implemented
    (`docs/REALDATA.md`) but **has not been run**. Present it only as future
    work unless it is run before submission.
  - Structured or non-Gaussian noise.

## Figures and tables

| Item | Content | Source | Note |
|---|---|---|---|
| Fig. 1 | Coherent field vs IAAFT surrogate, matched spectrum and histogram | `results/figures/fields_coherent_vs_iaaft.png` | Crop to one k_f row to save space |
| Fig. 2 | Margin of every configuration over `baselines_union`, per task | `results/figures/margin_vs_union.png` | Or a compact redraw with ~8 sets |
| Fig. 3 | Degraded fields: margin over `baselines_union` and over the matched control | `results/figures/robustness_margin_heatmap.png` | Alternative: `robustness_noise.png` |
| Table 1 | Feature sets: family, dimension | `results/hard_benchmark.md` (dim column) | Can merge into §3 text if space is tight |
| Table 2 | Accuracy, primary and 6-class tasks, key sets (leak detectors, OW, gradient, increments, union, 8 summaries 64²/256², X1, X2, X1+X2, multi-scale), margin vs union [CI] | `results/hard_benchmark.md` | Numbers in §4.3–4.4 |
| Table 3 | Degraded fields at −5 dB, 0 dB, ×16, 40% gaps, 2 bits: margin vs union and vs matched control | `results/noise_robustness.md` | Fill the gap and quantization cells from that file |

## Claims to avoid

- Topology "outperforms" or "is more robust than" classical diagnostics. The
  data show parity, and the robustness is a smoothing effect.
- Any result on real data. The ERA5 pipeline has not been run.
- That the stability theorem makes persistence *summaries* robust to noise.
  The X2 drop at 0 dB contradicts it in practice.
- The as-specified GO (+0.207) as a result. Use it only as a warning about
  weak baselines.

## References (verify every entry before submission)

- Adams, H. et al. (2017). Persistence images: a stable vector representation
  of persistent homology. *JMLR* 18(8).
- Cohen-Steiner, D., Edelsbrunner, H., Harer, J. (2007). Stability of
  persistence diagrams. *Discrete Comput. Geom.* 37.
- Davini, P. et al. (2012). Bidimensional diagnostics, variability, and
  trends of Northern Hemisphere blocking. *J. Climate* 25 (outlook only).
- Isern-Fontanet, J. et al. (2006). Vortices of the Mediterranean Sea: an
  altimetric perspective. *J. Phys. Oceanogr.* 36.
- Kramár, M. et al. (2016). Analysis of Kolmogorov flow and Rayleigh–Bénard
  convection using persistent homology. *Physica D* 334.
- Okubo, A. (1970). Horizontal dispersion of floatable particles in the
  vicinity of velocity singularities such as convergences. *Deep-Sea Res.*
  17.
- Reinoso & Escudero (2026). [Title and full reference of the PanAm 2026
  multiview-descriptor work, to complete.]
- Schreiber, T., Schmitz, A. (1996). Improved surrogate data for
  nonlinearity tests. *Phys. Rev. Lett.* 77.
- The GUDHI Project. GUDHI library, cubical complexes [version and citation
  to complete].
- Tibaldi, S., Molteni, F. (1990). On the operational predictability of
  blocking. *Tellus A* 42 (outlook only).
- Weiss, J. (1991). The dynamics of enstrophy transfer in two-dimensional
  hydrodynamics. *Physica D* 48.

## Open decisions for the authors

- **Authors and affiliations.** Collaborators named in `docs/PLAN.md`: Jaime
  Lincovil, Enrique Cayún.
- **Format.** Language, template and figure limits from the SPMAC author
  guidelines. They are not in this repo.
- **Scope of X1.** Recommended: keep it. It is the descriptor of the PanAm
  talk, and the honest result (weak alone, beats the Laplacian view) belongs
  in the record.
- **ERA5.** Whether to run the pipeline before the deadline. If it runs,
  §4.6 and Fig. 4 would be added and the page budget would be cut elsewhere.

## Where each number comes from

| Number | Source |
|---|---|
| 240 + 240 fields, solver and IAAFT parameters, 3×10⁻¹⁶, 0.023 σ, 4×10⁻⁴ | `docs/RESULTS.md`, *What was tested* |
| Leak detectors 0.500 / 0.503, spectral view 0.580 | `docs/RESULTS.md`, *What was tested* and *Does the topological view add value* |
| 0.983 / 1.000 (8 summaries), descriptor 0.989, −0.011 [−0.018, −0.005] | `docs/RESULTS.md`, *Verdict*; `results/hard_benchmark.md` |
| Gradient 1.000, Okubo–Weiss 0.974, increments 0.976, union 0.999 / 0.998 | `results/hard_benchmark.md`, primary and 6-class tables |
| 6-class 0.982 vs 0.926 (+0.056) | `docs/RESULTS.md`, item 4 |
| Images 0.993, Betti 0.988, multi-scale 1.000 | `docs/RESULTS.md`, *Richer topological vectorizations* |
| X1 0.69–0.75, vs Laplacian +0.117 [+0.063, +0.171], 21 × 6 configurations, max +0.011 | `docs/RESULTS.md`, *Authentic multiview views* |
| +0.207 vs the originally specified baselines | `docs/RESULTS.md`, *Verdict* |
| X2 1.000 → 0.802 at 0 dB, +0.287 [+0.253, +0.322], −0.006 [−0.017, +0.004], ≥ 0.995, −0.139 [−0.176, −0.101], 552 comparisons | `docs/NOISE_ROBUSTNESS.md`, *Results*; `results/noise_robustness.md` |
| Round-off leak (moments 1.000), ~10⁻⁷ above cutoff | `docs/RESULTS.md`, *Methodological findings* |
