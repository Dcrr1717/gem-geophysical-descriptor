# Results — hard structure-only benchmark (GEM Go/No-Go, 1 Nov 2026)

## Verdict

**NO-GO for the paper's core claim.** The descriptor separates coherent flow
structure almost perfectly, but it does **not** do so better than standard,
non-topological physical diagnostics of coherent structures. Recommendation:
fall back to the **SPMAC extended abstract** (Plan B).

Results under the two criteria:

| Primary task: coherent vs IAAFT surrogate, 480 fields, grouped CV | descriptor | best baseline | margin [95% CI] | rule (>= +0.05) |
|---|---|---|---|---|
| vs baselines named in the brief (moments, spectral_slope, raw_pixels) | 0.989 | raw_pixels 0.782 | +0.207 [+0.173, +0.239] | GO |
| vs strict set (+ leak detectors + structure-aware physical baselines) | 0.989 | gradient_stats 1.000 | −0.011 [−0.018, −0.005] | **NO-GO** |

The as-specified GO is real but uninformative. The benchmark was built so that
spectrum and one-point statistics are blind by construction. The only
non-blind baseline in the brief is a 16×16 pixel classifier, which cannot
learn translation-invariant structure from 480 samples. A GEM reviewer in
geophysical fluids will ask for Okubo–Weiss / gradient / intermittency
diagnostics. Against those, the descriptor ties or loses on every task:

| Task | descriptor | best non-topological baseline | margin [95% CI] |
|---|---|---|---|
| binary, pooled (primary) | 0.989 | gradient_stats 1.000 | −0.011 [−0.018, −0.005] |
| binary, IAAFT histogram-exact variant | 0.989 | gradient_stats 1.000 | −0.011 [−0.018, −0.005] |
| binary, k_f = 4 | 1.000 | gradient_stats 1.000 | 0.000 |
| binary, k_f = 8 | 1.000 | okubo_weiss 1.000 | 0.000 |
| binary, k_f = 16 | 1.000 | okubo_weiss 1.000 | 0.000 |
| 6 classes (k_f × coherent/surrogate) | 0.991 | baselines_union 0.998 | −0.007 [−0.015, 0.000] |

**Follow-up: richer topological vectorizations, still NO-GO.** Persistence
images, Betti curves and a multi-scale filtration were added to the topological
view and evaluated on the same benchmark. None reaches +0.05 over
`baselines_union` on any task or noise condition; see
[Richer topological vectorizations](#richer-topological-vectorizations-follow-up).

Full tables: `results/hard_benchmark.md` / `.json`. Exploratory noise test:
`results/noise_sweep.md` / `.json`. Figures: `results/figures/`.

## What was tested

- **Data.** Forced 2D turbulence on a 256² grid, the repo's pseudo-spectral
  vorticity solver (with an opt-in IF-RK4 integrator and linear Ekman drag,
  needed for a stable, statistically stationary run at ν = 10⁻³). Three
  forcing scales k_f ∈ {4, 8, 16}, 8 independent trajectories each, 10
  snapshots per trajectory, 3 time units apart after 25 time units of spin-up:
  240 coherent fields.
- **Surrogates.** One 2D IAAFT surrogate per field (100 iterations, Hermitian
  symmetry enforced): 240 more fields. Primary variant: final spectral step.
  The Fourier amplitudes match to 3×10⁻¹⁶, including the exactly-zero
  dealiased modes; the histogram matches to 0.023 σ on average, in the tails.
  This is the conservative choice, because any residual leak helps the
  one-point baselines, not the descriptor. Robustness variant: final
  rank-order step (exact histogram, spectrum to 4×10⁻⁴). It gives identical
  conclusions.
- **Validity checks pass.** The full log power spectrum (85 bins) scores
  0.500 and the one-point quantiles score 0.503. The two classes are
  indistinguishable to any spectral or one-point statistic, so the task is
  structure-only.
- **Protocol.** Repeated StratifiedGroupKFold (4 folds × 5 repeats). The
  group is the trajectory, so a snapshot and its surrogate, and all
  correlated snapshots of one run, are always in the same fold. LogReg and
  RandomForest; the better of the two is reported for every feature set.
  Paired group-bootstrap CIs (2000 resamples). The descriptor configuration
  is the repo default, fixed before the run (`topo_downsample = 64`).

## Does the topological view add value beyond spectrum / moments?

**Yes in the narrow sense, no in the sense a paper needs.**

1. **Beyond spectrum and moments: yes, trivially.** The topological view
   reaches 0.983 against 0.50 for spectral slope, the full spectrum, moments
   and histogram (+0.48). The spectral view is near chance (0.580), so the
   descriptor's performance is essentially the topological view's; the
   graph-Laplacian part adds almost nothing.
2. **Beyond classical coherent-structure diagnostics: no.**
   - A 7-number gradient/Laplacian-statistics vector reaches 1.000 on the
     primary task (topological view 0.983, margin −0.017 [−0.028, −0.007]).
   - Okubo–Weiss statistics (0.974) and increment flatness (0.976) are on
     par with the topological view.
   - Per forcing scale, everything with phase sensitivity ties at ≈ 1.0.
3. **Incremental value: none.** Adding the topological view to all
   non-topological baselines changes accuracy by +0.000 (pooled binary),
   +0.001 (6 classes) and at most +0.011 per regime. Every CI includes 0 and
   all are far below the 0.05 rule.
4. **The one apparent win is explained by scale information.** In the
   6-class task the topological view beats the physical-structure family
   alone (0.982 vs 0.926, +0.056 [+0.027, +0.090]). This is not a structural
   advantage. Persistence counts also encode the forcing scale (the number of
   extrema grows with k_f), while the physical diagnostics are scale-free
   ratios. Once any spectral baseline is concatenated (`baselines_union`,
   0.998), the gap disappears. The like-for-like comparison is two-view
   descriptor (spectral + topological, 0.991) vs spectrum + physical (0.998).
5. **Noise does not rescue it (exploratory, post hoc).** Independent white
   observation noise was added to both classes, which keeps them matched in
   expected spectrum and histogram. Results:
   - At 0 dB SNR (η = 1) on raw fields, the topological view collapses
     (0.597, descriptor 0.665).
   - Okubo–Weiss (0.928) and all physical features together (0.939) remain
     strong, so topology is *less* noise-robust than the classical
     diagnostics here.
   - With a common Gaussian low-pass (σ = 2 px), everything recovers to
     ≥ 0.97 and gradient statistics (0.999) still match or beat the
     topological view (0.977).
   - No condition shows a ≥ 0.05 advantage or a positive incremental value
     for topology. See `results/figures/noise_sweep.png`.

The positive finding worth keeping: on a structure-only task, 8 persistence
summaries from a coarse 64² sublevel filtration detect coherent vortices at
98% (100% at full resolution). Topology does capture coherent structure. It
just does not capture it *better* than the diagnostics geophysicists already
use, so it does not support a "better than baselines" GEM paper.

## Richer topological vectorizations (follow-up)

The original topological view compresses each persistence diagram of a 64²
image into 8 numbers. The obvious objection is that a richer vectorization
could reveal an advantage the summaries hide. To test that,
`src/descriptor/topology_rich.py` adds three standard vectorizations of the
same sublevel-set cubical persistence:

- **Persistence images** (Adams et al. 2017): H0 and H1 diagrams of the
  full-resolution 256² field, (birth, persistence) coordinates, linear
  persistence weight, 2 × 10×10 pixels.
- **Betti curves**: β0(t) and β1(t) on 33 fixed thresholds.
- **A multi-scale filtration:** periodic Gaussian low-pass at σ = 1, 2, 4,
  8 grid cells, resampled to 256², 128², 64² and 32², with 16 persistence
  summaries per scale.
- **Combinations:** all three concatenated (`topo_rich_all`), and the
  spectral view plus all three (`descriptor_rich`).

Every hyper-parameter is a fixed constant chosen before any classification
was run, so nothing is fitted across folds. The old 8-summary view is kept
as a baseline: at 64² (pre-registered) and at 256² (the resolution-matched
control). The benchmark is unchanged: same cached fields and surrogates, same
grouped CV (4 folds × 5 repeats; 3 repeats for the noise conditions) and the
same paired group-bootstrap CIs. Rule: GO only if some richer vectorization
beats `baselines_union` (all non-topological baselines concatenated) by at
least +0.05.

**Verdict: NO-GO. No richer vectorization reaches +0.05 over
`baselines_union` on any of the 6 clean tasks or 6 noise conditions** (5
variants × 12 settings = 60 comparisons).

| Setting | `baselines_union` | max attainable margin | old view (64²) | 8 summaries (256²) | pers. images | Betti curves | multi-scale | rich, all | union + rich topo |
|---|---|---|---|---|---|---|---|---|---|
| binary, pooled (primary) | 0.999 | +0.001 | 0.983 | 1.000 | 0.993 | 0.988 | 1.000 | 0.999 | 0.999 |
| 6 classes | 0.998 | +0.002 | 0.982 | 1.000 | 0.997 | 0.991 | 1.000 | 0.998 | 0.998 |
| noise η = 1, raw | 0.955 | +0.045 | 0.597 | – | 0.896 | 0.845 | **0.999** | 0.991 | 1.000 |
| noise η = 1, common low-pass | 0.996 | +0.004 | 0.977 | – | 0.969 | 0.994 | 0.994 | 0.998 | 0.991 |

(Per-regime tasks, the IAAFT robustness variant and the η = 0.1 / 0.3
conditions are in `results/hard_benchmark.md` and `results/noise_sweep.md`;
the largest margin among them is +0.011.)

What this shows:

1. **The GO threshold is unreachable on this benchmark.** `baselines_union`
   scores ≥ 0.955 in every setting, so the largest margin *any* feature set
   could achieve is +0.045 (η = 1, raw). On the clean tasks it is at most
   +0.011. Making the topological view richer cannot produce a ≥ +0.05
   margin here, and none did.
2. **Clean fields: parity, and resolution matters more than vectorization.**
   Richer vectorizations lift topology from 0.983 to 0.99–1.00, tying
   `baselines_union`. On the primary task all margins lie in [−0.011, +0.001]
   and no CI lies above 0. The plain 8 summaries
   computed at 256² already score 1.000. The gain over the pre-registered
   64² view comes from resolution, not from the richer vectorization;
   persistence images (0.993) and Betti curves (0.988) do not beat the
   resolution-matched 8 summaries.
3. **Noise: the multi-scale filtration fixes the collapse of the old view,
   but its edge is denoising, not topology.**
   - At η = 1 on raw fields, the old 64² view collapses to 0.597 and
     multi-scale persistence reaches 0.999. That is +0.044 over
     `baselines_union` (CI [+0.024, +0.067]): below the threshold, at the
     +0.045 ceiling, and the only setting where any CI lies entirely above 0.
   - The multi-scale filtration low-passes the field internally. When every
     feature set gets the same Gaussian low-pass (η = 1, common low-pass),
     the baselines recover to 0.996 and the margin vanishes (−0.001 for
     multi-scale; +0.002 [−0.006, +0.012] for the best variant).
   - Single-resolution vectorizations of the noisy 256² field do worse than
     the baselines (persistence images −0.059, Betti curves −0.110).
4. **No incremental value either.** Adding all richer views to
   `baselines_union` changes accuracy by −0.005 to +0.011 on every setting
   except η = 1 raw (+0.045). That one gain disappears with the common
   low-pass (−0.005).

Taken together, the limitation listed earlier is closed. The negative result
is not an artefact of the 8-number summary: standard, richer vectorizations
also only reach parity with classical coherent-structure diagnostics. For the
SPMAC abstract this is a stronger statement, plus one positive,
reportable property: a multi-scale filtration makes the topological view
robust to 0 dB observation noise (0.597 → 0.999).

## Methodological findings (useful regardless of the verdict)

- **Round-off amplitude leak in standardization.** `(w − m) / (std + ε)` has
  standard deviation 1 − ε/std, so it encodes the original amplitude at
  round-off level, and `StandardScaler` amplifies that into a usable feature.
  On the hard benchmark this made `moments` score 1.000 with exactly matched
  histograms. It is fixed in the new pipeline: no ε, plus all features
  rounded to 10⁻⁸. The same issue existed in the original
  `src/eval/gonogo.py` (ε = 10⁻⁸) and is fixed there too. Its smoke-run
  `moments` baseline drops from 0.971 to 0.689; the verdict is unchanged
  (NO-GO).
- **Dealiasing and IAAFT.** A rank-order-final IAAFT surrogate puts ~10⁻⁷ of
  its variance above the solver's 2/3 cutoff, where the coherent field has
  exactly zero power. The benchmark therefore uses the spectrum-final
  variant, with the full spectrum as a leak detector. In practice both
  variants gave identical accuracies here.
- **Grouped vs stratified CV.** At this scale (8 trajectories × 10
  well-separated snapshots per regime), grouped and plain stratified CV agree
  within 0.01 for every set. On the small smoke configuration, raw pixels
  dropped from 0.84 (stratified) to 0.59 (grouped). Grouping matters when
  snapshots are close in time.

## Limitations

- Synthetic data only. The optional real-data part (ERA5 / public DNS) was
  **not run**: this cloud environment's egress proxy refused every external
  host tried (psl.noaa.gov, cds.climate.copernicus.eu, huggingface.co), and
  no CDS credentials were present.
- One flow family (Kolmogorov-forced 2D turbulence with drag). A different
  flow family could change the ranking, but the IAAFT task is a general
  nonlinearity test that any phase-sensitive statistic tends to pass.
- The pre-registered descriptor is the 8-summary, 64² view, with no tuning.
  Richer vectorizations (persistence images, Betti curves, a multi-scale
  filtration) were tested afterwards (see above) and do not change the
  verdict. Other vectorizations (persistence landscapes, kernels, learned
  vectorizations) were not tested. Their margin over `baselines_union` is
  bounded by the same ceiling (≤ +0.045 on this benchmark).
- Multiple comparisons: 5 richer variants × 12 settings. Reporting the best
  variant per setting is optimistic, and it still stays below the threshold.
- "Best of LogReg/RF" is slightly optimistic, equally for every feature set.

## Recommendation

1. **NO-GO for the GEM "better than baselines" paper** on the current
   evidence.
2. **Fallback: SPMAC extended abstract (3–6 pp.).** A defensible framing is
   "a structure-only benchmark for topological flow descriptors":
   - the IAAFT benchmark design and its validity checks;
   - the topological view detecting coherent structure at 98–100% where
     spectral and one-point statistics are at chance;
   - an honest parity result against Okubo–Weiss / gradient / intermittency
     diagnostics, which also holds for persistence images, Betti curves and a
     multi-scale filtration;
   - the noise-robustness of the multi-scale filtration (0.597 → 0.999 at
     0 dB SNR), with the caveat that a common low-pass gives the classical
     diagnostics the same robustness;
   - the round-off standardization pitfall.
3. Before reconsidering GEM, a pre-registered experiment would need to show a
   ≥ 0.05 margin over `baselines_union` on a task where classical diagnostics
   are known to struggle. Its baselines must also receive the same denoising
   or multi-scale treatment as the topology. Candidates are real reanalysis fields with
   structure-level labels (e.g. blocking vs zonal flow) or regime-transition
   detection. The code here (`src/eval/hard_benchmark.py`, `noise_sweep.py`)
   can be reused for that.

## Reproduce

```bash
pip install -r requirements.txt gudhi
python -m src.eval.hard_benchmark --config config/hard_benchmark.yaml   # ~1 h on 4 cores
python -m src.eval.noise_sweep   --config config/hard_benchmark.yaml   # ~20 min (uses the cache)
python -m src.eval.hard_benchmark --config config/hard_benchmark_smoke.yaml  # ~3 min pipeline check
```

All seeds are fixed in the configs. Simulated fields and surrogates are cached
under `data/` (git-ignored).
