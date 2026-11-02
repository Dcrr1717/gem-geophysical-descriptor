# Topological robustness on degraded geophysical fields

**Status: PRE-REGISTERED, results pending.** This file, `config/noise_robustness.yaml` and
the code in `src/eval/degradation.py`, `src/eval/robustness_sweep.py` and
`src/baselines/multiscale.py` are committed before any degraded condition is
evaluated. Results are added below the line in a later commit.

## Question

`docs/RESULTS.md` closed the clean-field Go/No-Go as **NO-GO**: on the
structure-only benchmark (coherent 2D turbulence vs IAAFT surrogates),
`baselines_union` reaches ≥ 0.989, so no descriptor can show a +0.05
margin. It left one exploratory lead. At 0 dB SNR the multi-scale topological
filtration (`topo_multiscale`) scored 0.999 vs 0.955 for `baselines_union`,
a margin of +0.044 [+0.024, +0.067]. The same analysis noted that this edge
vanished when every feature set got a common Gaussian low-pass, so it may be
denoising rather than topology.

This study turns that hint into a systematic test:

> Under realistic degradations of satellite / reanalysis fields, does any
> topological configuration show a **real, CI-backed ≥ +0.05** accuracy
> advantage over `baselines_union`? Does the multi-scale lead generalize
> across degradation types, and does it survive a baseline that gets the same
> multi-scale smoothing?

## Data and protocol (unchanged from the hard benchmark)

- **Fields.** The cached 256² benchmark: 240 coherent fields (k_f ∈ {4, 8, 16},
  8 trajectories each, 10 snapshots) and their 240 IAAFT surrogates
  (spectrum-final; same power spectrum, same histogram to 0.023 σ).
  `config/noise_robustness.yaml` inherits `config/hard_benchmark.yaml`, so the
  cache key and the fields are identical.
- **Task.** Binary, coherent vs surrogate, pooled. This is the primary task of
  the hard benchmark.
- **CV.** Repeated StratifiedGroupKFold (4 folds × 3 repeats; group =
  trajectory, so a field, its surrogate and every correlated snapshot of a run
  share a fold). For every feature set the better of LogReg and RandomForest
  is reported, as in the hard benchmark.
- **Uncertainty.** Paired group-bootstrap 95% CIs (2000 resamples of
  trajectories) for every margin, plus a one-sided bootstrap p (margin ≤ 0).
  Benjamini–Hochberg is applied over the whole family: every topological
  configuration × degraded condition × reference.

## Degradation models

All models are applied **identically to both classes**:

- masks and gap positions come from a seed that depends only on the pair, so
  a coherent field and its own surrogate lose exactly the same pixels;
- noise is independent per field, so the classes are matched in expectation.

Every degraded field is re-standardized. Four levels per model, plus the
clean reference:

| Model | Levels | Notes |
|---|---|---|
| 1. Additive white observation noise | SNR +20, +10, 0, −5 dB (η = 0.10, 0.32, 1.00, 1.78) | linear; spectrum and histogram stay matched in expectation |
| 2. Random missing pixels | 5, 10, 20, 40 % | harmonic (Laplace) infill, parameter-free |
| 3a. Contiguous gaps: swaths | ≈5, 10, 20, 40 % (2, 3, 6, 13 bands of 8 px; axis random per pair) | harmonic infill |
| 3b. Contiguous gaps: blocks | ≈5, 10, 20, 40 % (3, 6, 13, 26 blocks of 32 × 32) | harmonic infill |
| 4. Coarse quantization | 8, 4, 3, 2 bits on [−4σ, 4σ] | pointwise nonlinear: spectra of the two classes may differ after it (see leak detectors) |
| 5. Resolution loss | ×2, ×4, ×8, ×16 block-mean, periodic bilinear back to 256² | linear |

Models 1, 2, 3a, 3b and 5 are linear in the data for a fixed mask, so they
keep the classes matched in expected power spectrum. Quantization is
nonlinear. The two leak detectors (`power_spectrum`, `histogram_quantiles`)
are reported in every condition so any spectral or one-point cue stays
visible.

## Feature sets

| Group | Sets |
|---|---|
| Full descriptors | `descriptor` (legacy spectral + topological view), `x1x2_rich` (authentic X1 + X2 + all richer topology) |
| Authentic views | `x1x2`, `x1_graph` (X1), `x2_ph` (X2, `sig6` summaries) |
| Topology vectorizations | `topological_view_fullres` (8 summaries at 256²), `topo_persistence_image`, `topo_betti_curves`, `topo_multiscale`, `topo_rich_all` |
| Structure baselines | `okubo_weiss`, `gradient_stats`, `increment_flatness`, `physical_structure_all` |
| Reference | `baselines_union` (every non-topological baseline) |
| Matched-scale control (new) | `physical_structure_ms`, `baselines_union_ms` = `baselines_union` + Okubo–Weiss / gradient / increment / moments / quantiles on the **same** Gaussian pyramid as `topo_multiscale` (σ = 1, 2, 4, 8 px) |
| Leak detectors | `power_spectrum`, `histogram_quantiles` |
| Incremental | `baselines_union+topo_rich`, `baselines_union_ms+topo_multiscale` |

All descriptor and baseline hyper-parameters are the hard benchmark's
pre-registered constants. The matched-scale control reuses the
multi-scale filtration's own pyramid code (`_gaussian_lowpass_periodic`,
`MS_SCALES`), so the comparison is like for like.

## Decision rule (fixed before running)

- **Real advantage** of a configuration at one (model, level): margin over
  `baselines_union` ≥ +0.05, 95% CI lower bound > 0, and BH q < 0.05. The
  table also flags the stricter case where the CI lower bound itself is
  ≥ +0.05.
- **Topology-specific** advantage: the same three conditions hold against
  the matched-scale control `baselines_union_ms` too. Otherwise the edge is
  attributed to the built-in multi-scale smoothing, not to topology.
- **Generalization of the lead** (`topo_multiscale`): a real advantage in ≥ 3
  of the 5 degradation models, at any level. Swaths and blocks count as one
  model, "contiguous gaps".
- **Conclusions:**
  - *Supports a dedicated "topological robustness on degraded geophysical
    fields" paper* only if the lead generalizes **and** is topology-specific
    in ≥ 2 models.
  - *Generalizes, but as denoising* if it generalizes against
    `baselines_union` only.
  - Otherwise *does not generalize*.

---

## Results

*Pending: added after the pre-registered run.*
