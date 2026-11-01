# Hard structure-only benchmark - GEM Go/No-Go

## Overall verdict: **NO-GO**

Overall = strict verdict. On this benchmark the baselines named in the brief are blind by construction (that is its purpose), so beating them does not show superiority over strong baselines; the strict set adds the phase-sensitive physical diagnostics a geophysical reviewer would expect. Interpretation: `docs/RESULTS.md`.

**Verdict as specified in the brief (primary binary coherent-vs-IAAFT task, grouped CV, vs moments, spectral_slope, raw_pixels): GO** (margin +0.207, threshold 0.05).

**Strict verdict (adds leak detectors and structure-aware physical baselines): NO-GO** (margin -0.011 vs `gradient_stats`).

## Richer topological vectorizations vs `baselines_union`

**Verdict: NO-GO** on the primary task; tasks where any richer vectorization reaches >= +0.05 over `baselines_union`: none.

Cells: accuracy (margin vs `baselines_union` [95% paired group-bootstrap CI]). `max attainable margin` = 1 - accuracy of `baselines_union`: when it is below +0.05 the GO threshold cannot be reached by any feature set. Richer views are computed from the full-resolution 256^2 field; `topological_view_fullres` (8 summaries at 256^2) is the resolution-matched control in the task tables.

| Task | baselines_union | max attainable margin | old topo view (64^2) | `topo_persistence_image` | `topo_betti_curves` | `topo_multiscale` | `topo_rich_all` | `descriptor_rich` | union + rich topo (gain) | any >= +0.05? |
|---|---|---|---|---|---|---|---|---|---|---|
| binary_pooled | 0.999 | +0.001 | 0.983 | 0.993 (-0.007 [-0.017, +0.000]) | 0.988 (-0.011 [-0.025, -0.002]) | 1.000 (+0.001 [+0.000, +0.002]) | 0.999 (-0.000 [-0.003, +0.002]) | 0.999 (-0.000 [-0.003, +0.002]) | 0.999 (-0.000 [-0.002, +0.001]) | no |
| multiclass6 | 0.998 | +0.002 | 0.982 | 0.997 (-0.001 [-0.003, +0.001]) | 0.991 (-0.007 [-0.018, +0.002]) | 1.000 (+0.002 [+0.000, +0.006]) | 0.998 (-0.000 [-0.001, +0.001]) | 0.998 (-0.000 [-0.001, +0.001]) | 0.998 (-0.000 [-0.001, +0.001]) | no |
| binary_kf4 | 0.994 | +0.006 | 1.000 | 0.988 (-0.006 [-0.019, +0.000]) | 0.999 (+0.005 [+0.000, +0.015]) | 1.000 (+0.006 [+0.000, +0.019]) | 0.994 (+0.000 [+0.000, +0.000]) | 0.994 (+0.000 [+0.000, +0.000]) | 0.994 (+0.000 [+0.000, +0.000]) | no |
| binary_kf8 | 0.989 | +0.011 | 1.000 | 1.000 (+0.011 [+0.000, +0.034]) | 1.000 (+0.011 [+0.000, +0.034]) | 1.000 (+0.011 [+0.000, +0.034]) | 1.000 (+0.011 [+0.000, +0.034]) | 1.000 (+0.011 [+0.000, +0.034]) | 1.000 (+0.011 [+0.000, +0.034]) | no |
| binary_kf16 | 1.000 | +0.000 | 1.000 | 1.000 (+0.000 [+0.000, +0.000]) | 1.000 (+0.000 [+0.000, +0.000]) | 1.000 (+0.000 [+0.000, +0.000]) | 1.000 (+0.000 [+0.000, +0.000]) | 1.000 (+0.000 [+0.000, +0.000]) | 1.000 (+0.000 [+0.000, +0.000]) | no |
| binary_pooled_iaaft_amplitude | 0.999 | +0.001 | 0.983 | 0.992 (-0.007 [-0.016, -0.000]) | 0.987 (-0.013 [-0.026, -0.002]) | 1.000 (+0.001 [+0.000, +0.002]) | 0.997 (-0.002 [-0.005, +0.001]) | 0.999 (-0.000 [-0.003, +0.002]) | 0.998 (-0.001 [-0.004, +0.001]) | no |

## Setup

- Coherent fields: 240 snapshots of forced 2D turbulence (grid 256^2, IF-RK4, dt=0.005, nu=0.001, linear drag=0.1), regimes `kf4` (k_f=4, amp=0.6), `kf8` (k_f=8, amp=1.0), `kf16` (k_f=16, amp=1.5).
- 8 independent trajectories per regime, 10 snapshots each, 3.0 time units apart after 25.0 time units of spin-up. Every field standardized (zero mean, unit variance).
- Surrogates: IAAFT, 100 iterations, final step `spectrum` (primary). Relative Fourier-amplitude mismatch mean 3.0e-16 (max 3.8e-16); histogram mismatch mean 2.3e-02 std (max 7.1e-02); Hermitian error <= 3e-16.
- Robustness surrogates: final step `amplitude`: spectral mismatch mean 3.8e-04, histogram mismatch mean 2.2e-16.
- Total fields: 480 (240 coherent + 240 surrogates). CV: StratifiedGroupKFold, 4 folds x 5 repeats, group = trajectory (a snapshot and its surrogate always share a fold). Classifiers: LogReg and RandomForest(300); `best acc` = the better of the two (same rule for every feature set). Margin CIs: paired bootstrap over trajectories (2000 resamples).
- Descriptor configuration pre-registered as the repo default (topo_downsample=64); `*_fullres*` rows are a post-hoc sensitivity check and are not used for any verdict.
- Timing (wall clock, s): simulation 847, surrogates 405, features 273, total_this_run 1976. The cloud VM was intermittently suspended during the run, so these overstate compute; the solver needs ~15 ms per IF-RK4 step at 256^2 on one core (~160 s per trajectory).

## Tiers

- `ours`, `ours-ablation`: the descriptor and its two views.
- `baseline-specified`: the baselines named in the Go/No-Go brief.
- `baseline-validity`: strongest spectral / one-point baselines; must be ~chance if the benchmark is valid (leak detectors).
- `baseline-structure`: phase-sensitive physical diagnostics of coherent structures (Okubo-Weiss, gradient statistics, increment flatness) - no topology.
- `baseline-union` / `incremental`: all non-topological baselines, without / with the topological view added (incremental value of topology).
- `ours-rich`: richer vectorizations of the same sublevel-set persistent homology (`src/descriptor/topology_rich.py`): persistence images, Betti curves, a multi-scale Gaussian filtration, their concatenation, and the spectral view + all of them.

## Primary task: coherent vs IAAFT surrogate (all forcing scales pooled)

N = 480, classes = 2, trajectories = 24, chance = 0.500.

| Feature set | tier | dim | best acc | LogReg acc (fold std) | RF acc (fold std) | macro-F1 (best clf) | silhouette |
|---|---|---|---|---|---|---|---|
| **descriptor** | ours | 19 | 0.989 | 0.989 (0.011) | 0.989 (0.009) | 0.989 | 0.020 |
| spectral_view | ours-ablation | 11 | 0.580 | 0.580 (0.041) | 0.570 (0.047) | 0.580 | 0.006 |
| topological_view | ours-ablation | 8 | 0.983 | 0.889 (0.030) | 0.983 (0.008) | 0.983 | 0.037 |
| moments | baseline-specified | 6 | 0.500 | 0.500 (0.000) | 0.500 (0.003) | 0.333 | -0.004 |
| spectral_slope | baseline-specified | 1 | 0.500 | 0.500 (0.000) | 0.500 (0.000) | 0.333 | -0.004 |
| raw_pixels | baseline-specified | 256 | 0.782 | 0.700 (0.054) | 0.782 (0.056) | 0.780 | 0.010 |
| power_spectrum | baseline-validity | 85 | 0.500 | 0.500 (0.000) | 0.500 (0.000) | 0.333 | -0.004 |
| histogram_quantiles | baseline-validity | 19 | 0.503 | 0.501 (0.004) | 0.503 (0.014) | 0.503 | -0.004 |
| okubo_weiss | baseline-structure | 6 | 0.974 | 0.974 (0.019) | 0.965 (0.026) | 0.974 | 0.248 |
| gradient_stats | baseline-structure | 7 | 1.000 | 0.998 (0.004) | 1.000 (0.000) | 1.000 | 0.451 |
| increment_flatness | baseline-structure | 12 | 0.976 | 0.976 (0.012) | 0.972 (0.016) | 0.976 | 0.274 |
| physical_structure_all | baseline-structure | 25 | 0.997 | 0.997 (0.005) | 0.995 (0.008) | 0.997 | 0.309 |
| baselines_union | baseline-union | 136 | 0.999 | 0.999 (0.004) | 0.992 (0.013) | 0.999 | 0.077 |
| baselines_union+topo | incremental | 144 | 0.999 | 0.999 (0.004) | 0.993 (0.010) | 0.999 | 0.075 |
| baselines_union+topo_rich | incremental | 466 | 0.999 | 0.996 (0.005) | 0.999 (0.003) | 0.999 | 0.094 |
| topo_persistence_image | ours-rich | 200 | 0.993 | 0.989 (0.011) | 0.993 (0.009) | 0.992 | 0.129 |
| topo_betti_curves | ours-rich | 66 | 0.988 | 0.930 (0.027) | 0.988 (0.016) | 0.988 | 0.126 |
| topo_multiscale | ours-rich | 64 | 1.000 | 1.000 (0.000) | 1.000 (0.000) | 1.000 | 0.058 |
| topo_rich_all | ours-rich | 330 | 0.999 | 0.997 (0.004) | 0.999 (0.003) | 0.999 | 0.102 |
| descriptor_rich | ours-rich | 341 | 0.999 | 0.997 (0.004) | 0.999 (0.003) | 0.999 | 0.097 |
| topological_view_fullres | sensitivity | 8 | 1.000 | 1.000 (0.000) | 0.999 (0.002) | 1.000 | 0.117 |
| descriptor_fullres_topo | sensitivity | 19 | 1.000 | 1.000 (0.000) | 1.000 (0.002) | 1.000 | 0.052 |

Verdicts / key comparisons:

- vs specified baselines: descriptor 0.989 vs `raw_pixels` 0.782 -> margin +0.207, 95% CI [+0.173, +0.239] -> **GO**
- vs strict baselines: descriptor 0.989 vs `gradient_stats` 1.000 -> margin -0.011, 95% CI [-0.018, -0.005] -> **NO-GO**
- topological view vs best spectrum/one-point baseline (`histogram_quantiles`): 0.983 vs 0.503 (margin +0.480, CI [+0.467, +0.493])
- topological view vs best structure-aware physical baseline (`gradient_stats`): 0.983 vs 1.000 (margin -0.017, CI [-0.028, -0.007])
- incremental value of topology on top of all baselines: 0.999 -> 0.999 (gain +0.000, CI [+0.000, +0.000])
- richer topology: best variant `topo_multiscale` 1.000 vs `baselines_union` 0.999 -> margin +0.001, CI [+0.000, +0.002] (max attainable +0.001) -> **NO-GO**; adding all richer views to `baselines_union`: gain -0.000, CI [-0.002, +0.001]

## Multiclass: forcing scale x {coherent, surrogate} (6 classes)

N = 480, classes = 6, trajectories = 24, chance = 0.167.

| Feature set | tier | dim | best acc | LogReg acc (fold std) | RF acc (fold std) | macro-F1 (best clf) | silhouette |
|---|---|---|---|---|---|---|---|
| **descriptor** | ours | 19 | 0.991 | 0.991 (0.007) | 0.988 (0.008) | 0.991 | 0.059 |
| spectral_view | ours-ablation | 11 | 0.571 | 0.571 (0.054) | 0.558 (0.035) | 0.570 | 0.010 |
| topological_view | ours-ablation | 8 | 0.982 | 0.976 (0.009) | 0.982 (0.009) | 0.983 | 0.199 |
| moments | baseline-specified | 6 | 0.334 | 0.266 (0.027) | 0.334 (0.033) | 0.333 | -0.139 |
| spectral_slope | baseline-specified | 1 | 0.500 | 0.500 (0.000) | 0.500 (0.000) | 0.476 | -0.013 |
| raw_pixels | baseline-specified | 256 | 0.601 | 0.513 (0.032) | 0.601 (0.028) | 0.554 | 0.017 |
| power_spectrum | baseline-validity | 85 | 0.500 | 0.500 (0.000) | 0.500 (0.000) | 0.483 | -0.012 |
| histogram_quantiles | baseline-validity | 19 | 0.425 | 0.425 (0.027) | 0.398 (0.033) | 0.424 | -0.116 |
| okubo_weiss | baseline-structure | 6 | 0.791 | 0.791 (0.034) | 0.764 (0.040) | 0.789 | 0.035 |
| gradient_stats | baseline-structure | 7 | 0.800 | 0.786 (0.042) | 0.800 (0.022) | 0.797 | 0.080 |
| increment_flatness | baseline-structure | 12 | 0.830 | 0.830 (0.030) | 0.810 (0.040) | 0.828 | 0.031 |
| physical_structure_all | baseline-structure | 25 | 0.926 | 0.926 (0.033) | 0.897 (0.036) | 0.926 | 0.063 |
| baselines_union | baseline-union | 136 | 0.998 | 0.998 (0.004) | 0.984 (0.014) | 0.998 | 0.216 |
| baselines_union+topo | incremental | 144 | 0.999 | 0.999 (0.004) | 0.992 (0.009) | 0.999 | 0.214 |
| baselines_union+topo_rich | incremental | 466 | 0.998 | 0.995 (0.005) | 0.998 (0.004) | 0.998 | 0.213 |
| topo_persistence_image | ours-rich | 200 | 0.997 | 0.988 (0.009) | 0.997 (0.004) | 0.997 | 0.092 |
| topo_betti_curves | ours-rich | 66 | 0.991 | 0.944 (0.017) | 0.991 (0.007) | 0.991 | 0.130 |
| topo_multiscale | ours-rich | 64 | 1.000 | 1.000 (0.000) | 0.998 (0.004) | 1.000 | 0.284 |
| topo_rich_all | ours-rich | 330 | 0.998 | 0.994 (0.006) | 0.998 (0.004) | 0.998 | 0.182 |
| descriptor_rich | ours-rich | 341 | 0.998 | 0.995 (0.006) | 0.998 (0.004) | 0.998 | 0.173 |
| topological_view_fullres | sensitivity | 8 | 1.000 | 1.000 (0.000) | 1.000 (0.002) | 1.000 | 0.400 |
| descriptor_fullres_topo | sensitivity | 19 | 1.000 | 1.000 (0.000) | 1.000 (0.002) | 1.000 | 0.126 |

Verdicts / key comparisons:

- vs specified baselines: descriptor 0.991 vs `raw_pixels` 0.601 -> margin +0.390, 95% CI [+0.369, +0.410] -> **GO**
- vs strict baselines: descriptor 0.991 vs `baselines_union` 0.998 -> margin -0.007, 95% CI [-0.015, +0.000] -> **NO-GO**
- topological view vs best spectrum/one-point baseline (`spectral_slope`): 0.982 vs 0.500 (margin +0.482, CI [+0.470, +0.493])
- topological view vs best structure-aware physical baseline (`physical_structure_all`): 0.982 vs 0.926 (margin +0.056, CI [+0.027, +0.090])
- incremental value of topology on top of all baselines: 0.998 -> 0.999 (gain +0.001, CI [+0.000, +0.004])
- richer topology: best variant `topo_multiscale` 1.000 vs `baselines_union` 0.998 -> margin +0.002, CI [+0.000, +0.006] (max attainable +0.002) -> **NO-GO**; adding all richer views to `baselines_union`: gain -0.000, CI [-0.001, +0.001]

## Task `binary_kf4`

N = 160, classes = 2, trajectories = 8, chance = 0.500.

| Feature set | tier | dim | best acc | LogReg acc (fold std) | RF acc (fold std) | macro-F1 (best clf) | silhouette |
|---|---|---|---|---|---|---|---|
| **descriptor** | ours | 19 | 1.000 | 1.000 (0.000) | 0.971 (0.024) | 1.000 | 0.151 |
| spectral_view | ours-ablation | 11 | 0.679 | 0.662 (0.095) | 0.679 (0.057) | 0.678 | 0.041 |
| topological_view | ours-ablation | 8 | 1.000 | 1.000 (0.000) | 0.961 (0.033) | 1.000 | 0.281 |
| moments | baseline-specified | 6 | 0.500 | 0.500 (0.000) | 0.500 (0.000) | 0.333 | -0.012 |
| spectral_slope | baseline-specified | 1 | 0.500 | 0.500 (0.000) | 0.500 (0.000) | 0.333 | -0.012 |
| raw_pixels | baseline-specified | 256 | 0.770 | 0.728 (0.047) | 0.770 (0.061) | 0.769 | 0.020 |
| power_spectrum | baseline-validity | 85 | 0.500 | 0.500 (0.000) | 0.500 (0.000) | 0.333 | -0.012 |
| histogram_quantiles | baseline-validity | 19 | 0.506 | 0.501 (0.012) | 0.506 (0.013) | 0.504 | -0.012 |
| okubo_weiss | baseline-structure | 6 | 0.926 | 0.926 (0.035) | 0.910 (0.055) | 0.926 | 0.152 |
| gradient_stats | baseline-structure | 7 | 1.000 | 1.000 (0.000) | 0.995 (0.010) | 1.000 | 0.476 |
| increment_flatness | baseline-structure | 12 | 0.961 | 0.961 (0.039) | 0.956 (0.032) | 0.961 | 0.361 |
| physical_structure_all | baseline-structure | 25 | 0.996 | 0.996 (0.009) | 0.988 (0.015) | 0.996 | 0.327 |
| baselines_union | baseline-union | 136 | 0.994 | 0.994 (0.011) | 0.970 (0.027) | 0.994 | 0.063 |
| baselines_union+topo | incremental | 144 | 1.000 | 1.000 (0.000) | 0.981 (0.021) | 1.000 | 0.079 |
| baselines_union+topo_rich | incremental | 466 | 0.994 | 0.991 (0.012) | 0.994 (0.011) | 0.994 | 0.197 |
| topo_persistence_image | ours-rich | 200 | 0.988 | 0.980 (0.017) | 0.988 (0.013) | 0.987 | 0.215 |
| topo_betti_curves | ours-rich | 66 | 0.999 | 0.999 (0.005) | 0.996 (0.009) | 0.999 | 0.338 |
| topo_multiscale | ours-rich | 64 | 1.000 | 1.000 (0.000) | 0.994 (0.011) | 1.000 | 0.348 |
| topo_rich_all | ours-rich | 330 | 0.994 | 0.988 (0.017) | 0.994 (0.011) | 0.994 | 0.263 |
| descriptor_rich | ours-rich | 341 | 0.994 | 0.991 (0.012) | 0.994 (0.011) | 0.994 | 0.253 |
| topological_view_fullres | sensitivity | 8 | 1.000 | 1.000 (0.000) | 0.994 (0.011) | 1.000 | 0.412 |
| descriptor_fullres_topo | sensitivity | 19 | 1.000 | 1.000 (0.000) | 0.994 (0.011) | 1.000 | 0.213 |

Verdicts / key comparisons:

- vs specified baselines: descriptor 1.000 vs `raw_pixels` 0.770 -> margin +0.230, 95% CI [+0.179, +0.276] -> **GO**
- vs strict baselines: descriptor 1.000 vs `gradient_stats` 1.000 -> margin +0.000, 95% CI [+0.000, +0.000] -> **NO-GO**
- topological view vs best spectrum/one-point baseline (`histogram_quantiles`): 1.000 vs 0.506 (margin +0.494, CI [+0.484, +0.500])
- topological view vs best structure-aware physical baseline (`gradient_stats`): 1.000 vs 1.000 (margin +0.000, CI [+0.000, +0.000])
- incremental value of topology on top of all baselines: 0.994 -> 1.000 (gain +0.006, CI [+0.000, +0.019])
- richer topology: best variant `topo_multiscale` 1.000 vs `baselines_union` 0.994 -> margin +0.006, CI [+0.000, +0.019] (max attainable +0.006) -> **NO-GO**; adding all richer views to `baselines_union`: gain +0.000, CI [+0.000, +0.000]

## Task `binary_kf8`

N = 160, classes = 2, trajectories = 8, chance = 0.500.

| Feature set | tier | dim | best acc | LogReg acc (fold std) | RF acc (fold std) | macro-F1 (best clf) | silhouette |
|---|---|---|---|---|---|---|---|
| **descriptor** | ours | 19 | 1.000 | 1.000 (0.000) | 0.994 (0.011) | 1.000 | 0.172 |
| spectral_view | ours-ablation | 11 | 0.554 | 0.554 (0.059) | 0.521 (0.051) | 0.553 | 0.002 |
| topological_view | ours-ablation | 8 | 1.000 | 1.000 (0.000) | 0.994 (0.011) | 1.000 | 0.358 |
| moments | baseline-specified | 6 | 0.500 | 0.500 (0.000) | 0.500 (0.000) | 0.333 | -0.012 |
| spectral_slope | baseline-specified | 1 | 0.500 | 0.500 (0.000) | 0.500 (0.000) | 0.333 | -0.013 |
| raw_pixels | baseline-specified | 256 | 0.826 | 0.786 (0.059) | 0.826 (0.032) | 0.823 | 0.035 |
| power_spectrum | baseline-validity | 85 | 0.500 | 0.500 (0.000) | 0.500 (0.000) | 0.333 | -0.012 |
| histogram_quantiles | baseline-validity | 19 | 0.512 | 0.512 (0.019) | 0.505 (0.020) | 0.511 | -0.012 |
| okubo_weiss | baseline-structure | 6 | 1.000 | 0.994 (0.011) | 1.000 (0.000) | 1.000 | 0.286 |
| gradient_stats | baseline-structure | 7 | 0.988 | 0.988 (0.022) | 0.988 (0.022) | 0.987 | 0.560 |
| increment_flatness | baseline-structure | 12 | 0.981 | 0.981 (0.022) | 0.972 (0.032) | 0.981 | 0.402 |
| physical_structure_all | baseline-structure | 25 | 0.995 | 0.995 (0.015) | 0.980 (0.028) | 0.995 | 0.405 |
| baselines_union | baseline-union | 136 | 0.989 | 0.989 (0.020) | 0.983 (0.026) | 0.989 | 0.082 |
| baselines_union+topo | incremental | 144 | 1.000 | 1.000 (0.000) | 0.990 (0.018) | 1.000 | 0.101 |
| baselines_union+topo_rich | incremental | 466 | 1.000 | 0.994 (0.011) | 1.000 (0.000) | 1.000 | 0.242 |
| topo_persistence_image | ours-rich | 200 | 1.000 | 0.990 (0.015) | 1.000 (0.000) | 1.000 | 0.277 |
| topo_betti_curves | ours-rich | 66 | 1.000 | 1.000 (0.000) | 1.000 (0.000) | 1.000 | 0.413 |
| topo_multiscale | ours-rich | 64 | 1.000 | 1.000 (0.000) | 1.000 (0.000) | 1.000 | 0.396 |
| topo_rich_all | ours-rich | 330 | 1.000 | 0.991 (0.012) | 1.000 (0.000) | 1.000 | 0.323 |
| descriptor_rich | ours-rich | 341 | 1.000 | 0.991 (0.012) | 1.000 (0.000) | 1.000 | 0.311 |
| topological_view_fullres | sensitivity | 8 | 1.000 | 1.000 (0.000) | 1.000 (0.000) | 1.000 | 0.474 |
| descriptor_fullres_topo | sensitivity | 19 | 1.000 | 1.000 (0.000) | 1.000 (0.000) | 1.000 | 0.229 |

Verdicts / key comparisons:

- vs specified baselines: descriptor 1.000 vs `raw_pixels` 0.826 -> margin +0.174, 95% CI [+0.128, +0.216] -> **GO**
- vs strict baselines: descriptor 1.000 vs `okubo_weiss` 1.000 -> margin +0.000, 95% CI [+0.000, +0.000] -> **NO-GO**
- topological view vs best spectrum/one-point baseline (`histogram_quantiles`): 1.000 vs 0.512 (margin +0.488, CI [+0.476, +0.496])
- topological view vs best structure-aware physical baseline (`okubo_weiss`): 1.000 vs 1.000 (margin +0.000, CI [+0.000, +0.000])
- incremental value of topology on top of all baselines: 0.989 -> 1.000 (gain +0.011, CI [+0.000, +0.034])
- richer topology: best variant `topo_persistence_image` 1.000 vs `baselines_union` 0.989 -> margin +0.011, CI [+0.000, +0.034] (max attainable +0.011) -> **NO-GO**; adding all richer views to `baselines_union`: gain +0.011, CI [+0.000, +0.034]

## Task `binary_kf16`

N = 160, classes = 2, trajectories = 8, chance = 0.500.

| Feature set | tier | dim | best acc | LogReg acc (fold std) | RF acc (fold std) | macro-F1 (best clf) | silhouette |
|---|---|---|---|---|---|---|---|
| **descriptor** | ours | 19 | 1.000 | 1.000 (0.000) | 1.000 (0.000) | 1.000 | 0.236 |
| spectral_view | ours-ablation | 11 | 0.517 | 0.517 (0.076) | 0.508 (0.071) | 0.517 | 0.002 |
| topological_view | ours-ablation | 8 | 1.000 | 1.000 (0.000) | 1.000 (0.000) | 1.000 | 0.488 |
| moments | baseline-specified | 6 | 0.500 | 0.500 (0.000) | 0.500 (0.000) | 0.462 | -0.012 |
| spectral_slope | baseline-specified | 1 | 0.500 | 0.500 (0.000) | 0.500 (0.000) | 0.333 | -0.013 |
| raw_pixels | baseline-specified | 256 | 0.834 | 0.834 (0.039) | 0.804 (0.050) | 0.833 | 0.036 |
| power_spectrum | baseline-validity | 85 | 0.500 | 0.500 (0.000) | 0.500 (0.000) | 0.333 | -0.012 |
| histogram_quantiles | baseline-validity | 19 | 0.503 | 0.503 (0.011) | 0.495 (0.015) | 0.502 | -0.012 |
| okubo_weiss | baseline-structure | 6 | 1.000 | 1.000 (0.000) | 0.997 (0.008) | 1.000 | 0.423 |
| gradient_stats | baseline-structure | 7 | 1.000 | 1.000 (0.000) | 1.000 (0.000) | 1.000 | 0.646 |
| increment_flatness | baseline-structure | 12 | 1.000 | 1.000 (0.000) | 1.000 (0.000) | 1.000 | 0.444 |
| physical_structure_all | baseline-structure | 25 | 1.000 | 1.000 (0.000) | 1.000 (0.000) | 1.000 | 0.482 |
| baselines_union | baseline-union | 136 | 1.000 | 1.000 (0.000) | 1.000 (0.000) | 1.000 | 0.096 |
| baselines_union+topo | incremental | 144 | 1.000 | 1.000 (0.000) | 1.000 (0.000) | 1.000 | 0.121 |
| baselines_union+topo_rich | incremental | 466 | 1.000 | 1.000 (0.000) | 1.000 (0.000) | 1.000 | 0.242 |
| topo_persistence_image | ours-rich | 200 | 1.000 | 1.000 (0.000) | 1.000 (0.000) | 1.000 | 0.279 |
| topo_betti_curves | ours-rich | 66 | 1.000 | 1.000 (0.000) | 1.000 (0.000) | 1.000 | 0.404 |
| topo_multiscale | ours-rich | 64 | 1.000 | 1.000 (0.000) | 1.000 (0.000) | 1.000 | 0.376 |
| topo_rich_all | ours-rich | 330 | 1.000 | 1.000 (0.000) | 1.000 (0.000) | 1.000 | 0.315 |
| descriptor_rich | ours-rich | 341 | 1.000 | 1.000 (0.000) | 1.000 (0.000) | 1.000 | 0.304 |
| topological_view_fullres | sensitivity | 8 | 1.000 | 1.000 (0.000) | 1.000 (0.000) | 1.000 | 0.518 |
| descriptor_fullres_topo | sensitivity | 19 | 1.000 | 1.000 (0.000) | 1.000 (0.000) | 1.000 | 0.250 |

Verdicts / key comparisons:

- vs specified baselines: descriptor 1.000 vs `raw_pixels` 0.834 -> margin +0.166, 95% CI [+0.100, +0.224] -> **GO**
- vs strict baselines: descriptor 1.000 vs `okubo_weiss` 1.000 -> margin +0.000, 95% CI [+0.000, +0.000] -> **NO-GO**
- topological view vs best spectrum/one-point baseline (`histogram_quantiles`): 1.000 vs 0.503 (margin +0.497, CI [+0.491, +0.503])
- topological view vs best structure-aware physical baseline (`okubo_weiss`): 1.000 vs 1.000 (margin +0.000, CI [+0.000, +0.000])
- incremental value of topology on top of all baselines: 1.000 -> 1.000 (gain +0.000, CI [+0.000, +0.000])
- richer topology: best variant `topo_persistence_image` 1.000 vs `baselines_union` 1.000 -> margin +0.000, CI [+0.000, +0.000] (max attainable +0.000) -> **NO-GO**; adding all richer views to `baselines_union`: gain +0.000, CI [+0.000, +0.000]

## Task `binary_pooled_iaaft_amplitude`

N = 480, classes = 2, trajectories = 24, chance = 0.500.

| Feature set | tier | dim | best acc | LogReg acc (fold std) | RF acc (fold std) | macro-F1 (best clf) | silhouette |
|---|---|---|---|---|---|---|---|
| **descriptor** | ours | 19 | 0.989 | 0.989 (0.011) | 0.988 (0.009) | 0.989 | 0.020 |
| spectral_view | ours-ablation | 11 | 0.580 | 0.580 (0.041) | 0.575 (0.052) | 0.580 | 0.006 |
| topological_view | ours-ablation | 8 | 0.983 | 0.889 (0.030) | 0.983 (0.009) | 0.983 | 0.037 |
| moments | baseline-specified | 6 | 0.500 | 0.500 (0.000) | 0.500 (0.000) | 0.333 | -0.004 |
| spectral_slope | baseline-specified | 1 | 0.500 | 0.500 (0.000) | 0.500 (0.000) | 0.333 | -0.004 |
| raw_pixels | baseline-specified | 256 | 0.782 | 0.700 (0.054) | 0.782 (0.057) | 0.781 | 0.010 |
| power_spectrum | baseline-validity | 85 | 0.502 | 0.502 (0.003) | 0.501 (0.005) | 0.501 | -0.004 |
| histogram_quantiles | baseline-validity | 19 | 0.500 | 0.500 (0.000) | 0.500 (0.000) | 0.333 | -0.004 |
| okubo_weiss | baseline-structure | 6 | 0.974 | 0.974 (0.019) | 0.965 (0.025) | 0.974 | 0.248 |
| gradient_stats | baseline-structure | 7 | 1.000 | 0.998 (0.004) | 1.000 (0.000) | 1.000 | 0.451 |
| increment_flatness | baseline-structure | 12 | 0.976 | 0.976 (0.012) | 0.972 (0.016) | 0.976 | 0.274 |
| physical_structure_all | baseline-structure | 25 | 0.997 | 0.997 (0.005) | 0.995 (0.008) | 0.997 | 0.309 |
| baselines_union | baseline-union | 136 | 0.999 | 0.999 (0.004) | 0.989 (0.014) | 0.999 | 0.077 |
| baselines_union+topo | incremental | 144 | 0.999 | 0.999 (0.004) | 0.991 (0.013) | 0.999 | 0.075 |
| baselines_union+topo_rich | incremental | 466 | 0.998 | 0.996 (0.005) | 0.998 (0.004) | 0.998 | 0.094 |
| topo_persistence_image | ours-rich | 200 | 0.992 | 0.989 (0.012) | 0.992 (0.008) | 0.992 | 0.129 |
| topo_betti_curves | ours-rich | 66 | 0.987 | 0.930 (0.026) | 0.987 (0.016) | 0.987 | 0.126 |
| topo_multiscale | ours-rich | 64 | 1.000 | 1.000 (0.000) | 1.000 (0.000) | 1.000 | 0.058 |
| topo_rich_all | ours-rich | 330 | 0.997 | 0.997 (0.004) | 0.997 (0.005) | 0.997 | 0.102 |
| descriptor_rich | ours-rich | 341 | 0.999 | 0.996 (0.005) | 0.999 (0.003) | 0.999 | 0.097 |
| topological_view_fullres | sensitivity | 8 | 1.000 | 1.000 (0.000) | 0.999 (0.002) | 1.000 | 0.117 |
| descriptor_fullres_topo | sensitivity | 19 | 1.000 | 1.000 (0.000) | 1.000 (0.002) | 1.000 | 0.052 |

Verdicts / key comparisons:

- vs specified baselines: descriptor 0.989 vs `raw_pixels` 0.782 -> margin +0.207, 95% CI [+0.172, +0.239] -> **GO**
- vs strict baselines: descriptor 0.989 vs `gradient_stats` 1.000 -> margin -0.011, 95% CI [-0.018, -0.005] -> **NO-GO**
- topological view vs best spectrum/one-point baseline (`power_spectrum`): 0.983 vs 0.502 (margin +0.481, CI [+0.470, +0.492])
- topological view vs best structure-aware physical baseline (`gradient_stats`): 0.983 vs 1.000 (margin -0.017, CI [-0.029, -0.007])
- incremental value of topology on top of all baselines: 0.999 -> 0.999 (gain +0.000, CI [+0.000, +0.000])
- richer topology: best variant `topo_multiscale` 1.000 vs `baselines_union` 0.999 -> margin +0.001, CI [+0.000, +0.002] (max attainable +0.001) -> **NO-GO**; adding all richer views to `baselines_union`: gain -0.001, CI [-0.004, +0.001]

## Existing StratifiedKFold evaluator (no grouping) for comparison

| Task | Feature set | best acc (StratifiedKFold) | best acc (grouped CV) |
|---|---|---|---|
| binary_pooled | descriptor | 0.990 | 0.989 |
| binary_pooled | spectral_view | 0.596 | 0.580 |
| binary_pooled | topological_view | 0.983 | 0.983 |
| binary_pooled | moments | 0.463 | 0.500 |
| binary_pooled | spectral_slope | 0.475 | 0.500 |
| binary_pooled | raw_pixels | 0.783 | 0.782 |
| multiclass6 | descriptor | 0.992 | 0.991 |
| multiclass6 | spectral_view | 0.594 | 0.571 |
| multiclass6 | topological_view | 0.988 | 0.982 |
| multiclass6 | moments | 0.231 | 0.334 |
| multiclass6 | spectral_slope | 0.458 | 0.500 |
| multiclass6 | raw_pixels | 0.606 | 0.601 |

## Figures

- `figures/fields_coherent_vs_iaaft.png` - example coherent fields, their IAAFT surrogates, and the matched spectra / histograms.
- `figures/accuracy_binary.png`, `figures/accuracy_multiclass.png` - grouped-CV accuracy per feature set.
- `figures/topology_features.png` - distribution of the topological-view features for coherent vs surrogate fields.
- `figures/rich_topology_margins.png` - margin of the best richer topological vectorization over `baselines_union` per task, with the +0.05 GO threshold.

