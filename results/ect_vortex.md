# ECT / DECT study (generated tables)

Design, rule and interpretation: [`docs/ECT_RESULTS.md`](../docs/ECT_RESULTS.md). Repeated StratifiedGroupKFold (3 repeats x 4 folds, group = trajectory); best of LogReg / RF for feature views, the DECT CNN head for `dect`; paired group-bootstrap 95% CIs; ★ = margin >= +0.05 with CI above 0 and BH q < 0.05.

DECT enabled: True.

## Verdict per task

| Task | primary views | best primary view | margin [95% CI] | headroom | verdict |
|---|---|---|---|---|---|
| iaaft_binary | ect, dect | `ect` | -0.033 [-0.047, -0.022] | 0.000 | **NO-GO** |
| vortex_merger | ph_counts, ect, dect | `ph_counts` | +0.000 [+0.000, +0.000] | 0.000 | **NO-GO** |

### Task A: coherent turbulence vs IAAFT surrogate (structure-only)

`baselines_union` = 1.000; headroom (largest attainable margin) = 0.000. Best structure baseline: `gradient_stats`.

| View | accuracy | margin vs `baselines_union` [95% CI] | BH q | margin vs best structure baseline [95% CI] |
|---|---|---|---|---|
| `ph_counts` (H0/H1 counts (persistence >= 0.5, 1, 2)) | 0.999 | -0.001 [-0.002, +0.000] | 1.000 | -0.001 [-0.002, +0.000] |
| `x2_ph` (X2: sig6 PH summaries) | 1.000 | +0.000 [+0.000, +0.000] | 1.000 | +0.000 [+0.000, +0.000] |
| `topological_view_fullres` (8 PH summaries, 256^2) | 1.000 | +0.000 [+0.000, +0.000] | 1.000 | +0.000 [+0.000, +0.000] |
| `topo_betti_curves` (Betti curves) | 0.987 | -0.013 [-0.026, -0.002] | 1.000 | -0.013 [-0.026, -0.002] |
| `topo_persistence_image` (persistence images) | 0.991 | -0.009 [-0.021, -0.001] | 1.000 | -0.009 [-0.021, -0.001] |
| `topo_multiscale` (multi-scale filtration) | 1.000 | +0.000 [+0.000, +0.000] | 1.000 | +0.000 [+0.000, +0.000] |
| `ect` (ECT (exact, injective)) | 0.967 | -0.033 [-0.047, -0.022] | 1.000 | -0.033 [-0.047, -0.022] |
| `dect` (DECT (learned directions/thresholds)) | 0.888 | -0.112 [-0.141, -0.084] | 1.000 | -0.112 [-0.141, -0.084] |
| `okubo_weiss` (Okubo-Weiss) | 0.974 | – | – | – |
| `gradient_stats` (gradient statistics) | 1.000 | – | – | – |
| `increment_flatness` (increment flatness) | 0.976 | – | – | – |
| `physical_structure_all` (all 3 structure baselines) | 0.998 | – | – | – |
| `moments` (moments) | 0.501 | – | – | – |
| `power_spectrum` (power spectrum) | 0.500 | – | – | – |
| `histogram_quantiles` (histogram quantiles) | 0.504 | – | – | – |
| `baselines_union` (baselines_union) | 1.000 | – | – | – |
| `baselines_union+ect` (baselines_union + ECT) | 0.998 | -0.002 [-0.006, +0.000] | – | -0.002 [-0.006, +0.000] |
| `baselines_union+ph_counts` (baselines_union + H0/H1 counts) | 1.000 | +0.000 [+0.000, +0.000] | – | +0.000 [+0.000, +0.000] |

### Task B: merged vs separate vortex pair (connectivity)

`baselines_union` = 1.000; headroom (largest attainable margin) = 0.000. Best structure baseline: `physical_structure_all`.

| View | accuracy | margin vs `baselines_union` [95% CI] | BH q | margin vs best structure baseline [95% CI] |
|---|---|---|---|---|
| `ph_counts` (H0/H1 counts (persistence >= 0.5, 1, 2)) | 1.000 | +0.000 [+0.000, +0.000] | 1.000 | +0.001 [+0.000, +0.002] |
| `x2_ph` (X2: sig6 PH summaries) | 1.000 | +0.000 [+0.000, +0.000] | 1.000 | +0.001 [+0.000, +0.002] |
| `topological_view_fullres` (8 PH summaries, 256^2) | 0.986 | -0.014 [-0.028, -0.003] | 1.000 | -0.013 [-0.028, -0.002] |
| `topo_betti_curves` (Betti curves) | 0.840 | -0.160 [-0.211, -0.111] | 1.000 | -0.159 [-0.210, -0.111] |
| `topo_persistence_image` (persistence images) | 0.958 | -0.042 [-0.065, -0.022] | 1.000 | -0.042 [-0.065, -0.021] |
| `topo_multiscale` (multi-scale filtration) | 0.988 | -0.012 [-0.024, -0.003] | 1.000 | -0.012 [-0.024, -0.002] |
| `ect` (ECT (exact, injective)) | 0.944 | -0.056 [-0.081, -0.033] | 1.000 | -0.056 [-0.081, -0.032] |
| `dect` (DECT (learned directions/thresholds)) | 0.690 | -0.310 [-0.388, -0.235] | 1.000 | -0.310 [-0.388, -0.235] |
| `okubo_weiss` (Okubo-Weiss) | 0.951 | – | – | – |
| `gradient_stats` (gradient statistics) | 0.986 | – | – | – |
| `increment_flatness` (increment flatness) | 0.994 | – | – | – |
| `physical_structure_all` (all 3 structure baselines) | 0.999 | – | – | – |
| `moments` (moments) | 0.872 | – | – | – |
| `power_spectrum` (power spectrum) | 1.000 | – | – | – |
| `histogram_quantiles` (histogram quantiles) | 0.976 | – | – | – |
| `baselines_union` (baselines_union) | 1.000 | – | – | – |
| `baselines_union+ect` (baselines_union + ECT) | 0.998 | -0.002 [-0.006, +0.000] | – | -0.001 [-0.006, +0.002] |
| `baselines_union+ph_counts` (baselines_union + H0/H1 counts) | 1.000 | +0.000 [+0.000, +0.000] | – | +0.001 [+0.000, +0.002] |

Elapsed: 8973.2 s.

