# Matched core-count study (generated tables)

Design, rule and interpretation: [`docs/MATCHED_CORE_COUNT.md`](../docs/MATCHED_CORE_COUNT.md). Repeated StratifiedGroupKFold (3 repeats x 4 folds, group = trajectory); best of LogReg / RF for feature views, the DECT CNN head for `dect`; paired group-bootstrap 95% CIs; ★ = margin >= +0.05 with CI above 0 and BH q < 0.05.

DECT enabled: True.

## Verdict per task

| Task | primary views | best primary view | margin [95% CI] | headroom | verdict |
|---|---|---|---|---|---|
| matched_core_count | ph_counts, ect, ect_boundary_aware, dect | `ph_counts` | -0.001 [-0.017, +0.017] | 0.023 | **NO-GO** |

### Task C: one core vs two cores, scale-matched (connectivity)

`baselines_union` = 0.977; headroom (largest attainable margin) = 0.023. Best structure baseline: `physical_structure_all`.

| View | accuracy | margin vs `baselines_union` [95% CI] | BH q | margin vs best structure baseline [95% CI] |
|---|---|---|---|---|
| `ph_counts` (H0/H1 counts (persistence >= 0.5, 1, 2)) | 0.976 | -0.001 [-0.017, +0.017] | 1.000 | +0.010 [-0.010, +0.028] |
| `x2_ph` (X2: sig6 PH summaries) | 1.000 | +0.023 [+0.009, +0.041] | 0.005 | +0.033 [+0.017, +0.051] |
| `topological_view_fullres` (8 PH summaries, 256^2) | 0.827 | -0.150 [-0.189, -0.111] | 1.000 | -0.140 [-0.181, -0.101] |
| `topo_betti_curves` (Betti curves) | 0.924 | -0.053 [-0.080, -0.025] | 1.000 | -0.042 [-0.067, -0.017] |
| `topo_persistence_image` (persistence images) | 0.918 | -0.059 [-0.090, -0.028] | 1.000 | -0.049 [-0.081, -0.020] |
| `topo_multiscale` (multi-scale filtration) | 0.919 | -0.058 [-0.092, -0.029] | 1.000 | -0.048 [-0.083, -0.017] |
| `ect` (ECT (exact, injective)) | 0.779 | -0.198 [-0.243, -0.156] | 1.000 | -0.187 [-0.230, -0.147] |
| `ect_boundary_aware` (ECT, boundary-aware (periodic roll)) | 0.869 | -0.109 [-0.144, -0.079] | 1.000 | -0.098 [-0.133, -0.066] |
| `dect` (DECT (learned directions/thresholds)) | 0.795 | -0.182 [-0.223, -0.146] | 1.000 | -0.171 [-0.211, -0.134] |
| `okubo_weiss` (Okubo-Weiss) | 0.877 | – | – | – |
| `gradient_stats` (gradient statistics) | 0.884 | – | – | – |
| `increment_flatness` (increment flatness) | 0.872 | – | – | – |
| `physical_structure_all` (all 3 structure baselines) | 0.967 | – | – | – |
| `moments` (moments) | 0.834 | – | – | – |
| `power_spectrum` (power spectrum) | 0.901 | – | – | – |
| `histogram_quantiles` (histogram quantiles) | 0.900 | – | – | – |
| `baselines_union` (baselines_union) | 0.977 | – | – | – |
| `baselines_union+ect` (baselines_union + ECT) | 0.936 | -0.041 [-0.064, -0.020] | – | -0.031 [-0.051, -0.013] |
| `baselines_union+ph_counts` (baselines_union + H0/H1 counts) | 0.987 | +0.010 [+0.002, +0.019] | – | +0.020 [+0.003, +0.037] |
| `design_stats` (matching statistics (sanity)) | 0.579 | – | – | – |

Elapsed: 2517.9 s.

