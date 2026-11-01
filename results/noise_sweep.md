# Noise stress test (exploratory, post hoc - not part of the verdict)

Noise model: x + eta * N(0,1) per pixel, independent for every field (coherent and surrogate), then re-standardized; 'smoothed' adds a periodic Gaussian low-pass, sigma = 2.0 px, applied identically before every feature set.

Binary coherent-vs-IAAFT task, 480 fields, grouped CV (3 repeats x 4 folds, group = trajectory). Entries: best of LogReg / RF accuracy.

| Feature set | eta0.1 raw | eta0.1 smoothed | eta0.3 raw | eta0.3 smoothed | eta1 raw | eta1 smoothed |
|---|---|---|---|---|---|---|
| **descriptor** (ours) | 0.983 | 0.996 | 0.965 | 0.997 | 0.665 | 0.972 |
| spectral_view (ours-ablation) | 0.584 | 0.583 | 0.557 | 0.585 | 0.519 | 0.601 |
| topological_view (ours-ablation) | 0.980 | 0.994 | 0.966 | 0.995 | 0.597 | 0.977 |
| moments (baseline-specified) | 0.512 | 0.759 | 0.504 | 0.756 | 0.500 | 0.759 |
| spectral_slope (baseline-specified) | 0.501 | 0.500 | 0.502 | 0.512 | 0.515 | 0.537 |
| raw_pixels (baseline-specified) | 0.769 | 0.779 | 0.772 | 0.774 | 0.707 | 0.787 |
| power_spectrum (baseline-validity) | 0.511 | 0.515 | 0.500 | 0.522 | 0.508 | 0.519 |
| histogram_quantiles (baseline-validity) | 0.527 | 0.771 | 0.496 | 0.767 | 0.505 | 0.773 |
| okubo_weiss (baseline-structure) | 0.975 | 0.940 | 0.977 | 0.942 | 0.928 | 0.944 |
| gradient_stats (baseline-structure) | 0.998 | 0.996 | 0.956 | 0.998 | 0.612 | 0.999 |
| increment_flatness (baseline-structure) | 0.986 | 0.977 | 0.971 | 0.980 | 0.851 | 0.980 |
| physical_structure_all (baseline-structure) | 1.000 | 0.999 | 0.999 | 1.000 | 0.939 | 0.998 |
| baselines_union (baseline-union) | 1.000 | 0.994 | 0.996 | 0.993 | 0.955 | 0.996 |
| baselines_union+topo (incremental) | 1.000 | 0.997 | 1.000 | 0.996 | 0.944 | 0.996 |
| topo_persistence_image (ours-rich) | 0.992 | 0.983 | 0.976 | 0.981 | 0.896 | 0.969 |
| topo_betti_curves (ours-rich) | 0.964 | 0.997 | 0.933 | 0.995 | 0.845 | 0.994 |
| topo_multiscale (ours-rich) | 1.000 | 0.999 | 1.000 | 0.997 | 0.999 | 0.994 |
| topo_rich_all (ours-rich) | 1.000 | 0.997 | 0.999 | 0.996 | 0.991 | 0.998 |
| descriptor_rich (ours-rich) | 1.000 | 0.998 | 0.999 | 0.995 | 0.993 | 0.998 |
| baselines_union+topo_rich (incremental) | 0.999 | 0.992 | 0.998 | 0.991 | 1.000 | 0.991 |

Key comparisons per condition (margin, 95% paired group-bootstrap CI):

- `eta0.1_raw`: descriptor - best baseline (`physical_structure_all`) = -0.017 [-0.026, -0.008] (NO-GO at the 0.05 rule); topo view - best structure baseline (`physical_structure_all`) = -0.020 [-0.034, -0.008]; topology added to all baselines: +0.000 [+0.000, +0.000]
- `eta0.1_smoothed`: descriptor - best baseline (`physical_structure_all`) = -0.003 [-0.008, +0.001] (NO-GO at the 0.05 rule); topo view - best structure baseline (`physical_structure_all`) = -0.005 [-0.011, +0.000]; topology added to all baselines: +0.002 [+0.000, +0.006]
- `eta0.3_raw`: descriptor - best baseline (`physical_structure_all`) = -0.033 [-0.049, -0.019] (NO-GO at the 0.05 rule); topo view - best structure baseline (`physical_structure_all`) = -0.033 [-0.048, -0.019]; topology added to all baselines: +0.004 [+0.000, +0.010]
- `eta0.3_smoothed`: descriptor - best baseline (`physical_structure_all`) = -0.003 [-0.006, +0.000] (NO-GO at the 0.05 rule); topo view - best structure baseline (`physical_structure_all`) = -0.005 [-0.010, +0.000]; topology added to all baselines: +0.003 [+0.000, +0.006]
- `eta1_raw`: descriptor - best baseline (`baselines_union`) = -0.290 [-0.334, -0.243] (NO-GO at the 0.05 rule); topo view - best structure baseline (`physical_structure_all`) = -0.342 [-0.373, -0.311]; topology added to all baselines: -0.011 [-0.031, +0.009]
- `eta1_smoothed`: descriptor - best baseline (`gradient_stats`) = -0.026 [-0.038, -0.016] (NO-GO at the 0.05 rule); topo view - best structure baseline (`gradient_stats`) = -0.022 [-0.032, -0.012]; topology added to all baselines: +0.000 [+0.000, +0.000]

## Richer topological vectorizations vs `baselines_union`

Cells: accuracy (margin vs `baselines_union` [95% CI]); `max attainable margin` = 1 - accuracy of `baselines_union`.

| Task | baselines_union | max attainable margin | old topo view (64^2) | `topo_persistence_image` | `topo_betti_curves` | `topo_multiscale` | `topo_rich_all` | `descriptor_rich` | union + rich topo (gain) | any >= +0.05? |
|---|---|---|---|---|---|---|---|---|---|---|
| eta0.1_raw | 1.000 | +0.000 | 0.980 | 0.992 (-0.008 [-0.017, -0.002]) | 0.964 (-0.036 [-0.058, -0.016]) | 1.000 (+0.000 [+0.000, +0.000]) | 1.000 (+0.000 [+0.000, +0.000]) | 1.000 (+0.000 [+0.000, +0.000]) | 0.999 (-0.001 [-0.003, +0.000]) | no |
| eta0.1_smoothed | 0.994 | +0.006 | 0.994 | 0.983 (-0.011 [-0.022, -0.001]) | 0.997 (+0.003 [-0.004, +0.012]) | 0.999 (+0.005 [-0.001, +0.014]) | 0.997 (+0.003 [-0.003, +0.011]) | 0.998 (+0.003 [-0.001, +0.011]) | 0.992 (-0.003 [-0.013, +0.009]) | no |
| eta0.3_raw | 0.996 | +0.004 | 0.966 | 0.976 (-0.020 [-0.033, -0.009]) | 0.933 (-0.063 [-0.095, -0.037]) | 1.000 (+0.004 [+0.000, +0.010]) | 0.999 (+0.003 [+0.000, +0.008]) | 0.999 (+0.003 [+0.000, +0.009]) | 0.998 (+0.002 [+0.000, +0.006]) | no |
| eta0.3_smoothed | 0.993 | +0.007 | 0.995 | 0.981 (-0.012 [-0.027, -0.001]) | 0.995 (+0.002 [-0.006, +0.013]) | 0.997 (+0.003 [-0.003, +0.013]) | 0.996 (+0.003 [-0.004, +0.012]) | 0.995 (+0.002 [-0.006, +0.012]) | 0.991 (-0.002 [-0.013, +0.010]) | no |
| eta1_raw | 0.955 | +0.045 | 0.597 | 0.896 (-0.059 [-0.091, -0.030]) | 0.845 (-0.110 [-0.143, -0.076]) | 0.999 (+0.044 [+0.024, +0.067]) | 0.991 (+0.036 [+0.016, +0.059]) | 0.993 (+0.038 [+0.017, +0.061]) | 1.000 (+0.045 [+0.024, +0.068]) | no |
| eta1_smoothed | 0.996 | +0.004 | 0.977 | 0.969 (-0.026 [-0.051, -0.008]) | 0.994 (-0.001 [-0.010, +0.010]) | 0.994 (-0.001 [-0.012, +0.010]) | 0.998 (+0.002 [-0.006, +0.012]) | 0.998 (+0.002 [-0.006, +0.012]) | 0.991 (-0.005 [-0.010, -0.001]) | no |

Elapsed: 1891.9 s (wall clock).

