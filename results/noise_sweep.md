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

Key comparisons per condition (margin, 95% paired group-bootstrap CI):

- `eta0.1_raw`: descriptor - best baseline (`physical_structure_all`) = -0.017 [-0.026, -0.008] (NO-GO at the 0.05 rule); topo view - best structure baseline (`physical_structure_all`) = -0.020 [-0.034, -0.008]; topology added to all baselines: +0.000 [+0.000, +0.000]
- `eta0.1_smoothed`: descriptor - best baseline (`physical_structure_all`) = -0.003 [-0.008, +0.001] (NO-GO at the 0.05 rule); topo view - best structure baseline (`physical_structure_all`) = -0.005 [-0.011, +0.000]; topology added to all baselines: +0.002 [+0.000, +0.006]
- `eta0.3_raw`: descriptor - best baseline (`physical_structure_all`) = -0.033 [-0.049, -0.019] (NO-GO at the 0.05 rule); topo view - best structure baseline (`physical_structure_all`) = -0.033 [-0.048, -0.019]; topology added to all baselines: +0.004 [+0.000, +0.010]
- `eta0.3_smoothed`: descriptor - best baseline (`physical_structure_all`) = -0.003 [-0.006, +0.000] (NO-GO at the 0.05 rule); topo view - best structure baseline (`physical_structure_all`) = -0.005 [-0.010, +0.000]; topology added to all baselines: +0.003 [+0.000, +0.006]
- `eta1_raw`: descriptor - best baseline (`baselines_union`) = -0.290 [-0.334, -0.243] (NO-GO at the 0.05 rule); topo view - best structure baseline (`physical_structure_all`) = -0.342 [-0.373, -0.311]; topology added to all baselines: -0.011 [-0.031, +0.009]
- `eta1_smoothed`: descriptor - best baseline (`gradient_stats`) = -0.026 [-0.038, -0.016] (NO-GO at the 0.05 rule); topo view - best structure baseline (`gradient_stats`) = -0.022 [-0.032, -0.012]; topology added to all baselines: +0.000 [+0.000, +0.000]

Elapsed: 1558.6 s (wall clock).

