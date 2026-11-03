# Task C: a scale-matched core-count task (one vortex core vs two)

**Status: PRE-REGISTERED, results pending.** The fields, the outcome-blind
design phase (matching and balance gate), the views and the decision rule
below are committed before any descriptor view or benchmark classifier is
computed on task-C data.
- Code: `src/sims/core_count.py` and `src/eval/matched_core_count.py`.
- Configuration: `config/matched_core_count.yaml`.
- Results are added below the line in a later commit.

## Why this task

On task B (merged vs separate vortex pairs, `docs/ECT_RESULTS.md`), the H0/H1
counts were perfect, but so was the power spectrum alone. A merger conserves
circulation and roughly area, so the merged core is about twice the size of
each separate core, and the spectrum reads that scale change.

For identical round cores, the count is fixed by non-topological statistics:
the spectrum fixes the core size, the one-point histogram fixes the occupied
area, and count = area / size². Task C breaks that identity:
- the label is still the number of coherent cores (1 vs 2);
- the shapes are randomized so that single cores and pairs overlap in scale
  and occupancy;
- the classes are matched one to one on descriptor-free design statistics
  before any classifier sees them.

This is the follow-up proposed at the end of `docs/ECT_RESULTS.md`.

## Fields (`src/sims/core_count.py`)

Every core has the profile ω(ρ) = A exp(−ρ^p). Here ρ is the (elliptical)
radius in units of the core radius, and p is the steepness: p = 2 is a
Gaussian, a large p is flat-topped.

| | class 0: one core | class 1: two cores |
|---|---|---|
| shape | area-preserving ellipse, aspect ratio U[1.5, 3.5], random orientation | two round cores |
| size | equivalent radius U[0.30, 0.75] | larger radius U[0.30, 0.60]; smaller / larger U[0.35, 0.80] |
| steepness p | U[3, 8] | U[2, 5] |
| peak | A = 10 | 10 for the stronger core; weaker / stronger U[0.6, 1.0] (which core is weaker is random) |
| placement | random | random centre and orientation; separation U[4, 7] × mean radius, at most 3.0 |

**Why these families.** A single core reaches the area-to-scale ratio of a
pair only when it is elongated and flat-topped. A pair reaches the ratio of a
single core only when its cores are unequal. Round monotone profiles are
stable. Elongated cores can pinch off or shed fragments; the validity check
below removes those snapshots.

**Shared setup (both classes):**
- Background: weak broadband vorticity (rms 5% of the peak, |k| ∈ [4, 24]),
  as in task B.
- Solver: unforced IF-RK4 at 256² (ν = 2·10⁻⁴, dt = 0.005).
- Snapshots: t = 1.0, 1.5, …, 4.5 (8 per trajectory).
- Pool: 120 trajectories per class, 1 920 snapshots in total.
- Every field is standardized (zero mean, unit variance).
- The CV group is the trajectory.

## Design phase (outcome-blind; Rubin 2007)

Only the core count and four design statistics are computed in this phase. No
descriptor view and no benchmark classifier is involved.

1. **Validity.** Keep a snapshot only if its descriptor-free core count
   (connected regions of {ω > 0.5 max ω} on the torus) equals the design: 1
   or 2.
2. **Design statistics.**
   - spectral centroid: power-weighted mean |k|;
   - spectral width: power-weighted SD of |k|;
   - occupied area: fraction of the domain above half the peak;
   - standardized peak: max / std.

   Area and peak enter in logs.
3. **Propensity-score matching** (Rosenbaum & Rubin 1983).
   - A logistic regression of the class on the standardized design
     statistics, expanded to linear, squared and pairwise terms (so variances
     are balanced too), is fitted on the valid pool.
   - Matching is greedy and one to one on the logit, without replacement,
     closest pairs first.
   - The caliper is 0.2 SD of the logit (Austin 2011).
4. **Balance gate** on the matched set. The task is analysed only if all four
   checks pass:
   - every |standardized mean difference| < 0.1;
   - every variance ratio in [0.5, 2];
   - at least 120 matched pairs;
   - a grouped-CV logistic regression on the four design statistics predicts
     the class with AUC ≤ 0.60.
5. **Freeze.** The matched set is stored with a hash of its indices. The
   analysis refuses to run if the gate failed or the set changed.

**Pilots** (different seed, 24 trajectories × 4 snapshots per class;
`results/core_count_pilot_v{1,2,3}.json` and figures):

| pilot | single core | pair | matching | outcome |
|---|---|---|---|---|
| v1 | round | independent radii in [0.2, 0.5] | Mahalanobis, caliper 0.25 | 0 pairs. The classes differ jointly: area × centroid² has median 0.32 vs 0.47, the count = area / size² identity. |
| v2 | elliptical, aspect up to 3 | same as v1 | same | 0 pairs. Propensity AUC of the design statistics stays about 0.96 (at t = 0). |
| v3 | ranges above | ranges above | linear propensity | 50 pairs. Means balanced, but variance ratios about 0.5. |

- The ranges for v3 were chosen at t = 0, by the propensity AUC of the design
  statistics (0.94 → 0.66).
- After v3, the propensity model became quadratic, chosen on the cached v3
  pool.
- All pilot decisions used design statistics and core counts only.

**Design phase on the final pool** (seed 0; `results/matched_core_count_design.json`,
`results/figures/core_count_matching.png`). It was run once with the settings
above and passed the gate on the first attempt. Nothing was iterated on the
final pool.

- **Pool:** 240 trajectories, 1 920 snapshots (840 s on 4 cores).
- **Validity:** 93.3% of one-core snapshots and 97.3% of two-core snapshots
  have the designed core count. The rest are dropped.
  - Dropped one-core snapshots: counts of 2 (45), 3 (13), 4 (4), 5 and 8
    (1 each). Elongated cores pinch off or shed fragments above half the peak.
  - Dropped two-core snapshots: counts of 1 (6) and 3 to 5 (20).
- **Balance:**

  | design statistic | SMD before | SMD after | variance ratio before | variance ratio after |
  |---|---|---|---|---|
  | spectral centroid | +0.15 | −0.002 | 0.72 | 1.04 |
  | spectral width | −0.49 | +0.027 | 0.65 | 0.99 |
  | occupied area (log) | −0.24 | +0.026 | 0.59 | 1.08 |
  | standardized peak (log) | +0.36 | −0.008 | 0.54 | 1.06 |
  | grouped-CV AUC of LogReg on the four statistics | 0.83 | **0.46** | | |

- **Matched set:** 436 pairs, so 872 fields.
  - 200 trajectories: 107 one-core and 93 two-core.
  - All 8 snapshot times are represented (86–142 fields each).
  - Set hash `dcb2434ec859aba3`.
- **The matched set is typical of the pool.** Median generation parameters
  barely move:
  - one-core aspect ratio 2.71 → 2.61, steepness 5.29 → 5.23;
  - two-core size ratio 0.56 → 0.54, amplitude ratio 0.76 → 0.75.

## DECT head (developed on task B only)

On task B, the original DECT under-fitted. Its head ends in global average
pooling ("gap"), which discards where along the threshold axis the Euler
characteristic changes (`docs/ECT_RESULTS.md`, diagnostics).

- **Candidate.** `src/descriptor/dect.py` gains a "flat" head: the same two
  convolutions, then flatten and a linear layer. "gap" stays the default, so
  tasks A and B are reproducible.
- **Development.** `src/eval/dect_dev.py` trains both heads on two task-B
  splits (repeat 0, folds 0 and 1), with the unchanged training budget:
  30 epochs, batch 32, Adam lr 3·10⁻³, weight decay 10⁻⁴.
- **Selection rule, written before the development run finished:** use "flat"
  for task C if its mean final test accuracy on the two task-B splits is
  higher than that of "gap"; otherwise keep "gap".
- No task-C data is used for this choice.

**Development result** (`results/dect_head_dev.json`,
`results/figures/dect_head_dev.png`):

| head | final train accuracy (split 0 / 1) | final test accuracy (split 0 / 1) | mean test |
|---|---|---|---|
| gap (original) | 0.783 / 0.775 | 0.700 / 0.717 | 0.708 |
| **flat** | 1.000 / 1.000 | 0.908 / 0.917 | **0.912** |

**Task C uses `head: flat`**, by the rule above.
- The original head's split-0 numbers reproduce the earlier task-B
  diagnostic exactly.
- Task B's exact ECT with LogReg still scores higher (0.944 in the full CV).
  So DECT remains a weaker learner here, but it no longer under-fits.

## Views

| Group | Views |
|---|---|
| Topological | `ph_counts` (H0 / H1 counts, periodic), `x2_ph`, `topological_view_fullres`, `topo_betti_curves`, `topo_persistence_image`, `topo_multiscale`, **`ect`**, **`ect_boundary_aware`**, **`dect`** |
| Non-topological | `okubo_weiss`, `gradient_stats`, `increment_flatness`, `physical_structure_all`, `moments`, `power_spectrum`, `histogram_quantiles`, **`baselines_union`** |
| Incremental | `baselines_union+ect`, `baselines_union+ph_counts` |
| Sanity | `design_stats` (the four matching statistics; expected near chance) |

- **`ect_boundary_aware`** is new here. It is the exact ECT after a periodic
  roll that puts the quietest row and column on the domain edges. On task B
  it was a post-hoc diagnostic; here it is pre-registered.
- **Protocol.** As in the ECT study: repeated StratifiedGroupKFold
  (4 folds × 3 repeats; group = trajectory); best of LogReg / RF for feature
  views; paired group-bootstrap 95% CIs (2 000 resamples).

## Decision rule (fixed before running)

- **GO** if at least one primary view reaches all three:
  - margin over `baselines_union` ≥ +0.05;
  - 95% CI lower bound > 0;
  - Benjamini–Hochberg q < 0.05, over the 9 topological views of task C.
- **Primary views:** `ph_counts`, `ect`, `ect_boundary_aware`, `dect`.
- **Headroom** (1 − accuracy of `baselines_union`) is reported. If it is
  below 0.05, a GO is arithmetically impossible and the verdict says so.
- All other views, and the margins over the best single structure baseline,
  are reported but not used for the verdict.

**Stated in advance: what could still give the label away.**
- **Two-point statistics.** Matching the spectral centroid and width does not
  remove every two-point signature of a pair. Two cores at distance d add a
  cross-correlation peak at lag d, which shows up as oscillations
  ∝ J0(k d) in the radially averaged spectrum; a single elliptical core has
  none.
- **Edges and gradients.** Two cores have more edge per unit area than one,
  which gradient and increment statistics can see.

The baselines may therefore still separate the classes. If they do, the
verdict is NO-GO, and the report names the baseline that does it.

---

## Results

*Pending: added after the pre-registered run.*
