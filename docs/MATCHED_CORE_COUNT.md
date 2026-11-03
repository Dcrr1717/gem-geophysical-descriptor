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
single core only when its cores are unequal. Monotone profiles are stable, so
no core splits or sheds strong filaments within the window.

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
| v1 | round | equal | Mahalanobis, caliper 0.25 | 0 pairs. The classes differ jointly: area × centroid² has median 0.32 vs 0.47, the count = area / size² identity. |
| v2 | elliptical, aspect up to 3 | equal | same | 0 pairs. Propensity AUC of the design statistics stays about 0.96. |
| v3 | ranges above | ranges above | linear propensity | 50 pairs. Means balanced, but variance ratios about 0.5. |

- The ranges for v3 were chosen at t = 0, by the propensity AUC of the design
  statistics (0.94 → 0.66).
- After v3, the propensity model became quadratic, chosen on the cached v3
  pool.
- All pilot decisions used design statistics and core counts only.

@@DESIGN@@

## DECT head (developed on task B only)

@@DECT@@

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
