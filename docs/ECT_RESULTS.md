# Richer topology (ECT, DECT) and a genuinely topological task

**Status: PRE-REGISTERED, results pending.** The design, views and decision
rule below are committed before any ECT/DECT classification was run. The
code is in `src/descriptor/ect.py`, `src/descriptor/dect.py`,
`src/sims/vortex_events.py` and `src/eval/ect_benchmark.py`; the
configuration is `config/ect_vortex.yaml`. Results are added below the line
in a later commit.

## Why this study

The clean structure-only benchmark (coherent 2D turbulence vs IAAFT
surrogates) is a confirmed NO-GO for a "better than baselines" claim
(`docs/RESULTS.md`, `docs/NOISE_ROBUSTNESS.md`). Two reasons were identified,
and both match the diagnosis of Turkeš, Montúfar & Otter, "On the
Effectiveness of Persistent Homology" (NeurIPS 2022): persistent homology
helps when the task is genuinely topological, and how the diagrams are
represented matters.

1. **The task.** Separating coherent fields from phase-randomized surrogates
   is a nonlinearity / intermittency test. Okubo–Weiss and gradient
   statistics already solve it (~1.000), so there is no headroom.
2. **The representation.** The descriptor compressed persistence into a few
   summaries.

This study acts on both:

- **A.** It adds an injective, richer topological representation (ECT) and a
  learnable one (DECT), and re-tests them on the existing benchmark.
- **B.** It builds a task whose label depends on global connectivity.

## A. Richer topological representations

### ECT: Euler Characteristic Transform (`src/descriptor/ect.py`)

For a shape K ⊂ ℝ², ECT_K(v, t) = χ(K ∩ {x : ⟨x, v⟩ ≤ t}) for every direction
v ∈ S¹ and threshold t (Turner, Mukherjee & Boyer 2014; Munch 2023). It is
injective on constructible shapes, unlike a handful of persistence
summaries.

- **Shapes.** A field is not a shape, so the ECT is applied to its sublevel
  sets K_h = {f ≤ h} at 5 fixed levels h ∈ {−1.5, −0.75, 0, +0.75, +1.5}
  (field-std units), each as a cubical complex. Every pixel is a closed
  square with its edges and vertices.
- **Grid.** 16 directions uniform on the circle and 24 thresholds on
  [−1.42, 1.42], with coordinates in [−1, 1]².
- **Resolution.** The field is block-averaged to 128² (66 049 cells).
- **Output.** A 5 × 16 × 24 image, 1 920 features.
- **Computation.** Cells enter at the maximum height of their vertices
  (lower-star filtration). For a fixed grid, the whole ECT is one matrix
  product: sublevel-set cell masks @ signed indicator [height ≤ t].
- **Validation.** Unit tests check that the Euler characteristic equals
  8-connected components − 4-connected holes on random images, that an
  annulus gives χ = 0, and that rotating the image by 90° shifts the
  direction axis by D/4.

### DECT: Differentiable ECT (`src/descriptor/dect.py`, optional torch)

Röell & Rieck (2023), arXiv:2310.07630.

- **The layer.** The indicator [height_v(c) ≤ t] becomes
  σ(λ (t − height_v(c))). The 16 directions, the 16 thresholds and the
  sharpness λ (initially 40) are learnable parameters.
- **The input.** The same cubical sublevel-set complexes as the ECT: the
  same 5 fixed levels at 128².
- **The model.** A small CNN head (batch-norm; two 3 × 3 convolutions, the
  direction axis wrap-padded; global average; linear) classifies the DECT
  image. Everything is trained end to end with Adam (lr 3·10⁻³, weight decay
  10⁻⁴, 30 epochs, batch 32) on the same grouped CV splits as every other
  view.
- **Validation.** With very sharp sigmoids the layer reproduces the exact
  ECT (unit test), and gradients reach the directions and thresholds.
- **Optional.** torch is optional: `ect_study.dect.enabled` or `--no-dect`.

## B. A genuinely topological task: merged vs separate vortex pairs

`src/sims/vortex_events.py`.

**Setup.**
- Two identical co-rotating Gaussian vortices (peak 10, radius a = 0.3 in
  the 2π box) are seeded on weak broadband background vorticity (rms 5% of
  the peak, |k| ∈ [4, 24]).
- Position and orientation are random.
- Each pair is evolved, unforced, with the repo's IF-RK4 solver at 256²
  (ν = 2·10⁻⁴, dt = 0.005). The solver gained an optional initial condition
  `w0`; its default path is bit-identical, and a unit test checks this.

**Classes.** Co-rotating vortices merge when their separation is below a
critical d/a (Melander, Zabusky & McWilliams 1988; Meunier et al. 2005):

| class | d/a (uniform) | state in the snapshot window |
|---|---|---|
| 0 "merging" | [2.0, 2.6] | one coherent core |
| 1 "separate" | [5.5, 6.5] | two coherent cores |

The label is the initial separation regime, a physical parameter no
descriptor sees. Everything else is drawn from the same distributions.

**Window and size.** 24 trajectories per class, 10 snapshots each at
t = 10, 12, …, 28: 480 fields and 48 groups for the grouped CV. Every field
is standardized (zero mean, unit variance), as in the IAAFT benchmark.

**Descriptor-free pilot** (`src/eval/vortex_pilot.py`): the number of
coherent cores is counted as the connected regions of {ω > 0.5 max ω}. This
is a validity check, never a feature.

- **Pilot v1** (ν = 5·10⁻⁴, background 10%): every pair merged by
  t ≈ 17–22, even at d/a = 5.4. With those settings the "separate" class
  would not have been separate (`results/vortex_pilot_v1.json`,
  `results/figures/vortex_pilot_v1.png`).
- **Pilot v2** (ν = 2·10⁻⁴, background 5%): d/a ≤ 2.6 is one core from t ≈ 7
  on, and d/a ≥ 5.5 keeps two cores up to t = 29
  (`results/vortex_pilot.json`, `results/figures/vortex_pilot.png`).
- The class ranges and the window were fixed from v2, before any features
  were computed.

## Views and protocol (both tasks)

| Group | Views |
|---|---|
| Topological | `ph_counts` (numbers of H0 / H1 pairs with persistence ≥ 0.5, 1, 2 σ, periodic sublevel filtration), `x2_ph` (sig6 summaries), `topological_view_fullres`, `topo_betti_curves`, `topo_persistence_image`, `topo_multiscale`, **`ect`**, **`dect`** |
| Non-topological | `okubo_weiss`, `gradient_stats`, `increment_flatness`, `physical_structure_all`, `moments`, `power_spectrum`, `histogram_quantiles`, **`baselines_union`** |
| Incremental | `baselines_union+ect`, `baselines_union+ph_counts` |

- **CV.** Repeated StratifiedGroupKFold, 4 folds × 3 repeats; the group is
  the trajectory.
- **Classifiers.** Feature views report the better of LogReg and
  RandomForest. DECT uses its CNN head on the same splits.
- **Uncertainty.** Paired group-bootstrap 95% CIs (2000 resamples) for the
  margin of every topological view over `baselines_union` and over the best
  structure baseline.
- **Multiplicity.** Benjamini–Hochberg over the 16 comparisons (8
  topological views × 2 tasks) against `baselines_union`.

## Decision rule (fixed before running)

- **GO for a task:** at least one of its primary views reaches a margin over
  `baselines_union` of ≥ +0.05, with the 95% CI lower bound > 0 and BH
  q < 0.05.
- **Primary views:**
  - Task A (IAAFT): `ect`, `dect`. These are the new representations under
    test. An honest tie is the expected outcome, because the headroom of
    `baselines_union` on this task was 0.001.
  - Task B (vortex events): `ph_counts`, `ect`, `dect`, i.e. H0/H1 counts,
    ECT and DECT.
- **Reporting.** All other views are reported with their margins and are not
  used for the verdict. Headroom (1 − accuracy of `baselines_union`) is
  reported per task: if it is below 0.05, a GO is arithmetically impossible
  and the verdict says so.

**Disclosure (smoke check, not a result).** Before this commit, only the
offline CI configuration (`config/ect_vortex_smoke.yaml`: 64², 24 fields per
task, 2 DECT epochs) was run, to check the pipeline end to end. On that toy
grid several non-topological baselines (gradient statistics, power spectrum)
already separate merged from separate pairs. So `baselines_union` may
saturate on Task B, and the headroom clause above would then decide the
verdict. The design was not changed after seeing this.

---

## Results

*Pending: added after the pre-registered run.*
