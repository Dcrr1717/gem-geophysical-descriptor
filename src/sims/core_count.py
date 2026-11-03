"""Matched-scale core-count fields: one coherent vortex core vs two (Task C).

Task B (src/sims/vortex_events.py) showed that a vortex merger changes the
core *scale* as well as the core *count*. The single merged core is about
twice as large as each pre-merger core, so the power spectrum alone
classifies merged vs separate pairs perfectly. This module builds a core-count
task in which scale and one-point statistics are decoupled from the count:

  class 0  "one core"   one vortex core
  class 1  "two cores"  two vortex cores of comparable strength

Every core has its own random radius R and profile steepness p, with profile

    omega(r) = A exp(-(r / R)^p),   p in [2, p_max]   (p = 2: Gaussian; large p: flat top).

The two classes draw (R, p) from wide ranges, so a single core can be
compact or broad, smooth or flat-topped, and so can each core of a pair.
Monotone profiles are stable, so cores neither split nor shed filaments.
The fields are seeded on weak broadband background vorticity and evolved,
unforced, with the repo's IF-RK4 solver, like Task B.

Matching happens afterwards, in src/eval/matched_core_count.py. A larger
pool is simulated, and snapshots are paired one-to-one across classes on
descriptor-free design statistics (`design_stats`: spectral centroid,
spectral width, occupied area, standardized peak). The classes then agree on
scale, occupancy and peak height, and the label is the number of cores.
`core_count` (src/sims/vortex_events.py) verifies the label on every field;
it is a validity check, never a feature.
"""
from __future__ import annotations

import numpy as np

from .kolmogorov import simulate
from .vortex_events import background

DESIGN_STATS = ("spectral_centroid", "spectral_width", "occupied_area", "standardized_peak")


def periodic_core(n, center, radius, steepness, amp, aspect=1.0, angle=0.0):
    """amp * exp(-rho^steepness) with minimum-image distances (axis 0 = x), where
    rho is the elliptical radius of an area-preserving ellipse: semi-axes
    radius * sqrt(aspect) along `angle` and radius / sqrt(aspect) across it."""
    x = np.linspace(0, 2 * np.pi, n, endpoint=False)
    X, Y = np.meshgrid(x, x, indexing="ij")
    dx = (X - center[0] + np.pi) % (2 * np.pi) - np.pi
    dy = (Y - center[1] + np.pi) % (2 * np.pi) - np.pi
    u = dx * np.cos(angle) + dy * np.sin(angle)
    v = -dx * np.sin(angle) + dy * np.cos(angle)
    rho = np.hypot(u / np.sqrt(aspect), v * np.sqrt(aspect)) / radius
    return amp * np.exp(-rho ** steepness)


def draw_cores(label, cc, rng):
    """Core parameters for one trajectory. `cc` = the `core_count` config block.

    label 0: one elliptical core with equivalent radius in cc['single_radius'] and
    aspect ratio in cc['single_aspect'] (random orientation);
    label 1: two cores, the larger with radius in cc['pair_radius'] and the
    other smaller by a factor in cc['pair_size_ratio'] (which of the two is
    the weaker one is random), amplitude ratio of the weaker core in
    cc['pair_amp_ratio'], centre distance d with
    d / mean radius in cc['pair_separation'] (capped at cc['max_separation'])."""
    amp = float(cc["amplitude"])
    p_lo, p_hi = cc.get("single_steepness" if label == 0 else "pair_steepness", cc["steepness"])
    center = rng.uniform(0, 2 * np.pi, size=2)
    if label == 0:
        return [{"center": center, "radius": float(rng.uniform(*cc["single_radius"])),
                 "steepness": float(rng.uniform(p_lo, p_hi)), "amp": amp,
                 "aspect": float(rng.uniform(*cc.get("single_aspect", (1.0, 1.0)))),
                 "angle": float(rng.uniform(0, np.pi))}]
    r_big = float(rng.uniform(*cc["pair_radius"]))
    r = np.array([r_big, r_big * float(rng.uniform(*cc.get("pair_size_ratio", (1.0, 1.0))))])
    rng.shuffle(r)
    p = rng.uniform(p_lo, p_hi, size=2)
    ratio = float(rng.uniform(*cc["pair_amp_ratio"]))
    d = min(float(rng.uniform(*cc["pair_separation"])) * r.mean(), float(cc["max_separation"]))
    angle = rng.uniform(0, np.pi)
    e = np.array([np.cos(angle), np.sin(angle)])
    return [{"center": center - 0.5 * d * e, "radius": float(r[0]), "steepness": float(p[0]),
             "amp": amp},
            {"center": center + 0.5 * d * e, "radius": float(r[1]), "steepness": float(p[1]),
             "amp": amp * ratio}]


def initial_condition(n, cores, bg_rms, bg_band, rng):
    w = sum(periodic_core(n, c["center"], c["radius"], c["steepness"], c["amp"],
                          c.get("aspect", 1.0), c.get("angle", 0.0)) for c in cores)
    w = w + background(n, bg_rms * max(c["amp"] for c in cores), bg_band, rng)
    return w - w.mean()       # zero total circulation on the torus


def core_jobs(cc, seed=0):
    """Independent trajectories (the unit of grouped CV), `runs_per_class` per class."""
    jobs = []
    for label in (0, 1):
        for run in range(int(cc["runs_per_class"])):
            rng = np.random.default_rng(np.random.SeedSequence([seed, 3141, label, run]))
            cores = draw_cores(label, cc, rng)
            jobs.append({"label": label, "run": run,
                         "cores": [{k: (v.tolist() if isinstance(v, np.ndarray) else v)
                                    for k, v in c.items()} for c in cores],
                         "seed_words": [seed, 3141, label, run, 1]})
    return jobs


def simulate_cores(job, cc):
    """Evolve one seeded configuration; returns (job, list of snapshots)."""
    n = int(cc["grid"])
    rng = np.random.default_rng(np.random.SeedSequence(job["seed_words"]))
    w0 = initial_condition(n, job["cores"], cc["background_rms"], tuple(cc["background_band"]),
                           rng)
    dt = float(cc["dt"])
    spinup = int(round(cc["first_snapshot_time"] / dt))
    stride = int(round(cc["snapshot_interval"] / dt))
    n_snap = int(cc["snapshots_per_run"])
    snaps = simulate(n=n, nu=float(cc["nu"]), forcing_amp=0.0, dt=dt,
                     n_steps=spinup + stride * n_snap, spinup=spinup, n_snapshots=n_snap,
                     drag=0.0, scheme="rk4", w0=w0)
    return job, snaps


def design_stats(w):
    """Descriptor-free design statistics of a standardized field, used only to
    match the classes (never as classifier features):

    - spectral_centroid: power-weighted mean |k| of the radial spectrum (k >= 1);
    - spectral_width: power-weighted standard deviation of |k|;
    - occupied_area: fraction of the domain where w > 0.5 max w;
    - standardized_peak: max w / std w."""
    n = w.shape[0]
    k = np.fft.fftfreq(n) * n
    K = np.hypot(*np.meshgrid(k, k, indexing="ij"))
    P = np.abs(np.fft.fft2(w)) ** 2
    P[0, 0] = 0.0
    tot = P.sum()
    kc = (K * P).sum() / tot
    kw = np.sqrt(((K - kc) ** 2 * P).sum() / tot)
    return np.array([kc, kw, float((w > 0.5 * w.max()).mean()), w.max() / w.std()])


# ---------------------------------------------------------------------------
# Matching on the design statistics (descriptor-free)
# ---------------------------------------------------------------------------
def _design_space(S):
    """Matching coordinates: centroid and width as is, area and peak in logs
    (both are scale-like and right-skewed)."""
    S = np.asarray(S, dtype=float)
    return np.column_stack([S[:, 0], S[:, 1], np.log(S[:, 2]), np.log(S[:, 3])])


def match_mahalanobis(S, y, caliper):
    """Greedy one-to-one nearest-neighbour matching on the Mahalanobis distance
    in the design space (pilots v1, v2 only; the study uses `match_propensity`).
    Returns (indices of class-0 members, indices of class-1 members, distances)."""
    Z = _design_space(S)
    y = np.asarray(y)
    VI = np.linalg.inv(np.cov(Z, rowvar=False))
    i0, i1 = np.flatnonzero(y == 0), np.flatnonzero(y == 1)
    D = Z[i0][:, None, :] - Z[i1][None, :, :]
    return _greedy(np.sqrt(np.einsum("abi,ij,abj->ab", D, VI, D)), i0, i1, caliper)


def _greedy(dist, i0, i1, caliper):
    """Accept pairs in order of increasing distance while both members are
    unused and the distance is <= caliper."""
    order = np.argsort(dist, axis=None, kind="stable")
    used0, used1 = np.zeros(len(i0), bool), np.zeros(len(i1), bool)
    a_out, b_out, d_out = [], [], []
    for flat in order:
        a, b = np.unravel_index(flat, dist.shape)
        if dist[a, b] > caliper:
            break
        if used0[a] or used1[b]:
            continue
        used0[a] = used1[b] = True
        a_out.append(i0[a])
        b_out.append(i1[b])
        d_out.append(dist[a, b])
    return np.array(a_out, int), np.array(b_out, int), np.array(d_out)


def propensity_logit(S, y, degree=2):
    """Logit of the propensity score P(class 1 | design statistics): logistic
    regression (scikit-learn defaults, L2 with C = 1) on the standardized
    design-space coordinates expanded to polynomial terms of `degree`
    (2 = linear, squares and pairwise products, so that variances are balanced
    as well as means), fitted on the whole pool."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import PolynomialFeatures, StandardScaler
    Z = _design_space(S)
    model = make_pipeline(StandardScaler(), PolynomialFeatures(degree, include_bias=False),
                          StandardScaler(), LogisticRegression(max_iter=5000)).fit(Z, y)
    return model.decision_function(Z)


def match_propensity(S, y, caliper_sd=0.2, degree=2):
    """Propensity-score matching (Rosenbaum & Rubin 1983): greedy one-to-one
    nearest-neighbour matching on the logit of the propensity score, without
    replacement, closest pairs first, caliper = caliper_sd x SD of the logit
    (Austin 2011). Returns (class-0 indices, class-1 indices, logit distances)."""
    y = np.asarray(y)
    lg = propensity_logit(S, y, degree)
    i0, i1 = np.flatnonzero(y == 0), np.flatnonzero(y == 1)
    dist = np.abs(lg[i0][:, None] - lg[i1][None, :])
    return _greedy(dist, i0, i1, caliper_sd * lg.std())


match_pairs = match_mahalanobis      # name used by pilots v1 and v2


def balance(S, y):
    """Standardized mean difference (class 1 - class 0, pooled SD) and variance
    ratio (class 1 / class 0) of every design statistic, in the design space."""
    Z = _design_space(S)
    y = np.asarray(y)
    out = {}
    for j, name in enumerate(DESIGN_STATS):
        a, b = Z[y == 0, j], Z[y == 1, j]
        sd = np.sqrt(0.5 * (a.var(ddof=1) + b.var(ddof=1)))
        out[name] = {"smd": float((b.mean() - a.mean()) / sd) if sd > 0 else 0.0,
                     "variance_ratio": float(b.var(ddof=1) / a.var(ddof=1))
                     if a.var(ddof=1) > 0 else float("nan")}
    return out


def design_auc(S, y, groups, n_splits=5, seed=0):
    """Cross-validated ROC AUC of a logistic regression predicting the class
    from the design statistics (StratifiedGroupKFold, groups = trajectories).
    About 0.5 means the design statistics no longer carry the label."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    Z = _design_space(S)
    y = np.asarray(y)
    groups = np.asarray(groups)
    n_splits = max(2, min(n_splits, *(len(np.unique(groups[y == c])) for c in (0, 1))))
    score = np.empty(len(y))
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for tr, te in cv.split(Z, y, groups):
        m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)).fit(Z[tr], y[tr])
        score[te] = m.decision_function(Z[te])
    return float(roc_auc_score(y, score))


def balance_gate(S, y, groups, gate):
    """The pre-registered balance gate on a matched set: every |SMD| below
    gate['max_abs_smd'], every variance ratio inside gate['variance_ratio'],
    at least gate['min_pairs'] pairs, and design AUC <= gate['max_design_auc']."""
    bal = balance(S, y)
    auc = design_auc(S, y, groups)
    n_pairs = int(min((np.asarray(y) == 0).sum(), (np.asarray(y) == 1).sum()))
    lo, hi = gate["variance_ratio"]
    checks = {"smd": all(abs(v["smd"]) < gate["max_abs_smd"] for v in bal.values()),
              "variance_ratio": all(lo <= v["variance_ratio"] <= hi for v in bal.values()),
              "pairs": n_pairs >= int(gate["min_pairs"]),
              "design_auc": auc <= float(gate["max_design_auc"])}
    return {"passed": all(checks.values()), "checks": checks, "balance": bal,
            "design_auc": auc, "n_pairs": n_pairs}
