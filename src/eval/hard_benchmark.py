"""Hard, structure-only Go/No-Go benchmark (coherent turbulence vs IAAFT surrogates).

Question: does the topological-spectral multiview descriptor detect coherent
flow structure better than strong physical / statistical baselines, when the
power spectrum and the one-point distribution are matched across classes?

Data
  Coherent 2D forced turbulence (src/sims/kolmogorov.simulate) at several
  forcing scales, many independent trajectories, snapshots after spin-up.
  For every coherent snapshot an IAAFT surrogate (src/sims/surrogate.iaaft) with
  the same power spectrum and the same value histogram but random phases.

Tasks
  binary_pooled   coherent (0) vs surrogate (1), all forcing scales pooled [PRIMARY]
  binary_<regime> the same task within one forcing scale
  multiclass6     (forcing scale) x (coherent / surrogate) -> 6 classes
  binary_pooled_iaaft_<variant>  robustness: other IAAFT final step

Protocol
  Repeated StratifiedGroupKFold; group = simulated trajectory (its snapshots
  AND their surrogates), so correlated snapshots never straddle train/test.
  LogReg + RandomForest; accuracy / macro-F1 / silhouette. The existing
  StratifiedKFold `evaluate` is also reported for comparison.

Verdict
  GO iff descriptor accuracy - best baseline accuracy >= go_margin (0.05).
  Reported against (a) the baselines named in the brief and (b) a strict set
  that adds leak detectors and structure-aware physical diagnostics.

Authentic views and richer topology (extensions)
  Richer vectorizations of the topological view (src/descriptor/topology_rich.py)
  and the authentic multiview views ported verbatim from the reference
  (src/descriptor/authentic.py): X1 (functional graph of 8x8 mesh regions),
  X2 (sig6 persistent homology), X1+X2, the X3 fusion and X1+X2+richer. The
  existing Laplacian-spectrum graph view alone (`laplacian_view`) is kept as
  the baseline for X1. `union_verdicts` compares EVERY non-baseline
  configuration with `baselines_union`: GO iff some configuration reaches
  accuracy - acc(baselines_union) >= go_margin on some task.

Usage
  python -m src.eval.hard_benchmark --config config/hard_benchmark.yaml
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import yaml

from ..sims.datasets import simulate_trajectory, trajectory_jobs
from ..sims.surrogate import iaaft, spectral_mismatch, histogram_mismatch
from ..descriptor.spectral import spectral_features
from ..descriptor.topology import topology_features, _HAS_GUDHI
from ..descriptor.topology_rich import rich_topology_features
from ..descriptor.authentic import x1_patch_coherence, x2_snapshot_ph, desc_x3
from ..baselines.features import BASELINES
from ..baselines.structure import VALIDITY_BASELINES, okubo_weiss, gradient_stats, \
    increment_flatness
from .classify import evaluate, evaluate_grouped

# ---------------------------------------------------------------------------
# Feature-set registry: name -> (tier, description)
# ---------------------------------------------------------------------------
FEATURE_SETS = {
    "descriptor": ("ours", "full multiview descriptor (spectral + topological view)"),
    "spectral_view": ("ours-ablation", "spectral view only (spectrum slope/peak + graph-Laplacian eigenvalues)"),
    "topological_view": ("ours-ablation", "topological view only (sublevel-set PH, H0/H1 statistics)"),
    "laplacian_view": ("graph-laplacian", "existing graph view alone: 8 smallest normalized-Laplacian eigenvalues of the 16x16 similarity grid graph"),
    "moments": ("baseline-specified", "vorticity moments / enstrophy"),
    "spectral_slope": ("baseline-specified", "inertial-range spectral slope"),
    "raw_pixels": ("baseline-specified", "raw down-sampled pixels (16x16)"),
    "power_spectrum": ("baseline-validity", "full log radial power spectrum (leak detector)"),
    "histogram_quantiles": ("baseline-validity", "one-point distribution quantiles (leak detector)"),
    "okubo_weiss": ("baseline-structure", "Okubo-Weiss parameter statistics"),
    "gradient_stats": ("baseline-structure", "higher-order vorticity-gradient / Laplacian statistics"),
    "increment_flatness": ("baseline-structure", "flatness of vorticity increments, r = 1..32"),
    "physical_structure_all": ("baseline-structure", "Okubo-Weiss + gradient + increment features"),
    "baselines_union": ("baseline-union", "all non-topological baselines concatenated (no raw pixels)"),
    "baselines_union+topo": ("incremental", "baselines_union + topological view"),
    "baselines_union+topo_rich": ("incremental", "baselines_union + all richer topological views"),
    "baselines_union+x1": ("incremental", "baselines_union + X1"),
    "baselines_union+x1x2": ("incremental", "baselines_union + X1 + X2"),
    "baselines_union+x1x2_rich": ("incremental", "baselines_union + X1 + X2 + all richer topological views"),
    "topo_persistence_image": ("ours-rich", "persistence images of H0/H1 (2 x 10x10), full-resolution field"),
    "topo_betti_curves": ("ours-rich", "Betti curves beta0/beta1 on 33 thresholds, full-resolution field"),
    "topo_multiscale": ("ours-rich", "multi-scale filtration (Gaussian sigma 1/2/4/8 px), 16 summaries per scale"),
    "topo_rich_all": ("ours-rich", "persistence images + Betti curves + multi-scale filtration"),
    "descriptor_rich": ("ours-rich", "spectral view + all richer topological views"),
    "x1_graph": ("authentic", "X1: 10 functional-graph invariants, |Pearson| > 0.40 between 8x8 = 64 mesh regions (snapshot mode)"),
    "x2_ph": ("authentic", "X2: sig6 summaries of H0/H1, periodic sublevel-set PH at full resolution"),
    "x1x2": ("authentic", "authentic descriptor X1 + X2 (concatenated)"),
    "x3_fusion": ("authentic", "X3 = [X1/|X1|, X2/|X2|], the original label-free fusion"),
    "x1x2_rich": ("authentic", "X1 + X2 + all richer topological views"),
    "topological_view_fullres": ("sensitivity", "topological view at full 256^2 resolution (not pre-registered)"),
    "descriptor_fullres_topo": ("sensitivity", "descriptor with full-resolution topological view (not pre-registered)"),
}
SPECIFIED = [k for k, v in FEATURE_SETS.items() if v[0] == "baseline-specified"]
STRICT = [k for k, v in FEATURE_SETS.items() if v[0].startswith("baseline")]
SPECTRUM_MOMENTS = ["moments", "spectral_slope", "power_spectrum", "histogram_quantiles"]
STRUCTURE = [k for k, v in FEATURE_SETS.items() if v[0] == "baseline-structure"]
RICH = [k for k, v in FEATURE_SETS.items() if v[0] == "ours-rich"]
# Every non-baseline configuration is tested against baselines_union (incl. post-hoc ones).
CANDIDATES = [k for k, v in FEATURE_SETS.items() if not v[0].startswith("baseline")]


def _key(obj) -> str:
    return hashlib.sha1(json.dumps(obj, sort_keys=True).encode()).hexdigest()[:12]


def _standardize(w):
    """Zero mean, unit variance. No additive epsilon: (w - m) / (s + eps) has
    std = s / (s + eps) = 1 - eps / s, which leaks the original amplitude s at
    round-off level - enough for a StandardScaler-based classifier to separate
    coherent fields (s ~ 5) from already-standardized surrogates (s ~ 1)."""
    s = w.std()
    return (w - w.mean()) / (s if s > 0 else 1.0)


# ---------------------------------------------------------------------------
# Data generation (cached)
# ---------------------------------------------------------------------------
def generate_coherent(cfg, cache_dir, workers):
    sim = cfg["simulation"]
    jobs = trajectory_jobs(
        cfg["regimes"], n=sim["grid"], dt=sim["dt"], nu=sim["nu"],
        drag=sim.get("drag", 0.0), scheme=sim.get("scheme", "rk4"),
        spinup_time=sim["spinup_time"], snapshot_interval=sim["snapshot_interval"],
        snapshots_per_run=sim["snapshots_per_run"],
        runs_per_regime=sim["runs_per_regime"], seed=cfg.get("seed", 0))
    key = _key({"sim": sim, "regimes": cfg["regimes"], "seed": cfg.get("seed", 0),
                "standardize": "no-eps-v2"})
    path = Path(cache_dir) / f"coherent_{key}.npz"
    if path.exists():
        print(f"[data] loading cached coherent fields {path}")
        d = np.load(path)
        return {k: d[k] for k in d.files}, jobs

    t0 = time.time()
    print(f"[data] simulating {len(jobs)} trajectories "
          f"({jobs[0]['n_steps']} steps each, grid {sim['grid']}) on {workers} workers")
    results = []
    with Pool(workers) as pool:
        for job, snaps in pool.imap_unordered(simulate_trajectory, jobs):
            results.append((job, snaps))
            print(f"  regime {job['regime_name']:>6s} run {job['run']} done "
                  f"({len(results)}/{len(jobs)}, {time.time() - t0:.0f}s)", flush=True)
    results.sort(key=lambda t: (t[0]["regime"], t[0]["run"]))
    fields, regime, run, snap, group = [], [], [], [], []
    for g, (job, snaps) in enumerate(results):
        for s_i, w in enumerate(snaps):
            fields.append(_standardize(w))
            regime.append(job["regime"])
            run.append(job["run"])
            snap.append(s_i)
            group.append(g)
    data = {"fields": np.asarray(fields), "regime": np.asarray(regime),
            "run": np.asarray(run), "snap": np.asarray(snap), "group": np.asarray(group),
            "sim_seconds": np.asarray(time.time() - t0)}
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **data)
    print(f"[data] {len(fields)} coherent snapshots in {time.time() - t0:.0f}s -> {path}")
    return data, jobs


def _surrogate_worker(args):
    w, seed_words, n_iter, final = args
    rng = np.random.default_rng(np.random.SeedSequence(seed_words))
    s, info = iaaft(w, n_iter=n_iter, seed=rng, final=final, return_info=True)
    s = _standardize(s)
    return s, {"iterations": info["iterations"], "converged": info["converged"],
               "spectral_mismatch": spectral_mismatch(w, s),
               "histogram_mismatch": histogram_mismatch(w, s),
               "hermitian_error": info["hermitian_error"]}


def generate_surrogates(coh, cfg, final, cache_dir, workers):
    sc = cfg["surrogate"]
    key = _key({"coh": _key(coh["fields"][:, ::17, ::17].round(10).tolist()),
                "n": int(len(coh["fields"])), "n_iter": sc["n_iter"], "final": final,
                "seed": cfg.get("seed", 0)})
    path = Path(cache_dir) / f"surrogates_{final}_{key}.pkl"
    if path.exists():
        print(f"[surrogate] loading cached IAAFT ({final}) {path}")
        return pickle.loads(path.read_bytes())
    t0 = time.time()
    seed = cfg.get("seed", 0)
    tasks = [(w, [seed, 7919, i], sc["n_iter"], final) for i, w in enumerate(coh["fields"])]
    with Pool(workers) as pool:
        res = pool.map(_surrogate_worker, tasks, chunksize=4)
    out = {"fields": np.asarray([r[0] for r in res]), "info": [r[1] for r in res],
           "seconds": time.time() - t0}
    path.write_bytes(pickle.dumps(out))
    print(f"[surrogate] {len(res)} IAAFT surrogates ({final}) in {time.time() - t0:.0f}s")
    return out


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------
def _features_worker(args):
    w, topo_ds, fullres, rich, authentic = args
    f = {"spectral_view": spectral_features(w),
         "topological_view": topology_features(w, downsample=topo_ds),
         "okubo_weiss": okubo_weiss(w),
         "gradient_stats": gradient_stats(w),
         "increment_flatness": increment_flatness(w)}
    for b, fn in BASELINES.items():
        f[b] = fn(w)
    for b, fn in VALIDITY_BASELINES.items():
        f[b] = fn(w)
    if fullres:
        f["topological_view_fullres"] = topology_features(w, downsample=w.shape[0])
    if rich:
        f.update(rich_topology_features(w))
    if authentic is not None:
        x1 = authentic.get("x1", {})
        f["x1_graph"] = x1_patch_coherence(w, theta=float(x1.get("theta", 0.40)),
                                           blocks=tuple(x1.get("blocks", (8, 8))),
                                           corr=x1.get("corr", "pearson"))
        f["x2_ph"] = x2_snapshot_ph(w, periodic=(True, True),
                                    min_pers=float(authentic.get("x2", {}).get("min_pers", 0.01)))
    return f


def compute_features(fields, topo_ds, workers, decimals=8, fullres=True, rich=True,
                     authentic=None):
    """All feature sets for a stack of fields. Every feature is rounded to
    `decimals` absolute decimals: no feature here is meaningful below 1e-8, and
    rounding removes floating-point round-off cues (e.g. the mean of a
    standardized field is ~1e-17 noise that a StandardScaler would amplify).

    `authentic` (a dict of X1 / X2 parameters, may be empty; None = off) adds
    the authentic X1, X2, X1+X2, X3 views and their combinations."""
    with Pool(workers) as pool:
        rows = pool.map(_features_worker,
                        [(w, topo_ds, fullres, rich, authentic) for w in fields], chunksize=4)
    F = {k: np.vstack([r[k] for r in rows]) for k in rows[0]}
    F["descriptor"] = np.hstack([F["spectral_view"], F["topological_view"]])
    # the existing graph view on its own (spectral_view = 3 spectrum numbers + 8 eigenvalues)
    F["laplacian_view"] = F["spectral_view"][:, 3:]
    if fullres:
        F["descriptor_fullres_topo"] = np.hstack([F["spectral_view"],
                                                  F["topological_view_fullres"]])
    F["physical_structure_all"] = np.hstack([F["okubo_weiss"], F["gradient_stats"],
                                             F["increment_flatness"]])
    F["baselines_union"] = np.hstack([F["moments"], F["spectral_slope"], F["power_spectrum"],
                                      F["histogram_quantiles"], F["physical_structure_all"]])
    F["baselines_union+topo"] = np.hstack([F["baselines_union"], F["topological_view"]])
    if rich:
        F["topo_rich_all"] = np.hstack([F["topo_persistence_image"], F["topo_betti_curves"],
                                        F["topo_multiscale"]])
        F["descriptor_rich"] = np.hstack([F["spectral_view"], F["topo_rich_all"]])
        F["baselines_union+topo_rich"] = np.hstack([F["baselines_union"], F["topo_rich_all"]])
    if authentic is not None:
        F["x1x2"] = np.hstack([F["x1_graph"], F["x2_ph"]])
        F["x3_fusion"] = np.vstack([desc_x3(a, b) for a, b in zip(F["x1_graph"], F["x2_ph"])])
        F["baselines_union+x1"] = np.hstack([F["baselines_union"], F["x1_graph"]])
        F["baselines_union+x1x2"] = np.hstack([F["baselines_union"], F["x1x2"]])
        if rich:
            F["x1x2_rich"] = np.hstack([F["x1x2"], F["topo_rich_all"]])
            F["baselines_union+x1x2_rich"] = np.hstack([F["baselines_union"], F["x1x2_rich"]])
    if decimals is not None:
        F = {k: np.round(np.nan_to_num(v), decimals) for k, v in F.items()}
    return F


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
def paired_bootstrap_margin(res_a, res_b, y, groups, n_boot=2000, seed=0):
    """95% CI of accuracy(A) - accuracy(B) using each set's best classifier,
    resampling trajectories (groups) with replacement; paired because both
    sets share the same folds."""
    y = np.asarray(y)
    groups = np.asarray(groups)
    ug, ginv = np.unique(groups, return_inverse=True)
    G = len(ug)
    n_g = np.bincount(ginv, minlength=G).astype(float)

    def correct_counts(res):
        oof = np.asarray(res[res["best_classifier"]]["oof"])      # (repeats, N)
        return np.stack([np.bincount(ginv, weights=(o == y).astype(float), minlength=G)
                         for o in oof])                            # (repeats, G)
    ca, cb = correct_counts(res_a), correct_counts(res_b)
    rng = np.random.default_rng(seed)
    W = rng.multinomial(G, np.full(G, 1.0 / G), size=n_boot).astype(float)  # (B, G)
    tot = W @ n_g
    acc_a = (W @ ca.T) / tot[:, None]
    acc_b = (W @ cb.T) / tot[:, None]
    diff = (acc_a - acc_b).mean(axis=1)
    point = float((ca.sum(1) / n_g.sum()).mean() - (cb.sum(1) / n_g.sum()).mean())
    lo, hi = np.percentile(diff, [2.5, 97.5])
    return {"margin": point, "ci95": [float(lo), float(hi)]}


def run_task(name, F, y, groups, sets, ev, chance):
    print(f"\n[task] {name}: N={len(y)}, classes={len(np.unique(y))}, "
          f"groups={len(np.unique(groups))}, chance={chance:.3f}")
    res = {}
    for s in sets:
        t0 = time.time()
        res[s] = evaluate_grouped(F[s], y, groups, seed=ev["seed"],
                                  n_splits=ev["n_splits"], n_repeats=ev["n_repeats"])
        r = res[s]
        print(f"  {s:>26s}: best={r['best_accuracy']:.3f} "
              f"(LR {r['logreg']['accuracy']:.3f}+-{r['logreg']['accuracy_std']:.3f}, "
              f"RF {r['rf']['accuracy']:.3f}+-{r['rf']['accuracy_std']:.3f}) "
              f"[{time.time() - t0:.1f}s]", flush=True)
    return {"n": int(len(y)), "n_classes": int(len(np.unique(y))),
            "n_groups": int(len(np.unique(groups))), "chance": chance, "results": res}


def verdicts(task, y, groups, go_margin, n_boot, seed):
    R = task["results"]

    def best_of(names):
        names = [n for n in names if n in R]
        b = max(names, key=lambda n: R[n]["best_accuracy"])
        return b, R[b]["best_accuracy"]

    out = {}
    d = R["descriptor"]["best_accuracy"]
    for label, pool_ in (("specified", SPECIFIED), ("strict", STRICT)):
        b, acc = best_of(pool_)
        ci = paired_bootstrap_margin(R["descriptor"], R[b], y, groups, n_boot, seed)
        margin = d - acc
        out[label] = {"descriptor_accuracy": d, "best_baseline": b,
                      "best_baseline_accuracy": acc, "margin": margin,
                      "margin_ci95": ci["ci95"],
                      "verdict": "GO" if margin >= go_margin else "NO-GO"}
    # Core claim: does the topological view add value?
    t = R["topological_view"]["best_accuracy"]
    b, acc = best_of(SPECTRUM_MOMENTS)
    out["topology_vs_spectrum_moments"] = {
        "topological_view_accuracy": t, "best_spectrum_moment_baseline": b,
        "best_accuracy": acc, "margin": t - acc,
        "margin_ci95": paired_bootstrap_margin(R["topological_view"], R[b], y, groups,
                                               n_boot, seed)["ci95"]}
    b, acc = best_of(STRUCTURE)
    out["topology_vs_structure_baselines"] = {
        "topological_view_accuracy": t, "best_structure_baseline": b,
        "best_accuracy": acc, "margin": t - acc,
        "margin_ci95": paired_bootstrap_margin(R["topological_view"], R[b], y, groups,
                                               n_boot, seed)["ci95"]}
    u, ut = R["baselines_union"]["best_accuracy"], R["baselines_union+topo"]["best_accuracy"]
    out["incremental_topology"] = {
        "baselines_union_accuracy": u, "baselines_union_plus_topo_accuracy": ut,
        "gain": ut - u,
        "gain_ci95": paired_bootstrap_margin(R["baselines_union+topo"], R["baselines_union"],
                                             y, groups, n_boot, seed)["ci95"]}
    s = R["spectral_view"]["best_accuracy"]
    out["view_ablation"] = {"descriptor": d, "spectral_view": s, "topological_view": t,
                            "descriptor_minus_spectral_view": d - s,
                            "descriptor_minus_topological_view": d - t}
    if all(k in R for k in RICH + ["baselines_union+topo_rich"]):
        out["rich_topology"] = rich_verdict(R, y, groups, go_margin, n_boot, seed)
    return out


def rich_verdict(R, y, groups, go_margin, n_boot, seed):
    """Does ANY richer topological vectorization beat `baselines_union` (all
    non-topological baselines concatenated) by >= go_margin?"""
    u = R["baselines_union"]["best_accuracy"]
    sb = max(STRUCTURE, key=lambda n: R[n]["best_accuracy"])
    variants = {}
    for k in RICH:
        a = R[k]["best_accuracy"]
        variants[k] = {
            "accuracy": a,
            "margin_vs_baselines_union": a - u,
            "margin_vs_baselines_union_ci95": paired_bootstrap_margin(
                R[k], R["baselines_union"], y, groups, n_boot, seed)["ci95"],
            "margin_vs_best_structure_baseline": a - R[sb]["best_accuracy"],
            "margin_vs_old_topological_view": a - R["topological_view"]["best_accuracy"],
            "go": bool(a - u >= go_margin)}
    best = max(RICH, key=lambda k: variants[k]["accuracy"])
    ui = R["baselines_union+topo_rich"]["best_accuracy"]
    return {"baselines_union_accuracy": u,
            "max_attainable_margin": 1.0 - u,
            "best_structure_baseline": sb,
            "best_structure_baseline_accuracy": R[sb]["best_accuracy"],
            "best_rich_variant": best, "variants": variants,
            "incremental": {"baselines_union_plus_topo_rich_accuracy": ui, "gain": ui - u,
                            "gain_ci95": paired_bootstrap_margin(
                                R["baselines_union+topo_rich"], R["baselines_union"], y,
                                groups, n_boot, seed)["ci95"]},
            "any_go": any(v["go"] for v in variants.values()),
            "verdict": "GO" if any(v["go"] for v in variants.values()) else "NO-GO"}


def union_verdicts(task, y, groups, go_margin, n_boot, seed):
    """Every non-baseline configuration vs `baselines_union` (the rule asked for
    the authentic / richer-topology extension), plus the head-to-head of the
    two graph views and of the authentic vs the legacy descriptor.

    GO for a configuration iff its accuracy - baselines_union accuracy >=
    go_margin. `headroom` = 1 - acc(baselines_union) is the largest margin any
    configuration can reach on this task; if it is below go_margin, a GO is
    arithmetically impossible whatever the descriptor."""
    R = task["results"]
    if "baselines_union" not in R:
        return None
    U = R["baselines_union"]
    u = U["best_accuracy"]

    def cmp(a, b_res, b_acc):
        ci = paired_bootstrap_margin(R[a], b_res, y, groups, n_boot, seed)
        return {"accuracy": R[a]["best_accuracy"], "margin": R[a]["best_accuracy"] - b_acc,
                "margin_ci95": ci["ci95"]}

    rows = {}
    for c in CANDIDATES:
        if c in R:
            rows[c] = cmp(c, U, u)
            rows[c]["reaches_go_margin"] = bool(rows[c]["margin"] >= go_margin)
    best = max(rows, key=lambda c: rows[c]["margin"])
    out = {"baselines_union_accuracy": u, "headroom": 1.0 - u,
           "go_margin": go_margin, "go_arithmetically_possible": bool(1.0 - u >= go_margin),
           "configs": rows, "best_config": best, "best_margin": rows[best]["margin"],
           "best_margin_ci95": rows[best]["margin_ci95"],
           "configs_reaching_go_margin": [c for c, r in rows.items() if r["reaches_go_margin"]],
           # stricter reading: the max over many configurations x tasks is optimistic, so
           # also flag the configurations whose margin CI excludes zero
           "configs_reaching_go_margin_ci_above_0": [
               c for c, r in rows.items() if r["reaches_go_margin"] and r["margin_ci95"][0] > 0]}
    out["verdict"] = "GO" if out["configs_reaching_go_margin"] else "NO-GO"
    pairs = {"x1_graph_vs_laplacian_view": ("x1_graph", "laplacian_view"),
             "x1x2_vs_descriptor": ("x1x2", "descriptor"),
             "x2_ph_vs_topological_view": ("x2_ph", "topological_view"),
             "x2_ph_vs_topological_view_fullres": ("x2_ph", "topological_view_fullres"),
             "x1x2_rich_vs_x1x2": ("x1x2_rich", "x1x2"),
             "x1x2_rich_vs_physical_structure_all": ("x1x2_rich", "physical_structure_all")}
    out["head_to_head"] = {}
    for name, (a, b) in pairs.items():
        if a in R and b in R:
            out["head_to_head"][name] = {"a": a, "b": b, "accuracy_b": R[b]["best_accuracy"],
                                         **cmp(a, R[b], R[b]["best_accuracy"])}
    return out


def union_summary(tasks):
    """Across all tasks: does any configuration reach the GO margin over baselines_union?"""
    per_task, best = {}, None
    for name, t in tasks.items():
        uv = t.get("union_verdicts")
        if not uv:
            continue
        per_task[name] = {"verdict": uv["verdict"], "best_config": uv["best_config"],
                          "best_margin": uv["best_margin"],
                          "best_margin_ci95": uv["best_margin_ci95"],
                          "headroom": uv["headroom"],
                          "go_arithmetically_possible": uv["go_arithmetically_possible"],
                          "configs_reaching_go_margin": uv["configs_reaching_go_margin"],
                          "configs_reaching_go_margin_ci_above_0":
                              uv["configs_reaching_go_margin_ci_above_0"]}
        if best is None or uv["best_margin"] > best[2]:
            best = (name, uv["best_config"], uv["best_margin"])
    if best is None:
        return {"verdict": "n/a", "best_task": None, "best_config": None, "best_margin": np.nan,
                "per_task": per_task}
    any_go = any(v["verdict"] == "GO" for v in per_task.values())
    return {"verdict": "GO" if any_go else "NO-GO", "best_task": best[0],
            "best_config": best[1], "best_margin": best[2], "per_task": per_task}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run(cfg):
    t_start = time.time()
    workers = int(cfg.get("n_workers", os.cpu_count() or 1))
    cache_dir = cfg.get("cache_dir", "data/hard_benchmark")
    out_dir = Path(cfg.get("out_dir", "results"))
    out_dir.mkdir(parents=True, exist_ok=True)
    seed = cfg.get("seed", 0)
    ev = dict(cfg["evaluation"])
    ev.setdefault("seed", seed)
    n_boot = int(ev.get("bootstrap", 2000))
    go_margin = float(cfg.get("go_margin", 0.05))
    if not _HAS_GUDHI:
        raise RuntimeError("gudhi is required for the hard benchmark (pip install gudhi)")

    coh, _ = generate_coherent(cfg, cache_dir, workers)
    regimes = [r["name"] for r in cfg["regimes"]]
    sc = cfg["surrogate"]
    sur = generate_surrogates(coh, cfg, sc["final"], cache_dir, workers)
    rob_final = sc.get("robustness_final")
    sur_rob = generate_surrogates(coh, cfg, rob_final, cache_dir, workers) if rob_final else None

    t0 = time.time()
    topo_ds = cfg["descriptor"]["topo_downsample"]
    dec = cfg.get("feature_decimals", 8)
    auth = {k: cfg["descriptor"].get(k, {}) for k in ("x1", "x2")}
    Fc = compute_features(coh["fields"], topo_ds, workers, dec, authentic=auth)
    Fs = compute_features(sur["fields"], topo_ds, workers, dec, authentic=auth)
    Fr = (compute_features(sur_rob["fields"], topo_ds, workers, dec, authentic=auth)
          if sur_rob else None)
    feat_seconds = time.time() - t0
    print(f"[features] computed in {feat_seconds:.0f}s")

    N = len(coh["fields"])
    reg, grp = coh["regime"], coh["group"]
    sets = list(FEATURE_SETS)

    def stack(Fa, Fb):
        return {k: np.vstack([Fa[k], Fb[k]]) for k in Fa}

    F_bin = stack(Fc, Fs)
    y_bin = np.r_[np.zeros(N, int), np.ones(N, int)]
    g_bin = np.r_[grp, grp]
    r_bin = np.r_[reg, reg]

    tasks, task_data = {}, {}
    tasks["binary_pooled"] = run_task("binary_pooled", F_bin, y_bin, g_bin, sets, ev, 0.5)
    task_data["binary_pooled"] = (y_bin, g_bin)
    y6 = 2 * r_bin + y_bin
    tasks["multiclass6"] = run_task("multiclass6", F_bin, y6, g_bin, sets, ev,
                                    1.0 / (2 * len(regimes)))
    task_data["multiclass6"] = (y6, g_bin)
    for i, rn in enumerate(regimes):
        m = r_bin == i
        Fi = {k: v[m] for k, v in F_bin.items()}
        tasks[f"binary_{rn}"] = run_task(f"binary_{rn}", Fi, y_bin[m], g_bin[m], sets, ev, 0.5)
        task_data[f"binary_{rn}"] = (y_bin[m], g_bin[m])
    if Fr is not None:
        name = f"binary_pooled_iaaft_{rob_final}"
        tasks[name] = run_task(name, stack(Fc, Fr), y_bin, g_bin, sets, ev, 0.5)
        task_data[name] = (y_bin, g_bin)

    for name, task in tasks.items():
        y, g = task_data[name]
        task["verdicts"] = verdicts(task, y, g, go_margin, n_boot, seed)
        task["union_verdicts"] = union_verdicts(task, y, g, go_margin, n_boot, seed)

    # Existing (non-grouped) StratifiedKFold evaluator, for comparison only.
    strat = {}
    for name, (Fx, yx) in {"binary_pooled": (F_bin, y_bin), "multiclass6": (F_bin, y6)}.items():
        strat[name] = {}
        for s in ["descriptor", "spectral_view", "topological_view"] + SPECIFIED:
            strat[name][s] = evaluate(Fx[s], yx, seed=seed)
        print(f"[stratified-kfold] {name}: " + ", ".join(
            f"{s}={v['best_accuracy']:.3f}" for s, v in strat[name].items()))

    def surrogate_summary(S):
        info = S["info"]
        sm = np.array([i["spectral_mismatch"] for i in info])
        hm = np.array([i["histogram_mismatch"] for i in info])
        it = np.array([i["iterations"] for i in info])
        he = np.array([i["hermitian_error"] for i in info])
        return {"n": len(info), "spectral_mismatch_mean": float(sm.mean()),
                "spectral_mismatch_max": float(sm.max()),
                "histogram_mismatch_mean": float(hm.mean()),
                "histogram_mismatch_max": float(hm.max()),
                "iterations_mean": float(it.mean()),
                "converged_fraction": float(np.mean([i["converged"] for i in info])),
                "hermitian_error_max": float(he.max()), "seconds": round(S["seconds"], 1)}

    primary = tasks["binary_pooled"]["verdicts"]
    res = {
        "benchmark": "structure-only: coherent 2D turbulence vs IAAFT surrogates",
        "config": cfg,
        "n_coherent": int(N), "n_total_fields": int(2 * N),
        "regimes": regimes,
        "surrogate_primary": {"final": sc["final"], **surrogate_summary(sur)},
        "surrogate_robustness": ({"final": rob_final, **surrogate_summary(sur_rob)}
                                 if sur_rob else None),
        "feature_sets": {k: {"tier": v[0], "description": v[1],
                             "dim": int(F_bin[k].shape[1])} for k, v in FEATURE_SETS.items()},
        "tasks": tasks,
        "stratified_kfold_comparison": strat,
        "go_margin": go_margin,
        "verdict_specified": primary["specified"]["verdict"],
        "verdict_strict": primary["strict"]["verdict"],
        "verdict_overall": primary["strict"]["verdict"],
        "verdict_rich_topology": primary.get("rich_topology", {}).get("verdict"),
        "rich_topology_go_tasks": [k for k, t in tasks.items()
                                   if t["verdicts"].get("rich_topology", {}).get("any_go")],
        "verdict_overall_rule": "overall = strict verdict: the specified baselines are blind by "
                                "construction on this benchmark, so only the strict comparison "
                                "tests superiority over strong baselines",
        "verdict_vs_union_any_config": union_summary(tasks)["verdict"],
        "vs_union_summary": union_summary(tasks),
        "timing_seconds": {"simulation": float(coh.get("sim_seconds", np.nan)),
                           "surrogates": round(sur["seconds"], 1),
                           "features": round(feat_seconds, 1),
                           "total_this_run": round(time.time() - t_start, 1)},
    }
    stem = cfg.get("report_stem", "hard_benchmark")
    (out_dir / f"{stem}.json").write_text(json.dumps(_strip_oof(res), indent=2))
    (Path(cache_dir) / f"{stem}_full.pkl").write_bytes(pickle.dumps(res))
    write_report(out_dir / f"{stem}.md", res)
    if cfg.get("figures", True):
        from .hard_figures import make_figures
        make_figures(res, coh, sur, out_dir / "figures")
    print(f"\n=== PRIMARY (binary_pooled) VERDICT vs specified baselines: "
          f"{res['verdict_specified']} (margin {primary['specified']['margin']:+.3f}); "
          f"vs strict baselines: {res['verdict_strict']} "
          f"(margin {primary['strict']['margin']:+.3f}) ===")
    us = res["vs_union_summary"]
    print(f"=== ANY configuration >= +{go_margin:.2f} over baselines_union on ANY task: "
          f"{us['verdict']} (best: `{us['best_config']}` on `{us['best_task']}`, margin "
          f"{us['best_margin']:+.3f}) ===")
    return res


def _strip_oof(obj):
    if isinstance(obj, dict):
        return {k: _strip_oof(v) for k, v in obj.items() if k != "oof"}
    if isinstance(obj, list):
        return [_strip_oof(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------
def _fmt_ci(ci):
    return f"[{ci[0]:+.3f}, {ci[1]:+.3f}]"


def _task_table(task):
    R = task["results"]
    L = ["| Feature set | tier | dim | best acc | LogReg acc (fold std) | RF acc (fold std) "
         "| macro-F1 (best clf) | silhouette |",
         "|---|---|---|---|---|---|---|---|"]
    for s, r in R.items():
        tier = FEATURE_SETS[s][0]
        name = f"**{s}**" if s == "descriptor" else s
        bc = r["best_classifier"]
        L.append(f"| {name} | {tier} | {task.get('_dims', {}).get(s, '')} | "
                 f"{r['best_accuracy']:.3f} | "
                 f"{r['logreg']['accuracy']:.3f} ({r['logreg']['accuracy_std']:.3f}) | "
                 f"{r['rf']['accuracy']:.3f} ({r['rf']['accuracy_std']:.3f}) | "
                 f"{r[bc]['f1_macro']:.3f} | {r['silhouette']:.3f} |")
    return L


def _verdict_lines(v):
    s, st = v["specified"], v["strict"]
    tm, ts, inc = (v["topology_vs_spectrum_moments"], v["topology_vs_structure_baselines"],
                   v["incremental_topology"])
    return [
        f"- vs specified baselines: descriptor {s['descriptor_accuracy']:.3f} vs "
        f"`{s['best_baseline']}` {s['best_baseline_accuracy']:.3f} -> margin "
        f"{s['margin']:+.3f}, 95% CI {_fmt_ci(s['margin_ci95'])} -> **{s['verdict']}**",
        f"- vs strict baselines: descriptor {st['descriptor_accuracy']:.3f} vs "
        f"`{st['best_baseline']}` {st['best_baseline_accuracy']:.3f} -> margin "
        f"{st['margin']:+.3f}, 95% CI {_fmt_ci(st['margin_ci95'])} -> **{st['verdict']}**",
        f"- topological view vs best spectrum/one-point baseline (`{tm['best_spectrum_moment_baseline']}`): "
        f"{tm['topological_view_accuracy']:.3f} vs {tm['best_accuracy']:.3f} "
        f"(margin {tm['margin']:+.3f}, CI {_fmt_ci(tm['margin_ci95'])})",
        f"- topological view vs best structure-aware physical baseline (`{ts['best_structure_baseline']}`): "
        f"{ts['topological_view_accuracy']:.3f} vs {ts['best_accuracy']:.3f} "
        f"(margin {ts['margin']:+.3f}, CI {_fmt_ci(ts['margin_ci95'])})",
        f"- incremental value of topology on top of all baselines: "
        f"{inc['baselines_union_accuracy']:.3f} -> {inc['baselines_union_plus_topo_accuracy']:.3f} "
        f"(gain {inc['gain']:+.3f}, CI {_fmt_ci(inc['gain_ci95'])})",
    ] + (_rich_lines(v["rich_topology"]) if "rich_topology" in v else [])


def _rich_lines(r):
    b = r["best_rich_variant"]
    bv = r["variants"][b]
    inc = r["incremental"]
    return [
        f"- richer topology: best variant `{b}` {bv['accuracy']:.3f} vs `baselines_union` "
        f"{r['baselines_union_accuracy']:.3f} -> margin {bv['margin_vs_baselines_union']:+.3f}, "
        f"CI {_fmt_ci(bv['margin_vs_baselines_union_ci95'])} (max attainable "
        f"{r['max_attainable_margin']:+.3f}) -> **{r['verdict']}**; adding all richer views "
        f"to `baselines_union`: gain {inc['gain']:+.3f}, CI {_fmt_ci(inc['gain_ci95'])}",
    ]


def rich_summary_table(tasks, order):
    """One row per task: can any richer topological vectorization beat
    baselines_union by >= 0.05?"""
    L = ["| Task | baselines_union | max attainable margin | old topo view (64^2) | "
         + " | ".join(f"`{k}`" for k in RICH)
         + " | union + rich topo (gain) | any >= +0.05? |",
         "|---|---|---|---|" + "---|" * len(RICH) + "---|---|"]
    for name in order:
        v = tasks[name]["verdicts"].get("rich_topology")
        if v is None:
            continue
        R = tasks[name]["results"]
        cells = []
        for k in RICH:
            x = v["variants"][k]
            ci = x["margin_vs_baselines_union_ci95"]
            cells.append(f"{x['accuracy']:.3f} ({x['margin_vs_baselines_union']:+.3f} "
                         f"[{ci[0]:+.3f}, {ci[1]:+.3f}])")
        inc = v["incremental"]
        L.append(f"| {name} | {v['baselines_union_accuracy']:.3f} | "
                 f"{v['max_attainable_margin']:+.3f} | "
                 f"{R['topological_view']['best_accuracy']:.3f} | " + " | ".join(cells)
                 + f" | {inc['baselines_union_plus_topo_rich_accuracy']:.3f} "
                   f"({inc['gain']:+.3f} [{inc['gain_ci95'][0]:+.3f}, {inc['gain_ci95'][1]:+.3f}]) | "
                 + ("**YES**" if v["any_go"] else "no") + " |")
    return L


_TASK_SHORT = {"binary_pooled": "binary (primary)", "multiclass6": "6 classes"}


def _union_section(res):
    us = res.get("vs_union_summary")
    if not us or not us.get("per_task"):
        return []
    gm = res["go_margin"]
    tnames = list(us["per_task"])
    T = res["tasks"]
    short = [_TASK_SHORT.get(t, t.replace("binary_pooled_iaaft_", "IAAFT-").replace("binary_", ""))
             for t in tnames]
    L = ["## Authentic X1/X2 and richer topology vs baselines_union", "",
         f"Rule: a configuration is GO on a task iff accuracy(config) - "
         f"accuracy(`baselines_union`) >= +{gm:.2f}. Every non-baseline configuration is tested, "
         f"including the post-hoc `sensitivity` rows. Headroom = 1 - accuracy(`baselines_union`) "
         f"is the largest margin any configuration can reach; where it is below {gm:.2f}, GO is "
         f"impossible on that task whatever the descriptor. The answer below is a maximum over "
         f"all configurations and tasks, which is optimistic (multiple comparisons), so the "
         f"table also counts configurations whose margin CI excludes zero.", "",
         f"**Answer: {'YES' if us['verdict'] == 'GO' else 'NO'} - verdict {us['verdict']}.** "
         f"Best margin over all configurations and tasks: `{us['best_config']}` on "
         f"`{us['best_task']}`, {us['best_margin']:+.3f}.", "",
         "| Task | baselines_union acc | headroom | GO possible? | best configuration | "
         f"margin [95% CI] | configs >= +{gm:.2f} (of which CI > 0) | verdict |",
         "|---|---|---|---|---|---|---|---|"]
    for t in tnames:
        v = us["per_task"][t]
        uv = T[t]["union_verdicts"]
        L.append(f"| {t} | {uv['baselines_union_accuracy']:.3f} | {v['headroom']:.3f} | "
                 f"{'yes' if v['go_arithmetically_possible'] else 'no'} | `{v['best_config']}` | "
                 f"{v['best_margin']:+.3f} {_fmt_ci(v['best_margin_ci95'])} | "
                 f"{len(v['configs_reaching_go_margin'])} "
                 f"({len(v.get('configs_reaching_go_margin_ci_above_0', []))}) | "
                 f"**{v['verdict']}** |")
    L += ["", "Accuracy (grouped CV, better of LogReg / RF) per configuration and task:", "",
          "| Configuration | tier | " + " | ".join(short) + " |",
          "|---|---|" + "---|" * len(tnames)]
    show = CANDIDATES + ["physical_structure_all", "baselines_union"]
    for c in show:
        if c not in T[tnames[0]]["results"]:
            continue
        L.append(f"| {c} | {FEATURE_SETS[c][0]} | " + " | ".join(
            f"{T[t]['results'][c]['best_accuracy']:.3f}" for t in tnames) + " |")
    L += ["", "Margin vs `baselines_union` [95% paired group-bootstrap CI]:", "",
          "| Configuration | " + " | ".join(short) + " |",
          "|---|" + "---|" * len(tnames)]
    for c in CANDIDATES:
        if c not in T[tnames[0]]["union_verdicts"]["configs"]:
            continue
        cells = []
        for t in tnames:
            r = T[t]["union_verdicts"]["configs"][c]
            cells.append(f"{r['margin']:+.3f} {_fmt_ci(r['margin_ci95'])}")
        L.append(f"| {c} | " + " | ".join(cells) + " |")
    L += ["", "Head-to-head comparisons (margin = accuracy(a) - accuracy(b) [95% CI]):", "",
          "| Comparison | " + " | ".join(short) + " |", "|---|" + "---|" * len(tnames)]
    for h in T[tnames[0]]["union_verdicts"]["head_to_head"]:
        cells = []
        for t in tnames:
            r = T[t]["union_verdicts"]["head_to_head"][h]
            cells.append(f"{r['accuracy']:.3f} vs {r['accuracy_b']:.3f}: {r['margin']:+.3f} "
                         f"{_fmt_ci(r['margin_ci95'])}")
        a, b = T[tnames[0]]["union_verdicts"]["head_to_head"][h]["a"], \
            T[tnames[0]]["union_verdicts"]["head_to_head"][h]["b"]
        L.append(f"| `{a}` vs `{b}` | " + " | ".join(cells) + " |")
    L.append("")
    return L


def write_report(path, res):
    cfg = res["config"]
    sim = cfg["simulation"]
    P = res["tasks"]["binary_pooled"]["verdicts"]
    sp, so = res["surrogate_primary"], res["surrogate_robustness"]
    dims = {k: v["dim"] for k, v in res["feature_sets"].items()}
    for t in res["tasks"].values():
        t["_dims"] = dims
    overall = res.get("verdict_overall", res["verdict_strict"])
    L = ["# Hard structure-only benchmark - GEM Go/No-Go", "",
         f"## Overall verdict: **{overall}**", "",
         "Overall = strict verdict. On this benchmark the baselines named in the brief are "
         "blind by construction (that is its purpose), so beating them does not show "
         "superiority over strong baselines; the strict set adds the phase-sensitive "
         "physical diagnostics a geophysical reviewer would expect. Interpretation: "
         "`docs/RESULTS.md`.", ""]
    L += [f"**Verdict as specified in the brief (primary binary coherent-vs-IAAFT task, "
          f"grouped CV, vs moments, spectral_slope, raw_pixels): {res['verdict_specified']}** "
          f"(margin {P['specified']['margin']:+.3f}, threshold {res['go_margin']}).", "",
          f"**Strict verdict (adds leak detectors and structure-aware physical baselines): "
          f"{res['verdict_strict']}** (margin {P['strict']['margin']:+.3f} vs "
          f"`{P['strict']['best_baseline']}`).", ""]
    order = ["binary_pooled", "multiclass6"] + [k for k in res["tasks"]
                                                 if k not in ("binary_pooled", "multiclass6")]
    us = res.get("vs_union_summary")
    if us and us.get("per_task"):
        L += [f"**Every view, the authentic X1+X2 descriptor and X1+X2+richer: does ANY "
              f"configuration beat `baselines_union` by >= +{res['go_margin']:.2f} on ANY task? "
              f"{'YES' if us['verdict'] == 'GO' else 'NO'} -> {us['verdict']}** (best: "
              f"`{us['best_config']}` on `{us['best_task']}`, margin {us['best_margin']:+.3f}). "
              f"See *Authentic X1/X2 and richer topology vs baselines_union* below.", ""]
    if "rich_topology" in P:
        go_tasks = res.get("rich_topology_go_tasks", [])
        L += ["## Richer topological vectorizations vs `baselines_union`", "",
              f"**Verdict: {P['rich_topology']['verdict']}** on the primary task; tasks where any "
              f"richer vectorization reaches >= +{res['go_margin']:.2f} over `baselines_union`: "
              f"{', '.join(go_tasks) if go_tasks else 'none'}.", "",
              "Cells: accuracy (margin vs `baselines_union` [95% paired group-bootstrap CI]). "
              "`max attainable margin` = 1 - accuracy of `baselines_union`: when it is below "
              "+0.05 the GO threshold cannot be reached by any feature set. Richer views are "
              f"computed from the full-resolution {sim['grid']}^2 field; "
              f"`topological_view_fullres` (8 summaries at {sim['grid']}^2) is the "
              "resolution-matched control in the task tables.", ""]
        L += rich_summary_table(res["tasks"], order) + [""]
    L += ["## Setup", "",
          f"- Coherent fields: {res['n_coherent']} snapshots of forced 2D turbulence "
          f"(grid {sim['grid']}^2, IF-{sim.get('scheme', 'rk4').upper()}, dt={sim['dt']}, "
          f"nu={sim['nu']}, linear drag={sim.get('drag', 0.0)}), regimes "
          + ", ".join(f"`{r['name']}` (k_f={r['forcing_wavenumber']}, amp={r.get('forcing_amp')})"
                      for r in cfg["regimes"]) + ".",
          f"- {sim['runs_per_regime']} independent trajectories per regime, "
          f"{sim['snapshots_per_run']} snapshots each, {sim['snapshot_interval']} time units apart "
          f"after {sim['spinup_time']} time units of spin-up. Every field standardized "
          f"(zero mean, unit variance).",
          f"- Surrogates: IAAFT, {cfg['surrogate']['n_iter']} iterations, final step "
          f"`{sp['final']}` (primary). Relative Fourier-amplitude mismatch mean "
          f"{sp['spectral_mismatch_mean']:.1e} (max {sp['spectral_mismatch_max']:.1e}); "
          f"histogram mismatch mean {sp['histogram_mismatch_mean']:.1e} std "
          f"(max {sp['histogram_mismatch_max']:.1e}); Hermitian error <= {sp['hermitian_error_max']:.0e}.",
          ]
    if so:
        L.append(f"- Robustness surrogates: final step `{so['final']}`: spectral mismatch mean "
                 f"{so['spectral_mismatch_mean']:.1e}, histogram mismatch mean "
                 f"{so['histogram_mismatch_mean']:.1e}.")
    ev = cfg["evaluation"]
    L += [f"- Total fields: {res['n_total_fields']} ({res['n_coherent']} coherent + "
          f"{res['n_coherent']} surrogates). CV: StratifiedGroupKFold, {ev['n_splits']} folds x "
          f"{ev['n_repeats']} repeats, group = trajectory (a snapshot and its surrogate always "
          f"share a fold). Classifiers: LogReg and RandomForest(300); `best acc` = the better "
          f"of the two (same rule for every feature set). Margin CIs: paired bootstrap over "
          f"trajectories ({ev.get('bootstrap', 2000)} resamples).",
          f"- Descriptor configuration pre-registered as the repo default "
          f"(topo_downsample={cfg['descriptor']['topo_downsample']}); `*_fullres*` rows are a "
          f"post-hoc sensitivity check and are not used for any verdict.",
          "- Timing (wall clock, s): " + ", ".join(
              f"{k} {v:.0f}" for k, v in res["timing_seconds"].items())
          + ". The cloud VM was intermittently suspended during the run, so these overstate "
            "compute; the solver needs ~15 ms per IF-RK4 step at 256^2 on one core "
            "(~160 s per trajectory).", ""]
    L += ["## Tiers", "",
          "- `ours`, `ours-ablation`: the descriptor and its two views.",
          "- `baseline-specified`: the baselines named in the Go/No-Go brief.",
          "- `baseline-validity`: strongest spectral / one-point baselines; must be ~chance if the "
          "benchmark is valid (leak detectors).",
          "- `baseline-structure`: phase-sensitive physical diagnostics of coherent structures "
          "(Okubo-Weiss, gradient statistics, increment flatness) - no topology.",
          "- `baseline-union` / `incremental`: all non-topological baselines, without / with the "
          "topological view added (incremental value of topology).",
          "- `ours-rich`: richer vectorizations of the same sublevel-set persistent homology "
          "(`src/descriptor/topology_rich.py`): persistence images, Betti curves, a multi-scale "
          "Gaussian filtration, their concatenation, and the spectral view + all of them.",
          "- `graph-laplacian`: the existing graph view alone (Laplacian eigenvalues), kept as the "
          "baseline for the authentic functional graph X1.",
          "- `authentic`: X1 (functional graph), X2 (sig6 persistent homology, periodic), X1+X2, "
          "the X3 fusion and X1+X2+richer, ported verbatim from the multiview-nn-descriptor "
          "reference (`src/descriptor/authentic.py`).", ""]
    L += _union_section(res)
    titles = {"binary_pooled": "Primary task: coherent vs IAAFT surrogate (all forcing scales pooled)",
              "multiclass6": "Multiclass: forcing scale x {coherent, surrogate} (6 classes)"}
    for name in order:
        t = res["tasks"][name]
        title = titles.get(name, f"Task `{name}`")
        L += [f"## {title}", "",
              f"N = {t['n']}, classes = {t['n_classes']}, trajectories = {t['n_groups']}, "
              f"chance = {t['chance']:.3f}.", ""]
        L += _task_table(t)
        L += ["", "Verdicts / key comparisons:", ""] + _verdict_lines(t["verdicts"]) + [""]
    L += ["## Existing StratifiedKFold evaluator (no grouping) for comparison", "",
          "| Task | Feature set | best acc (StratifiedKFold) | best acc (grouped CV) |",
          "|---|---|---|---|"]
    for tname, sets in res["stratified_kfold_comparison"].items():
        for s, v in sets.items():
            L.append(f"| {tname} | {s} | {v['best_accuracy']:.3f} | "
                     f"{res['tasks'][tname]['results'][s]['best_accuracy']:.3f} |")
    L.append("")
    L += ["## Figures", "",
          "- `figures/fields_coherent_vs_iaaft.png` - example coherent fields, their IAAFT "
          "surrogates, and the matched spectra / histograms.",
          "- `figures/accuracy_binary.png`, `figures/accuracy_multiclass.png` - grouped-CV "
          "accuracy per feature set.",
          "- `figures/topology_features.png` - distribution of the topological-view features "
          "for coherent vs surrogate fields.",
          "- `figures/rich_topology_margins.png` - margin of the best richer topological "
          "vectorization over `baselines_union` per task, with the +0.05 GO threshold.",
          "- `figures/margin_vs_union.png` - margin of every non-baseline configuration over "
          "`baselines_union` per task, with the +0.05 GO line and the headroom.", ""]
    for t in res["tasks"].values():
        t.pop("_dims", None)
    path.write_text("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/hard_benchmark.yaml")
    ap.add_argument("--report-only", action="store_true",
                    help="re-render the Markdown report from the cached full results")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    if args.report_only:
        stem = cfg.get("report_stem", "hard_benchmark")
        res = pickle.loads((Path(cfg.get("cache_dir", "data/hard_benchmark"))
                            / f"{stem}_full.pkl").read_bytes())
        res.setdefault("verdict_overall", res["verdict_strict"])
        res.setdefault("verdict_overall_rule", "overall = strict verdict: the specified "
                       "baselines are blind by construction on this benchmark, so only the "
                       "strict comparison tests superiority over strong baselines")
        out = Path(cfg.get("out_dir", "results"))
        (out / f"{stem}.json").write_text(json.dumps(_strip_oof(res), indent=2))
        write_report(out / f"{stem}.md", res)
        return
    run(cfg)


if __name__ == "__main__":
    main()
