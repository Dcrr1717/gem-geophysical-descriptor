"""ECT / DECT study: a richer topological representation on two tasks.

Pre-registered in docs/ECT_RESULTS.md and config/ect_vortex.yaml.

Task A  iaaft_binary   the EXISTING structure-only benchmark (coherent 2D
                       turbulence vs IAAFT surrogates; cached 256^2 fields).
Task B  vortex_merger  a genuinely topological task: co-rotating vortex pairs
                       that merged (one coherent core) vs pairs that stayed
                       apart (two cores), evolved with the repo's solver
                       (src/sims/vortex_events.py).

Views compared on both tasks:
  topological  ph_counts (significant H0/H1 counts), x2_ph, 8 summaries at
               256^2, Betti curves, persistence images, multi-scale filtration,
               ECT (src/descriptor/ect.py) and DECT (src/descriptor/dect.py,
               optional torch);
  baselines    okubo_weiss, gradient_stats, increment_flatness, all three,
               moments, power_spectrum, histogram_quantiles and their union
               `baselines_union`;
  incremental  baselines_union + ECT, baselines_union + ph_counts.
Protocol: repeated StratifiedGroupKFold (group = trajectory), best of LogReg /
RF for feature views (DECT: its own CNN head on the same splits), paired
group-bootstrap 95% CIs, Benjamini-Hochberg over every topological view x
task comparison with baselines_union.

Usage: python -m src.eval.ect_benchmark --config config/ect_vortex.yaml
"""
from __future__ import annotations

import argparse
import json
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import yaml

from ..descriptor.authentic import sublevel_persistence
from ..descriptor.ect import ECT_LEVELS, ECT_RES, ect_features, level_masks
from ..sims.vortex_events import event_jobs, simulate_event
from .hard_benchmark import (_standardize, _strip_oof, compute_features, generate_coherent,
                             generate_surrogates, paired_bootstrap_margin, run_task)
from .robustness_sweep import bh_qvalues

TOPO = ["ph_counts", "x2_ph", "topological_view_fullres", "topo_betti_curves",
        "topo_persistence_image", "topo_multiscale", "ect"]
STRUCT = ["okubo_weiss", "gradient_stats", "increment_flatness", "physical_structure_all"]
BASE = STRUCT + ["moments", "power_spectrum", "histogram_quantiles", "baselines_union"]
INCR = ["baselines_union+ect", "baselines_union+ph_counts"]
PRIMARY = {"iaaft_binary": ["ect", "dect"], "vortex_merger": ["ph_counts", "ect", "dect"]}
PH_COUNT_THRESHOLDS = (0.5, 1.0, 2.0)   # persistence, field-std units
LABELS = {
    "ph_counts": "H0/H1 counts (persistence >= 0.5, 1, 2)",
    "x2_ph": "X2: sig6 PH summaries", "topological_view_fullres": "8 PH summaries, 256^2",
    "topo_betti_curves": "Betti curves", "topo_persistence_image": "persistence images",
    "topo_multiscale": "multi-scale filtration", "ect": "ECT (exact, injective)",
    "dect": "DECT (learned directions/thresholds)",
    "okubo_weiss": "Okubo-Weiss", "gradient_stats": "gradient statistics",
    "increment_flatness": "increment flatness", "physical_structure_all": "all 3 structure baselines",
    "moments": "moments", "power_spectrum": "power spectrum", "histogram_quantiles": "histogram quantiles",
    "baselines_union": "baselines_union", "baselines_union+ect": "baselines_union + ECT",
    "baselines_union+ph_counts": "baselines_union + H0/H1 counts",
}


# ---------------------------------------------------------------------------
# Extra views
# ---------------------------------------------------------------------------
def ph_counts(w, thresholds=PH_COUNT_THRESHOLDS):
    """Numbers of significant components (H0) and loops (H1) of the periodic
    sublevel-set filtration: the most direct 'how many coherent structures' view."""
    h0, h1 = sublevel_persistence(w, (True, True), min(thresholds))
    return np.array([float((h >= t).sum()) for h in (h0, h1) for t in thresholds])


def _extra_worker(w):
    return ect_features(w), ph_counts(w), level_masks(w)


def task_features(fields, cfg, workers, want_masks):
    dec = cfg.get("feature_decimals", 8)
    auth = {k: cfg["descriptor"].get(k, {}) for k in ("x1", "x2")}
    F = compute_features(fields, cfg["descriptor"]["topo_downsample"], workers, dec,
                         fullres=True, rich=True, authentic=auth)
    with Pool(workers) as pool:
        ex = pool.map(_extra_worker, list(fields), chunksize=4)
    F["ect"] = np.vstack([e[0] for e in ex])
    F["ph_counts"] = np.vstack([e[1] for e in ex])
    F["baselines_union+ect"] = np.hstack([F["baselines_union"], F["ect"]])
    F["baselines_union+ph_counts"] = np.hstack([F["baselines_union"], F["ph_counts"]])
    F = {k: v for k, v in F.items() if k in TOPO + BASE + INCR}
    masks = np.stack([e[2] for e in ex]) if want_masks else None
    return F, masks


# ---------------------------------------------------------------------------
# Task B data
# ---------------------------------------------------------------------------
def _event_worker(args):
    job, ev = args
    return simulate_event(job, ev)


def generate_vortex_events(cfg, workers):
    ev = cfg["vortex_events"]
    from .hard_benchmark import _key
    key = _key({"ev": {k: v for k, v in ev.items() if k != "cache_dir"},
                "seed": cfg.get("seed", 0), "v": 1})
    path = Path(ev.get("cache_dir", "data/vortex_events")) / f"events_{key}.npz"
    if path.exists():
        print(f"[vortex] loading cached events {path}")
        d = np.load(path)
        return {k: d[k] for k in d.files}
    t0 = time.time()
    jobs = event_jobs(ev, cfg.get("seed", 0))
    print(f"[vortex] simulating {len(jobs)} vortex-pair trajectories (grid {ev['grid']})")
    with Pool(workers) as pool:
        out = pool.map(_event_worker, [(j, ev) for j in jobs])
    fields, label, group, d_over_a, snap = [], [], [], [], []
    for g, (job, snaps) in enumerate(out):
        for s_i, w in enumerate(snaps):
            fields.append(_standardize(w))
            label.append(job["label"])
            group.append(g)
            d_over_a.append(job["d_over_a"])
            snap.append(s_i)
    data = {"fields": np.asarray(fields), "label": np.asarray(label),
            "group": np.asarray(group), "d_over_a": np.asarray(d_over_a),
            "snap": np.asarray(snap), "seconds": np.asarray(time.time() - t0)}
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **data)
    print(f"[vortex] {len(fields)} fields in {time.time() - t0:.0f}s -> {path}")
    return data


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate_task(name, F, masks, y, g, cfg, ev, dect_cfg, sets=None):
    sets = sets or TOPO + BASE + INCR
    task = run_task(name, F, y, g, sets, ev, 1.0 / len(np.unique(y)))
    if dect_cfg is not None:
        from ..descriptor.dect import train_eval_grouped
        t0 = time.time()
        n = int(round((np.sqrt(masks.shape[2]) - 1) / 2))
        task["results"]["dect"] = train_eval_grouped(
            masks, y, g, n, dect_cfg, seed=ev["seed"], n_splits=ev["n_splits"],
            n_repeats=ev["n_repeats"])
        print(f"  {'dect':>26s}: best={task['results']['dect']['best_accuracy']:.3f} "
              f"[{time.time() - t0:.0f}s]", flush=True)
    return task


def margins(task, y, g, n_boot, seed, topo=None):
    """Paired group-bootstrap margins of every topological view (`topo`, default
    TOPO, plus DECT and the incremental sets) over baselines_union and over the
    best structure baseline."""
    topo = topo or TOPO
    R = task["results"]
    u = R["baselines_union"]
    sb = max(STRUCT, key=lambda k: R[k]["best_accuracy"])
    out = {"baselines_union_accuracy": u["best_accuracy"], "headroom": 1 - u["best_accuracy"],
           "best_structure_baseline": sb, "vs_union": {}, "vs_best_structure": {}}
    for k in [c for c in topo + ["dect"] + INCR if c in R]:
        m = paired_bootstrap_margin(R[k], u, y, g, n_boot, seed)
        out["vs_union"][k] = {"accuracy": R[k]["best_accuracy"], "margin": m["margin"],
                              "ci95": m["ci95"], "p_le_0": m["p_le_0"]}
        m2 = paired_bootstrap_margin(R[k], R[sb], y, g, n_boot, seed)
        out["vs_best_structure"][k] = {"margin": m2["margin"], "ci95": m2["ci95"]}
    return out


def decide(tasks, thr, primary=None, topo=None):
    """Pre-registered rule: BH over all (task, topological view) comparisons vs
    baselines_union; GO for a task if a primary view has margin >= thr, CI
    lower bound > 0 and BH q < 0.05."""
    primary = primary or PRIMARY
    topo = topo or TOPO
    fam = [(t, k) for t in tasks for k in tasks[t]["margins"]["vs_union"]
           if k in topo + ["dect"]]
    q = bh_qvalues([tasks[t]["margins"]["vs_union"][k]["p_le_0"] for t, k in fam])
    for (t, k), qq in zip(fam, q):
        tasks[t]["margins"]["vs_union"][k]["q_bh"] = float(qq)
    verdict = {}
    for t, task in tasks.items():
        mu = task["margins"]["vs_union"]
        real = {k: bool(e["margin"] >= thr and e["ci95"][0] > 0 and e.get("q_bh", 1) < 0.05)
                for k, e in mu.items() if k in topo + ["dect"]}
        prim = [k for k in primary[t] if k in mu]
        best = max(prim, key=lambda k: mu[k]["margin"]) if prim else None
        verdict[t] = {"primary_views": prim, "best_primary_view": best,
                      "best_primary_margin": mu[best]["margin"] if best else None,
                      "best_primary_ci95": mu[best]["ci95"] if best else None,
                      "headroom": task["margins"]["headroom"],
                      "go_arithmetically_possible": task["margins"]["headroom"] >= thr,
                      "views_with_real_advantage": [k for k, v in real.items() if v],
                      "verdict": "GO" if any(real.get(k) for k in prim) else "NO-GO"}
    return verdict


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def load_config(path):
    cfg = yaml.safe_load(Path(path).read_text())
    if "base_config" in cfg:
        base = yaml.safe_load(Path(cfg["base_config"]).read_text())
        base.update({k: v for k, v in cfg.items() if k != "base_config"})
        cfg = base
    return cfg


def run(cfg):
    t_start = time.time()
    es = cfg["ect_study"]
    workers = int(cfg.get("n_workers", 4))
    seed = cfg.get("seed", 0)
    ev = dict(cfg["evaluation"])
    ev.setdefault("seed", seed)
    ev["n_repeats"] = es.get("n_repeats", ev["n_repeats"])
    n_boot = int(ev.get("bootstrap", 2000))
    out_dir = Path(cfg.get("out_dir", "results"))
    dect_cfg = es.get("dect", {})
    if dect_cfg.get("enabled", False):
        from ..descriptor.dect import HAS_TORCH
        if not HAS_TORCH:
            print("[dect] torch not installed: DECT view skipped (pip install torch)")
            dect_cfg = None
    else:
        dect_cfg = None

    # Task A: the existing structure-only IAAFT benchmark (cached fields)
    cache = cfg.get("cache_dir", "data/hard_benchmark")
    coh, _ = generate_coherent(cfg, cache, workers)
    sur = generate_surrogates(coh, cfg, cfg["surrogate"]["final"], cache, workers)
    N = len(coh["fields"])
    FA, MA = task_features(np.concatenate([coh["fields"], sur["fields"]]), cfg, workers,
                           dect_cfg is not None)
    yA = np.r_[np.zeros(N, int), np.ones(N, int)]
    gA = np.r_[coh["group"], coh["group"]]
    # Task B: vortex events
    vb = generate_vortex_events(cfg, workers)
    FB, MB = task_features(vb["fields"], cfg, workers, dect_cfg is not None)
    yB, gB = vb["label"].astype(int), vb["group"].astype(int)

    tasks = {}
    for name, F, M, y, g in (("iaaft_binary", FA, MA, yA, gA),
                             ("vortex_merger", FB, MB, yB, gB)):
        tasks[name] = evaluate_task(name, F, M, y, g, cfg, ev, dect_cfg)
        tasks[name]["margins"] = margins(tasks[name], y, g, n_boot, seed)
        tasks[name]["class_mean_ect"] = [F["ect"][y == c].mean(0).tolist() for c in (0, 1)]
        tasks[name]["class_std_ect"] = [F["ect"][y == c].std(0).tolist() for c in (0, 1)]
    verdict = decide(tasks, float(es["rule"]["min_margin"]))
    res = {"study": "ECT / DECT on the IAAFT structure-only task and the vortex-merger task",
           "config": cfg, "labels": LABELS, "ect": {"levels": list(ECT_LEVELS), "res": ECT_RES},
           "dect_enabled": dect_cfg is not None, "tasks": tasks, "verdict": verdict,
           "seconds": round(time.time() - t_start, 1)}
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = es.get("report_stem", "ect_vortex")
    (out_dir / f"{stem}.json").write_text(json.dumps(_strip_oof(res), indent=2))
    from .ect_report import make_figures, write_report
    write_report(res, out_dir / f"{stem}.md")
    if cfg.get("figures", True):
        make_figures(res, vb, coh, sur, out_dir / "figures")
    for t, v in verdict.items():
        print(f"=== {t}: {v['verdict']} (best primary view `{v['best_primary_view']}`, "
              f"margin {v['best_primary_margin']:+.3f}, headroom {v['headroom']:.3f}) ===")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/ect_vortex.yaml")
    ap.add_argument("--no-dect", action="store_true", help="skip the optional torch DECT view")
    a = ap.parse_args()
    cfg = load_config(a.config)
    if a.no_dect:
        cfg["ect_study"].setdefault("dect", {})["enabled"] = False
    run(cfg)


if __name__ == "__main__":
    main()
