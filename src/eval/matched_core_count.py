"""Task C: matched-scale core count (one vortex core vs two), pre-registered.

Pre-registration and results: docs/MATCHED_CORE_COUNT.md. Two phases, after
Rubin (2007), "The design versus the analysis of observational studies":

1. Design phase (`--design`). This phase is outcome-blind: no descriptor view
   and no benchmark classifier is computed.
   - Simulate the pool: `runs_per_class` one-core and two-core trajectories
     (src/sims/core_count.py), 8 snapshots each.
   - Keep the snapshots whose descriptor-free core count equals the design
     (1 or 2).
   - Match the classes one-to-one on the propensity score of the four design
     statistics: spectral centroid, spectral width, occupied area,
     standardized peak.
   - Check the pre-registered balance gate.
   - Writes results/matched_core_count_design.json,
     results/figures/core_count_matching.png and the frozen matched set
     data/core_count/matched_<key>.npz.
2. Analysis phase (default). This phase refuses to run unless the gate
   passed.
   - Compute every view on the matched set: the topological views,
     `ect_boundary_aware`, DECT, the baselines, `baselines_union` and the
     design statistics themselves as a sanity view.
   - Run the same grouped CV, paired group-bootstrap margins and BH
     correction as the ECT study, and apply the pre-registered rule.

Usage:
    python -m src.eval.matched_core_count --config config/matched_core_count.yaml --design
    python -m src.eval.matched_core_count --config config/matched_core_count.yaml
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from ..descriptor.ect import ect_features, roll_quiet_to_edge
from ..sims.core_count import (DESIGN_STATS, balance, balance_gate, core_jobs, design_auc,
                               design_stats, match_propensity, simulate_cores)
from ..sims.vortex_events import core_count
from .ect_benchmark import (BASE, INCR, LABELS, TOPO, decide, evaluate_task, load_config,
                            margins, task_features)
from .hard_benchmark import _key, _standardize, _strip_oof

TASK = "matched_core_count"
TOPO_C = TOPO + ["ect_boundary_aware"]
PRIMARY_C = ["ph_counts", "ect", "ect_boundary_aware", "dect"]
SANITY = ["design_stats"]


# ---------------------------------------------------------------------------
# Design phase (outcome-blind)
# ---------------------------------------------------------------------------
def _pool_worker(args):
    job, cc = args
    job, snaps = simulate_cores(job, cc)
    fields = [_standardize(w) for w in snaps]
    return (job, np.asarray(fields, dtype=np.float32), [design_stats(w) for w in fields],
            [core_count(w) for w in fields])


def _pool_key(cfg):
    cc = cfg["core_count"]
    return _key({"cc": {k: v for k, v in cc.items() if k != "cache_dir"},
                 "seed": cfg.get("seed", 0), "v": 1})


def generate_pool(cfg, workers):
    cc = cfg["core_count"]
    path = Path(cc.get("cache_dir", "data/core_count")) / f"pool_{_pool_key(cfg)}.npz"
    if path.exists():
        print(f"[pool] loading cached pool {path}")
        d = np.load(path)
        return {k: d[k] for k in d.files}
    t0 = time.time()
    jobs = core_jobs(cc, cfg.get("seed", 0))
    print(f"[pool] simulating {len(jobs)} trajectories (grid {cc['grid']})", flush=True)
    with Pool(workers) as pool:
        out = pool.map(_pool_worker, [(j, cc) for j in jobs])
    F, S, cnt, y, g, snap, jid = [], [], [], [], [], [], []
    for gi, (job, fields, stats, counts) in enumerate(out):
        for si in range(len(fields)):
            F.append(fields[si])
            S.append(stats[si])
            cnt.append(counts[si])
            y.append(job["label"])
            g.append(gi)
            snap.append(si)
    data = {"fields": np.asarray(F, dtype=np.float32), "S": np.asarray(S),
            "core_count": np.asarray(cnt), "label": np.asarray(y), "group": np.asarray(g),
            "snap": np.asarray(snap), "jobs": np.asarray(json.dumps(
                [{k: v for k, v in j.items() if k != "seed_words"} for j, *_ in out])),
            "seconds": np.asarray(time.time() - t0)}
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **data)
    print(f"[pool] {len(F)} snapshots in {time.time() - t0:.0f}s -> {path}", flush=True)
    return data


def design(cfg, workers):
    pool = generate_pool(cfg, workers)
    S, y, g, cnt = pool["S"], pool["label"], pool["group"], pool["core_count"]
    valid = cnt == np.where(y == 0, 1, 2)
    iv = np.flatnonzero(valid)
    mc = cfg["matching"]
    a, b, dist = match_propensity(S[iv], y[iv], float(mc["caliper_sd"]), int(mc.get("degree", 2)))
    sel = np.sort(np.r_[iv[a], iv[b]])
    pair_id = np.full(len(y), -1)
    pair_id[iv[a]] = np.arange(len(a))
    pair_id[iv[b]] = np.arange(len(a))
    gate = balance_gate(S[sel], y[sel], g[sel], mc["gate"])
    digest = hashlib.sha256(sel.astype(np.int64).tobytes()).hexdigest()[:16]
    res = {"phase": "design (outcome-blind: no descriptor view, no benchmark classifier)",
           "pool_snapshots": int(len(y)), "pool_trajectories": int(len(np.unique(g))),
           "valid_fraction_by_class": [float(valid[y == c].mean()) for c in (0, 1)],
           "invalid_core_counts": {str(c): {str(int(k)): int(n) for k, n in zip(
                                       *np.unique(cnt[(y == c) & ~valid], return_counts=True))}
                                   for c in (0, 1)},
           "balance_before": balance(S[iv], y[iv]),
           "design_auc_before": design_auc(S[iv], y[iv], g[iv]),
           "matching": mc, "n_pairs": int(len(a)),
           "matched_trajectories": int(len(np.unique(g[sel]))),
           "matched_snapshots_by_time": np.bincount(pool["snap"][sel]).tolist(),
           "gate": gate, "matched_set_sha256_16": digest}
    out = Path(cfg.get("out_dir", "results"))
    out.mkdir(parents=True, exist_ok=True)
    (out / "matched_core_count_design.json").write_text(json.dumps(res, indent=2))
    matching_figure(S[iv], y[iv], np.isin(iv, sel), out / "figures" / "core_count_matching.png")
    if gate["passed"]:
        path = Path(cfg["core_count"].get("cache_dir", "data/core_count")) / \
            f"matched_{_pool_key(cfg)}.npz"
        np.savez(path, fields=pool["fields"][sel], label=y[sel], group=g[sel], S=S[sel],
                 snap=pool["snap"][sel], pair_id=pair_id[sel], pool_index=sel)
        print(f"[design] gate PASSED: {len(a)} pairs, set {digest} -> {path}")
    else:
        print(f"[design] gate FAILED: {gate['checks']}")
    print(json.dumps({k: res[k] for k in ("valid_fraction_by_class", "n_pairs",
                                          "matched_trajectories", "design_auc_before")}))
    print(json.dumps(gate, indent=1))
    return res


def matching_figure(S, y, matched, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = ["spectral centroid", "spectral width", "occupied area", "standardized peak"]
    logs = [False, False, True, True]
    fig, axes = plt.subplots(2, 4, figsize=(17, 7))
    cols = {0: "#2a78d6", 1: "#e8743b"}
    labs = {0: "one core", 1: "two cores"}
    for row, mask_sel in enumerate((np.ones(len(y), bool), matched)):
        for j, ax in enumerate(axes[row]):
            v = np.log10(S[:, j]) if logs[j] else S[:, j]
            bins = np.linspace(v.min(), v.max(), 40)
            for c in (0, 1):
                ax.hist(v[(y == c) & mask_sel], bins=bins, color=cols[c], alpha=0.5,
                        density=True, label=labs[c])
            ax.set_xlabel(("log10 " if logs[j] else "") + names[j])
            ax.set_yticks([])
        axes[row, 0].set_ylabel("pool (valid)" if row == 0 else "matched set")
    axes[0, 0].legend(frameon=False)
    fig.suptitle("Task C design statistics before (top) and after (bottom) propensity matching "
                 "(descriptor-free)")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Analysis phase (pre-registered)
# ---------------------------------------------------------------------------
def _ect_rolled(w):
    return ect_features(roll_quiet_to_edge(w))


def load_matched(cfg):
    design_json = Path(cfg.get("out_dir", "results")) / "matched_core_count_design.json"
    d_res = json.loads(design_json.read_text())
    if not d_res["gate"]["passed"]:
        raise SystemExit("balance gate failed: no analysis (docs/MATCHED_CORE_COUNT.md)")
    path = Path(cfg["core_count"].get("cache_dir", "data/core_count")) / \
        f"matched_{_pool_key(cfg)}.npz"
    d = np.load(path)
    digest = hashlib.sha256(d["pool_index"].astype(np.int64).tobytes()).hexdigest()[:16]
    if digest != d_res["matched_set_sha256_16"]:
        raise SystemExit("matched set does not match the design record")
    return {k: d[k] for k in d.files}, d_res


def run(cfg, use_dect=True):
    t_start = time.time()
    es = cfg["ect_study"]
    workers = int(cfg.get("n_workers", 4))
    ev = dict(cfg["evaluation"])
    ev.setdefault("seed", cfg.get("seed", 0))
    ev["n_repeats"] = es.get("n_repeats", ev["n_repeats"])
    n_boot = int(ev.get("bootstrap", 2000))
    dect_cfg = es.get("dect", {})
    if use_dect and dect_cfg.get("enabled", False):
        from ..descriptor.dect import HAS_TORCH
        if not HAS_TORCH:
            print("[dect] torch not installed: DECT view skipped (pip install torch)")
            dect_cfg = None
    else:
        dect_cfg = None
    data, d_res = load_matched(cfg)
    fields = data["fields"].astype(np.float64)
    y, g = data["label"].astype(int), data["group"].astype(int)
    print(f"[analysis] matched set {d_res['matched_set_sha256_16']}: {len(y)} fields, "
          f"{len(np.unique(g))} trajectories", flush=True)
    F, M = task_features(fields, cfg, workers, dect_cfg is not None)
    with Pool(workers) as pool:
        F["ect_boundary_aware"] = np.vstack(pool.map(_ect_rolled, list(fields), chunksize=4))
    F["design_stats"] = np.asarray(data["S"])
    task = evaluate_task(TASK, F, M, y, g, cfg, ev, dect_cfg,
                         sets=TOPO_C + BASE + INCR + SANITY)
    task["margins"] = margins(task, y, g, n_boot, ev["seed"], topo=TOPO_C)
    task["class_mean_ect"] = [F["ect"][y == c].mean(0).tolist() for c in (0, 1)]
    task["class_std_ect"] = [F["ect"][y == c].std(0).tolist() for c in (0, 1)]
    tasks = {TASK: task}
    verdict = decide(tasks, float(es["rule"]["min_margin"]), primary={TASK: PRIMARY_C},
                     topo=TOPO_C)
    res = {"study": "Task C: matched-scale core count (one vortex core vs two)",
           "config": cfg, "labels": {**LABELS, **EXTRA_LABELS}, "design": d_res,
           "dect_enabled": dect_cfg is not None, "tasks": tasks, "verdict": verdict,
           "seconds": round(time.time() - t_start, 1)}
    out_dir = Path(cfg.get("out_dir", "results"))
    (out_dir / "matched_core_count.json").write_text(json.dumps(_strip_oof(res), indent=2))
    from .ect_report import accuracy_figure, ect_figure, write_report
    write_report(res, out_dir / "matched_core_count.md",
                 title="Matched core-count study (generated tables)",
                 doc="docs/MATCHED_CORE_COUNT.md")
    if cfg.get("figures", True):
        fig_dir = out_dir / "figures"
        fig_dir.mkdir(parents=True, exist_ok=True)
        accuracy_figure(res, TASK, fig_dir / "core_count_accuracy.png")
        ect_figure(res, fig_dir / "core_count_ect_images.png")
        examples_figure(data, fig_dir / "core_count_examples.png")
    v = verdict[TASK]
    print(f"=== {TASK}: {v['verdict']} (best primary view `{v['best_primary_view']}`, "
          f"margin {v['best_primary_margin']:+.3f}, headroom {v['headroom']:.3f}) ===")
    return res


EXTRA_LABELS = {"ect_boundary_aware": "ECT, boundary-aware (periodic roll)",
                "design_stats": "matching statistics (sanity)"}


def examples_figure(data, path, n=4):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    y, pid = data["label"], data["pair_id"]
    rng = np.random.default_rng(0)
    ids = np.unique(pid[pid >= 0])
    n = min(n, len(ids))
    pairs = rng.choice(ids, size=n, replace=False)
    fig, axes = plt.subplots(2, n, figsize=(4 * n, 8.2), squeeze=False)
    for j, p in enumerate(pairs):
        for row, c in enumerate((0, 1)):
            i = np.flatnonzero((pid == p) & (y == c))[0]
            w = data["fields"][i]
            lim = np.abs(w).max()
            ax = axes[row, j]
            ax.imshow(w.T, origin="lower", cmap="RdBu_r", vmin=-lim, vmax=lim)
            s = data["S"][i]
            ax.set_title(f"{'one core' if c == 0 else 'two cores'}: k_c {s[0]:.1f}, "
                         f"area {s[2]:.3f}, peak {s[3]:.1f}", fontsize=9, loc="left")
            ax.set_xticks([])
            ax.set_yticks([])
    fig.suptitle("Task C: matched pairs (columns) - one core (top) vs two cores (bottom), "
                 "with matched spectral centroid, occupied area and standardized peak")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/matched_core_count.yaml")
    ap.add_argument("--design", action="store_true", help="outcome-blind design phase only")
    ap.add_argument("--no-dect", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.design:
        design(cfg, int(cfg.get("n_workers", 4)))
    else:
        run(cfg, use_dect=not args.no_dect)


if __name__ == "__main__":
    main()
