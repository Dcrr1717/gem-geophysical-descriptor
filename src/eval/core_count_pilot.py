"""Descriptor-free pilot for the matched core-count task (Task C).

Simulates a small pool of one-core and two-core trajectories with the
`core_count` config, overridden by its `pilot` block. For every snapshot it
computes only:
- the core count (validity check), and
- the four design statistics (src/sims/core_count.design_stats).

It then reports how far apart the classes are before matching, and the
balance and number of pairs after one-to-one matching. No descriptor view and
no classifier is computed. The pilot only checks that the parameter ranges
let the classes be matched; the ranges are then fixed in the config.

Usage: python -m src.eval.core_count_pilot --config config/matched_core_count.yaml
"""
from __future__ import annotations

import argparse
import json
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from ..sims.core_count import (DESIGN_STATS, balance, balance_gate, core_jobs, design_stats,
                               match_mahalanobis, match_propensity, simulate_cores)
from ..sims.vortex_events import core_count
from .ect_benchmark import load_config
from .hard_benchmark import _standardize


def _worker(args):
    job, cc = args
    job, snaps = simulate_cores(job, cc)
    fields = [_standardize(w) for w in snaps]
    return job, [design_stats(w) for w in fields], [core_count(w) for w in fields]


def run_pool(cc, seed, workers):
    jobs = core_jobs(cc, seed)
    with Pool(workers) as pool:
        out = pool.map(_worker, [(j, cc) for j in jobs])
    S, y, g, cnt = [], [], [], []
    for gi, (job, stats, counts) in enumerate(out):
        for s, c in zip(stats, counts):
            S.append(s)
            y.append(job["label"])
            g.append(gi)
            cnt.append(c)
    return np.array(S), np.array(y), np.array(g), np.array(cnt)


def _match(S, y, mcfg):
    if mcfg.get("method", "mahalanobis") == "propensity":
        return match_propensity(S, y, float(mcfg["caliper_sd"]), int(mcfg.get("degree", 2)))
    return match_mahalanobis(S, y, float(mcfg["caliper"]))


def summarize(S, y, g, cnt, mcfg):
    valid = cnt == np.where(y == 0, 1, 2)
    Sv, yv, gv = S[valid], y[valid], g[valid]
    a, b, d = _match(Sv, yv, mcfg)
    idx = np.r_[a, b]
    per_class = {}
    for c in (0, 1):
        m = yv == c
        per_class[str(c)] = {n: [float(np.percentile(Sv[m, j], q)) for q in (5, 50, 95)]
                             for j, n in enumerate(DESIGN_STATS)}
    return {"n_snapshots": int(len(y)),
            "valid_fraction_by_class": [float(valid[y == c].mean()) for c in (0, 1)],
            "design_stats_5_50_95_by_class": per_class,
            "balance_before": balance(Sv, yv),
            "n_pairs_matched": int(len(a)),
            "balance_after": balance(Sv[idx], yv[idx]) if len(a) > 2 else None,
            "gate_after": (balance_gate(Sv[idx], yv[idx], gv[idx], mcfg["gate"])
                           if len(a) > 10 and "gate" in mcfg else None)}


def figure(S, y, cnt, mcfg, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    valid = cnt == np.where(y == 0, 1, 2)
    Sv, yv = S[valid], y[valid]
    a, b, _ = _match(Sv, yv, mcfg)
    names = ["spectral centroid", "spectral width", "occupied area", "standardized peak"]
    pairs = [(0, 2), (0, 1), (2, 3)]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    for ax, (i, j) in zip(axes, pairs):
        for c, col, lab in ((0, "#2a78d6", "one core"), (1, "#e8743b", "two cores")):
            m = yv == c
            ax.scatter(Sv[m, i], Sv[m, j], s=8, color=col, alpha=0.25, label=f"{lab} (pool)")
            sel = a if c == 0 else b
            ax.scatter(Sv[sel, i], Sv[sel, j], s=12, color=col, edgecolor="k", linewidth=0.3,
                       label=f"{lab} (matched)")
        ax.set_xlabel(names[i])
        ax.set_ylabel(names[j])
        if j == 2 or i == 2:
            ax.set_yscale("log") if j == 2 else ax.set_xscale("log")
        ax.grid(alpha=0.3)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Core-count pilot: design statistics before and after one-to-one matching "
                 "(descriptor-free)")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/matched_core_count.yaml")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    cfg = load_config(args.config)
    cc = dict(cfg["core_count"])
    cc.update(cfg.get("core_count_pilot", {}))
    t0 = time.time()
    pool_path = Path(cc.get("cache_dir", "data/core_count")) / f"pilot_pool{args.tag}.npz"
    if pool_path.exists():
        d = np.load(pool_path)
        S, y, g, cnt = d["S"], d["y"], d["g"], d["cnt"]
    else:
        S, y, g, cnt = run_pool(cc, cfg.get("seed", 0) + 99, int(cfg.get("n_workers", 4)))
        pool_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(pool_path, S=S, y=y, g=g, cnt=cnt)
    mcfg = cfg["matching"]
    res = {"params": cc, "matching": mcfg, **summarize(S, y, g, cnt, mcfg),
           "seconds": round(time.time() - t0, 1)}
    out = Path(cfg.get("out_dir", "results"))
    stem = f"core_count_pilot{args.tag}"
    (out / f"{stem}.json").write_text(json.dumps(res, indent=2))
    figure(S, y, cnt, mcfg, out / "figures" / f"{stem}.png")
    print(json.dumps({k: v for k, v in res.items() if k != "params"}, indent=1))


if __name__ == "__main__":
    main()
