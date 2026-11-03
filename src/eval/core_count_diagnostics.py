"""Exploratory, post-hoc diagnostics for task C (NOT part of the verdict).

Run after the pre-registered analysis, on the same frozen matched set:

    python -m src.eval.core_count_diagnostics --config config/matched_core_count.yaml

The verdict on task C is NO-GO because `baselines_union` still reaches 0.977
after matching. This script asks which non-topological cue that is.

1. Drop-one-family ablation of `baselines_union`. The union is rebuilt
   without, in turn:
   - the one-point statistics (moments, histogram quantiles);
   - the spectrum (spectral slope, power spectrum);
   - the structure statistics (Okubo-Weiss, gradient statistics, increment
     flatness).
2. Accuracy by snapshot time: out-of-fold accuracy of `baselines_union`,
   `x2_ph` and `ph_counts` at each of the 8 snapshot times (t = 1.0 ... 4.5).
   It tests whether the residual cue grows as the cores stir their
   surroundings.

Same grouped CV as the analysis (StratifiedGroupKFold, 4 folds x 3 repeats,
group = trajectory, best of LogReg / RF). Writes
results/matched_core_count_diagnostics.json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .classify import evaluate_grouped
from .ect_benchmark import load_config, task_features
from .matched_core_count import load_matched

FAMILIES = {"one_point": ["moments", "histogram_quantiles"],
            "spectrum": ["spectral_slope", "power_spectrum"],
            "structure": ["okubo_weiss", "gradient_stats", "increment_flatness"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/matched_core_count.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    workers = int(cfg.get("n_workers", 4))
    ev = dict(cfg["evaluation"])
    seed = int(ev.get("seed", cfg.get("seed", 0)))
    n_rep = int(cfg["ect_study"].get("n_repeats", ev["n_repeats"]))
    data, d_res = load_matched(cfg)
    fields = data["fields"].astype(np.float64)
    y, g, snap = data["label"].astype(int), data["group"].astype(int), data["snap"]
    from .hard_benchmark import compute_features
    dec = cfg.get("feature_decimals", 8)
    auth = {k: cfg["descriptor"].get(k, {}) for k in ("x1", "x2")}
    F = compute_features(fields, cfg["descriptor"]["topo_downsample"], workers, dec,
                         fullres=False, rich=False, authentic=auth)
    Fx, _ = task_features(fields, cfg, workers, False)
    F["ph_counts"] = Fx["ph_counts"]
    kw = dict(seed=seed, n_splits=ev["n_splits"], n_repeats=n_rep)

    def best(X):
        r = evaluate_grouped(X, y, g, **kw)
        k = r["best_classifier"]
        return r, k
    res = {"note": "exploratory, post hoc; not part of the pre-registered verdict",
           "matched_set": d_res["matched_set_sha256_16"], "ablation": {}, "by_time": {}}
    all_fams = sum(FAMILIES.values(), [])
    full, kf = best(np.hstack([F[k] for k in all_fams]))
    res["ablation"]["baselines_union"] = {"accuracy": full[kf]["accuracy"], "classifier": kf}
    for fam, members in FAMILIES.items():
        keep = [k for k in all_fams if k not in members]
        r, k = best(np.hstack([F[m] for m in keep]))
        res["ablation"][f"without_{fam}"] = {"accuracy": r[k]["accuracy"], "classifier": k}
        r, k = best(np.hstack([F[m] for m in members]))
        res["ablation"][f"only_{fam}"] = {"accuracy": r[k]["accuracy"], "classifier": k}
    print(json.dumps(res["ablation"], indent=1), flush=True)
    for name, X in (("baselines_union", np.hstack([F[k] for k in all_fams])),
                    ("x2_ph", F["x2_ph"]), ("ph_counts", F["ph_counts"])):
        r, k = best(X)
        correct = np.mean([np.asarray(o) == y for o in r[k]["oof"]], axis=0)
        res["by_time"][name] = {"classifier": k,
                                "accuracy_by_snapshot": [float(correct[snap == s].mean())
                                                         for s in range(int(snap.max()) + 1)],
                                "n_by_snapshot": np.bincount(snap).tolist()}
    print(json.dumps(res["by_time"], indent=1), flush=True)
    out = Path(cfg.get("out_dir", "results"))
    (out / "matched_core_count_diagnostics.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
