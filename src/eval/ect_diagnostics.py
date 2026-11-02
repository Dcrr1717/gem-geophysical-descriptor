"""Exploratory, post-hoc diagnostics for the ECT / DECT study (NOT part of the verdict).

Run after src/eval/ect_benchmark.py, on the same cached fields:

    python -m src.eval.ect_diagnostics --config config/ect_vortex.yaml

1. Boundary cuts (task B). The ECT is computed on the planar square [-1, 1]^2,
   while the flow is periodic: a vortex core that crosses the domain edge is
   cut open, so it is no longer a hole of the sublevel set {f <= h}. For every
   field this records whether a core region {f > 1.5} touches the edge (a
   descriptor-free mask, never a feature). It then splits the out-of-fold
   errors of the ECT view by that flag, on the benchmark's own grouped CV
   splits.
2. Boundary-aware ECT (task B). The same ECT after a periodic roll that puts
   the quietest row and column (smallest max |f|) on the domain edges. This
   is a post-hoc variant: it is reported to locate the ECT shortfall, not to
   change the verdict.
3. DECT learning curves (both tasks). The pre-registered DECT is trained on
   the first split (repeat 0, fold 0) with train and test accuracy recorded
   after each epoch. The exact ECT with LogReg / RF is scored on the same
   split for reference.

Writes results/ect_diagnostics.json and results/figures/dect_learning_curves.png.
"""
from __future__ import annotations

import argparse
import json
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from ..descriptor.ect import ect_features, level_masks
from .classify import evaluate_grouped
from .ect_benchmark import generate_vortex_events, load_config

CORE_LEVEL = 1.5            # = the top ECT level (field-std units)


def touches_edge(w, level=CORE_LEVEL):
    m = w > level
    return bool(m[0].any() or m[-1].any() or m[:, 0].any() or m[:, -1].any())


def roll_quiet_to_edge(w):
    """Periodic roll that moves the row / column with the smallest max |w| to index 0."""
    a = np.abs(w)
    return np.roll(np.roll(w, -int(a.max(1).argmin()), axis=0), -int(a.max(0).argmin()), axis=1)


def _ect_pair(w):
    return ect_features(w), ect_features(roll_quiet_to_edge(w))


def _best(res):
    k = res["best_classifier"]
    return k, res[k]


def boundary_diagnostic(vb, ev, workers):
    F, y, g = vb["fields"], vb["label"].astype(int), vb["group"].astype(int)
    touch = np.array([touches_edge(w) for w in F])
    with Pool(workers) as pool:
        pairs = pool.map(_ect_pair, list(F))
    E = np.stack([p[0] for p in pairs])
    E_roll = np.stack([p[1] for p in pairs])
    touch_roll = np.array([touches_edge(roll_quiet_to_edge(w)) for w in F])
    kw = dict(seed=ev["seed"], n_splits=ev["n_splits"], n_repeats=ev["n_repeats"])
    out = {"core_level": CORE_LEVEL}
    for name, X, t in (("ect", E, touch), ("ect_boundary_aware", E_roll, touch_roll)):
        res = evaluate_grouped(X, y, g, **kw)
        clf, r = _best(res)
        err = np.mean([np.asarray(o) != y for o in r["oof"]], axis=0)     # per-field error rate
        out[name] = {
            "accuracy": r["accuracy"], "classifier": clf,
            "fraction_fields_core_on_edge": float(t.mean()),
            "fraction_on_edge_by_class": [float(t[y == c].mean()) for c in (0, 1)],
            "error_rate_core_on_edge": float(err[t].mean()) if t.any() else None,
            "error_rate_core_interior": float(err[~t].mean()) if (~t).any() else None,
            "n_on_edge": int(t.sum()), "n_interior": int((~t).sum())}
    return out


def dect_curves(name, masks, F_ect, y, g, ev, hp):
    import torch
    from sklearn.model_selection import StratifiedGroupKFold
    from ..descriptor.dect import _fit_predict, _predict
    from .classify import _classifiers
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    cv = StratifiedGroupKFold(n_splits=ev["n_splits"], shuffle=True, random_state=ev["seed"])
    tr, te = next(cv.split(np.zeros(len(y)), y, g))
    curve = {"epoch": [], "train_acc": [], "test_acc": [], "sharpness": []}

    def rec(ep, model):
        curve["epoch"].append(ep + 1)
        curve["train_acc"].append(float((_predict(model, masks, tr) == y[tr]).mean()))
        curve["test_acc"].append(float((_predict(model, masks, te) == y[te]).mean()))
        curve["sharpness"].append(float(model.dect.log_lam.detach().exp()))
    n = int(round((np.sqrt(masks.shape[2]) - 1) / 2))
    t0 = time.time()
    _fit_predict(masks, y, tr, te, n, len(np.unique(y)), hp, seed=1000 * ev["seed"], on_epoch=rec)
    ref = {}
    for clf_name, clf in _classifiers(ev["seed"]):
        pipe = make_pipeline(StandardScaler(), clf).fit(F_ect[tr], y[tr])
        ref[clf_name] = {"train_acc": float(pipe.score(F_ect[tr], y[tr])),
                         "test_acc": float(pipe.score(F_ect[te], y[te]))}
    print(f"[dect curve] {name}: final train {curve['train_acc'][-1]:.3f}, test "
          f"{curve['test_acc'][-1]:.3f} (ECT LR test {ref['logreg']['test_acc']:.3f}, RF "
          f"{ref['rf']['test_acc']:.3f}) [{time.time() - t0:.0f}s, torch threads "
          f"{torch.get_num_threads()}]", flush=True)
    return {"split": "repeat 0, fold 0", "n_train": int(len(tr)), "n_test": int(len(te)),
            "dect": curve, "ect_same_split": ref}


def curves_figure(curves, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    titles = {"iaaft_binary": "Task A: coherent vs IAAFT",
              "vortex_merger": "Task B: merged vs separate pair"}
    fig, axes = plt.subplots(1, len(curves), figsize=(6 * len(curves), 4), squeeze=False)
    for ax, (name, c) in zip(axes[0], curves.items()):
        d = c["dect"]
        ax.plot(d["epoch"], d["train_acc"], color="#2a78d6", lw=2, label="DECT train")
        ax.plot(d["epoch"], d["test_acc"], color="#2a78d6", lw=2, ls="--", label="DECT test")
        best = max(c["ect_same_split"].values(), key=lambda r: r["test_acc"])["test_acc"]
        ax.axhline(best, color="#008300", lw=1.5, label="exact ECT + LR/RF, test")
        ax.axhline(0.5, color="#888888", lw=1, ls=":")
        ax.set_ylim(0.4, 1.02)
        ax.set_xlabel("epoch")
        ax.set_ylabel("accuracy")
        ax.set_title(f"{titles.get(name, name)} (one split)", loc="left", fontsize=11)
        ax.grid(alpha=0.3)
        ax.legend(frameon=False, fontsize=9, loc="lower right")
    fig.suptitle("DECT learning curves (pre-registered hyperparameters; exploratory)")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/ect_vortex.yaml")
    ap.add_argument("--no-dect", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    from .hard_benchmark import generate_coherent, generate_surrogates
    es = cfg["ect_study"]
    workers = int(cfg.get("n_workers", 4))
    ev = dict(cfg["evaluation"])
    ev.setdefault("seed", cfg.get("seed", 0))
    ev["n_repeats"] = es.get("n_repeats", ev["n_repeats"])
    out_dir = Path(cfg.get("out_dir", "results"))

    vb = generate_vortex_events(cfg, workers)
    res = {"note": "exploratory, post hoc; not part of the pre-registered verdict",
           "boundary_task_b": boundary_diagnostic(vb, ev, workers)}
    print("[boundary]", json.dumps(res["boundary_task_b"], indent=1), flush=True)

    dect = es.get("dect", {})
    if dect.get("enabled", False) and not args.no_dect:
        from ..descriptor.dect import HAS_TORCH
        if HAS_TORCH:
            cache = cfg.get("cache_dir", "data/hard_benchmark")
            coh, _ = generate_coherent(cfg, cache, workers)
            sur = generate_surrogates(coh, cfg, cfg["surrogate"]["final"], cache, workers)
            N = len(coh["fields"])
            tasks = {"iaaft_binary": (np.concatenate([coh["fields"], sur["fields"]]),
                                      np.r_[np.zeros(N, int), np.ones(N, int)],
                                      np.r_[coh["group"], coh["group"]]),
                     "vortex_merger": (vb["fields"], vb["label"].astype(int),
                                       vb["group"].astype(int))}
            curves = {}
            for name, (F, y, g) in tasks.items():
                with Pool(workers) as pool:
                    masks = np.stack(pool.map(level_masks, list(F)))
                    E = np.stack(pool.map(ect_features, list(F)))
                curves[name] = dect_curves(name, masks, E, y, g, ev, dect)
            res["dect_learning_curves"] = curves
            curves_figure(curves, out_dir / "figures" / "dect_learning_curves.png")
    stem = es.get("report_stem", "ect_vortex")
    (out_dir / f"{stem}_diagnostics.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
