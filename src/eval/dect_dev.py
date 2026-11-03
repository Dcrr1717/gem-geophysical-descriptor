"""DECT head development on task B (development data only; never on task C).

Task B (merged vs separate vortex pairs) showed that the original DECT head
(global average pooling) under-fits: training accuracy is about 0.78 while the
exact ECT with LogReg fits the training set perfectly
(results/ect_vortex_diagnostics.json). This script compares the original head
("gap") with a head that keeps the threshold position ("flat") on two
task-B splits (repeat 0, folds 0 and 1). It uses the pre-registered training
budget of the ECT study and records train and test accuracy after every
epoch. The head chosen here is fixed in config/matched_core_count.yaml before
any task-C view is computed.

Usage: python -m src.eval.dect_dev --config config/ect_vortex.yaml
"""
from __future__ import annotations

import argparse
import json
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from ..descriptor.ect import level_masks
from .ect_benchmark import generate_vortex_events, load_config


def curves(masks, y, tr, te, hp, seed):
    from ..descriptor.dect import _fit_predict, _predict
    c = {"train_acc": [], "test_acc": []}

    def rec(ep, model):
        c["train_acc"].append(float((_predict(model, masks, tr) == y[tr]).mean()))
        c["test_acc"].append(float((_predict(model, masks, te) == y[te]).mean()))
    n = int(round((np.sqrt(masks.shape[2]) - 1) / 2))
    _fit_predict(masks, y, tr, te, n, 2, hp, seed=seed, on_epoch=rec)
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/ect_vortex.yaml")
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    import torch
    from sklearn.model_selection import StratifiedGroupKFold
    torch.set_num_threads(args.threads)
    cfg = load_config(args.config)
    workers = int(cfg.get("n_workers", 4))
    vb = generate_vortex_events(cfg, workers)
    y, g = vb["label"].astype(int), vb["group"].astype(int)
    with Pool(workers) as pool:
        masks = np.stack(pool.map(level_masks, list(vb["fields"])))
    ev = cfg["evaluation"]
    seed = int(cfg.get("seed", 0))
    cv = StratifiedGroupKFold(n_splits=ev["n_splits"], shuffle=True, random_state=seed)
    splits = list(cv.split(np.zeros(len(y)), y, g))[:2]
    hp0 = dict(cfg["ect_study"]["dect"])
    res = {"task": "vortex_merger (task B, development data)", "hp": hp0, "heads": {}}
    for head in ("gap", "flat"):
        hp = dict(hp0, head=head)
        res["heads"][head] = []
        for f, (tr, te) in enumerate(splits):
            t0 = time.time()
            c = curves(masks, y, tr, te, hp, seed=1000 * seed + f)
            res["heads"][head].append({"fold": f, **c})
            print(f"[dect dev] head {head} fold {f}: final train {c['train_acc'][-1]:.3f}, "
                  f"test {c['test_acc'][-1]:.3f} [{time.time() - t0:.0f}s]", flush=True)
    out = Path(cfg.get("out_dir", "results"))
    (out / "dect_head_dev.json").write_text(json.dumps(res, indent=2))
    figure(res, out / "figures" / "dect_head_dev.png")


def figure(res, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    cols = {"gap": "#8a8a8a", "flat": "#2a78d6"}
    for ax, f in zip(axes, (0, 1)):
        for head, runs in res["heads"].items():
            c = runs[f]
            ep = np.arange(1, len(c["train_acc"]) + 1)
            ax.plot(ep, c["train_acc"], color=cols[head], lw=2, label=f"{head} train")
            ax.plot(ep, c["test_acc"], color=cols[head], lw=2, ls="--", label=f"{head} test")
        ax.axhline(0.5, color="#bbbbbb", lw=1, ls=":")
        ax.set_ylim(0.4, 1.02)
        ax.set_xlabel("epoch")
        ax.set_title(f"task B (development), split {f}", loc="left", fontsize=11)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("accuracy")
    axes[0].legend(frameon=False, fontsize=9, loc="lower right")
    fig.suptitle("DECT head development: global average pooling (gap) vs flatten (flat)")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    main()
