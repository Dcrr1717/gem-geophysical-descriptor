"""Differentiable Euler Characteristic Transform (DECT) - optional, needs torch.

Roell & Rieck (2023), "Differentiable Euler Characteristic Transforms for
Shape Classification", arXiv:2310.07630. The ECT indicator [height_v(c) <= t]
is replaced by a sigmoid sigma(lambda * (t - height_v(c))), so the transform
is differentiable in the directions v, the thresholds t and the sharpness
lambda. These are learned end to end with a small CNN head that classifies
the resulting ECT image.

Here the shapes are the same cubical sublevel-set complexes as the exact ECT
(src/descriptor/ect.py): fixed field levels ECT_LEVELS, field block-averaged
to ECT_RES^2. Because cell coordinates are shared by every field, one forward
pass is

    DECT[b, level, dir, t] = masks[b, level, :] @ (sign * sigmoid(lambda (t - H(theta))))

with `masks` the binary cell-membership masks (precomputed, no gradient) and
H(theta) the lower-star cell heights (max over a cell's vertices of <x, v>),
differentiable in the learnable angles theta.

`train_eval_grouped` runs the same repeated StratifiedGroupKFold splits as
src/eval/classify.evaluate_grouped and returns the same result structure
(accuracy, out-of-fold predictions), so the paired group-bootstrap margins
of the hard benchmark apply unchanged.
"""
from __future__ import annotations

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except Exception:          # torch is optional
    HAS_TORCH = False

from .ect import ECT_LEVELS, T_MAX, grid_cells


if HAS_TORCH:
    class DECTLayer(nn.Module):
        """Learnable-direction / learnable-threshold ECT of cubical sublevel sets."""

        def __init__(self, n, n_dirs=16, n_t=16, sharpness=40.0):
            super().__init__()
            corners, sign = grid_cells(n)
            self.register_buffer("corners", torch.from_numpy(corners))       # (C, 4, 2)
            self.register_buffer("sign", torch.from_numpy(sign))             # (C,)
            self.theta = nn.Parameter(torch.linspace(0, 2 * np.pi, n_dirs + 1)[:-1].clone())
            self.t = nn.Parameter(torch.linspace(-T_MAX, T_MAX, n_t))
            self.log_lam = nn.Parameter(torch.tensor(float(np.log(sharpness))))

        def heights(self):
            v = torch.stack([torch.cos(self.theta), torch.sin(self.theta)], dim=0)   # (2, D)
            return (self.corners @ v).amax(dim=1)                                    # (C, D)

        def forward(self, masks):
            """masks: (B, L, C) float in {0, 1} -> (B, L, D, T)."""
            B, L, C = masks.shape
            H = self.heights()
            S = torch.sigmoid(self.log_lam.exp() * (self.t[None, None, :] - H[:, :, None]))
            S = (S * self.sign[:, None, None]).reshape(C, -1)                       # (C, D*T)
            out = masks.reshape(B * L, C) @ S
            return out.reshape(B, L, len(self.theta), len(self.t))

    class DECTClassifier(nn.Module):
        """DECT layer + small CNN head on the (levels, directions, thresholds) image."""

        def __init__(self, n, n_levels, n_classes, n_dirs=16, n_t=16, sharpness=40.0):
            super().__init__()
            self.dect = DECTLayer(n, n_dirs, n_t, sharpness)
            self.norm = nn.BatchNorm2d(n_levels)
            self.conv1 = nn.Conv2d(n_levels, 16, 3, padding=1)
            self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
            self.head = nn.Linear(32, n_classes)

        def forward(self, masks):
            x = self.norm(self.dect(masks))
            # directions are circular: wrap-pad that axis, zero-pad the threshold axis
            x = F.pad(x, (0, 0, 1, 1), mode="circular")
            x = F.relu(self.conv1(F.pad(x, (1, 1, 0, 0))))
            x = F.max_pool2d(x, 2)
            x = F.pad(x, (0, 0, 1, 1), mode="circular")
            x = F.relu(self.conv2(F.pad(x, (1, 1, 0, 0))))
            return self.head(x.mean(dim=(2, 3)))


def _predict(model, masks, idx, chunk=64):
    """Class predictions of `model` (eval mode) for masks[idx]."""
    with torch.no_grad():
        X = torch.from_numpy(masks[idx])
        return torch.cat([model(X[i:i + chunk].float()).argmax(1)
                          for i in range(0, len(idx), chunk)]).numpy()


def _fit_predict(masks, y, tr, te, n, n_classes, hp, seed, on_epoch=None):
    """Train one DECT classifier on `tr`, predict `te`. `on_epoch(epoch, model)`,
    if given, is called after every epoch with the model in eval mode (it does
    not change the training trajectory)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = DECTClassifier(n, masks.shape[1], n_classes, hp["n_dirs"], hp["n_t"],
                           hp["sharpness"])
    opt = torch.optim.Adam(model.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"])
    Xtr = torch.from_numpy(masks[tr])
    ytr = torch.from_numpy(np.asarray(y)[tr]).long()
    g = torch.Generator().manual_seed(seed)
    bs = int(hp["batch_size"])
    model.train()
    for ep in range(int(hp["epochs"])):
        perm = torch.randperm(len(tr), generator=g)
        for i in range(0, len(tr), bs):
            idx = perm[i:i + bs]
            if len(idx) < 2:            # BatchNorm needs > 1 sample
                continue
            opt.zero_grad()
            loss = F.cross_entropy(model(Xtr[idx].float()), ytr[idx])
            loss.backward()
            opt.step()
        if on_epoch is not None:
            model.eval()
            on_epoch(ep, model)
            model.train()
    model.eval()
    return _predict(model, masks, te), model


def train_eval_grouped(masks, y, groups, n, hp, seed=0, n_splits=4, n_repeats=5,
                       verbose=False):
    """Repeated StratifiedGroupKFold training/evaluation of the DECT classifier.

    `masks`: (N, L, C) uint8 cell masks (src/descriptor/ect.level_masks).
    Returns a dict shaped like src/eval/classify.evaluate_grouped's output,
    with a single "dect" classifier entry (plus the learned directions of the
    last fold, for inspection)."""
    if not HAS_TORCH:
        raise RuntimeError("DECT needs torch (pip install torch); disable it with dect.enabled: false")
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.model_selection import StratifiedGroupKFold
    torch.set_num_threads(max(1, torch.get_num_threads()))
    y = np.asarray(y)
    groups = np.asarray(groups)
    n_classes = len(np.unique(y))
    fold_acc, rep_acc, rep_f1, oof_all, thetas = [], [], [], [], []
    for r in range(n_repeats):
        cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed + r)
        oof = np.empty_like(y)
        for f, (tr, te) in enumerate(cv.split(np.zeros(len(y)), y, groups)):
            pred, model = _fit_predict(masks, y, tr, te, n, n_classes, hp,
                                       seed=1000 * (seed + r) + f)
            oof[te] = pred
            fold_acc.append(accuracy_score(y[te], pred))
            thetas.append(np.sort(np.mod(model.dect.theta.detach().numpy(), 2 * np.pi)).tolist())
            if verbose:
                print(f"    [dect] repeat {r} fold {f}: acc {fold_acc[-1]:.3f}", flush=True)
        rep_acc.append(accuracy_score(y, oof))
        rep_f1.append(f1_score(y, oof, average="macro"))
        oof_all.append(oof)
    res = {"dect": {"accuracy": float(np.mean(rep_acc)), "accuracy_std": float(np.std(fold_acc)),
                    "accuracy_std_repeats": float(np.std(rep_acc)),
                    "f1_macro": float(np.mean(rep_f1)),
                    "fold_accuracies": [float(a) for a in fold_acc],
                    "oof": [o.tolist() for o in oof_all]},
           "learned_directions_last_fold": thetas[-1], "silhouette": float("nan")}
    res["best_classifier"] = "dect"
    res["best_accuracy"] = res["dect"]["accuracy"]
    return res


DECT_LEVELS = ECT_LEVELS
