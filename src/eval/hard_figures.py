"""Figures for the hard structure-only benchmark (static PNGs for the paper)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from ..baselines.structure import radial_power_spectrum  # noqa: E402
from ..descriptor.topology import topology_features  # noqa: E402

# Validated categorical palette (fixed order) + chrome.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
TIER_COLORS = {
    "ours": "#2a78d6", "ours-ablation": "#2a78d6",
    "baseline-specified": "#eb6834",
    "baseline-validity": "#1baf7a",
    "baseline-structure": "#eda100",
    "baseline-union": "#e87ba4", "incremental": "#e87ba4",
    "ours-rich": "#008300",
    "sensitivity": "#c3c2b7",
    "graph-laplacian": "#2a78d6",
    "authentic": "#6250d6",
}
TIER_LABELS = {
    "ours": "descriptor and its views", "baseline-specified": "baselines named in brief",
    "baseline-validity": "leak detectors (spectrum / histogram)",
    "baseline-structure": "structure-aware physical baselines",
    "baseline-union": "all baselines, without / with topology",
    "ours-rich": "richer topological vectorizations",
    "authentic": "authentic X1 / X2 / X1+X2 / X3 (+ richer)",
    "sensitivity": "post-hoc sensitivity (not used for verdict)",
}
COH, SUR = "#2a78d6", "#eb6834"
VORT_CMAP = LinearSegmentedColormap.from_list(
    "vort", ["#104281", "#3987e5", "#cde2fb", "#f0efec", "#f6c7c6", "#e34948", "#8f1f1f"])


def _style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def fields_figure(res, coh, sur, path):
    regimes = res["regimes"]
    fig, axes = plt.subplots(len(regimes), 4, figsize=(13, 3.2 * len(regimes)),
                             facecolor=SURFACE)
    axes = np.atleast_2d(axes)
    for i, rn in enumerate(regimes):
        idx = np.where((coh["regime"] == i) & (coh["run"] == 0))[0][-1]
        w, s = coh["fields"][idx], sur["fields"][idx]
        for j, (f, t) in enumerate(((w, "coherent"), (s, "IAAFT surrogate"))):
            ax = axes[i, j]
            ax.imshow(f.T, origin="lower", cmap=VORT_CMAP, vmin=-3, vmax=3,
                      interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"{rn}: {t}", fontsize=10, color=INK)
        ax = axes[i, 2]
        _style(ax)
        k = np.arange(1, w.shape[0] // 3 + 1)
        ax.loglog(k, radial_power_spectrum(w), color=COH, lw=2, label="coherent")
        ax.loglog(k, radial_power_spectrum(s), color=SUR, lw=0, marker="o", ms=3,
                  markevery=3, label="surrogate")
        ax.set_xlabel("wavenumber k", fontsize=8, color=INK2)
        ax.set_ylabel("power", fontsize=8, color=INK2)
        ax.set_title("radial power spectrum", fontsize=10, color=INK)
        ax = axes[i, 3]
        _style(ax)
        bins = np.linspace(-4, 6, 81)
        ax.hist(w.ravel(), bins=bins, histtype="step", color=COH, lw=2, label="coherent")
        ax.hist(s.ravel(), bins=bins, histtype="step", color=SUR, lw=1.2, label="surrogate")
        ax.set_yscale("log")
        ax.set_xlabel("standardized vorticity", fontsize=8, color=INK2)
        ax.set_title("one-point distribution", fontsize=10, color=INK)
        if i == 0:
            axes[i, 2].legend(frameon=False, fontsize=8)
            axes[i, 3].legend(frameon=False, fontsize=8)
    fig.suptitle("Coherent 2D turbulence vs IAAFT surrogate: same spectrum, same histogram, "
                 "random phases", fontsize=12, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def accuracy_figure(task, feature_sets, path, title):
    R = task["results"]
    names = list(R)[::-1]                       # top-to-bottom = registry order
    acc = [R[n]["best_accuracy"] for n in names]
    err = [R[n][R[n]["best_classifier"]]["accuracy_std"] for n in names]
    tiers = [feature_sets[n]["tier"] for n in names]
    colors = [TIER_COLORS[t] for t in tiers]
    fig, ax = plt.subplots(figsize=(9, 0.38 * len(names) + 1.6), facecolor=SURFACE)
    _style(ax)
    ax.grid(True, axis="x", color=GRID, linewidth=0.8)
    ax.grid(False, axis="y")
    y = np.arange(len(names))
    ax.barh(y, acc, height=0.62, color=colors, edgecolor=SURFACE, linewidth=2)
    ax.errorbar(acc, y, xerr=err, fmt="none", ecolor=INK2, elinewidth=1, capsize=2)
    for yi, a, e in zip(y, acc, err):
        ax.text(min(a + e + 0.012, 1.08), yi, f"{a:.3f}", va="center", fontsize=8, color=INK)
    ax.axvline(task["chance"], color=INK2, lw=1)
    ax.text(task["chance"], len(names) - 0.35, f" chance {task['chance']:.2f}",
            fontsize=8, color=INK2, va="bottom")
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=9, color=INK)
    for lab, n in zip(ax.get_yticklabels(), names):
        if n == "descriptor":
            lab.set_fontweight("bold")
    ax.set_xlim(0, 1.12)
    ax.set_ylim(-0.6, len(names) - 0.1)
    ax.set_xlabel("accuracy (grouped CV, better of LogReg / RF; bar = fold std)",
                  fontsize=9, color=INK2)
    handles = []
    seen = set()
    present = set(tiers)
    for t in ["ours", "baseline-specified", "baseline-validity", "baseline-structure",
              "baseline-union", "ours-rich", "authentic", "sensitivity"]:
        if t in seen or (t in ("ours-rich", "authentic") and t not in present):
            continue
        seen.add(t)
        handles.append(plt.Rectangle((0, 0), 1, 1, color=TIER_COLORS[t], label=TIER_LABELS[t]))
    ax.legend(handles=handles, frameon=False, fontsize=8, loc="upper center",
              bbox_to_anchor=(0.45, -0.09 - 0.6 / len(names)), ncol=2)
    ax.set_title(title, fontsize=11, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


_TOPO_NAMES = ["H0 count", "H0 total persistence", "H0 max persistence", "H0 entropy",
               "H1 count", "H1 total persistence", "H1 max persistence", "H1 entropy"]


def topology_figure(res, coh, sur, path):
    ds = res["config"]["descriptor"]["topo_downsample"]
    Tc = np.vstack([topology_features(w, downsample=ds) for w in coh["fields"]])
    Ts = np.vstack([topology_features(w, downsample=ds) for w in sur["fields"]])
    fig, axes = plt.subplots(2, 4, figsize=(13, 5.4), facecolor=SURFACE)
    for j, ax in enumerate(axes.ravel()):
        _style(ax)
        lo = min(Tc[:, j].min(), Ts[:, j].min())
        hi = max(Tc[:, j].max(), Ts[:, j].max())
        bins = np.linspace(lo, hi + 1e-9, 30)
        ax.hist(Tc[:, j], bins=bins, histtype="step", color=COH, lw=2, label="coherent")
        ax.hist(Ts[:, j], bins=bins, histtype="step", color=SUR, lw=2, label="surrogate")
        ax.set_title(_TOPO_NAMES[j], fontsize=9, color=INK)
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle(f"Topological-view features (sublevel-set PH, {ds}x{ds} image), all forcing "
                 f"scales pooled", fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def rich_margin_figure(entries, path, title, go_margin=0.05):
    """Dot plot: margin of the best richer topological vectorization over
    baselines_union per task (95% CI), the GO threshold, and the largest
    margin any feature set could reach (1 - accuracy of baselines_union)."""
    entries = [(n, v) for n, v in entries if v is not None][::-1]
    fig, ax = plt.subplots(figsize=(9, 0.5 * len(entries) + 1.8), facecolor=SURFACE)
    _style(ax)
    ax.grid(True, axis="x", color=GRID, linewidth=0.8)
    ax.grid(False, axis="y")
    y = np.arange(len(entries))
    labels = []
    for yi, (name, v) in zip(y, entries):
        b = v["best_rich_variant"]
        m = v["variants"][b]["margin_vs_baselines_union"]
        lo, hi = v["variants"][b]["margin_vs_baselines_union_ci95"]
        ax.plot([lo, hi], [yi, yi], color=INK2, lw=1.2, zorder=2)
        ax.scatter([m], [yi], s=64, color="#008300", edgecolor=SURFACE, linewidth=2,
                   zorder=3, label="best richer topology (95% CI)" if yi == 0 else None)
        ax.scatter([v["max_attainable_margin"]], [yi], s=60, marker="|", color=INK, lw=2,
                   zorder=3, label="max attainable margin (1 - union acc)" if yi == 0 else None)
        labels.append(f"{name}  ({b})")
    ax.axvline(0.0, color=AXIS, lw=1)
    ax.axvline(go_margin, color=INK2, lw=1)
    ax.text(go_margin, len(entries) - 0.4, f" GO threshold +{go_margin:.2f}", fontsize=8,
            color=INK2, va="bottom")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8, color=INK)
    lo_all = min(min(v["variants"][v["best_rich_variant"]]["margin_vs_baselines_union_ci95"][0]
                     for _, v in entries), -0.05)
    ax.set_xlim(lo_all - 0.02, 0.12)
    ax.set_ylim(-0.6, len(entries) - 0.1)
    ax.set_xlabel("accuracy margin over baselines_union (grouped CV)", fontsize=9, color=INK2)
    ax.legend(frameon=False, fontsize=8, loc="upper center",
              bbox_to_anchor=(0.45, -0.12 - 0.5 / len(entries)), ncol=2)
    ax.set_title(title, fontsize=11, color=INK, loc="left")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def make_figures(res, coh, sur, out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    fields_figure(res, coh, sur, out / "fields_coherent_vs_iaaft.png")
    accuracy_figure(res["tasks"]["binary_pooled"], res["feature_sets"],
                    out / "accuracy_binary.png",
                    "Coherent vs IAAFT surrogate (binary, all forcing scales)")
    accuracy_figure(res["tasks"]["multiclass6"], res["feature_sets"],
                    out / "accuracy_multiclass.png",
                    "Forcing scale x {coherent, surrogate} (6 classes)")
    topology_figure(res, coh, sur, out / "topology_features.png")
    entries = [(k, t["verdicts"].get("rich_topology")) for k, t in res["tasks"].items()]
    if any(v is not None for _, v in entries):
        rich_margin_figure(entries, out / "rich_topology_margins.png",
                           "Richer topology vs all non-topological baselines (clean fields)",
                           res.get("go_margin", 0.05))
    if res.get("vs_union_summary", {}).get("per_task"):
        margin_figure(res, out / "margin_vs_union.png")
    print(f"[figures] written to {out}")


def margin_figure(res, path):
    """Margin over baselines_union of every non-baseline configuration, per task."""
    us = res["vs_union_summary"]
    tnames = list(us["per_task"])
    T = res["tasks"]
    cfgs = list(T[tnames[0]]["union_verdicts"]["configs"])[::-1]
    gm = res["go_margin"]
    fig, axes = plt.subplots(1, len(tnames), figsize=(2.3 * len(tnames) + 2.6,
                                                       0.30 * len(cfgs) + 1.9),
                             facecolor=SURFACE, sharey=True)
    axes = np.atleast_1d(axes)
    y = np.arange(len(cfgs))
    lo_all = min(min(T[t]["union_verdicts"]["configs"][c]["margin_ci95"][0] for c in cfgs)
                 for t in tnames)
    xmin = max(-0.6, min(-0.06, lo_all - 0.02))
    hi_all = max(max(T[t]["union_verdicts"]["configs"][c]["margin_ci95"][1] for c in cfgs)
                 for t in tnames)
    xmax = max(0.12, hi_all + 0.02)
    for ax, t in zip(axes, tnames):
        _style(ax)
        ax.grid(False, axis="y")
        uv = T[t]["union_verdicts"]
        m = np.array([uv["configs"][c]["margin"] for c in cfgs])
        ci = np.array([uv["configs"][c]["margin_ci95"] for c in cfgs])
        cols = [TIER_COLORS[res["feature_sets"][c]["tier"]] for c in cfgs]
        ax.axvspan(uv["headroom"], xmax, color=GRID, alpha=0.6, lw=0)
        ax.axvline(0, color=INK2, lw=1)
        ax.axvline(gm, color="#e34948", lw=1.2, ls="--")
        ax.errorbar(m, y, xerr=[m - ci[:, 0], ci[:, 1] - m], fmt="none", ecolor=MUTED,
                    elinewidth=1, capsize=0)
        ax.scatter(m, y, c=cols, s=22, zorder=3, edgecolors=SURFACE, linewidths=0.6)
        ax.set_xlim(xmin, xmax)
        ax.set_title(f"{t}\nunion {uv['baselines_union_accuracy']:.3f}", fontsize=8.5,
                     color=INK, loc="left")
        ax.tick_params(axis="x", labelsize=7)
    axes[0].set_yticks(y)
    # display names X1 / X2 / X3 (as in the reference) for readability
    axes[0].set_yticklabels([c.replace("x1x2", "X1+X2").replace("x1_graph", "X1_graph")
                             .replace("x2_ph", "X2_ph").replace("x3_fusion", "X3_fusion")
                             .replace("+x1", "+X1") for c in cfgs], fontsize=8, color=INK)
    fig.supxlabel(f"accuracy(config) - accuracy(baselines_union), 95% CI; dashed = +{gm:.2f} "
                  "GO line; shaded = unreachable (above headroom)", fontsize=8.5, color=INK2)
    fig.suptitle("Every non-baseline configuration vs baselines_union (grouped CV)",
                 fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
