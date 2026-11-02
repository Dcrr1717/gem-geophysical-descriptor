"""Tables and figures for the ECT / DECT study (src/eval/ect_benchmark.py)."""
from __future__ import annotations

from pathlib import Path

import numpy as np

TASK_TITLES = {"iaaft_binary": "Task A: coherent turbulence vs IAAFT surrogate (structure-only)",
               "vortex_merger": "Task B: merged vs separate vortex pair (connectivity)"}
CLASS_NAMES = {"iaaft_binary": ("coherent", "IAAFT surrogate"),
               "vortex_merger": ("merged (1 core)", "separate (2 cores)")}


def _rows(task):
    from .ect_benchmark import BASE, INCR, TOPO
    R = task["results"]
    return [k for k in TOPO + ["dect"] + BASE + INCR if k in R]


def task_table(name, task, thr):
    from .ect_benchmark import LABELS, TOPO
    R, M = task["results"], task["margins"]
    L = [f"### {TASK_TITLES[name]}", "",
         f"`baselines_union` = {M['baselines_union_accuracy']:.3f}; headroom (largest "
         f"attainable margin) = {M['headroom']:.3f}. Best structure baseline: "
         f"`{M['best_structure_baseline']}`.", "",
         "| View | accuracy | margin vs `baselines_union` [95% CI] | BH q | "
         "margin vs best structure baseline [95% CI] |", "|---|---|---|---|---|"]
    for k in _rows(task):
        acc = R[k]["best_accuracy"]
        if k in M["vs_union"]:
            e, e2 = M["vs_union"][k], M["vs_best_structure"][k]
            star = " ★" if (e["margin"] >= thr and e["ci95"][0] > 0
                            and e.get("q_bh", 1) < 0.05) else ""
            q = f"{e['q_bh']:.3f}" if "q_bh" in e else "–"
            L.append(f"| `{k}` ({LABELS.get(k, k)}) | {acc:.3f} | {e['margin']:+.3f} "
                     f"[{e['ci95'][0]:+.3f}, {e['ci95'][1]:+.3f}]{star} | {q} | "
                     f"{e2['margin']:+.3f} [{e2['ci95'][0]:+.3f}, {e2['ci95'][1]:+.3f}] |")
        else:
            L.append(f"| `{k}` ({LABELS.get(k, k)}) | {acc:.3f} | – | – | – |")
    return L


def write_report(res, path):
    thr = float(res["config"]["ect_study"]["rule"]["min_margin"])
    L = ["# ECT / DECT study (generated tables)", "",
         "Design, rule and interpretation: [`docs/ECT_RESULTS.md`](../docs/ECT_RESULTS.md). "
         f"Repeated StratifiedGroupKFold ({res['config']['ect_study']['n_repeats']} repeats x "
         f"{res['config']['evaluation']['n_splits']} folds, group = trajectory); best of LogReg "
         "/ RF for feature views, the DECT CNN head for `dect`; paired group-bootstrap 95% CIs; "
         "★ = margin >= +0.05 with CI above 0 and BH q < 0.05.", "",
         f"DECT enabled: {res['dect_enabled']}.", "", "## Verdict per task", "",
         "| Task | primary views | best primary view | margin [95% CI] | headroom | verdict |",
         "|---|---|---|---|---|---|"]
    for t, v in res["verdict"].items():
        ci = v["best_primary_ci95"]
        L.append(f"| {t} | {', '.join(v['primary_views'])} | `{v['best_primary_view']}` | "
                 f"{v['best_primary_margin']:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}] | "
                 f"{v['headroom']:.3f} | **{v['verdict']}** |")
    L += [""]
    for t, task in res["tasks"].items():
        L += task_table(t, task, thr) + [""]
    L += [f"Elapsed: {res['seconds']} s.", ""]
    path.write_text("\n".join(L) + "\n")


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


FAMILY_COLORS = {"topo": "#2a78d6", "ect": "#008300", "struct": "#eda100",
                 "spectral": "#1baf7a", "union": "#e87ba4"}


def _family(k):
    if k in ("ect", "dect"):
        return "ect"
    if k.startswith("baselines_union"):
        return "union"
    if k in ("okubo_weiss", "gradient_stats", "increment_flatness", "physical_structure_all"):
        return "struct"
    if k in ("moments", "power_spectrum", "histogram_quantiles"):
        return "spectral"
    return "topo"


def accuracy_figure(res, name, path):
    plt = _plt()
    from .hard_figures import INK, INK2, SURFACE, _style
    task = res["tasks"][name]
    R, M = task["results"], task["margins"]
    rows = _rows(task)
    thr = float(res["config"]["ect_study"]["rule"]["min_margin"])
    fig, axes = plt.subplots(1, 2, figsize=(13, 0.32 * len(rows) + 1.6), facecolor=SURFACE,
                             gridspec_kw={"width_ratios": [1.1, 1]})
    y = np.arange(len(rows))[::-1]
    ax = axes[0]
    _style(ax)
    acc = [R[k]["best_accuracy"] for k in rows]
    ax.barh(y, acc, color=[FAMILY_COLORS[_family(k)] for k in rows], height=0.7,
            edgecolor=SURFACE, linewidth=2)
    for yi, a in zip(y, acc):
        ax.text(a + 0.005, yi, f"{a:.3f}", va="center", fontsize=7, color=INK2)
    ax.set_yticks(y)
    ax.set_yticklabels(rows, fontsize=8, color=INK2)
    ax.axvline(0.5 if len(task["class_mean_ect"]) == 2 else 0, color=INK2, lw=1)
    ax.set_xlim(0.4, 1.06)
    ax.set_xlabel("accuracy (grouped CV)", fontsize=9, color=INK2)
    ax.set_title("Accuracy per view", fontsize=10, color=INK, loc="left")
    ax = axes[1]
    _style(ax)
    keys = [k for k in rows if k in M["vs_union"]]
    yy = np.arange(len(keys))[::-1]
    for yi, k in zip(yy, keys):
        e = M["vs_union"][k]
        col = FAMILY_COLORS[_family(k)]
        ax.plot(e["ci95"], [yi, yi], color=col, lw=2)
        ax.plot(e["margin"], yi, "o", color=col, ms=6)
    ax.axvline(0, color=INK2, lw=1)
    ax.axvline(thr, color=INK2, lw=1, ls="--")
    ax.set_yticks(yy)
    ax.set_yticklabels(keys, fontsize=8, color=INK2)
    ax.set_xlabel("margin over baselines_union (95% CI); dashed = +0.05", fontsize=9, color=INK2)
    ax.set_title(f"Topology vs baselines_union (= {M['baselines_union_accuracy']:.3f})",
                 fontsize=10, color=INK, loc="left")
    fig.suptitle(TASK_TITLES[name], fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def ect_figure(res, path):
    """Mean ECT image per class and the standardized difference, at the level
    where the classes differ most, for both tasks."""
    plt = _plt()
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
    from ..descriptor.ect import ECT_DIRECTIONS, ECT_LEVELS, ECT_THRESHOLDS, T_MAX
    from .hard_figures import INK, INK2, SURFACE
    seq = LinearSegmentedColormap.from_list("seq", ["#f0efec", "#86b6ef", "#2a78d6", "#104281"])
    div = LinearSegmentedColormap.from_list("div", ["#e34948", "#f6c7c6", "#f0efec",
                                                    "#cde2fb", "#2a78d6"])
    shape = (len(ECT_LEVELS), ECT_DIRECTIONS, ECT_THRESHOLDS)
    names = list(res["tasks"])
    fig, axes = plt.subplots(len(names), 3, figsize=(13, 3.6 * len(names)), facecolor=SURFACE,
                             squeeze=False)
    for r, name in enumerate(names):
        t = res["tasks"][name]
        m0, m1 = (np.asarray(x).reshape(shape) for x in t["class_mean_ect"])
        s0, s1 = (np.asarray(x).reshape(shape) for x in t["class_std_ect"])
        eff = (m1 - m0) / np.sqrt(0.5 * (s0 ** 2 + s1 ** 2) + 1e-9)
        lv = int(np.argmax(np.abs(eff).reshape(len(ECT_LEVELS), -1).mean(1)))
        vmin, vmax = min(m0[lv].min(), m1[lv].min()), max(m0[lv].max(), m1[lv].max())
        ext = [-T_MAX, T_MAX, 360, 0]
        # chi is signed (components minus holes): diverging scale when both signs occur
        kw = (dict(cmap=div, norm=TwoSlopeNorm(0, vmin, vmax)) if vmin < 0 < vmax
              else dict(cmap=seq, vmin=vmin, vmax=vmax))
        for c, (img, title) in enumerate(((m0[lv], CLASS_NAMES[name][0]),
                                          (m1[lv], CLASS_NAMES[name][1]))):
            ax = axes[r, c]
            im = ax.imshow(img, aspect="auto", extent=ext, **kw)
            ax.set_title(f"{title}: mean ECT, level h = {ECT_LEVELS[lv]:+g}", fontsize=9,
                         color=INK, loc="left")
            fig.colorbar(im, ax=ax, fraction=0.04).ax.tick_params(labelsize=7, colors=INK2)
        ax = axes[r, 2]
        lim = float(np.nanmax(np.abs(eff[lv]))) or 1.0
        im = ax.imshow(eff[lv], aspect="auto", cmap=div, norm=TwoSlopeNorm(0, -lim, lim),
                       extent=ext)
        ax.set_title("standardized difference (class 1 − class 0)", fontsize=9, color=INK,
                     loc="left")
        fig.colorbar(im, ax=ax, fraction=0.04).ax.tick_params(labelsize=7, colors=INK2)
        for ax in axes[r]:
            ax.set_xlabel("threshold t on <x, v>", fontsize=8, color=INK2)
            ax.set_ylabel("direction v (degrees)", fontsize=8, color=INK2)
            ax.tick_params(labelsize=7, colors=INK2)
        axes[r, 0].text(-0.28, 0.5, TASK_TITLES[name].split(":")[0], transform=axes[r, 0].transAxes,
                        rotation=90, va="center", fontsize=10, color=INK)
    fig.suptitle("Euler Characteristic Transform of the sublevel sets {f <= h}: "
                 "Euler characteristic per direction and threshold", fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def vortex_examples_figure(vb, path, n_cols=4):
    plt = _plt()
    from .hard_figures import INK, INK2, SURFACE, VORT_CMAP
    fig, axes = plt.subplots(2, n_cols, figsize=(3.2 * n_cols, 6.6), facecolor=SURFACE)
    for r, lab in enumerate((0, 1)):
        groups = np.unique(vb["group"][vb["label"] == lab])[:n_cols]
        for c, g in enumerate(groups):
            idx = np.flatnonzero(vb["group"] == g)
            i = idx[len(idx) // 2]
            w = vb["fields"][i]
            lim = np.abs(w).max()
            ax = axes[r, c]
            ax.imshow(w.T, origin="lower", cmap=VORT_CMAP, vmin=-lim, vmax=lim)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"{('merged', 'separate')[lab]}, d/a = {vb['d_over_a'][i]:.2f}",
                         fontsize=9, color=INK, loc="left")
    fig.suptitle("Vortex-event task: standardized vorticity at mid snapshot window "
                 "(top: merging pairs, one core; bottom: separate pairs, two cores)",
                 fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def make_figures(res, vb, coh, sur, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in res["tasks"]:
        accuracy_figure(res, name, out_dir / f"ect_accuracy_{name}.png")
    ect_figure(res, out_dir / "ect_images.png")
    vortex_examples_figure(vb, out_dir / "vortex_event_examples.png")
