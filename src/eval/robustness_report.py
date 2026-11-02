"""Tables and figures for the robustness study (src/eval/robustness_sweep.py)."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .degradation import MODELS

MODEL_TITLES = {
    "noise": "Additive white observation noise (SNR)",
    "random_mask": "Missing pixels at random, harmonic infill",
    "swath_gaps": "Contiguous gaps: 8-px swaths, harmonic infill",
    "block_gaps": "Contiguous gaps: 32x32 blocks, harmonic infill",
    "quantize": "Coarse quantization (bit depth)",
    "resample": "Resolution loss (block mean, bilinear back)",
}
TABLE_ROWS = ["descriptor", "x1x2_rich", "x1x2", "x1_graph", "x2_ph", "topological_view_fullres",
              "topo_persistence_image", "topo_betti_curves", "topo_multiscale", "topo_rich_all",
              "okubo_weiss", "gradient_stats", "increment_flatness", "physical_structure_all",
              "baselines_union", "physical_structure_ms", "baselines_union_ms",
              "power_spectrum", "histogram_quantiles",
              "baselines_union+topo_rich", "baselines_union_ms+topo_multiscale"]
MARGIN_ROWS = ["descriptor", "x1x2_rich", "x1x2", "x1_graph", "x2_ph", "topological_view_fullres",
               "topo_persistence_image", "topo_betti_curves", "topo_multiscale", "topo_rich_all",
               "baselines_union+topo_rich"]


def _conds_of(res, model):
    return ["clean"] + sorted([c for c, v in res["conditions"].items() if v["model"] == model],
                              key=lambda c: int(c.rsplit("_", 1)[1]))


def _real(e, thr):
    return e["margin"] >= thr and e["ci95"][0] > 0 and e.get("q_bh", 1.0) < 0.05


def _cell(e, thr):
    if e is None:
        return "–"
    mark = " **★**" if _real(e, thr) else ""
    return f"{e['margin']:+.3f} [{e['ci95'][0]:+.3f}, {e['ci95'][1]:+.3f}]{mark}"


def model_tables(res, model):
    thr = float(res["config"]["robustness"]["rule"]["min_margin"])
    C = res["conditions"]
    conds = _conds_of(res, model)
    head = ["clean"] + [C[c]["label"].split(" ", 1)[1] for c in conds[1:]]
    L = [f"### {MODEL_TITLES[model]}", ""]
    fr = [C[c].get("missing_fraction") for c in conds[1:]]
    if any(f is not None for f in fr):
        L += ["Achieved missing fraction: " + ", ".join(f"{100 * f:.1f}%" for f in fr), ""]
    L += ["Accuracy (best of LogReg / RF, grouped CV):", "",
          "| Feature set | " + " | ".join(head) + " |", "|---|" + "---|" * len(head)]
    for s in TABLE_ROWS:
        row = [f"{C[c]['results'][s]['best_accuracy']:.3f}" for c in conds]
        L.append(f"| `{s}` | " + " | ".join(row) + " |")
    for ref, title in (("baselines_union", "Margin over `baselines_union` [95% CI]"),
                       ("baselines_union_ms", "Margin over the matched-scale control "
                                              "`baselines_union_ms` [95% CI]")):
        L += ["", f"{title}; ★ = margin ≥ +{thr:.2f}, CI above 0 and BH q < 0.05:", "",
              "| Configuration | " + " | ".join(head) + " |", "|---|" + "---|" * len(head)]
        for s in MARGIN_ROWS:
            cells = [_cell(C[c]["margins"][ref].get(s), thr) for c in conds]
            L.append(f"| `{s}` | " + " | ".join(cells) + " |")
        ra = [f"{C[c]['results'][ref]['best_accuracy']:.3f}" for c in conds]
        L.append(f"| *reference accuracy* | " + " | ".join(ra) + " |")
    return L


def write_report(res, path):
    d = res["decision"]
    thr = float(res["config"]["robustness"]["rule"]["min_margin"])
    L = ["# Robustness of topology on degraded fields (generated tables)", "",
         "Pre-registered study; design, rule and interpretation in "
         "[`docs/NOISE_ROBUSTNESS.md`](../docs/NOISE_ROBUSTNESS.md). Binary coherent-vs-IAAFT "
         f"task, {res['conditions']['clean']['n']} fields, "
         f"{res['conditions']['clean']['n_groups']} trajectories, repeated StratifiedGroupKFold "
         f"({res['config']['robustness']['n_repeats']} repeats x 4 folds, group = trajectory), "
         "paired group-bootstrap 95% CIs (2000 resamples), one-sided bootstrap p with "
         f"Benjamini-Hochberg over {d['n_comparisons_in_fdr_family']} comparisons.", "",
         f"**Verdict for `{d['lead_config']}`: {d['verdict']}.**", "",
         "## Where a topological configuration has a real advantage", "",
         f"Real advantage = margin ≥ +{thr:.2f}, 95% CI above 0 and BH q < 0.05. "
         "Degradation models listed with the levels where it holds.", "",
         "| Configuration | vs `baselines_union` | CI lower bound ≥ +0.05 | also vs matched-scale "
         "control |", "|---|---|---|---|"]
    for k, v in d["per_config"].items():
        def fmt(h):
            return "; ".join(f"{m}: {', '.join(x.split(' ', 1)[1] for x in lv)}"
                             for m, lv in h.items()) or "none"
        L.append(f"| `{k}` | {fmt(v['advantage_vs_union'])} | "
                 f"{fmt(v['advantage_ci_lower_ge_threshold'])} | "
                 f"{fmt(v['advantage_also_vs_matched_scale_control'])} |")
    L += ["", "## Per degradation model", ""]
    for m in MODELS:
        L += model_tables(res, m) + [""]
    L += [f"Elapsed: {res['seconds']} s (wall clock).", ""]
    path.write_text("\n".join(L) + "\n")


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
LINE_SETS = [("topo_multiscale", "#008300", "o", "multi-scale topology"),
             ("x1x2_rich", "#6250d6", "s", "full descriptor X1+X2+richer"),
             ("x2_ph", "#2a78d6", "D", "X2 (PH summaries)"),
             ("gradient_stats", "#eda100", "^", "gradient statistics"),
             ("baselines_union", "#e87ba4", "v", "baselines_union"),
             ("baselines_union_ms", "#eb6834", "P", "matched-scale control")]
MARGIN_SETS = [("topo_multiscale", "#008300", "o", "multi-scale topology"),
               ("x1x2_rich", "#6250d6", "s", "full descriptor X1+X2+richer"),
               ("x1x2", "#2a78d6", "D", "authentic X1+X2"),
               ("topo_persistence_image", "#eda100", "^", "persistence images")]


def model_figure(res, model, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from .hard_figures import INK, INK2, SURFACE, _style

    thr = float(res["config"]["robustness"]["rule"]["min_margin"])
    C = res["conditions"]
    conds = _conds_of(res, model)
    x = np.arange(len(conds))
    ticks = ["clean"] + [C[c]["label"].split(" ", 1)[1] for c in conds[1:]]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), facecolor=SURFACE)
    ax = axes[0]
    _style(ax)
    for s, col, mk, lab in LINE_SETS:
        ax.plot(x, [C[c]["results"][s]["best_accuracy"] for c in conds], color=col, lw=2,
                marker=mk, ms=6, label=lab)
    ax.axhline(0.5, color=INK2, lw=1)
    ax.set_ylim(0.45, 1.02)
    ax.set_ylabel("accuracy (grouped CV)", fontsize=9, color=INK2)
    ax.set_title("Accuracy", fontsize=10, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=7, loc="lower left")
    for ax, ref, title in ((axes[1], "baselines_union", "Margin over baselines_union"),
                           (axes[2], "baselines_union_ms",
                            "Margin over matched-scale control")):
        _style(ax)
        off = np.linspace(-0.15, 0.15, len(MARGIN_SETS))
        for (s, col, mk, lab), o in zip(MARGIN_SETS, off):
            e = [C[c]["margins"][ref][s] for c in conds]
            m = np.array([v["margin"] for v in e])
            lo = np.array([v["ci95"][0] for v in e])
            hi = np.array([v["ci95"][1] for v in e])
            ax.errorbar(x + o, m, yerr=[m - lo, hi - m], color=col, marker=mk, ms=5, lw=0,
                        elinewidth=1.5, capsize=2, label=lab)
        ax.axhline(0, color=INK2, lw=1)
        ax.axhline(thr, color=INK2, lw=1, ls="--")
        ax.text(x[-1] + 0.4, thr, f"+{thr:.2f}", fontsize=7, color=INK2, va="center")
        ax.set_title(title + " (95% CI)", fontsize=10, color=INK, loc="left")
        ax.set_ylabel("accuracy margin", fontsize=9, color=INK2)
    axes[1].legend(frameon=False, fontsize=7, loc="best")
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(ticks, fontsize=8)
    fig.suptitle(MODEL_TITLES[model], fontsize=11, color=INK)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def heatmap_figure(res, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
    from .hard_figures import INK, INK2, SURFACE

    thr = float(res["config"]["robustness"]["rule"]["min_margin"])
    C = res["conditions"]
    cols = []
    for m in MODELS:
        cols += _conds_of(res, m)[1:]
    labels = [C[c]["label"].replace("_", " ") for c in cols]
    cmap = LinearSegmentedColormap.from_list(
        "div", ["#e34948", "#f6c7c6", "#f0efec", "#cde2fb", "#2a78d6"])
    fig, axes = plt.subplots(2, 1, figsize=(15, 8.5), facecolor=SURFACE)
    for ax, ref, title in ((axes[0], "baselines_union", "margin over baselines_union"),
                           (axes[1], "baselines_union_ms", "margin over matched-scale control")):
        M = np.array([[C[c]["margins"][ref][k]["margin"] for c in cols] for k in MARGIN_ROWS])
        im = ax.imshow(M, cmap=cmap, norm=TwoSlopeNorm(0, -0.3, 0.3), aspect="auto")
        for i, k in enumerate(MARGIN_ROWS):
            for j, c in enumerate(cols):
                e = C[c]["margins"][ref][k]
                if _real(e, thr):
                    ax.text(j, i, "★", ha="center", va="center", fontsize=9, color=INK)
        ax.set_yticks(range(len(MARGIN_ROWS)))
        # spell out "x1x2" (some fonts render "1x2" as a multiplication sign)
        ax.set_yticklabels([k.replace("x1x2", "X1+X2").replace("x1_graph", "X1 graph")
                            .replace("x2_ph", "X2 PH") for k in MARGIN_ROWS],
                           fontsize=8, color=INK2)
        ax.set_xticks(range(len(cols)))
        ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=7, color=INK2)
        ax.set_title(f"Topological configurations: {title} (★ = ≥ +{thr:.2f}, CI > 0, q < 0.05)",
                     fontsize=10, color=INK, loc="left")
        cb = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01)
        cb.ax.tick_params(labelsize=7, colors=INK2)
        cb.set_label("accuracy margin (blue = topology better)", fontsize=8, color=INK2)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def main():
    """Re-render the figures from results/noise_robustness.json (no re-run)."""
    import json
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "results/noise_robustness.json"
    res = json.load(open(path))
    make_figures(res, Path(path).parent / "figures")


def make_figures(res, out_dir):
    out_dir = Path(out_dir)
    for m in MODELS:
        model_figure(res, m, out_dir / f"robustness_{m}.png")
    heatmap_figure(res, out_dir / "robustness_margin_heatmap.png")


if __name__ == "__main__":
    main()
