"""Post-hoc stress test: the structure-only benchmark under observation noise.

EXPLORATORY - NOT PART OF THE GO/NO-GO VERDICT. Motivation: on clean 256^2
fields the coherent-vs-IAAFT task is saturated by simple phase-sensitive
physical diagnostics (gradient statistics reach ~100%), so it cannot reveal
whether topology adds anything. Real geophysical fields (reanalysis,
altimetry) are noisy; derivative-based diagnostics are known to be fragile to
noise, persistence diagrams are provably stable to it. This sweep asks
whether the topological view keeps an advantage when the baselines are no
longer saturated.

Noise model: independent white Gaussian noise of standard deviation `eta`
(relative to the unit-variance field) is added to every coherent field AND to
every surrogate, then the field is re-standardized. Because the noise is
independent of the field, both classes keep the same expected power spectrum
(signal + flat noise floor) and the same one-point distribution (histogram
convolved with the same Gaussian), so the benchmark stays structure-only.
Two conditions per noise level:
  raw      - features computed on the noisy field;
  smoothed - every feature set sees the same periodic Gaussian low-pass
             (sigma = `smooth_sigma_px` grid cells) applied before features,
             i.e. the standard denoising a practitioner would apply before
             differentiating noisy data.

Usage: python -m src.eval.noise_sweep --config config/hard_benchmark.yaml
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import yaml

from .hard_benchmark import (generate_coherent, generate_surrogates, compute_features,
                             run_task, verdicts, _standardize, _strip_oof, FEATURE_SETS,
                             rich_summary_table)

SWEEP_SETS = ["descriptor", "spectral_view", "topological_view", "moments", "spectral_slope",
              "raw_pixels", "power_spectrum", "histogram_quantiles", "okubo_weiss",
              "gradient_stats", "increment_flatness", "physical_structure_all",
              "baselines_union", "baselines_union+topo",
              "topo_persistence_image", "topo_betti_curves", "topo_multiscale", "topo_rich_all",
              "descriptor_rich", "baselines_union+topo_rich"]


def _gaussian_lowpass(w, sigma_px):
    n = w.shape[0]
    k = np.fft.fftfreq(n)  # cycles per grid cell
    kx, ky = np.meshgrid(k, k, indexing="ij")
    H = np.exp(-2.0 * (np.pi * sigma_px) ** 2 * (kx ** 2 + ky ** 2))
    return np.real(np.fft.ifft2(np.fft.fft2(w) * H))


def observe(fields, eta, seed_words, smooth_sigma=None):
    out = []
    for i, w in enumerate(fields):
        rng = np.random.default_rng(np.random.SeedSequence(list(seed_words) + [i]))
        v = w + eta * rng.standard_normal(w.shape)
        if smooth_sigma:
            v = _gaussian_lowpass(v, smooth_sigma)
        out.append(_standardize(v))
    return np.asarray(out)


def run(cfg):
    t0 = time.time()
    ns = cfg["noise_sweep"]
    workers = int(cfg.get("n_workers", 4))
    cache_dir = cfg.get("cache_dir", "data/hard_benchmark")
    out_dir = Path(cfg.get("out_dir", "results"))
    seed = cfg.get("seed", 0)
    ev = dict(cfg["evaluation"])
    ev.setdefault("seed", seed)
    ev["n_repeats"] = ns.get("n_repeats", ev["n_repeats"])
    go_margin = float(cfg.get("go_margin", 0.05))
    n_boot = int(ev.get("bootstrap", 2000))

    coh, _ = generate_coherent(cfg, cache_dir, workers)
    sur = generate_surrogates(coh, cfg, cfg["surrogate"]["final"], cache_dir, workers)
    N = len(coh["fields"])
    y = np.r_[np.zeros(N, int), np.ones(N, int)]
    g = np.r_[coh["group"], coh["group"]]
    topo_ds = cfg["descriptor"]["topo_downsample"]
    dec = cfg.get("feature_decimals", 8)

    conditions = []
    for li, eta in enumerate(ns["noise_levels"]):
        for mode in ("raw", "smoothed"):
            conditions.append((li, float(eta), mode))
    results = {}
    for li, eta, mode in conditions:
        sig = ns["smooth_sigma_px"] if mode == "smoothed" else None
        Oc = observe(coh["fields"], eta, [seed, 31337, li, 0], sig)
        Os = observe(sur["fields"], eta, [seed, 31337, li, 1], sig)
        Fc = compute_features(Oc, topo_ds, workers, dec, fullres=False)
        Fs = compute_features(Os, topo_ds, workers, dec, fullres=False)
        F = {k: np.vstack([Fc[k], Fs[k]]) for k in Fc}
        name = f"eta{eta:g}_{mode}"
        task = run_task(name, F, y, g, SWEEP_SETS, ev, 0.5)
        task["verdicts"] = verdicts(task, y, g, go_margin, n_boot, seed)
        task["eta"], task["mode"] = eta, mode
        results[name] = task

    res = {"note": "EXPLORATORY post-hoc stress test; not part of the Go/No-Go verdict.",
           "noise_model": "x + eta * N(0,1) per pixel, independent for every field "
                          "(coherent and surrogate), then re-standardized; 'smoothed' adds a "
                          f"periodic Gaussian low-pass, sigma = {ns['smooth_sigma_px']} px, "
                          "applied identically before every feature set.",
           "noise_levels": ns["noise_levels"], "smooth_sigma_px": ns["smooth_sigma_px"],
           "n_repeats": ev["n_repeats"], "conditions": results,
           "seconds": round(time.time() - t0, 1)}
    (out_dir / "noise_sweep.json").write_text(json.dumps(_strip_oof(res), indent=2))
    write_report(out_dir / "noise_sweep.md", res)
    sweep_figure(res, out_dir / "figures" / "noise_sweep.png")
    from .hard_figures import rich_margin_figure
    entries = [(k, c["verdicts"].get("rich_topology")) for k, c in results.items()]
    if any(v is not None for _, v in entries):
        rich_margin_figure(entries, out_dir / "figures" / "rich_topology_margins_noise.png",
                           "Richer topology vs all non-topological baselines (noisy fields)",
                           go_margin)
    return res


def write_report(path, res):
    L = ["# Noise stress test (exploratory, post hoc - not part of the verdict)", "",
         f"Noise model: {res['noise_model']}", "",
         f"Binary coherent-vs-IAAFT task, 480 fields, grouped CV "
         f"({res['n_repeats']} repeats x 4 folds, group = trajectory). Entries: best of "
         f"LogReg / RF accuracy.", ""]
    conds = list(res["conditions"])
    L.append("| Feature set | " + " | ".join(c.replace("_", " ") for c in conds) + " |")
    L.append("|---|" + "---|" * len(conds))
    for s in SWEEP_SETS:
        row = [f"{res['conditions'][c]['results'][s]['best_accuracy']:.3f}" for c in conds]
        name = f"**{s}**" if s == "descriptor" else s
        L.append(f"| {name} ({FEATURE_SETS[s][0]}) | " + " | ".join(row) + " |")
    L += ["", "Key comparisons per condition (margin, 95% paired group-bootstrap CI):", ""]
    for c in conds:
        v = res["conditions"][c]["verdicts"]
        st, ts, inc = v["strict"], v["topology_vs_structure_baselines"], v["incremental_topology"]
        L.append(f"- `{c}`: descriptor - best baseline (`{st['best_baseline']}`) = "
                 f"{st['margin']:+.3f} [{st['margin_ci95'][0]:+.3f}, {st['margin_ci95'][1]:+.3f}] "
                 f"({st['verdict']} at the 0.05 rule); topo view - best structure baseline "
                 f"(`{ts['best_structure_baseline']}`) = {ts['margin']:+.3f} "
                 f"[{ts['margin_ci95'][0]:+.3f}, {ts['margin_ci95'][1]:+.3f}]; "
                 f"topology added to all baselines: {inc['gain']:+.3f} "
                 f"[{inc['gain_ci95'][0]:+.3f}, {inc['gain_ci95'][1]:+.3f}]")
    if all("rich_topology" in res["conditions"][c]["verdicts"] for c in conds):
        L += ["", "## Richer topological vectorizations vs `baselines_union`", "",
              "Cells: accuracy (margin vs `baselines_union` [95% CI]); `max attainable margin` = "
              "1 - accuracy of `baselines_union`.", ""]
        L += rich_summary_table(res["conditions"], conds)
    L += ["", f"Elapsed: {res['seconds']} s (wall clock).", ""]
    path.write_text("\n".join(L) + "\n")


def sweep_figure(res, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from .hard_figures import _style, SURFACE, INK, INK2

    show = [("descriptor", "#2a78d6", "o"), ("topological_view", "#6250d6", "s"),
            ("gradient_stats", "#eda100", "^"), ("physical_structure_all", "#eb6834", "D"),
            ("baselines_union", "#1baf7a", "v"), ("topo_rich_all", "#008300", "P")]
    etas = res["noise_levels"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), facecolor=SURFACE, sharey=True)
    for ax, mode in zip(axes, ("raw", "smoothed")):
        _style(ax)
        for s, col, mk in show:
            acc = [res["conditions"][f"eta{e:g}_{mode}"]["results"][s]["best_accuracy"]
                   for e in etas]
            ax.plot(etas, acc, color=col, lw=2, marker=mk, ms=7, label=s)
        ax.axhline(0.5, color=INK2, lw=1)
        ax.set_xscale("log")
        ax.set_xticks(etas)
        ax.set_xticklabels([f"{e:g}" for e in etas])
        ax.set_xlabel("observation noise eta (std, relative to field)", fontsize=9, color=INK2)
        ax.set_title(f"{mode} noisy fields" if mode == "raw" else
                     f"noisy fields, Gaussian low-pass ({res['smooth_sigma_px']} px)",
                     fontsize=10, color=INK, loc="left")
    axes[0].set_ylabel("accuracy (grouped CV)", fontsize=9, color=INK2)
    axes[0].set_ylim(0.45, 1.02)
    axes[1].legend(frameon=False, fontsize=8, loc="lower left")
    fig.suptitle("Coherent vs IAAFT under observation noise (exploratory)", fontsize=11, color=INK)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/hard_benchmark.yaml")
    args = ap.parse_args()
    run(yaml.safe_load(Path(args.config).read_text()))


if __name__ == "__main__":
    main()
