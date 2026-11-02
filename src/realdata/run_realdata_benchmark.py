"""Step 3: ERA5 winter BLOCKING vs ZONAL days through the existing Go/No-Go evaluation.

Input: the labelled file written by `label_blocking` (daily 500 hPa relative
vorticity + TM90 labels). Every BLOCKING / ZONAL day becomes one field:

1. crop to `benchmark.domain` (lat / lon box, may cross the dateline);
2. resample to a `grid` x `grid` square (Gaussian anti-aliasing where the
   grid is coarsened, then bilinear interpolation). Over the default
   150 deg x 55 deg Euro-Atlantic box the physical aspect ratio at 57.5N is
   about 1.4 : 1, so the square grid distorts little;
3. make the field periodic, because the descriptor and the baselines use FFTs,
   periodic shifts and periodic cubical complexes (designed for the doubly
   periodic simulations): `taper` (default) removes the mean and multiplies
   by a 2D Tukey window, so the field decays smoothly to zero at the edges;
   `mirror` resamples to grid/2 and reflects evenly into a grid x grid
   periodic field; `none` keeps the edge discontinuity;
4. standardize to zero mean and unit variance (`hard_benchmark._standardize`,
   the same function as for the synthetic fields).

Then the SAME feature registry (`hard_benchmark.FEATURE_SETS`: descriptor,
its views, the authentic X1 / X2 views, richer topology, and every baseline),
the same grouped cross-validation (`classify.evaluate_grouped`), the same
verdict rules (`verdicts`, `union_verdicts`) and the same Markdown tables as
the synthetic hard benchmark. Groups are time blocks (whole seasons by
default), so consecutive, autocorrelated days are never split between train
and test; the paired bootstrap of margins resamples the same groups. For
contrast, the report also gives the shuffled-day StratifiedKFold accuracy
(leaky) of a few feature sets.

Usage
  python -m src.realdata.run_realdata_benchmark --config config/realdata.yaml
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import time
from pathlib import Path

import numpy as np

from ..eval.classify import evaluate
from ..eval.hard_benchmark import (FEATURE_SETS, SPECIFIED, _fmt_ci, _standardize, _strip_oof,
                                   _task_table, _union_section, _verdict_lines,
                                   compute_features, rich_summary_table, run_task,
                                   union_summary, union_verdicts, verdicts)
from .common import load_config, lon_window, time_groups, validate_config
from .label_blocking import AMBIGUOUS, BLOCKING, ZONAL

PRIMARY = "blocking_vs_zonal"
LEAKAGE_SETS = ["descriptor", "spectral_view", "topological_view", *SPECIFIED,
                "physical_structure_all", "baselines_union"]


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------
def load_labelled(path) -> dict:
    with np.load(path, allow_pickle=False) as d:
        return {k: d[k] for k in d.files}


def crop_domain(fields, lat, lon, domain, tol=1e-6):
    """fields (N, nlat, nlon) with lat ascending -> the box, columns west -> east."""
    lat = np.asarray(lat, dtype=float)
    rows = np.flatnonzero((lat >= float(domain["south"]) - tol) &
                          (lat <= float(domain["north"]) + tol))
    cols, _ = lon_window(lon, domain["west"], domain["east"])
    if rows.size < 2 or cols.size < 2:
        raise ValueError(f"benchmark.domain {domain} selects fewer than 2 x 2 grid points")
    return fields[:, rows][:, :, cols], lat[rows], np.asarray(lon)[cols]


def resample_square(field, n):
    """(h, w) -> (n, n): Gaussian anti-aliasing along coarsened axes
    (sigma = (factor - 1) / 2 input pixels), then bilinear interpolation at
    the centres of the target pixels."""
    from scipy import ndimage

    f = np.asarray(field, dtype=float)
    h, w = f.shape
    sig = (max(0.0, (h / n - 1) / 2), max(0.0, (w / n - 1) / 2))
    if any(s > 0 for s in sig):
        f = ndimage.gaussian_filter(f, sig, mode="nearest")
    r = (np.arange(n) + 0.5) * h / n - 0.5
    c = (np.arange(n) + 0.5) * w / n - 0.5
    rr, cc = np.meshgrid(r, c, indexing="ij")
    return ndimage.map_coordinates(f, [rr, cc], order=1, mode="nearest")


def periodize(field, method="taper", alpha=0.25):
    """Make a limited-area field periodic (see the module docstring)."""
    f = np.asarray(field, dtype=float)
    if method == "taper":
        from scipy.signal.windows import tukey

        win = np.outer(tukey(f.shape[0], alpha), tukey(f.shape[1], alpha))
        return (f - f.mean()) * win
    if method == "mirror":
        return np.block([[f, f[:, ::-1]], [f[::-1, :], f[::-1, ::-1]]])
    if method == "none":
        return f
    raise ValueError(f"unknown periodize method {method!r}")


def preprocess(fields, lat, lon, bm):
    """Raw (N, nlat, nlon) -> (N, grid, grid) periodic, standardized fields."""
    n = int(bm.get("grid", 64))
    method = bm.get("periodize", "taper")
    alpha = float(bm.get("taper_alpha", 0.25))
    crop, _, _ = crop_domain(np.asarray(fields), lat, lon, bm["domain"])
    m = n // 2 if method == "mirror" else n
    out = np.empty((len(crop), n, n))
    for i, f in enumerate(crop):
        out[i] = _standardize(periodize(resample_square(f, m), method, alpha))
    return out, crop.shape[1:]


def balance_classes(y, seed=0):
    """Indices that undersample the majority class to the minority size
    (seeded, uniformly over all days), returned in their original order."""
    y = np.asarray(y)
    i0, i1 = np.flatnonzero(y == 0), np.flatnonzero(y == 1)
    k = min(i0.size, i1.size)
    rng = np.random.default_rng(seed)
    if i0.size > k:
        i0 = rng.choice(i0, k, replace=False)
    if i1.size > k:
        i1 = rng.choice(i1, k, replace=False)
    return np.sort(np.r_[i0, i1])


def check_groups(y, groups, n_splits):
    for c in np.unique(y):
        ng = np.unique(groups[y == c]).size
        if ng < n_splits:
            raise SystemExit(
                f"[realdata] class {c} occurs in only {ng} groups but n_splits = {n_splits}: "
                "download more seasons, use benchmark.grouping.mode = block, or lower n_splits")


def select_days(lab, cfg):
    """BLOCKING / ZONAL days kept for the benchmark -> (idx, y, groups)."""
    bm = cfg["benchmark"]
    g = bm.get("grouping", {})
    labels = lab["labels"]
    groups_all, keep = time_groups(lab["dates"], cfg["season"]["months"],
                                   g.get("mode", "season"), g.get("block_days", 30),
                                   g.get("gap_days", 5))
    idx = np.flatnonzero((labels != AMBIGUOUS) & keep)
    y = (labels[idx] == BLOCKING).astype(int)
    if bm.get("balance", "undersample") == "undersample":
        b = balance_classes(y, int(cfg.get("seed", 0)))
        idx, y = idx[b], y[b]
    groups = np.unique(groups_all[idx], return_inverse=True)[1]
    return idx, y, groups


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run(cfg):
    t_start = time.time()
    validate_config(cfg)
    bm = cfg["benchmark"]
    seed = int(cfg.get("seed", 0))
    workers = int(cfg.get("n_workers", os.cpu_count() or 1))
    go_margin = float(cfg.get("go_margin", 0.05))
    ev = dict(bm["evaluation"])
    ev.setdefault("seed", seed)
    n_boot = int(ev.get("bootstrap", 2000))
    out_dir = Path(cfg.get("out_dir", "results"))
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = bm.get("report_stem", "realdata_benchmark")

    src = Path(cfg["labeling"].get("output", "data/era5/labelled_blocking.npz"))
    if not src.exists():
        raise SystemExit(f"[realdata] {src} not found; run `python -m "
                         "src.realdata.label_blocking` with the same config first")
    lab = load_labelled(src)
    idx, y, groups = select_days(lab, cfg)
    if y.size == 0 or np.unique(y).size < 2:
        raise SystemExit("[realdata] need both BLOCKING and ZONAL days; check the labels report")
    check_groups(y, groups, int(ev["n_splits"]))
    print(f"[realdata] {y.size} fields ({int(y.sum())} BLOCKING, {int((1 - y).sum())} ZONAL) "
          f"in {np.unique(groups).size} groups")

    t0 = time.time()
    fields, crop_shape = preprocess(lab["vorticity"][idx], lab["lat"], lab["lon"], bm)
    dsc = bm.get("descriptor", {})
    topo_ds = int(dsc.get("topo_downsample", 64))
    auth = {k: dsc.get(k, {}) for k in ("x1", "x2")}
    F = compute_features(fields, topo_ds, workers, bm.get("feature_decimals", 8),
                         fullres=fields.shape[1] > topo_ds, rich=True, authentic=auth)
    feat_seconds = time.time() - t0
    print(f"[features] computed in {feat_seconds:.0f}s")

    sets = [k for k in FEATURE_SETS if k in F]
    chance = float(max(y.mean(), 1 - y.mean()))
    task = run_task(PRIMARY, F, y, groups, sets, ev, chance)
    task["verdicts"] = verdicts(task, y, groups, go_margin, n_boot, seed)
    task["union_verdicts"] = union_verdicts(task, y, groups, go_margin, n_boot, seed)
    tasks = {PRIMARY: task}
    leak = {s: evaluate(F[s], y, seed=seed, n_splits=int(ev["n_splits"]))
            for s in LEAKAGE_SETS if s in F}
    print("[stratified-kfold, shuffled days] " + ", ".join(
        f"{s}={v['best_accuracy']:.3f}" for s, v in leak.items()))

    labels_all = lab["labels"]
    dates = lab["dates"][idx]
    P = task["verdicts"]
    us = union_summary(tasks)
    res = {
        "benchmark": "real data: ERA5 500 hPa relative vorticity, winter BLOCKING vs ZONAL "
                     "days (Tibaldi-Molteni 1990 labels)",
        "config": cfg,
        "data": {
            "labelled_file": str(src),
            "spatial_key": str(lab.get("spatial_key", "")),
            "source_files": [str(s) for s in lab.get("source_files", [])],
            "labeling_params": json.loads(str(lab["params"])),
            "n_days_labelled": int(labels_all.size),
            "label_counts_all": {"BLOCKING": int((labels_all == BLOCKING).sum()),
                                 "ZONAL": int((labels_all == ZONAL).sum()),
                                 "AMBIGUOUS": int((labels_all == AMBIGUOUS).sum())},
            "n_used": int(y.size), "n_blocking": int(y.sum()), "n_zonal": int((1 - y).sum()),
            "n_groups": int(np.unique(groups).size),
            "first_day": str(dates.min()), "last_day": str(dates.max()),
            "crop_shape": [int(s) for s in crop_shape], "grid": int(fields.shape[1]),
        },
        "feature_sets": {k: {"tier": v[0], "description": v[1], "dim": int(F[k].shape[1])}
                         for k, v in FEATURE_SETS.items() if k in F},
        "tasks": tasks,
        "stratified_kfold_comparison": {PRIMARY: leak},
        "go_margin": go_margin,
        "verdict_specified": P["specified"]["verdict"],
        "verdict_strict": P["strict"]["verdict"],
        "verdict_overall": P["strict"]["verdict"],
        "verdict_rich_topology": P.get("rich_topology", {}).get("verdict"),
        "verdict_vs_union_any_config": us["verdict"],
        "vs_union_summary": us,
        "timing_seconds": {"features": round(feat_seconds, 1),
                           "total_this_run": round(time.time() - t_start, 1)},
    }
    (out_dir / f"{stem}.json").write_text(json.dumps(_strip_oof(res), indent=2, default=str))
    cache = Path(bm.get("cache_dir", "data/era5"))
    cache.mkdir(parents=True, exist_ok=True)
    (cache / f"{stem}_full.pkl").write_bytes(pickle.dumps(res))
    write_report(out_dir / f"{stem}.md", res)
    if cfg.get("figures", True):
        make_figures(res, fields, y, out_dir / "figures", stem)
    print(f"\n=== {PRIMARY}: vs specified baselines {res['verdict_specified']} (margin "
          f"{P['specified']['margin']:+.3f}); vs strict baselines {res['verdict_strict']} "
          f"(margin {P['strict']['margin']:+.3f} vs `{P['strict']['best_baseline']}`) ===")
    print(f"=== ANY configuration >= +{go_margin:.2f} over baselines_union: {us['verdict']} "
          f"(best `{us['best_config']}`, margin {us['best_margin']:+.3f}) ===")
    return res


# ---------------------------------------------------------------------------
# Report and figures
# ---------------------------------------------------------------------------
def write_report(path, res):
    cfg, d = res["config"], res["data"]
    bm, dl, se = cfg["benchmark"], cfg["download"], cfg["season"]
    lp = d["labeling_params"]
    T = res["tasks"][PRIMARY]
    P = T["verdicts"]
    sp, st = P["specified"], P["strict"]
    us = res["vs_union_summary"]
    gm = res["go_margin"]
    ev = bm["evaluation"]
    g = bm.get("grouping", {})
    gmode = g.get("mode", "season")
    gtext = ("one group per winter season" if gmode == "season" else
             f"blocks of {g.get('block_days', 30)} days inside each season, the last "
             f"{g.get('gap_days', 5)} days of each non-final block dropped")
    dom = bm["domain"]
    sec = lp["sector"]
    c = d["label_counts_all"]
    per = bm.get("periodize", "taper")
    ptext = {"taper": f"mean removed and 2D Tukey taper (alpha = {bm.get('taper_alpha', 0.25)})",
             "mirror": "even mirror reflection of the grid/2 field",
             "none": "none (edge discontinuity kept)"}[per]
    L = ["# Real-data benchmark - ERA5 winter blocking vs zonal flow (GEM Go/No-Go)", "",
         f"## Overall verdict: **{res['verdict_overall']}**", "",
         "Same rules as the synthetic hard benchmark (`results/hard_benchmark.md`): GO iff "
         f"descriptor accuracy - best baseline accuracy >= +{gm:.2f} under grouped CV. Overall = "
         "strict verdict (every baseline, including the structure-aware physical diagnostics). "
         "How to read it: `docs/REALDATA.md`.", "",
         f"**Verdict vs the baselines named in the brief (moments, spectral_slope, raw_pixels): "
         f"{sp['verdict']}** (descriptor {sp['descriptor_accuracy']:.3f} vs "
         f"`{sp['best_baseline']}` {sp['best_baseline_accuracy']:.3f}, margin "
         f"{sp['margin']:+.3f}, 95% CI {_fmt_ci(sp['margin_ci95'])}).", "",
         f"**Strict verdict (every baseline): {st['verdict']}** (descriptor "
         f"{st['descriptor_accuracy']:.3f} vs `{st['best_baseline']}` "
         f"{st['best_baseline_accuracy']:.3f}, margin {st['margin']:+.3f}, 95% CI "
         f"{_fmt_ci(st['margin_ci95'])}).", "",
         f"**Does ANY configuration (every view, authentic X1+X2, richer topology) beat "
         f"`baselines_union` by >= +{gm:.2f}? {'YES' if us['verdict'] == 'GO' else 'NO'} -> "
         f"{us['verdict']}** (best `{us['best_config']}`, margin {us['best_margin']:+.3f}).", "",
         "## Data and labels", "",
         f"- ERA5 reanalysis (Hersbach et al. 2020), CDS dataset `{dl['dataset']}`, 500 hPa "
         f"relative vorticity (`vo`) and geopotential (`z`), {', '.join(dl['times'])} UTC "
         f"(daily mean if several hours), grid {dl.get('grid') or 'native'} deg, winter months "
         f"{se['months']} of seasons {se['first']}-{se['last']} (season = year of its first "
         f"month). Days used: {d['first_day']} .. {d['last_day']}.",
         f"- Labels from 500 hPa geopotential height with the Tibaldi & Molteni (1990) index: "
         f"a longitude is blocked iff GHGS > {lp['ghgs_min']} and GHGN < {lp['ghgn_max']} m/deg "
         f"for some D in {lp['deltas']} (phi_N, phi_0, phi_S = {lp['phi_n']}, {lp['phi_0']}, "
         f"{lp['phi_s']} + D). Sector {sec['west']}..{sec['east']} deg E blocked iff adjacent "
         f"blocked longitudes span >= {lp['min_extent_deg']} deg; BLOCKING = >= "
         f"{lp['min_duration_days']} consecutive sector-blocked days; ZONAL = no blocked "
         f"longitude in the sector and > {lp['zonal_buffer_days']} days from any BLOCKING day"
         + (f", sector-mean GHGS <= {lp['zonal_max_ghgs']}" if lp.get("zonal_max_ghgs")
            is not None else "") + "; AMBIGUOUS days excluded. Details: "
         f"`{cfg['labeling'].get('report_stem', 'realdata_labels')}.md`.",
         f"- All labelled days: {d['n_days_labelled']} (BLOCKING {c['BLOCKING']}, ZONAL "
         f"{c['ZONAL']}, AMBIGUOUS {c['AMBIGUOUS']}). Used: {d['n_used']} fields = "
         f"{d['n_blocking']} BLOCKING + {d['n_zonal']} ZONAL (balance: "
         f"`{bm.get('balance', 'undersample')}`; chance = {T['chance']:.3f}).", "",
         "## Preprocessing and protocol", "",
         f"- Classifier input: relative vorticity only (the labels come from geopotential). "
         f"Domain {dom['south']}..{dom['north']} N, {dom['west']}..{dom['east']} E "
         f"({d['crop_shape'][0]} x {d['crop_shape'][1]} grid points) resampled to "
         f"{d['grid']} x {d['grid']} (anti-aliased bilinear); periodization: {ptext}; then "
         f"standardized to zero mean and unit variance (same function as the synthetic fields).",
         f"- Grouped CV: StratifiedGroupKFold, {ev['n_splits']} folds x {ev['n_repeats']} "
         f"repeats, group = {gtext} ({d['n_groups']} groups); consecutive days never straddle "
         f"train and test. LogReg + RandomForest(300); `best acc` = the better of the two for "
         f"every feature set. Margin CIs: paired bootstrap over groups "
         f"({ev.get('bootstrap', 2000)} resamples).",
         "- Descriptor and baselines: the unchanged feature registry of the synthetic benchmark "
         "(`src/eval/hard_benchmark.py`), with the same fixed hyper-parameters; nothing was tuned "
         "on ERA5.",
         "- Timing (s): " + ", ".join(f"{k} {v:.0f}" for k, v in res["timing_seconds"].items())
         + ".", ""]
    L += _union_section(res)
    if "rich_topology" in P:
        L += ["## Richer topological vectorizations vs `baselines_union`", "",
              f"**Verdict: {P['rich_topology']['verdict']}.** Cells: accuracy (margin vs "
              "`baselines_union` [95% paired group-bootstrap CI]).", ""]
        L += rich_summary_table(res["tasks"], [PRIMARY]) + [""]
    T["_dims"] = {k: v["dim"] for k, v in res["feature_sets"].items()}
    L += ["## Task `blocking_vs_zonal` (primary)", "",
          f"N = {T['n']}, classes = {T['n_classes']}, groups = {T['n_groups']}, "
          f"chance = {T['chance']:.3f}.", ""]
    L += _task_table(T)
    T.pop("_dims", None)
    L += ["", "Verdicts / key comparisons:", ""] + _verdict_lines(P) + [""]
    L += ["## Leakage check: shuffled-day StratifiedKFold vs grouped CV", "",
          "Shuffled days put the neighbours of every test day in the training set. If the "
          "shuffled accuracy is clearly higher than the grouped one, autocorrelation leaks "
          "through day-level splits; only the grouped numbers are used for the verdict.", "",
          "| Feature set | best acc (shuffled days) | best acc (grouped CV) | difference |",
          "|---|---|---|---|"]
    for s, v in res["stratified_kfold_comparison"][PRIMARY].items():
        a, b = v["best_accuracy"], T["results"][s]["best_accuracy"]
        L.append(f"| {s} | {a:.3f} | {b:.3f} | {a - b:+.3f} |")
    stem = Path(path).stem
    L += ["", "## Figures", "",
          f"- `figures/{stem}_fields.png` - mean and example preprocessed vorticity fields per "
          "class.",
          f"- `figures/{stem}_accuracy.png` - grouped-CV accuracy per feature set.",
          f"- `figures/{stem}_margin_vs_union.png` - margin of every non-baseline configuration "
          f"over `baselines_union`, with the +{gm:.2f} GO line.", ""]
    Path(path).write_text("\n".join(L) + "\n")


def fields_figure(fields, y, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from ..eval.hard_figures import SURFACE, INK, VORT_CMAP

    panels = [("mean ZONAL", fields[y == 0].mean(0)), ("mean BLOCKING", fields[y == 1].mean(0)),
              ("example ZONAL day", fields[np.flatnonzero(y == 0)[0]]),
              ("example BLOCKING day", fields[np.flatnonzero(y == 1)[0]])]
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.4), facecolor=SURFACE)
    for ax, (title, f) in zip(axes, panels):
        v = np.abs(f).max() or 1.0
        ax.imshow(f, origin="lower", cmap=VORT_CMAP, vmin=-v, vmax=v)
        ax.set_title(title, fontsize=9, color=INK)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("Preprocessed 500 hPa relative vorticity (standardized; south at the bottom, "
                 "west on the left)", fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def make_figures(res, fields, y, out_dir, stem):
    from ..eval.hard_figures import accuracy_figure, margin_figure

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    fields_figure(fields, y, out / f"{stem}_fields.png")
    accuracy_figure(res["tasks"][PRIMARY], res["feature_sets"], out / f"{stem}_accuracy.png",
                    "ERA5 winter BLOCKING vs ZONAL days (grouped CV)")
    margin_figure(res, out / f"{stem}_margin_vs_union.png")
    print(f"[figures] written to {out}")


def main(argv=None):
    ap = argparse.ArgumentParser(description="ERA5 blocking vs zonal Go/No-Go benchmark.")
    ap.add_argument("--config", default="config/realdata.yaml")
    ap.add_argument("--report-only", action="store_true",
                    help="re-render the Markdown report from the cached full results")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    if args.report_only:
        bm = cfg["benchmark"]
        stem = bm.get("report_stem", "realdata_benchmark")
        res = pickle.loads((Path(bm.get("cache_dir", "data/era5")) / f"{stem}_full.pkl")
                           .read_bytes())
        out = Path(cfg.get("out_dir", "results"))
        (out / f"{stem}.json").write_text(json.dumps(_strip_oof(res), indent=2, default=str))
        write_report(out / f"{stem}.md", res)
        return res
    return run(cfg)


if __name__ == "__main__":
    main()
