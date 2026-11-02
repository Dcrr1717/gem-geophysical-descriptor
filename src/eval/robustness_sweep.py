"""Systematic robustness study: topology vs structure baselines on degraded fields.

Pre-registered in config/noise_robustness.yaml and docs/NOISE_ROBUSTNESS.md
(levels, feature sets and decision rule fixed before this script was run).

Question. On clean fields the structure-only benchmark (coherent 2D
turbulence vs IAAFT surrogates) is saturated: `baselines_union` reaches
~1.0, so no descriptor can show a +0.05 margin. docs/RESULTS.md found one
exploratory lead: at 0 dB SNR the multi-scale topological filtration beat
`baselines_union` by +0.044. Does a CI-backed >= +0.05 advantage appear, and
generalize, under realistic observation degradations?

Design.
  * Data: the cached 256^2 benchmark (240 coherent fields, 240 IAAFT
    surrogates, 24 trajectories), exactly as in src/eval/hard_benchmark.py.
  * Degradations (src/eval/degradation.py): noise (SNR sweep), random masking,
    swath gaps, block gaps (all with harmonic infill), quantization,
    resolution loss; 4 levels each, plus the clean reference. Applied
    identically to both classes; masks are shared within each
    coherent/surrogate pair. Fields are re-standardized afterwards.
  * Features: every topological configuration (legacy descriptor, authentic
    X1, X2, X1+X2, X1+X2+richer, 8 summaries at 256^2, persistence images,
    Betti curves, multi-scale filtration, all richer views), the structure
    baselines (Okubo-Weiss, gradient, increment flatness, all three),
    `baselines_union`, and the matched-scale control `baselines_union_ms`
    (src/baselines/multiscale.py). Leak detectors (spectrum, quantiles).
  * Protocol: repeated StratifiedGroupKFold (group = trajectory), best of
    LogReg / RF, paired group-bootstrap 95% CIs (2000 resamples) for the
    margin of every topological configuration over `baselines_union` and over
    `baselines_union_ms`; one-sided bootstrap p-values with Benjamini-Hochberg
    across the whole family of comparisons.

Usage: python -m src.eval.robustness_sweep --config config/noise_robustness.yaml
"""
from __future__ import annotations

import argparse
import json
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import yaml

from ..baselines.multiscale import onepoint_multiscale, physical_structure_multiscale
from .degradation import MODELS, degrade_pair, level_label
from .hard_benchmark import (FEATURE_SETS, _standardize, _strip_oof, compute_features,
                             generate_coherent, generate_surrogates, paired_bootstrap_margin,
                             run_task)

# Topological configurations whose margin is tested ("ours")
TOPO = ["descriptor", "x1x2_rich", "x1x2", "x1_graph", "x2_ph", "topological_view_fullres",
        "topo_persistence_image", "topo_betti_curves", "topo_multiscale", "topo_rich_all"]
STRUCT = ["okubo_weiss", "gradient_stats", "increment_flatness", "physical_structure_all"]
CONTROL = ["physical_structure_ms", "baselines_union_ms"]
VALIDITY = ["power_spectrum", "histogram_quantiles"]
INCR = ["baselines_union+topo_rich", "baselines_union_ms+topo_multiscale"]
SETS = TOPO + STRUCT + ["baselines_union"] + CONTROL + VALIDITY + INCR
LABELS = {
    "descriptor": "full legacy descriptor (spectral + topological view)",
    "x1x2_rich": "full authentic descriptor X1 + X2 + richer topology",
    "x1x2": "authentic X1 + X2",
    "x1_graph": "authentic X1 (functional graph)",
    "x2_ph": "authentic X2 (sig6 PH summaries, 256^2)",
    "topological_view_fullres": "8 PH summaries, 256^2",
    "topo_persistence_image": "persistence images",
    "topo_betti_curves": "Betti curves",
    "topo_multiscale": "multi-scale filtration",
    "topo_rich_all": "all richer topology",
    "okubo_weiss": "Okubo-Weiss", "gradient_stats": "gradient statistics",
    "increment_flatness": "increment flatness",
    "physical_structure_all": "all three structure baselines",
    "baselines_union": "baselines_union (all non-topological baselines)",
    "physical_structure_ms": "structure baselines on the multi-scale pyramid",
    "baselines_union_ms": "baselines_union + pyramid statistics (matched-scale control)",
    "power_spectrum": "leak detector: power spectrum",
    "histogram_quantiles": "leak detector: one-point quantiles",
    "baselines_union+topo_rich": "baselines_union + all richer topology",
    "baselines_union_ms+topo_multiscale": "matched-scale control + multi-scale topology",
}


# ---------------------------------------------------------------------------
# Degraded data
# ---------------------------------------------------------------------------
def _pair_worker(args):
    wc, ws, model, level, seed_words = args
    dc, ds, frac = degrade_pair(wc, ws, model, level, seed_words)
    return _standardize(dc), _standardize(ds), frac


def _ms_worker(w):
    return physical_structure_multiscale(w), onepoint_multiscale(w)


def condition_features(coh, sur, model, level, li, cfg, cache_dir, workers):
    """Feature dict (coherent rows then surrogate rows) for one condition, cached."""
    name = "clean" if model is None else f"{model}_{li}"
    path = Path(cache_dir) / f"features_{name}.npz"
    if path.exists():
        d = np.load(path)
        print(f"[features] {name}: cached", flush=True)
        return {k: d[k] for k in d.files if k != "_missing_fraction"}, (
            float(d["_missing_fraction"]) if "_missing_fraction" in d.files else None)
    t0 = time.time()
    seed = cfg.get("seed", 0)
    if model is None:
        Oc, Os, frac = coh["fields"], sur["fields"], None
    else:
        mid = MODELS.index(model)
        jobs = [(wc, ws, model, level, [seed, 4242, mid, li, i])
                for i, (wc, ws) in enumerate(zip(coh["fields"], sur["fields"]))]
        with Pool(workers) as pool:
            out = pool.map(_pair_worker, jobs, chunksize=4)
        Oc = np.asarray([o[0] for o in out])
        Os = np.asarray([o[1] for o in out])
        fr = [o[2] for o in out if o[2] is not None]
        frac = float(np.mean(fr)) if fr else None
    dec = cfg.get("feature_decimals", 8)
    auth = {k: cfg["descriptor"].get(k, {}) for k in ("x1", "x2")}
    topo_ds = cfg["descriptor"]["topo_downsample"]
    F = {}
    for O in (Oc, Os):
        Fx = compute_features(O, topo_ds, workers, dec, fullres=True, rich=True, authentic=auth)
        with Pool(workers) as pool:
            ms = pool.map(_ms_worker, list(O), chunksize=4)
        Fx["physical_structure_ms"] = np.round(np.nan_to_num(np.vstack([m[0] for m in ms])), dec)
        onept = np.round(np.nan_to_num(np.vstack([m[1] for m in ms])), dec)
        Fx["baselines_union_ms"] = np.hstack([Fx["baselines_union"],
                                              Fx["physical_structure_ms"], onept])
        Fx["baselines_union_ms+topo_multiscale"] = np.hstack([Fx["baselines_union_ms"],
                                                              Fx["topo_multiscale"]])
        Fx = {k: v for k, v in Fx.items() if k in SETS}
        for k, v in Fx.items():
            F.setdefault(k, []).append(v)
    F = {k: np.vstack(v) for k, v in F.items()}
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **F, **({"_missing_fraction": frac} if frac is not None else {}))
    print(f"[features] {name}: {time.time() - t0:.0f}s", flush=True)
    return F, frac


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
def bh_qvalues(p):
    p = np.asarray(p, float)
    n = len(p)
    order = np.argsort(p)
    q = np.empty(n)
    prev = 1.0
    for rank, i in zip(range(n, 0, -1), order[::-1]):
        prev = min(prev, p[i] * n / rank)
        q[i] = prev
    return q


def margins(task, y, g, n_boot, seed):
    R = task["results"]
    out = {}
    for ref in ("baselines_union", "baselines_union_ms"):
        out[ref] = {}
        for c in TOPO + ["baselines_union+topo_rich", "baselines_union_ms+topo_multiscale"]:
            if c == "baselines_union_ms+topo_multiscale" and ref == "baselines_union":
                continue
            m = paired_bootstrap_margin(R[c], R[ref], y, g, n_boot, seed)
            out[ref][c] = {"accuracy": R[c]["best_accuracy"],
                           "reference_accuracy": R[ref]["best_accuracy"],
                           "margin": m["margin"], "ci95": m["ci95"], "p_le_0": m["p_le_0"]}
    return out


def decide(conds, rule):
    """Apply the pre-registered rule (docs/NOISE_ROBUSTNESS.md, 'Decision rule')."""
    thr = float(rule["min_margin"])
    fam = [(c, ref, k) for c in conds if c != "clean"
           for ref in ("baselines_union", "baselines_union_ms")
           for k in conds[c]["margins"][ref]]
    q = bh_qvalues([conds[c]["margins"][ref][k]["p_le_0"] for c, ref, k in fam])
    for (c, ref, k), qq in zip(fam, q):
        conds[c]["margins"][ref][k]["q_bh"] = float(qq)

    def real(e):
        return e["margin"] >= thr and e["ci95"][0] > 0 and e.get("q_bh", 1.0) < 0.05

    model_of = {"swath_gaps": "contiguous_gaps", "block_gaps": "contiguous_gaps"}
    per_config = {}
    for k in TOPO:
        hits, hits_ms, strong = {}, {}, {}
        for c, cd in conds.items():
            if c == "clean":
                continue
            fam_model = model_of.get(cd["model"], cd["model"])
            e = cd["margins"]["baselines_union"][k]
            if real(e):
                hits.setdefault(fam_model, []).append(cd["label"])
                if e["ci95"][0] >= thr:
                    strong.setdefault(fam_model, []).append(cd["label"])
                if real(cd["margins"]["baselines_union_ms"][k]):
                    hits_ms.setdefault(fam_model, []).append(cd["label"])
        per_config[k] = {"advantage_vs_union": hits, "advantage_ci_lower_ge_threshold": strong,
                         "advantage_also_vs_matched_scale_control": hits_ms,
                         "n_models_vs_union": len(hits), "n_models_vs_matched_control": len(hits_ms)}
    lead = rule["lead_config"]
    pc = per_config[lead]
    generalizes = pc["n_models_vs_union"] >= int(rule["generalization_min_models"])
    topo_specific = pc["n_models_vs_matched_control"] >= int(rule["topology_specific_min_models"])
    if generalizes and topo_specific:
        verdict = "GENERALIZES (topology-specific): supports a dedicated paper"
    elif generalizes:
        verdict = ("GENERALIZES vs baselines_union, but NOT against the matched-scale control: "
                   "the advantage is denoising/scale, not topology")
    else:
        verdict = "DOES NOT GENERALIZE"
    return {"per_config": per_config, "lead_config": lead, "generalizes": generalizes,
            "topology_specific": topo_specific, "verdict": verdict,
            "n_comparisons_in_fdr_family": len(fam)}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run(cfg):
    t_start = time.time()
    rc = cfg["robustness"]
    workers = int(cfg.get("n_workers", 4))
    cache_dir = cfg.get("cache_dir", "data/hard_benchmark")
    feat_dir = rc.get("feature_cache_dir", "data/robustness")
    out_dir = Path(cfg.get("out_dir", "results"))
    seed = cfg.get("seed", 0)
    ev = dict(cfg["evaluation"])
    ev.setdefault("seed", seed)
    ev["n_repeats"] = rc.get("n_repeats", ev["n_repeats"])
    n_boot = int(ev.get("bootstrap", 2000))

    coh, _ = generate_coherent(cfg, cache_dir, workers)
    sur = generate_surrogates(coh, cfg, cfg["surrogate"]["final"], cache_dir, workers)
    N = len(coh["fields"])
    y = np.r_[np.zeros(N, int), np.ones(N, int)]
    g = np.r_[coh["group"], coh["group"]]

    plan = [("clean", None, None, None)]
    for model in MODELS:
        for li, level in enumerate(rc["levels"][model]):
            plan.append((f"{model}_{li}", model, level, li))
    conds, dims = {}, {}
    for name, model, level, li in plan:
        F, frac = condition_features(coh, sur, model, level, li, cfg, feat_dir, workers)
        dims = dims or {k: int(F[k].shape[1]) for k in SETS}
        task = run_task(name, F, y, g, SETS, ev, 0.5)
        task["model"] = model or "clean"
        task["level"] = level
        task["label"] = "clean" if model is None else f"{model} {level_label(model, level)}"
        task["missing_fraction"] = frac
        task["margins"] = margins(task, y, g, n_boot, seed)
        conds[name] = task
        _checkpoint(conds, out_dir, rc)
    summary = decide(conds, rc["rule"])
    res = {"study": "robustness of topology vs structure baselines on degraded fields",
           "config": cfg, "sets": {k: LABELS[k] for k in SETS},
           "set_dims": dims,
           "conditions": conds, "decision": summary,
           "seconds": round(time.time() - t_start, 1)}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "noise_robustness.json").write_text(json.dumps(_strip_oof(res), indent=2))
    from .robustness_report import write_report, make_figures
    write_report(res, out_dir / "noise_robustness.md")
    make_figures(res, out_dir / "figures")
    print(f"\n=== {summary['verdict']} ===")
    return res


def _checkpoint(conds, out_dir, rc):
    """Partial results after every condition (resumable, inspectable)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / "noise_robustness_partial.json"
    p.write_text(json.dumps(_strip_oof({"conditions": conds}), indent=1))


def load_config(path):
    """The study config names its base benchmark config (`base_config`); the
    base supplies simulation, surrogate, descriptor and evaluation settings, so
    the cached fields are exactly those of the hard benchmark."""
    cfg = yaml.safe_load(Path(path).read_text())
    if "base_config" in cfg:
        base = yaml.safe_load(Path(cfg["base_config"]).read_text())
        base.update({k: v for k, v in cfg.items() if k != "base_config"})
        cfg = base
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/noise_robustness.yaml")
    args = ap.parse_args()
    run(load_config(args.config))


if __name__ == "__main__":
    main()
