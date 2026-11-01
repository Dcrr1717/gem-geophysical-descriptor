"""Go/No-Go experiment.

Question: does the topological-spectral multiview descriptor separate
geophysical-flow regimes better than physical / statistical baselines?
A GO verdict requires beating the best baseline by at least `go_margin`.
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path

import numpy as np
import yaml

from ..sims.datasets import build_dataset
from ..descriptor.multiview import descriptor_matrix
from ..baselines.features import baseline_matrix, BASELINES
from .classify import evaluate


def run(config):
    t0 = time.time()
    sim = config["simulation"]
    fields, y, names = build_dataset(
        config["regimes"], n=sim["grid"], dt=sim["dt"], n_steps=sim["n_steps"],
        spinup=sim["spinup"], n_snapshots=sim["n_snapshots"],
        forcing_wavenumber=sim["forcing_wavenumber"],
        forcing_amp=sim["forcing_amp"], runs_per_regime=sim.get("runs_per_regime", 1),
        default_nu=sim.get("nu"), seed=config.get("seed", 0))
    print(f"[data] {len(fields)} snapshots across {len(names)} regimes: {names}")

    # Standardize each field (zero mean, unit variance) so that raw amplitude /
    # energy cannot be a trivial shortcut: the task becomes one of structure.
    if config.get("standardize", True):
        fields = [(w - w.mean()) / (w.std() + 1e-8) for w in fields]
        print("[prep] fields standardized (amplitude cue removed)")

    Xd = descriptor_matrix(fields, topo_downsample=config["descriptor"]["topo_downsample"])
    res = {"regimes": names, "n_snapshots": int(len(fields)),
           "descriptor": evaluate(Xd, y, seed=config.get("seed", 0))}
    print(f"[descriptor] best acc = {res['descriptor']['best_accuracy']:.3f}")

    res["baselines"] = {}
    for b in BASELINES:
        res["baselines"][b] = evaluate(baseline_matrix(fields, b), y,
                                       seed=config.get("seed", 0))
        print(f"[baseline:{b}] best acc = {res['baselines'][b]['best_accuracy']:.3f}")

    best_base = max(v["best_accuracy"] for v in res["baselines"].values())
    margin = res["descriptor"]["best_accuracy"] - best_base
    res["best_baseline_accuracy"] = best_base
    res["margin_over_best_baseline"] = margin
    res["go_margin"] = config.get("go_margin", 0.05)
    res["verdict"] = "GO" if margin >= res["go_margin"] else "NO-GO"
    res["elapsed_sec"] = round(time.time() - t0, 1)

    out = Path(config.get("out_dir", "results"))
    out.mkdir(parents=True, exist_ok=True)
    (out / "gonogo.json").write_text(json.dumps(res, indent=2))
    _write_report(out / "gonogo.md", res)

    print(f"\n=== VERDICT: {res['verdict']} "
          f"(descriptor {res['descriptor']['best_accuracy']:.3f} vs best baseline "
          f"{best_base:.3f}, margin {margin:+.3f}) ===")
    return res


def _write_report(path, res):
    L = ["# GEM Go/No-Go - Topological-Spectral Descriptor for Flow Regimes", ""]
    L.append(f"**Verdict: {res['verdict']}** "
             f"(margin {res['margin_over_best_baseline']:+.3f} over the best baseline; "
             f"threshold {res['go_margin']}).")
    L.append("")
    L.append(f"- Regimes: {', '.join(res['regimes'])}")
    L.append(f"- Snapshots: {res['n_snapshots']}")
    L.append(f"- Elapsed: {res['elapsed_sec']} s")
    L.append("")
    L.append("| Feature set | best acc | logreg | rf | silhouette |")
    L.append("|---|---|---|---|---|")
    d = res["descriptor"]
    L.append(f"| **descriptor (ours)** | {d['best_accuracy']:.3f} | "
             f"{d['logreg']['accuracy']:.3f} | {d['rf']['accuracy']:.3f} | {d['silhouette']:.3f} |")
    for b, v in res["baselines"].items():
        L.append(f"| {b} | {v['best_accuracy']:.3f} | {v['logreg']['accuracy']:.3f} | "
                 f"{v['rf']['accuracy']:.3f} | {v['silhouette']:.3f} |")
    path.write_text("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/experiment.yaml")
    args = ap.parse_args()
    run(yaml.safe_load(Path(args.config).read_text()))


if __name__ == "__main__":
    main()
