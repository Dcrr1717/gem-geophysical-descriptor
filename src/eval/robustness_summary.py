"""Compact summary of results/noise_robustness.json for docs/NOISE_ROBUSTNESS.md.

One row per (model, level): accuracy of the lead (`topo_multiscale`), of the
best topological configuration, of `baselines_union`, of the matched-scale
control and of the best structure baseline; margins with 95% CIs; the leak
detectors. Prints Markdown.

Usage: python -m src.eval.robustness_summary [results/noise_robustness.json]
"""
from __future__ import annotations

import json
import sys

from .degradation import MODELS
from .robustness_sweep import STRUCT, TOPO


def _m(e):
    return f"{e['margin']:+.3f} [{e['ci95'][0]:+.3f}, {e['ci95'][1]:+.3f}]"


def summary_table(res):
    C = res["conditions"]
    thr = float(res["config"]["robustness"]["rule"]["min_margin"])
    rows = ["| Condition | `topo_multiscale` | best topology config | `baselines_union` | "
            "matched-scale control | best structure baseline | lead − union [95% CI] | "
            "lead − control [95% CI] | best topo − union [95% CI] | leak detectors "
            "(spectrum / quantiles) |",
            "|---|---|---|---|---|---|---|---|---|---|"]
    order = ["clean"] + [c for m in MODELS for c in sorted(
        (k for k, v in C.items() if v["model"] == m), key=lambda k: int(k.rsplit("_", 1)[1]))]
    for c in order:
        R, M = C[c]["results"], C[c]["margins"]
        acc = {k: R[k]["best_accuracy"] for k in R}
        bt = max(TOPO, key=lambda k: (M["baselines_union"][k]["margin"], k == "topo_multiscale"))
        bs = max(STRUCT, key=lambda k: acc[k])
        lead_u = M["baselines_union"]["topo_multiscale"]
        lead_c = M["baselines_union_ms"]["topo_multiscale"]
        star = lambda e: " ★" if (e["margin"] >= thr and e["ci95"][0] > 0
                                  and e.get("q_bh", 1) < 0.05) else ""
        rows.append(
            f"| {C[c]['label']} | {acc['topo_multiscale']:.3f} | `{bt}` {acc[bt]:.3f} | "
            f"{acc['baselines_union']:.3f} | {acc['baselines_union_ms']:.3f} | "
            f"`{bs}` {acc[bs]:.3f} | {_m(lead_u)}{star(lead_u)} | {_m(lead_c)}{star(lead_c)} | "
            f"{_m(M['baselines_union'][bt])}{star(M['baselines_union'][bt])} | "
            f"{acc['power_spectrum']:.3f} / {acc['histogram_quantiles']:.3f} |")
    return rows


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "results/noise_robustness.json"
    res = json.load(open(path))
    print("\n".join(summary_table(res)))
    print()
    print(f"Decision: {res['decision']['verdict']}")


if __name__ == "__main__":
    main()
