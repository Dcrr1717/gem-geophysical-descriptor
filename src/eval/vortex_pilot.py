"""Descriptor-free pilot for the vortex-event task.

Evolves seeded co-rotating vortex pairs over a grid of separation ratios d/a
and records, once per time unit, the number of coherent cores (connected
regions of {w > 0.5 max w}, a validity diagnostic only). It is used once,
before any classification, to fix the two class ranges of d/a and the
snapshot window: every "merging" run must have merged (one core) and no
"separate" run may have merged (two cores) inside the window.

Usage: python -m src.eval.vortex_pilot --config config/ect_vortex.yaml
Writes results/vortex_pilot.json and results/figures/vortex_pilot.png.
"""
from __future__ import annotations

import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import yaml

from ..sims.kolmogorov import simulate
from ..sims.vortex_events import core_count, initial_condition


def _run(args):
    ev, ratio, rep, T = args
    rng = np.random.default_rng(np.random.SeedSequence([99, 31, int(ratio * 100), rep]))
    w0 = initial_condition(int(ev["grid"]), ratio, ev["radius"], ev["amplitude"],
                           ev["background_rms"], tuple(ev["background_band"]), rng)
    dt = float(ev["dt"])
    steps = int(round(T / dt))
    snaps = simulate(n=int(ev["grid"]), nu=float(ev["nu"]), forcing_amp=0.0, dt=dt,
                     n_steps=steps, spinup=0, n_snapshots=int(T),
                     drag=float(ev.get("drag", 0.0)), scheme="rk4", w0=w0)
    return {"d_over_a": ratio, "rep": rep, "cores": [core_count(s) for s in snaps],
            "peak": [float(s.max()) for s in snaps]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/ect_vortex.yaml")
    a = ap.parse_args()
    cfg = yaml.safe_load(Path(a.config).read_text())
    ev, pl = cfg["vortex_events"], cfg["vortex_pilot"]
    jobs = [(ev, float(r), rep, float(pl["duration"])) for r in pl["ratios"]
            for rep in range(int(pl["repeats"]))]
    with Pool(int(cfg.get("n_workers", 4))) as pool:
        out = pool.map(_run, jobs)
    res = {"note": "descriptor-free pilot; core count = components of {w > 0.5 max w}",
           "vortex_events": ev, "pilot": pl, "runs": out}
    od = Path(cfg.get("out_dir", "results"))
    (od / "figures").mkdir(parents=True, exist_ok=True)
    (od / "vortex_pilot.json").write_text(json.dumps(res, indent=1))
    for r in out:
        print(f"d/a={r['d_over_a']:.2f} rep={r['rep']}: cores " + "".join(
            str(min(c, 9)) for c in r["cores"]))
    _figure(res, od / "figures" / "vortex_pilot.png")


def _figure(res, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from .hard_figures import INK, INK2, SURFACE, _style
    fig, ax = plt.subplots(figsize=(8, 3.6), facecolor=SURFACE)
    _style(ax)
    ratios = sorted({r["d_over_a"] for r in res["runs"]})
    cmap = plt.get_cmap("viridis")
    for i, q in enumerate(ratios):
        runs = [r for r in res["runs"] if r["d_over_a"] == q]
        m = np.mean([r["cores"] for r in runs], axis=0)
        ax.plot(np.arange(len(m)), m, color=cmap(i / max(1, len(ratios) - 1)), lw=1.8,
                label=f"d/a = {q:g}")
    ev = res["vortex_events"]
    t0 = ev["first_snapshot_time"]
    t1 = t0 + ev["snapshot_interval"] * (ev["snapshots_per_run"] - 1)
    ax.axvspan(t0, t1, color=INK2, alpha=0.08, lw=0)
    ax.set_xlabel("time", fontsize=9, color=INK2)
    ax.set_ylabel("coherent cores (mean over runs)", fontsize=9, color=INK2)
    ax.set_title("Pilot: co-rotating vortex pairs, merged (1 core) vs separate (2 cores); "
                 "shaded = snapshot window", fontsize=10, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


if __name__ == "__main__":
    main()
