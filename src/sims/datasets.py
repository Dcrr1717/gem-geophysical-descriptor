"""Build a labelled dataset of vorticity snapshots across flow regimes."""
from __future__ import annotations
import numpy as np
from .kolmogorov import simulate


def build_dataset(regimes, n=64, dt=2e-3, n_steps=2500, spinup=1200,
                  n_snapshots=10, forcing_wavenumber=4, forcing_amp=0.6,
                  runs_per_regime=1, default_nu=None, seed=0):
    """Return (fields, labels, names).

    `regimes` is a list of dicts with a `name` and any of the per-regime
    overrides `nu`, `forcing_wavenumber`, `forcing_amp`. Each regime contributes
    `n_snapshots` fields, optionally spread over `runs_per_regime` independent
    trajectories (different seeds) to reduce temporal correlation so classifiers
    cannot memorize a single trajectory.
    """
    fields, labels, names = [], [], []
    for i, r in enumerate(regimes):
        nu = r.get("nu", default_nu)
        fw = r.get("forcing_wavenumber", forcing_wavenumber)
        fa = r.get("forcing_amp", forcing_amp)
        if nu is None:
            raise KeyError(f"regime '{r.get('name', i)}' needs a 'nu' (or a default)")
        per_run = max(1, n_snapshots // runs_per_regime)
        for run in range(runs_per_regime):
            snaps = simulate(n=n, nu=nu, forcing_wavenumber=fw, forcing_amp=fa,
                             dt=dt, n_steps=n_steps, spinup=spinup,
                             n_snapshots=per_run, seed=seed + 1000 * i + run)
            for s in snaps:
                fields.append(s)
                labels.append(i)
        names.append(r["name"])
    return fields, np.array(labels), names


def simulate_trajectory(job):
    """Worker-friendly wrapper: run one trajectory described by a dict `job`
    (keys = `simulate` kwargs plus bookkeeping keys `regime`, `run`) and return
    (job, snapshots). Used by the hard benchmark to parallelize over runs."""
    kw = {k: v for k, v in job.items() if k not in ("regime", "run", "regime_name")}
    return job, simulate(**kw)


def trajectory_jobs(regimes, n=256, dt=5e-3, nu=1e-3, drag=0.0, scheme="rk4",
                    spinup_time=25.0, snapshot_interval=3.0, snapshots_per_run=10,
                    runs_per_regime=8, forcing_amp=0.6, seed=0):
    """List of independent-trajectory jobs for `simulate_trajectory`.

    Every (regime, run) pair is an independent trajectory with its own random
    initial condition (seed = seed + 1000 * regime_index + run, the same
    convention as `build_dataset`). Snapshots of one trajectory are
    `snapshot_interval` time units apart after `spinup_time` time units of
    spin-up; trajectories are the unit for grouped cross-validation.
    """
    spinup = int(round(spinup_time / dt))
    stride = int(round(snapshot_interval / dt))
    n_steps = spinup + stride * snapshots_per_run
    jobs = []
    for i, r in enumerate(regimes):
        for run in range(runs_per_regime):
            jobs.append(dict(
                regime=i, regime_name=r["name"], run=run,
                n=n, nu=r.get("nu", nu), drag=r.get("drag", drag), scheme=scheme,
                forcing_wavenumber=r["forcing_wavenumber"],
                forcing_amp=r.get("forcing_amp", forcing_amp),
                dt=dt, n_steps=n_steps, spinup=spinup,
                n_snapshots=snapshots_per_run, seed=seed + 1000 * i + run))
    return jobs
