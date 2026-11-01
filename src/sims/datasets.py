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
