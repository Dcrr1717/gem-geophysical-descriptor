"""Vortex-event fields for a genuinely topological task: merged vs separate vortices.

Two identical co-rotating Gaussian vortices are seeded in a periodic box on
top of weak broadband background vorticity and evolved, unforced, with the
repo's pseudo-spectral IF-RK4 solver (src/sims/kolmogorov.simulate, w0=...).
Co-rotating vortices merge when their separation is below a critical value
(d / a ~ 3.3 for Gaussian cores in inviscid flow: Melander, Zabusky &
McWilliams 1988; Meunier, Le Dizes & Leweke 2005) and keep co-rotating as
two separate cores otherwise. The two classes are set by the initial
separation regime, a physical parameter that no descriptor sees:

  class 0  "merging"  d / a in MERGE_RANGE    -> one coherent core at sampling time
  class 1  "separate" d / a in SEPARATE_RANGE -> two coherent cores at sampling time

Everything else is drawn from the same distributions for both classes:
vortex radius and amplitude, position, orientation, background. Snapshots are
taken in a time window fixed from a descriptor-free pilot
(src/eval/vortex_pilot.py), so that every merging run has merged and no
separate run has. The label depends on the number of coherent cores, a
global connectivity property. A field is standardized (zero mean, unit
variance) before any feature is computed, as in the IAAFT benchmark.
"""
from __future__ import annotations

import numpy as np

from .kolmogorov import simulate


def periodic_gaussian_pair(n, d, a, amp, center, angle):
    """Two Gaussian vortices amp * exp(-r^2 / a^2) at center +- (d/2) e(angle),
    using minimum-image distances on the 2*pi-periodic box (axis 0 = x)."""
    x = np.linspace(0, 2 * np.pi, n, endpoint=False)
    X, Y = np.meshgrid(x, x, indexing="ij")
    w = np.zeros((n, n))
    for s in (-0.5, 0.5):
        cx = center[0] + s * d * np.cos(angle)
        cy = center[1] + s * d * np.sin(angle)
        dx = (X - cx + np.pi) % (2 * np.pi) - np.pi
        dy = (Y - cy + np.pi) % (2 * np.pi) - np.pi
        w += amp * np.exp(-(dx ** 2 + dy ** 2) / a ** 2)
    return w


def background(n, rms, k_band, rng):
    """Random-phase vorticity with flat amplitude in |k| in k_band, given rms."""
    k = np.fft.fftfreq(n) * n
    kx, ky = np.meshgrid(k, k, indexing="ij")
    kk = np.sqrt(kx ** 2 + ky ** 2)
    band = (kk >= k_band[0]) & (kk <= k_band[1])
    g = np.real(np.fft.ifft2(np.fft.fft2(rng.standard_normal((n, n))) * band))
    return rms * g / (g.std() if g.std() > 0 else 1.0)


def initial_condition(n, d_over_a, a, amp, bg_rms, bg_band, rng):
    center = rng.uniform(0, 2 * np.pi, size=2)
    angle = rng.uniform(0, np.pi)
    w = periodic_gaussian_pair(n, d_over_a * a, a, amp, center, angle)
    w += background(n, bg_rms * amp, bg_band, rng)
    return w - w.mean()       # zero total circulation on the torus


def event_jobs(ev, seed=0):
    """Independent trajectories for both classes (the unit of grouped CV).

    `ev` = the `vortex_events` config block. The separation ratio of every run
    is drawn uniformly from its class range with a seeded generator."""
    jobs = []
    for label, key in ((0, "merge_range"), (1, "separate_range")):
        for run in range(int(ev["runs_per_class"])):
            rng = np.random.default_rng(np.random.SeedSequence([seed, 2718, label, run]))
            jobs.append({"label": label, "run": run, "d_over_a": float(rng.uniform(*ev[key])),
                         "seed_words": [seed, 2718, label, run, 1]})
    return jobs


def simulate_event(job, ev):
    """Evolve one seeded vortex pair; returns (job, list of snapshots)."""
    n = int(ev["grid"])
    rng = np.random.default_rng(np.random.SeedSequence(job["seed_words"]))
    w0 = initial_condition(n, job["d_over_a"], ev["radius"], ev["amplitude"],
                           ev["background_rms"], tuple(ev["background_band"]), rng)
    dt = float(ev["dt"])
    spinup = int(round(ev["first_snapshot_time"] / dt))
    stride = int(round(ev["snapshot_interval"] / dt))
    n_snap = int(ev["snapshots_per_run"])
    snaps = simulate(n=n, nu=float(ev["nu"]), forcing_amp=0.0, dt=dt,
                     n_steps=spinup + stride * n_snap, spinup=spinup, n_snapshots=n_snap,
                     drag=float(ev.get("drag", 0.0)), scheme="rk4", w0=w0)
    return job, snaps


def core_count(w, frac=0.5):
    """Validity diagnostic only (never a feature): number of connected regions
    of {w > frac * max w} on the periodic grid."""
    from scipy import ndimage
    m = w > frac * w.max()
    lab, k = ndimage.label(m)
    # merge labels that touch across the periodic boundaries
    parent = list(range(k + 1))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for a_, b_ in ((lab[0, :], lab[-1, :]), (lab[:, 0], lab[:, -1])):
        for p, q in zip(a_, b_):
            if p and q:
                parent[find(p)] = find(q)
    return len({find(i) for i in range(1, k + 1)})
