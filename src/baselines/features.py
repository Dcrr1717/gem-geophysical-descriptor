"""Baseline feature sets the descriptor must beat for a GO verdict."""
from __future__ import annotations
import numpy as np
from scipy import stats
from ..descriptor.spectral import energy_spectrum


def moment_features(w):
    """Statistical / physical moments of the vorticity field."""
    v = w.ravel()
    return np.array([v.mean(), v.std(), stats.skew(v), stats.kurtosis(v),
                     np.mean(v ** 2), np.mean(np.abs(v))])  # enstrophy proxy = mean(v^2)


def spectral_slope_only(w):
    _, slope = energy_spectrum(w)
    return np.array([slope if np.isfinite(slope) else 0.0])


def raw_downsampled(w, size=16):
    n = w.shape[0]
    step = max(1, n // size)
    return w[::step, ::step].ravel()


BASELINES = {
    "moments": moment_features,
    "spectral_slope": spectral_slope_only,
    "raw_pixels": raw_downsampled,
}


def baseline_matrix(fields, name):
    fn = BASELINES[name]
    return np.vstack([fn(w) for w in fields])
