"""Additional baselines for the structure-only (IAAFT) benchmark.

Two groups:

1. *Validity checks / strongest spectral and one-point baselines.*
   `power_spectrum` (full radially averaged log spectrum over the resolved band)
   and `histogram_quantiles` (quantiles of the one-point distribution). On a
   correct spectrum- and histogram-matched benchmark they must sit at chance;
   if not, the surrogate leaks a non-structural cue.

2. *Structure-aware physical baselines.* Standard, phase-sensitive diagnostics
   of coherent structures in 2D / geophysical turbulence that do NOT use
   topology:
   - `okubo_weiss`: statistics of the Okubo-Weiss parameter
     W = s_n^2 + s_s^2 - omega^2 (Okubo 1970; Weiss 1991), with the usual
     vortex-core threshold W < -0.2 std(W) (Isern-Fontanet et al. 2006).
   - `gradient_stats`: higher-order statistics of vorticity gradients and
     Laplacian (their variances are fixed by the spectrum, their shapes are not).
   - `increment_flatness`: flatness of vorticity increments at separations
     r = 1..32 grid cells along x and y (classical intermittency diagnostic;
     the second-order structure function is fixed by the spectrum, the fourth
     is not).
   These are the baselines a geophysical-fluids reviewer would expect a
   topological descriptor to beat.

All functions take a single 2D periodic field (n x n) and return a 1D vector.
"""
from __future__ import annotations
import numpy as np
from scipy import stats


def _k_grid(n):
    k = np.fft.fftfreq(n) * n
    kx, ky = np.meshgrid(k, k, indexing="ij")
    return kx, ky


def radial_power_spectrum(w, kmax=None):
    """Radially averaged power |FFT(w)|^2 in unit-width shells k = 1..kmax."""
    n = w.shape[0]
    kmax = int(n // 3) if kmax is None else int(kmax)
    kx, ky = _k_grid(n)
    kr = np.rint(np.sqrt(kx ** 2 + ky ** 2)).astype(int).ravel()
    power = (np.abs(np.fft.fft2(w)) ** 2).ravel()
    sums = np.bincount(kr, weights=power, minlength=kmax + 1)
    counts = np.bincount(kr, minlength=kmax + 1)
    spec = sums[1:kmax + 1] / np.maximum(counts[1:kmax + 1], 1)
    return spec


def power_spectrum(w):
    """log10 radial spectrum over the resolved (dealiased) band k <= n/3,
    floored at 1e-12 of the total power so that round-off noise in exactly-zero
    modes cannot act as a cue."""
    spec = radial_power_spectrum(w)
    floor = 1e-12 * spec.sum() + 1e-300
    return np.log10(spec + floor)


_QUANTILES = np.array([0.5, 1, 2.5, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90,
                       95, 97.5, 99, 99.5])


def histogram_quantiles(w):
    """Quantiles of the one-point distribution plus its extremes."""
    v = w.ravel()
    return np.concatenate([np.percentile(v, _QUANTILES), [v.min(), v.max()]])


def _velocity_gradients(w):
    """Velocity-gradient tensor from vorticity (box 2*pi, solver convention:
    psi_hat = omega_hat / k^2, u = d(psi)/dy, v = -d(psi)/dx)."""
    n = w.shape[0]
    kx, ky = _k_grid(n)
    k2 = kx ** 2 + ky ** 2
    k2[0, 0] = 1.0
    wh = np.fft.fft2(w)
    psih = wh / k2
    psih[0, 0] = 0.0

    def d(h):
        return np.real(np.fft.ifft2(h))
    ux = d(-kx * ky * psih)          # d/dx (d psi/dy)  -> (i kx)(i ky) psi
    uy = d(-ky * ky * psih)
    vx = d(kx * kx * psih)           # d/dx (-d psi/dx) -> -(i kx)^2 psi
    vy = d(kx * ky * psih)
    return ux, uy, vx, vy


def okubo_weiss(w, thresh=0.2):
    ux, uy, vx, vy = _velocity_gradients(w)
    sn = ux - vy
    ss = vx + uy
    om = vx - uy                      # equals w up to round-off
    W = sn ** 2 + ss ** 2 - om ** 2
    sW = W.std() + 1e-12
    core = W < -thresh * sW
    strain = W > thresh * sW
    z = om ** 2
    ztot = z.sum() + 1e-12
    return np.array([
        core.mean(),                           # area fraction of vortex cores
        strain.mean(),                         # area fraction of strain cells
        z[core].sum() / ztot,                  # enstrophy fraction in cores
        stats.skew(W.ravel()),
        stats.kurtosis(W.ravel()),
        (W / sW).min(),                        # deepest vortex core (normalized)
    ])


def gradient_stats(w):
    n = w.shape[0]
    kx, ky = _k_grid(n)
    wh = np.fft.fft2(w)
    wx = np.real(np.fft.ifft2(1j * kx * wh))
    wy = np.real(np.fft.ifft2(1j * ky * wh))
    lap = np.real(np.fft.ifft2(-(kx ** 2 + ky ** 2) * wh))
    g2 = wx ** 2 + wy ** 2
    g = np.sqrt(g2)
    return np.array([
        stats.skew(g.ravel()), stats.kurtosis(g.ravel()),
        stats.kurtosis(wx.ravel()), stats.kurtosis(wy.ravel()),
        stats.skew(lap.ravel()), stats.kurtosis(lap.ravel()),
        np.corrcoef((w ** 2).ravel(), g2.ravel())[0, 1],
    ])


_SEPARATIONS = (1, 2, 4, 8, 16, 32)


def increment_flatness(w):
    feats = []
    for axis in (0, 1):
        for r in _SEPARATIONS:
            d = (np.roll(w, -r, axis=axis) - w).ravel()
            m2 = np.mean(d ** 2)
            feats.append(np.mean(d ** 4) / (m2 ** 2 + 1e-300))
    return np.array(feats)


def physical_structure_all(w):
    return np.concatenate([okubo_weiss(w), gradient_stats(w), increment_flatness(w)])


VALIDITY_BASELINES = {
    "power_spectrum": power_spectrum,
    "histogram_quantiles": histogram_quantiles,
}

STRUCTURE_BASELINES = {
    "okubo_weiss": okubo_weiss,
    "gradient_stats": gradient_stats,
    "increment_flatness": increment_flatness,
    "physical_structure_all": physical_structure_all,
}
