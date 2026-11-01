"""Spectral view: energy spectrum + normalized graph-Laplacian spectrum."""
from __future__ import annotations
import numpy as np
from numpy.linalg import eigvalsh


def energy_spectrum(w):
    """Radially averaged spectrum and its inertial-range log-log slope."""
    n = w.shape[0]
    wh = np.fft.fft2(w)
    k = np.fft.fftfreq(n) * n
    kx, ky = np.meshgrid(k, k, indexing="ij")
    kr = np.sqrt(kx ** 2 + ky ** 2).ravel()
    power = (np.abs(wh) ** 2).ravel()
    kbins = np.arange(1, n // 2)
    spec = np.zeros(kbins.size, dtype=float)
    for i, kb in enumerate(kbins):
        sel = (kr >= kb - 0.5) & (kr < kb + 0.5)
        if np.any(sel):
            spec[i] = power[sel].mean()
    lo, hi = 2, max(4, n // 8)
    sel = (kbins >= lo) & (kbins <= hi) & (spec > 0)
    slope = np.nan
    if sel.sum() >= 2:
        slope = np.polyfit(np.log(kbins[sel]), np.log(spec[sel]), 1)[0]
    return spec, slope


def laplacian_spectrum_features(w, downsample=16, k=8):
    """Top-k eigenvalues of the normalized Laplacian of a coarse grid graph
    whose edge weights encode vorticity similarity (coherent-structure geometry)."""
    n = w.shape[0]
    step = max(1, n // downsample)
    g = w[::step, ::step]
    m = g.shape[0]
    v = g.ravel()
    N = v.size
    s = np.std(v) + 1e-8
    W = np.zeros((N, N))
    for i in range(m):
        for j in range(m):
            a = i * m + j
            for di, dj in ((1, 0), (0, 1)):
                ii, jj = i + di, j + dj
                if ii < m and jj < m:
                    b = ii * m + jj
                    wgt = np.exp(-((v[a] - v[b]) ** 2) / (2 * s * s))
                    W[a, b] = W[b, a] = wgt
    d = W.sum(1)
    Dm = np.diag(1.0 / np.sqrt(d + 1e-12))
    L = np.eye(N) - Dm @ W @ Dm
    ev = np.sort(eigvalsh(L))[:k]
    if ev.size < k:
        ev = np.pad(ev, (0, k - ev.size))
    return ev


def spectral_features(w):
    spec, slope = energy_spectrum(w)
    lap = laplacian_spectrum_features(w)
    peak_k = float(np.argmax(spec) + 1)
    head = np.array([slope if np.isfinite(slope) else 0.0,
                     np.log(spec.sum() + 1e-12), peak_k])
    return np.concatenate([head, lap])
