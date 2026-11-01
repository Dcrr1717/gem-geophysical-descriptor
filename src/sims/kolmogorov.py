"""2D forced-dissipative turbulence (vorticity form) via a pseudo-spectral solver.

Generates labelled snapshots for distinct flow regimes by varying the viscosity
(i.e. the Reynolds number). Periodic box [0, 2*pi)^2. Integrating-factor Euler
time stepping with 2/3-rule dealiasing. NumPy + FFT only.
"""
from __future__ import annotations
import numpy as np


def _wavenumbers(n: int, L: float = 2 * np.pi):
    k = np.fft.fftfreq(n, d=L / n) * 2 * np.pi
    kx, ky = np.meshgrid(k, k, indexing="ij")
    k2 = kx ** 2 + ky ** 2
    k2_nozero = k2.copy()
    k2_nozero[0, 0] = 1.0
    return kx, ky, k2, k2_nozero


def _dealias_mask(n: int):
    k = np.fft.fftfreq(n) * n
    kmax = n / 3.0
    m1 = np.abs(k) <= kmax
    return np.outer(m1, m1)


def simulate(n: int = 64, nu: float = 0.02, forcing_wavenumber: int = 4,
             forcing_amp: float = 0.6, dt: float = 2e-3, n_steps: int = 2500,
             spinup: int = 1200, n_snapshots: int = 10, seed: int = 0):
    """Run the solver and return a list of vorticity fields (physical space).

    Parameters mirror the experiment config. Regimes are obtained by changing
    `nu`: larger nu -> laminar, smaller nu -> turbulent.
    """
    rng = np.random.default_rng(seed)
    kx, ky, k2, k2n = _wavenumbers(n)
    mask = _dealias_mask(n)
    x = np.linspace(0, 2 * np.pi, n, endpoint=False)
    _, Y = np.meshgrid(x, x, indexing="ij")

    # Kolmogorov-type steady forcing applied to the vorticity field.
    f = forcing_amp * forcing_wavenumber * np.cos(forcing_wavenumber * Y)
    fh = np.fft.fft2(f)

    # Smooth random low-wavenumber initial vorticity.
    wh = np.fft.fft2(rng.standard_normal((n, n))) * mask
    wh *= np.exp(-k2 / (2.0 * forcing_wavenumber ** 2))

    E = np.exp(-nu * k2 * dt)  # integrating factor for the viscous term
    usable = max(1, n_steps - spinup)
    stride = max(1, usable // n_snapshots)
    snaps = []
    for step in range(n_steps):
        psih = wh / k2n
        u = np.real(np.fft.ifft2(1j * ky * psih))
        v = np.real(np.fft.ifft2(-1j * kx * psih))
        wx = np.real(np.fft.ifft2(1j * kx * wh))
        wy = np.real(np.fft.ifft2(1j * ky * wh))
        Nh = np.fft.fft2(u * wx + v * wy) * mask            # advection, dealiased
        wh = E * (wh + dt * (-Nh + fh))
        if step >= spinup and (step - spinup) % stride == 0 and len(snaps) < n_snapshots:
            snaps.append(np.real(np.fft.ifft2(wh)).copy())
        if not np.all(np.isfinite(wh)):
            raise FloatingPointError(
                "Simulation diverged; reduce dt / forcing_amp or increase nu.")
    return snaps
