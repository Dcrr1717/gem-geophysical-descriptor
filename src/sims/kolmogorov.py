"""2D forced-dissipative turbulence (vorticity form) via a pseudo-spectral solver.

Generates labelled snapshots for distinct flow regimes by varying the viscosity
(i.e. the Reynolds number) or the forcing scale. Periodic box [0, 2*pi)^2 with
2/3-rule dealiasing. NumPy + FFT only.

Two time integrators are available:

* ``scheme="euler"`` (default, original behaviour): integrating-factor forward
  Euler. Cheap, but forward Euler is unconditionally unstable for pure
  advection, so it only works when viscosity damps the highest resolved
  wavenumbers (small grids / large ``nu``).
* ``scheme="rk4"``: integrating-factor (Lawson) fourth-order Runge-Kutta on
  real FFTs. Its stability region contains the imaginary axis up to |z| ~ 2.8,
  so it is stable for well-resolved turbulence at 256^2 and small ``nu``.
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


def _rfft_wavenumbers(n: int):
    """Wavenumbers / dealias mask for the half-spectrum layout of np.fft.rfft2
    (axis 0 = x, full; axis 1 = y, non-negative frequencies only)."""
    kxf = np.fft.fftfreq(n) * n
    kyf = np.fft.rfftfreq(n) * n
    kx, ky = np.meshgrid(kxf, kyf, indexing="ij")
    k2 = kx ** 2 + ky ** 2
    k2n = k2.copy()
    k2n[0, 0] = 1.0
    mask = np.outer(np.abs(kxf) <= n / 3.0, np.abs(kyf) <= n / 3.0)
    return kx, ky, k2, k2n, mask


def simulate(n: int = 64, nu: float = 0.02, forcing_wavenumber: int = 4,
             forcing_amp: float = 0.6, dt: float = 2e-3, n_steps: int = 2500,
             spinup: int = 1200, n_snapshots: int = 10, seed: int = 0,
             drag: float = 0.0, scheme: str = "euler"):
    """Run the solver and return a list of vorticity fields (physical space).

    Parameters mirror the experiment config. Regimes are obtained by changing
    `nu`: larger nu -> laminar, smaller nu -> turbulent.

    `drag` is an optional linear (Ekman / bottom-friction) damping rate `mu`
    acting on all modes, d(omega)/dt = ... - mu * omega. It arrests the inverse
    energy cascade so that long runs reach a statistically stationary state, as
    in standard forced 2D geophysical turbulence. The default 0.0 reproduces the
    original drag-free solver exactly.

    `scheme` selects the time integrator ("euler" = original, "rk4" = stable
    integrating-factor RK4, see module docstring). Snapshots are taken every
    `(n_steps - spinup) // n_snapshots` steps starting at step `spinup`.
    """
    if scheme == "rk4":
        return _simulate_ifrk4(n=n, nu=nu, forcing_wavenumber=forcing_wavenumber,
                               forcing_amp=forcing_amp, dt=dt, n_steps=n_steps,
                               spinup=spinup, n_snapshots=n_snapshots, seed=seed,
                               drag=drag)
    if scheme != "euler":
        raise ValueError(f"unknown scheme '{scheme}' (use 'euler' or 'rk4')")

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

    E = np.exp(-(nu * k2 + drag) * dt)  # integrating factor: viscosity + linear drag
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


def _simulate_ifrk4(n, nu, forcing_wavenumber, forcing_amp, dt, n_steps, spinup,
                    n_snapshots, seed, drag):
    """Integrating-factor (Lawson) RK4 on the rfft2 half spectrum.

    Same equations, forcing, initial condition (identical random draw) and
    snapshot schedule as the Euler path; only the time integrator differs.
    """
    rng = np.random.default_rng(seed)
    kx, ky, k2, k2n, mask = _rfft_wavenumbers(n)
    shape = (n, n)
    x = np.linspace(0, 2 * np.pi, n, endpoint=False)
    _, Y = np.meshgrid(x, x, indexing="ij")
    fh = np.fft.rfft2(forcing_amp * forcing_wavenumber * np.cos(forcing_wavenumber * Y))

    wh = np.fft.rfft2(rng.standard_normal(shape)) * mask
    wh *= np.exp(-k2 / (2.0 * forcing_wavenumber ** 2))

    ikx, iky = 1j * kx, 1j * ky

    def rhs(w):
        psih = w / k2n
        u = np.fft.irfft2(iky * psih, s=shape)
        v = np.fft.irfft2(-ikx * psih, s=shape)
        wx = np.fft.irfft2(ikx * w, s=shape)
        wy = np.fft.irfft2(iky * w, s=shape)
        return -np.fft.rfft2(u * wx + v * wy) * mask + fh

    lin = -(nu * k2 + drag)
    E = np.exp(lin * dt)
    E2 = np.exp(lin * dt / 2.0)
    usable = max(1, n_steps - spinup)
    stride = max(1, usable // n_snapshots)
    snaps = []
    for step in range(n_steps):
        k1 = dt * rhs(wh)
        k2_ = dt * rhs(E2 * (wh + 0.5 * k1))
        k3 = dt * rhs(E2 * wh + 0.5 * k2_)
        k4 = dt * rhs(E * wh + E2 * k3)
        wh = E * wh + (E * k1 + 2.0 * E2 * (k2_ + k3) + k4) / 6.0
        if step >= spinup and (step - spinup) % stride == 0 and len(snaps) < n_snapshots:
            snaps.append(np.fft.irfft2(wh, s=shape).copy())
        if step % 50 == 0 and not np.all(np.isfinite(wh)):
            raise FloatingPointError(
                "Simulation diverged; reduce dt / forcing_amp or increase nu.")
    if not np.all(np.isfinite(wh)):
        raise FloatingPointError("Simulation diverged; reduce dt / forcing_amp or increase nu.")
    return snaps
