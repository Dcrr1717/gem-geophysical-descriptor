"""2D IAAFT surrogates (Iterative Amplitude Adjusted Fourier Transform).

Given a real 2D field x, an IAAFT surrogate s has (to high accuracy) the same
power spectrum |FFT(x)|^2 AND exactly the same one-point value distribution
(histogram) as x, but randomized Fourier phases. Coherent spatial structure
(vortices, filaments, fronts - everything encoded in the phases) is destroyed,
while every spectral and one-point statistic is preserved.

Reference: Schreiber & Schmitz, "Improved surrogate data for nonlinearity
tests", Phys. Rev. Lett. 77 (1996) 635; 2D/image extension as in Venema et al.,
Nonlin. Processes Geophys. 13 (2006) 321.

Implementation notes
--------------------
* Works on the rfft2 half spectrum. Hermitian symmetry (needed for a real
  field) is enforced explicitly after every spectral step: the self-conjugate
  columns ky = 0 and ky = n/2 are symmetrized in kx and the four
  self-conjugate points are made real (see `enforce_hermitian`).
* Iteration (Schreiber & Schmitz): start from a random permutation of the
  values; repeat (a) impose the target Fourier amplitudes keeping the current
  phases, (b) rank-order remap onto the original sorted values. Stops early if
  the rank order no longer changes.
* ``final="amplitude"`` (default) returns the output of step (b): the histogram
  is exact and the spectrum is matched up to a small residual.
  ``final="spectrum"`` returns the output of step (a): the spectrum is exact
  (including any exactly-zero, e.g. dealiased, modes) and the histogram is
  matched up to a small residual.
"""
from __future__ import annotations
import numpy as np


def enforce_hermitian(H: np.ndarray, n0: int, n1: int) -> np.ndarray:
    """Make an rfft2-layout half spectrum H (shape n0 x (n1//2+1)) exactly the
    half spectrum of a real n0 x n1 field.

    For rfft2 only the ky=0 column (and the ky=n1/2 Nyquist column when n1 is
    even) carry redundant information: they must satisfy
    H[-kx, c] = conj(H[kx, c]). Self-conjugate points (kx in {0, n0/2}) are
    therefore real.
    """
    H = H.copy()
    cols = [0]
    if n1 % 2 == 0:
        cols.append(n1 // 2)
    neg = (-np.arange(n0)) % n0
    for c in cols:
        col = H[:, c]
        H[:, c] = 0.5 * (col + np.conj(col[neg]))
    return H


def hermitian_error(field: np.ndarray) -> float:
    """Relative violation of Hermitian symmetry of the full FFT of `field`
    (0 for any real field up to round-off). Diagnostic only."""
    F = np.fft.fft2(field)
    Fc = np.conj(np.roll(np.flip(F, axis=(0, 1)), shift=(1, 1), axis=(0, 1)))
    return float(np.linalg.norm(F - Fc) / (np.linalg.norm(F) + 1e-300))


def iaaft(x, n_iter: int = 100, seed=None, final: str = "amplitude",
          return_info: bool = False):
    """IAAFT surrogate of a real 2D field.

    Parameters
    ----------
    x : (n0, n1) array_like, real.
    n_iter : maximum number of iterations (~100 is standard).
    seed : int or np.random.Generator for the random initial permutation.
    final : "amplitude" (exact histogram) or "spectrum" (exact spectrum).
    return_info : also return a dict with iteration diagnostics.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 2:
        raise ValueError("iaaft expects a 2D field")
    if final not in ("amplitude", "spectrum"):
        raise ValueError("final must be 'amplitude' or 'spectrum'")
    rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
    n0, n1 = x.shape
    shape = x.shape

    sorted_vals = np.sort(x, axis=None)
    target_amp = np.abs(np.fft.rfft2(x))

    # Random initial condition: a random permutation of the original values
    # (random phases, exact histogram).
    s_amp = rng.permutation(x.ravel()).reshape(shape)
    prev_rank = None
    n_done = 0
    converged = False
    spec_err_hist = []
    for it in range(n_iter):
        # (a) impose target Fourier amplitudes, keep current phases
        S = np.fft.rfft2(s_amp)
        mag = np.abs(S)
        phase = np.where(mag > 0, S / np.where(mag > 0, mag, 1.0), 1.0)
        H = enforce_hermitian(target_amp * phase, n0, n1)
        s_spec = np.fft.irfft2(H, s=shape)
        # (b) rank-order remap onto the original value distribution
        rank = np.argsort(s_spec, axis=None, kind="stable")
        s_amp = np.empty(x.size)
        s_amp[rank] = sorted_vals
        s_amp = s_amp.reshape(shape)
        n_done = it + 1
        if return_info:
            spec_err_hist.append(spectral_mismatch(x, s_amp))
        if prev_rank is not None and np.array_equal(rank, prev_rank):
            converged = True
            break
        prev_rank = rank

    out = s_amp if final == "amplitude" else s_spec
    if not return_info:
        return out
    info = {"iterations": n_done, "converged": converged, "final": final,
            "spectral_mismatch": spectral_mismatch(x, out),
            "histogram_mismatch": histogram_mismatch(x, out),
            "hermitian_error": hermitian_error(out),
            "spectral_mismatch_history": spec_err_hist}
    return out, info


def spectral_mismatch(x, s) -> float:
    """Relative L2 difference of the 2D Fourier amplitude |FFT| of s vs x."""
    a = np.abs(np.fft.rfft2(x))
    b = np.abs(np.fft.rfft2(s))
    return float(np.linalg.norm(a - b) / (np.linalg.norm(a) + 1e-300))


def histogram_mismatch(x, s) -> float:
    """Max absolute difference of sorted values, in units of std(x)."""
    return float(np.max(np.abs(np.sort(x, axis=None) - np.sort(s, axis=None)))
                 / (np.std(x) + 1e-300))
