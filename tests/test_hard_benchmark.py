import numpy as np
import pytest

from src.sims.kolmogorov import simulate
from src.sims.surrogate import (iaaft, enforce_hermitian, hermitian_error,
                                spectral_mismatch, histogram_mismatch)
from src.baselines.structure import (okubo_weiss, gradient_stats, increment_flatness,
                                     power_spectrum, histogram_quantiles,
                                     _velocity_gradients)
from src.baselines.features import moment_features
from src.eval.classify import evaluate_grouped
from src.eval.hard_benchmark import _standardize, paired_bootstrap_margin


def _nongaussian_field(n=64, seed=0):
    """Smooth random field pushed through a nonlinearity (skewed histogram)."""
    rng = np.random.default_rng(seed)
    k = np.fft.fftfreq(n) * n
    kx, ky = np.meshgrid(k, k, indexing="ij")
    filt = np.exp(-(kx ** 2 + ky ** 2) / (2 * 6.0 ** 2))
    g = np.real(np.fft.ifft2(np.fft.fft2(rng.standard_normal((n, n))) * filt))
    g = (g - g.mean()) / g.std()
    return np.exp(g) - np.exp(g).mean()


def test_iaaft_amplitude_final_exact_histogram():
    x = _nongaussian_field()
    s, info = iaaft(x, n_iter=100, seed=1, final="amplitude", return_info=True)
    assert s.shape == x.shape and np.isrealobj(s)
    np.testing.assert_array_equal(np.sort(s, axis=None), np.sort(x, axis=None))
    assert info["spectral_mismatch"] < 0.02
    assert info["hermitian_error"] < 1e-12


def test_iaaft_spectrum_final_exact_spectrum():
    x = _nongaussian_field()
    s = iaaft(x, n_iter=100, seed=1, final="spectrum")
    assert spectral_mismatch(x, s) < 1e-10
    assert histogram_mismatch(x, s) < 0.5


def test_iaaft_randomizes_phases_and_is_deterministic():
    x = _nongaussian_field()
    s1 = iaaft(x, n_iter=50, seed=3)
    s2 = iaaft(x, n_iter=50, seed=3)
    s3 = iaaft(x, n_iter=50, seed=4)
    np.testing.assert_array_equal(s1, s2)
    assert not np.array_equal(s1, s3)
    assert abs(np.corrcoef(x.ravel(), s1.ravel())[0, 1]) < 0.3


@pytest.mark.parametrize("n", [16, 17])
def test_enforce_hermitian_gives_real_field(n):
    rng = np.random.default_rng(0)
    H = rng.standard_normal((n, n // 2 + 1)) + 1j * rng.standard_normal((n, n // 2 + 1))
    Hh = enforce_hermitian(H, n, n)
    f = np.fft.irfft2(Hh, s=(n, n))
    np.testing.assert_allclose(np.fft.rfft2(f), Hh, atol=1e-10)
    assert hermitian_error(f) < 1e-12


def test_rk4_consistent_with_euler():
    kw = dict(n=32, nu=0.02, forcing_wavenumber=3, forcing_amp=0.6, n_snapshots=1, seed=2)
    a = simulate(dt=5e-4, n_steps=1000, spinup=999, scheme="euler", **kw)[0]
    b = simulate(dt=5e-4, n_steps=1000, spinup=999, scheme="rk4", **kw)[0]
    assert np.linalg.norm(a - b) / np.linalg.norm(b) < 1e-3


def test_drag_damps_energy():
    kw = dict(n=32, nu=0.02, forcing_wavenumber=3, forcing_amp=0.6, dt=5e-3,
              n_steps=400, spinup=399, n_snapshots=1, seed=0, scheme="rk4")
    w0 = simulate(drag=0.0, **kw)[0]
    w1 = simulate(drag=0.5, **kw)[0]
    assert np.mean(w1 ** 2) < np.mean(w0 ** 2)


def test_velocity_gradients_recover_vorticity():
    w = _nongaussian_field()
    ux, uy, vx, vy = _velocity_gradients(w)
    np.testing.assert_allclose(vx - uy, w - w.mean(), atol=1e-8)
    np.testing.assert_allclose(ux + vy, 0.0, atol=1e-8)


def test_structure_baseline_shapes_finite():
    w = _standardize(_nongaussian_field())
    for fn, d in ((okubo_weiss, 6), (gradient_stats, 7), (increment_flatness, 12),
                  (histogram_quantiles, 19)):
        v = fn(w)
        assert v.shape == (d,) and np.all(np.isfinite(v))
    assert power_spectrum(w).shape == (64 // 3,)


def test_standardize_has_no_amplitude_leak():
    """Moments of standardized fields must not encode the original amplitude
    (after the benchmark's 1e-8 feature rounding)."""
    x = _nongaussian_field()
    a = np.round(moment_features(_standardize(x)), 8)
    b = np.round(moment_features(_standardize(7.3 * x)), 8)
    np.testing.assert_array_equal(a, b)


def test_evaluate_grouped_separable_vs_random():
    rng = np.random.default_rng(0)
    groups = np.repeat(np.arange(12), 10)
    y = np.tile([0, 1], 60)
    X_sep = np.c_[y + 0.05 * rng.standard_normal(120), rng.standard_normal(120)]
    X_rnd = rng.standard_normal((120, 3))
    r_sep = evaluate_grouped(X_sep, y, groups, n_splits=3, n_repeats=2)
    r_rnd = evaluate_grouped(X_rnd, y, groups, n_splits=3, n_repeats=2)
    assert r_sep["best_accuracy"] > 0.95
    assert r_rnd["best_accuracy"] < 0.7
    ci = paired_bootstrap_margin(r_sep, r_rnd, y, groups, n_boot=200)
    assert ci["ci95"][0] > 0.2
