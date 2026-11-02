import numpy as np

from src.baselines.multiscale import onepoint_multiscale, physical_structure_multiscale
from src.eval.degradation import (HarmonicInfill, block_mask, degrade_pair, quantize,
                                  random_mask, resample, snr_to_eta, swath_mask)
from src.eval.robustness_sweep import bh_qvalues


def _field(n=64, seed=0):
    rng = np.random.default_rng(seed)
    k = np.fft.fftfreq(n) * n
    kx, ky = np.meshgrid(k, k, indexing="ij")
    filt = np.exp(-(kx ** 2 + ky ** 2) / (2 * 5.0 ** 2))
    g = np.real(np.fft.ifft2(np.fft.fft2(rng.standard_normal((n, n))) * filt))
    return (g - g.mean()) / g.std()


def test_snr_to_eta():
    assert np.isclose(snr_to_eta(0), 1.0)
    assert np.isclose(snr_to_eta(20), 0.1)
    assert np.isclose(snr_to_eta(-5), 10 ** 0.25)


def test_mask_fractions_are_exact():
    rng = np.random.default_rng(0)
    assert random_mask((256, 256), 0.2, rng).sum() == round(0.2 * 256 * 256)
    assert np.isclose(swath_mask((256, 256), 0.2, rng).mean(), 6 * 8 / 256)
    assert np.isclose(block_mask((256, 256), 0.4, rng).mean(), 26 * 32 * 32 / 256 ** 2)


def test_swath_mask_is_whole_bands():
    m = swath_mask((64, 64), 0.25, np.random.default_rng(3))
    assert np.all(m.all(axis=0) | ~m.any(axis=0)) or np.all(m.all(axis=1) | ~m.any(axis=1))


def test_harmonic_infill_solves_discrete_laplace_and_keeps_data():
    w = _field()
    m = random_mask(w.shape, 0.3, np.random.default_rng(1))
    f = HarmonicInfill(m)(w)
    np.testing.assert_array_equal(f[~m], w[~m])
    nb = (np.roll(f, 1, 0) + np.roll(f, -1, 0) + np.roll(f, 1, 1) + np.roll(f, -1, 1)) / 4
    np.testing.assert_allclose(f[m], nb[m], atol=1e-10)
    assert np.abs(f[m] - w[m]).mean() < 0.2 * np.abs(w).mean()   # smooth field: close infill


def test_harmonic_infill_large_gaps_are_finite():
    w = _field()
    m = block_mask((64, 64), 0.5, np.random.default_rng(2))
    f = HarmonicInfill(m)(w)
    assert np.all(np.isfinite(f)) and f[m].min() >= w[~m].min() - 1e-9   # maximum principle
    assert f[m].max() <= w[~m].max() + 1e-9


def test_quantize_levels_and_error():
    w = _field()
    q = quantize(w, 3)
    assert len(np.unique(q)) <= 8
    inside = np.abs(w) < 4
    assert np.abs(q - w)[inside].max() <= 0.5 * 8 / 8 + 1e-12


def test_resample_preserves_mean_and_shape():
    w = _field()
    r = resample(w, 4)
    assert r.shape == w.shape and np.isclose(r.mean(), w.mean(), atol=1e-10)
    np.testing.assert_array_equal(resample(w, 1), w)
    x = np.linspace(0, 2 * np.pi, 64, endpoint=False)
    c = np.cos(x)[:, None] * np.ones(64)[None, :]
    assert np.abs(resample(c, 2) - c).max() < 0.05      # large scales survive


def test_pair_degradation_is_identical_for_identical_inputs_except_noise():
    w = _field()
    for model, level in (("random_mask", 0.2), ("swath_gaps", 0.2), ("block_gaps", 0.25),
                         ("quantize", 3), ("resample", 4)):
        a, b, _ = degrade_pair(w, w, model, level, [0, 1, 2])
        np.testing.assert_array_equal(a, b)
    a, b, _ = degrade_pair(w, w, "noise", 0, [0, 1, 2])
    assert not np.allclose(a, b)
    assert np.isclose(np.std(a - w), 1.0, atol=0.05)


def test_multiscale_baseline_shapes():
    w = _field(n=64)
    assert physical_structure_multiscale(w).shape == (100,)
    assert onepoint_multiscale(w).shape == (100,)


def test_bh_qvalues():
    # Benjamini-Hochberg step-up, worked by hand (same as statsmodels' fdr_bh)
    p = [0.01, 0.04, 0.03, 0.5, 0.002]
    np.testing.assert_allclose(bh_qvalues(p), [0.025, 0.05, 0.05, 0.5, 0.01])
