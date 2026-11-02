"""Observation-degradation models for the robustness study (docs/NOISE_ROBUSTNESS.md).

Each model maps a clean, standardized 256^2 periodic field to a degraded
observation of the same shape, mimicking a realistic defect of satellite or
reanalysis products. The study applies every model IDENTICALLY to both
classes (coherent field and its IAAFT surrogate):

* the random parts (masks, gap positions) are drawn from a seed that depends
  only on the PAIR index, so a coherent field and its own surrogate lose
  exactly the same pixels;
* observation noise is drawn independently per field (as in
  `noise_sweep.py`), so both classes are matched in expectation.

The linear models (noise, masking + harmonic infill, resolution loss) keep
the two classes matched in expected power spectrum and, for noise, in
one-point distribution. Quantization is a pointwise nonlinearity, so the
degraded classes can differ in spectrum because their phase structure
differs. That is still a structural difference, and the leak-detector
baselines (`power_spectrum`, `histogram_quantiles`) are reported for every
condition so it stays visible.

Models (`MODELS`; levels fixed in config/noise_robustness.yaml):
  noise        additive white Gaussian noise at a given SNR in dB
  random_mask  a fraction of pixels missing at random, harmonic infill
  swath_gaps   whole 8-pixel-wide bands missing (satellite swath gaps),
               harmonic infill; band axis drawn per pair
  block_gaps   32 x 32 blocks missing (cloud-like gaps), harmonic infill
  quantize     uniform quantization to b bits on [-4, 4] field std
  resample     block-mean downsampling by a factor f, periodic bilinear
               upsampling back to the original grid (resolution loss)

Every degraded field is re-standardized (zero mean, unit variance, no
epsilon; see `hard_benchmark._standardize`).
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.ndimage import zoom
from scipy.sparse.linalg import splu

SWATH_WIDTH = 8        # pixels per missing band
BLOCK_SIZE = 32        # side of a missing block
QUANT_RANGE = 4.0      # quantizer covers [-4, 4] field std; values outside are clipped


def snr_to_eta(snr_db):
    """Noise std for a unit-variance field at the given SNR (power ratio in dB)."""
    return float(10.0 ** (-snr_db / 20.0))


def add_noise(w, snr_db, rng):
    return w + snr_to_eta(snr_db) * rng.standard_normal(w.shape)


# ---------------------------------------------------------------------------
# Masks (True = missing)
# ---------------------------------------------------------------------------
def random_mask(shape, fraction, rng):
    n = shape[0] * shape[1]
    k = int(round(fraction * n))
    m = np.zeros(n, bool)
    m[rng.choice(n, size=k, replace=False)] = True
    return m.reshape(shape)


def swath_mask(shape, fraction, rng):
    """Missing bands of SWATH_WIDTH pixels along a random axis, at distinct
    band slots, so the coverage is exactly n_bands * width / size."""
    size = shape[0]
    slots = size // SWATH_WIDTH
    n_bands = max(1, int(round(fraction * slots)))
    chosen = rng.choice(slots, size=n_bands, replace=False)
    m = np.zeros(shape, bool)
    axis = int(rng.integers(2))
    for c in chosen:
        sl = slice(c * SWATH_WIDTH, (c + 1) * SWATH_WIDTH)
        if axis == 0:
            m[sl, :] = True
        else:
            m[:, sl] = True
    return m


def block_mask(shape, fraction, rng):
    """Missing BLOCK_SIZE x BLOCK_SIZE blocks at distinct slots of a regular
    tiling, so the coverage is exactly n_blocks * BLOCK_SIZE^2 / size^2."""
    nb = shape[0] // BLOCK_SIZE
    n_blocks = max(1, int(round(fraction * nb * nb)))
    chosen = rng.choice(nb * nb, size=n_blocks, replace=False)
    m = np.zeros(shape, bool)
    for c in chosen:
        i, j = divmod(int(c), nb)
        m[i * BLOCK_SIZE:(i + 1) * BLOCK_SIZE, j * BLOCK_SIZE:(j + 1) * BLOCK_SIZE] = True
    return m


# ---------------------------------------------------------------------------
# Harmonic (Laplace) infill on the periodic grid
# ---------------------------------------------------------------------------
class HarmonicInfill:
    """Fill missing pixels with the discrete harmonic interpolant: every
    missing value equals the mean of its 4 periodic neighbours, the known
    values are boundary conditions. Parameter-free, linear in the data, and
    the factorization depends only on the mask, so it is shared by a field and
    its surrogate."""

    def __init__(self, mask):
        self.mask = np.asarray(mask, bool)
        n0, n1 = self.mask.shape
        miss = np.flatnonzero(self.mask.ravel())
        self.miss = miss
        idx = -np.ones(n0 * n1, int)
        idx[miss] = np.arange(len(miss))
        r, c = np.divmod(miss, n1)
        rows, cols, vals = [np.arange(len(miss))], [np.arange(len(miss))], [np.full(len(miss), 4.0)]
        self.nbrs = []
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nb = ((r + dr) % n0) * n1 + (c + dc) % n1
            self.nbrs.append(nb)
            unk = idx[nb] >= 0
            rows.append(np.arange(len(miss))[unk])
            cols.append(idx[nb][unk])
            vals.append(-np.ones(unk.sum()))
        A = sp.csc_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                          shape=(len(miss), len(miss)))
        self.lu = splu(A) if len(miss) else None
        self.known_nbr = [~(idx[nb] >= 0) for nb in self.nbrs]

    def __call__(self, w):
        if self.lu is None:
            return np.array(w, float)
        flat = np.array(w, float).ravel()
        rhs = np.zeros(len(self.miss))
        for nb, kn in zip(self.nbrs, self.known_nbr):
            rhs[kn] += flat[nb[kn]]
        flat[self.miss] = self.lu.solve(rhs)
        return flat.reshape(self.mask.shape)


# ---------------------------------------------------------------------------
# Quantization and resolution loss
# ---------------------------------------------------------------------------
def quantize(w, bits, vrange=QUANT_RANGE):
    """Uniform mid-rise quantizer with 2**bits levels on [-vrange, vrange]."""
    L = 2 ** int(bits)
    step = 2.0 * vrange / L
    q = np.clip(np.floor((w + vrange) / step), 0, L - 1)
    return -vrange + (q + 0.5) * step


def resample(w, factor):
    """Block-mean downsampling by `factor`, then periodic bilinear upsampling
    back to the original grid (a coarser sensor footprint, then regridding)."""
    f = int(factor)
    if f == 1:
        return np.array(w, float)
    n0, n1 = w.shape
    coarse = w.reshape(n0 // f, f, n1 // f, f).mean(axis=(1, 3))
    return zoom(coarse, f, order=1, mode="grid-wrap", grid_mode=True)


MASKS = {"random_mask": random_mask, "swath_gaps": swath_mask, "block_gaps": block_mask}
MODELS = ("noise", "random_mask", "swath_gaps", "block_gaps", "quantize", "resample")


def degrade_pair(w_coh, w_sur, model, level, pair_seed_words):
    """Degrade a coherent field and its surrogate identically. Returns the two
    degraded (not yet re-standardized) fields and the achieved missing fraction
    (masking models) or None."""
    rng_pair = np.random.default_rng(np.random.SeedSequence(list(pair_seed_words)))
    if model == "noise":
        rc = np.random.default_rng(np.random.SeedSequence(list(pair_seed_words) + [0]))
        rs = np.random.default_rng(np.random.SeedSequence(list(pair_seed_words) + [1]))
        return add_noise(w_coh, level, rc), add_noise(w_sur, level, rs), None
    if model in MASKS:
        m = MASKS[model](w_coh.shape, level, rng_pair)
        fill = HarmonicInfill(m)
        return fill(w_coh), fill(w_sur), float(m.mean())
    if model == "quantize":
        return quantize(w_coh, level), quantize(w_sur, level), None
    if model == "resample":
        return resample(w_coh, level), resample(w_sur, level), None
    raise ValueError(model)


def level_label(model, level):
    return {"noise": f"{level:+g} dB", "random_mask": f"{100 * level:g}%",
            "swath_gaps": f"{100 * level:g}%", "block_gaps": f"{100 * level:g}%",
            "quantize": f"{level:g} bit", "resample": f"x{level:g}"}[model]
