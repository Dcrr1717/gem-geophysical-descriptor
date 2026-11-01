"""Richer vectorizations of the topological view (sublevel-set persistent homology).

The original topological view (`topology.py`) compresses each persistence
diagram of a 64x64 image into 8 numbers (count, total, max, entropy for H0 and
H1). This module provides the richer, standard vectorizations from the TDA
literature, all computed with the SAME sublevel-set cubical filtration of the
(standardized) vorticity field:

* `persistence_image_features` - persistence images (Adams et al., JMLR 18,
  2017) of the H0 and H1 diagrams of the full-resolution field, in
  (birth, persistence) coordinates with a linear persistence weight;
* `betti_curve_features` - Betti curves beta_0(t), beta_1(t) on a fixed grid
  of thresholds t (Umeda 2017; Chazal & Michel 2021);
* `multiscale_features` - a multi-scale filtration: the field is low-passed
  with a periodic Gaussian at sigma in {1, 2, 4, 8} grid cells, resampled to
  the matching resolution (256, 128, 64, 32), and each scale's diagrams are
  summarized (8 classic summaries + counts of features above persistence
  thresholds).

All hyper-parameters (ranges, resolutions, kernel widths, thresholds, scales)
are FIXED constants chosen a priori for unit-variance fields, not fitted to
data, so no information can leak across cross-validation folds. Infinite
(essential) bars are dropped from images and summaries and kept in Betti
curves (one essential H0 class per field, identical for all samples).
"""
from __future__ import annotations

import numpy as np
import gudhi

from .topology import _stats

# Persistence images
PI_BIRTH_RANGE = (-4.0, 4.0)
PI_PERS_RANGE = (0.0, 5.0)
PI_RES = 10
PI_SIGMA = 0.35
PI_WEIGHT_SATURATION = 2.0     # weight w(p) = min(p / 2, 1): small (noise) pairs count less
# Betti curves
BETTI_THRESHOLDS = np.linspace(-4.0, 4.0, 33)
# Multi-scale filtration
MS_SCALES = (1, 2, 4, 8)      # Gaussian sigma in grid cells of the 256^2 field
MS_PERSISTENCE_THRESHOLDS = (0.25, 0.5, 1.0, 2.0)


def sublevel_diagrams(img):
    """Sublevel-set cubical persistence of a 2D image -> (H0, H1) arrays of
    (birth, death) pairs; H0 may contain one infinite death."""
    cc = gudhi.CubicalComplex(top_dimensional_cells=np.asarray(img, dtype=float))
    cc.compute_persistence()
    h0 = np.asarray(cc.persistence_intervals_in_dimension(0), dtype=float).reshape(-1, 2)
    h1 = np.asarray(cc.persistence_intervals_in_dimension(1), dtype=float).reshape(-1, 2)
    return h0, h1


def _finite(pairs):
    return pairs[np.isfinite(pairs[:, 1])] if len(pairs) else pairs


def persistence_image(pairs, birth_range=PI_BIRTH_RANGE, pers_range=PI_PERS_RANGE,
                      res=PI_RES, sigma=PI_SIGMA, saturation=PI_WEIGHT_SATURATION):
    """Persistence image (res x res, flattened) of one diagram.

    Each finite pair (b, d) becomes a point (b, p = d - b), weighted by
    min(p / saturation, 1) and spread with an isotropic Gaussian of width
    `sigma`; the surface is evaluated at the pixel centres of a fixed grid.
    Points outside the grid are clipped onto its border so no mass is lost.
    """
    P = _finite(np.asarray(pairs, dtype=float).reshape(-1, 2))
    if len(P) == 0:
        return np.zeros(res * res)
    b = np.clip(P[:, 0], *birth_range)
    p = np.clip(P[:, 1] - P[:, 0], *pers_range)
    w = np.minimum(p / saturation, 1.0)
    bc = np.linspace(*birth_range, res + 1)
    pc = np.linspace(*pers_range, res + 1)
    bc = 0.5 * (bc[:-1] + bc[1:])
    pc = 0.5 * (pc[:-1] + pc[1:])
    gb = np.exp(-((bc[None, :] - b[:, None]) ** 2) / (2 * sigma ** 2))   # (n, res)
    gp = np.exp(-((pc[None, :] - p[:, None]) ** 2) / (2 * sigma ** 2))   # (n, res)
    img = (w[:, None] * gb).T @ gp / (2 * np.pi * sigma ** 2)           # (res_b, res_p)
    return img.ravel()


def betti_curve(pairs, thresholds=BETTI_THRESHOLDS):
    """Number of bars alive at each threshold t: birth <= t < death."""
    P = np.asarray(pairs, dtype=float).reshape(-1, 2)
    if len(P) == 0:
        return np.zeros(len(thresholds))
    t = np.asarray(thresholds)[:, None]
    return ((P[None, :, 0] <= t) & (t < P[None, :, 1])).sum(axis=1).astype(float)


def _gaussian_lowpass_periodic(w, sigma_px):
    n0, n1 = w.shape
    kx = np.fft.fftfreq(n0)[:, None]
    ky = np.fft.rfftfreq(n1)[None, :]
    H = np.exp(-2.0 * (np.pi * sigma_px) ** 2 * (kx ** 2 + ky ** 2))
    return np.fft.irfft2(np.fft.rfft2(w) * H, s=w.shape)


def _scale_summary(h0, h1):
    feats = _stats([d - b for b, d in h0]) + _stats([d - b for b, d in h1])
    for P in (h0, h1):
        pers = (P[:, 1] - P[:, 0]) if len(P) else np.zeros(0)
        pers = pers[np.isfinite(pers)]
        feats += [float((pers > t).sum()) for t in MS_PERSISTENCE_THRESHOLDS]
    return feats


def persistence_image_features(w):
    h0, h1 = sublevel_diagrams(w)
    return np.concatenate([persistence_image(h0), persistence_image(h1)])


def betti_curve_features(w):
    h0, h1 = sublevel_diagrams(w)
    return np.concatenate([betti_curve(h0), betti_curve(h1)])


def multiscale_features(w, scales=MS_SCALES):
    feats = []
    for s in scales:
        img = _gaussian_lowpass_periodic(w, s)[::s, ::s]
        feats += _scale_summary(*sublevel_diagrams(img))
    return np.asarray(feats, dtype=float)


def rich_topology_features(w):
    """All richer topological feature blocks for one field (one PH computation
    of the full-resolution field is shared by the image and the Betti curves)."""
    h0, h1 = sublevel_diagrams(w)
    return {
        "topo_persistence_image": np.concatenate([persistence_image(h0),
                                                  persistence_image(h1)]),
        "topo_betti_curves": np.concatenate([betti_curve(h0), betti_curve(h1)]),
        "topo_multiscale": multiscale_features(w),
    }
