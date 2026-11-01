"""Topological view: sublevel-set persistent homology of the vorticity field
(treated as a grayscale image). Uses GUDHI when available, with a light fallback."""
from __future__ import annotations
import numpy as np

try:
    import gudhi as _gudhi
    _HAS_GUDHI = True
except Exception:
    _HAS_GUDHI = False


def _stats(pers):
    pers = np.asarray([p for p in pers if np.isfinite(p)], dtype=float)
    if pers.size == 0:
        return [0.0, 0.0, 0.0, 0.0]
    prob = pers / (pers.sum() + 1e-12)
    entropy = float(-(prob * np.log(prob + 1e-12)).sum())
    return [float(pers.size), float(pers.sum()), float(pers.max()), entropy]


def topology_features(w, downsample=64):
    n = w.shape[0]
    step = max(1, n // downsample)
    img = w[::step, ::step].astype(float)
    if _HAS_GUDHI:
        cc = _gudhi.CubicalComplex(top_dimensional_cells=img)
        cc.compute_persistence()
        h0 = [d - b for (b, d) in cc.persistence_intervals_in_dimension(0)]
        h1 = [d - b for (b, d) in cc.persistence_intervals_in_dimension(1)]
        return np.array(_stats(h0) + _stats(h1), dtype=float)
    # Fallback without GUDHI: connected-component counts at vorticity quantiles.
    from scipy import ndimage
    feats = []
    for q in (25, 50, 75):
        t = np.percentile(img, q)
        _, ncomp = ndimage.label(img <= t)
        feats.append(float(ncomp))
    feats.append(float(img.std()))
    return np.array(feats, dtype=float)
