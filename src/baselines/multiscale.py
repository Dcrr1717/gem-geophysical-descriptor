"""Matched-scale control for the multi-scale topological filtration.

`topo_multiscale` (src/descriptor/topology_rich.py) low-passes the field with
a periodic Gaussian at sigma in MS_SCALES = (1, 2, 4, 8) grid cells, resamples
it, and summarizes the persistence of each scale. Under noise, part of its
advantage can come from that built-in denoising rather than from topology.
This module gives the non-topological baselines EXACTLY the same pyramid
(same low-pass, same decimation), so the two can be compared like for like:

* `physical_structure_multiscale` - Okubo-Weiss + gradient + increment
  statistics at every scale of the pyramid (4 x 25 = 100 numbers);
* `onepoint_multiscale` - moments and one-point quantiles of every scale
  (4 x 25 = 100 numbers). Low-passing is linear but the one-point
  distribution of a low-passed field depends on phase structure, so these
  are legitimate structure-sensitive statistics, not leak detectors.

`baselines_union_ms` in the robustness sweep = `baselines_union` (full
resolution) + both blocks above.
"""
from __future__ import annotations

import numpy as np

from ..descriptor.topology_rich import MS_SCALES, _gaussian_lowpass_periodic
from .features import moment_features
from .structure import gradient_stats, histogram_quantiles, increment_flatness, okubo_weiss


def pyramid(w, scales=MS_SCALES):
    """The multi-scale filtration's images: low-pass at sigma = s, keep every s-th pixel."""
    return [_gaussian_lowpass_periodic(w, s)[::s, ::s] for s in scales]


def physical_structure_multiscale(w, scales=MS_SCALES):
    return np.concatenate([np.concatenate([okubo_weiss(img), gradient_stats(img),
                                           increment_flatness(img)])
                           for img in pyramid(w, scales)])


def onepoint_multiscale(w, scales=MS_SCALES):
    return np.concatenate([np.concatenate([moment_features(img), histogram_quantiles(img)])
                           for img in pyramid(w, scales)])
