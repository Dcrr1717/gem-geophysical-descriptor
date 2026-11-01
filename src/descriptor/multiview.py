"""Multiview descriptor = spectral view (+) topological view, concatenated."""
from __future__ import annotations
import numpy as np
from .spectral import spectral_features
from .topology import topology_features


def descriptor(w, topo_downsample=64):
    return np.concatenate([spectral_features(w),
                           topology_features(w, downsample=topo_downsample)])


def descriptor_matrix(fields, topo_downsample=64):
    return np.vstack([descriptor(w, topo_downsample=topo_downsample) for w in fields])
