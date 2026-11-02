"""Euler Characteristic Transform (ECT) of the sublevel sets of a 2D field.

The Euler Characteristic Transform of a shape K in R^2 (Turner, Mukherjee &
Boyer 2014; Munch 2023) is the function

    ECT_K(v, t) = chi( K  intersect  { x : <x, v> <= t } ),   v in S^1, t in R,

i.e. the Euler characteristic curve of the height filtration in every
direction v. Unlike a handful of persistence summaries, the ECT is injective
on constructible shapes: two shapes with the same ECT are equal.

A vorticity field is not a shape, so we apply the ECT to its sublevel sets:
for each level h in ECT_LEVELS the shape is K_h = {x : f(x) <= h} of the
standardized field, represented as a cubical complex (every pixel is a
closed unit square with its edges and vertices). The descriptor stacks
ECT_{K_h}(v_j, t_k) over levels h, directions v_j (uniform on the circle) and
thresholds t_k into an image of shape (levels, directions, thresholds).

Implementation. A cell (vertex, edge, square) enters the height filtration
in direction v at the maximum height of its vertices (lower-star
filtration), so

    chi(K_h, v, t) = sum over cells c of K_h of (-1)^dim(c) * [height_v(c) <= t].

The cell heights do not depend on the field, only on the grid, so for a
fixed grid the whole ECT is one matrix product: (cell-membership masks of the
sublevel sets) @ (signed indicator matrix of [height_v(c) <= t]). The
differentiable version (src/descriptor/dect.py) replaces the indicator by a
sigmoid with learnable directions and thresholds.

All constants are fixed a priori for unit-variance fields; nothing is fitted.
The field is block-averaged to ECT_RES^2 pixels (2 x 2 blocks for the 256^2
benchmark) to keep the complex at ~66k cells.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np

ECT_RES = 128
ECT_LEVELS = (-1.5, -0.75, 0.0, 0.75, 1.5)   # sublevel sets {f <= h}, field-std units
ECT_DIRECTIONS = 16                          # uniform on [0, 2*pi)
ECT_THRESHOLDS = 24                          # uniform on [-T_MAX, T_MAX]
T_MAX = 1.42                                 # coordinates in [-1, 1]^2 -> |<x, v>| <= sqrt(2)


def directions(n_dirs=ECT_DIRECTIONS):
    return np.linspace(0.0, 2 * np.pi, n_dirs, endpoint=False)


def thresholds(n_t=ECT_THRESHOLDS):
    return np.linspace(-T_MAX, T_MAX, n_t)


@lru_cache(maxsize=4)
def grid_cells(n):
    """Cells of the full n x n cubical grid in a fixed order
    [vertices, horizontal edges, vertical edges, squares].

    Returns (corners, sign): `corners` has shape (n_cells, 4, 2) with the (x, y)
    coordinates, in [-1, 1], of the vertices of every cell (repeated to 4 for
    vertices and edges), and `sign` = (-1)^dim(cell).
    Pixel (i, j) (row i, column j) is the square [j, j+1] x [i, i+1]."""
    c = (np.arange(n + 1) - n / 2.0) / (n / 2.0)            # vertex coordinates
    I, J = np.meshgrid(np.arange(n + 1), np.arange(n + 1), indexing="ij")

    def xy(i, j):
        return np.stack([c[j], c[i]], axis=-1)

    v = xy(I, J).reshape(-1, 1, 2).repeat(4, axis=1)
    hi, hj = np.meshgrid(np.arange(n + 1), np.arange(n), indexing="ij")
    he = np.stack([xy(hi, hj), xy(hi, hj + 1)], axis=-2).reshape(-1, 2, 2)
    he = np.concatenate([he, he], axis=1)
    vi, vj = np.meshgrid(np.arange(n), np.arange(n + 1), indexing="ij")
    ve = np.stack([xy(vi, vj), xy(vi + 1, vj)], axis=-2).reshape(-1, 2, 2)
    ve = np.concatenate([ve, ve], axis=1)
    fi, fj = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    sq = np.stack([xy(fi, fj), xy(fi, fj + 1), xy(fi + 1, fj), xy(fi + 1, fj + 1)],
                  axis=-2).reshape(-1, 4, 2)
    corners = np.concatenate([v, he, ve, sq]).astype(np.float32)
    sign = np.concatenate([np.ones(len(v)), -np.ones(len(he) + len(ve)),
                           np.ones(len(sq))]).astype(np.float32)
    return corners, sign


def cell_masks(pixels):
    """Membership of every grid cell in the cubical complex of a binary pixel
    set (a pixel brings its closed square: its 4 edges and 4 vertices).
    Order matches `grid_cells`."""
    P = np.asarray(pixels, bool)
    n = P.shape[0]
    Pp = np.zeros((n + 2, n + 2), bool)
    Pp[1:-1, 1:-1] = P
    vert = Pp[:-1, :-1] | Pp[:-1, 1:] | Pp[1:, :-1] | Pp[1:, 1:]          # (n+1, n+1)
    hedge = Pp[:-1, 1:-1] | Pp[1:, 1:-1]                                  # (n+1, n)
    vedge = Pp[1:-1, :-1] | Pp[1:-1, 1:]                                  # (n, n+1)
    return np.concatenate([vert.ravel(), hedge.ravel(), vedge.ravel(), P.ravel()])


@lru_cache(maxsize=4)
def _indicator(n, n_dirs, n_t):
    """Signed indicator matrix (n_cells, n_dirs * n_t): sign(c) * [height_v(c) <= t]."""
    corners, sign = grid_cells(n)
    th = directions(n_dirs)
    V = np.stack([np.cos(th), np.sin(th)], axis=0).astype(np.float32)   # (2, D)
    H = (corners @ V).max(axis=1)                                         # (n_cells, D)
    t = thresholds(n_t).astype(np.float32)
    ind = (H[:, :, None] <= t[None, None, :] + 1e-6).astype(np.float32)
    return (ind * sign[:, None, None]).reshape(len(sign), n_dirs * n_t)


def downsample(w, res=ECT_RES):
    n = w.shape[0]
    f = max(1, n // res)
    if f == 1:
        return np.asarray(w, float)
    return np.asarray(w, float).reshape(n // f, f, n // f, f).mean(axis=(1, 3))


def level_masks(w, levels=ECT_LEVELS, res=ECT_RES):
    """(n_levels, n_cells) uint8 cell masks of the sublevel sets of the
    block-averaged, re-standardized field."""
    img = downsample(w, res)
    img = (img - img.mean()) / (img.std() if img.std() > 0 else 1.0)
    return np.stack([cell_masks(img <= h) for h in levels]).astype(np.uint8)


def ect_image(w, levels=ECT_LEVELS, n_dirs=ECT_DIRECTIONS, n_t=ECT_THRESHOLDS, res=ECT_RES):
    """ECT image of shape (n_levels, n_dirs, n_t) (integer Euler characteristics)."""
    M = level_masks(w, levels, res).astype(np.float32)
    n = _grid_size_from_cells(M.shape[1])
    out = M @ _indicator(n, n_dirs, n_t)
    return np.rint(out).reshape(len(levels), n_dirs, n_t)


def _grid_size_from_cells(n_cells):
    # n_cells = (n+1)^2 + 2 n (n+1) + n^2 = 4 n^2 + 4 n + 1 = (2n + 1)^2
    return int(round((np.sqrt(n_cells) - 1) / 2))


def ect_features(w):
    """Flattened ECT image (the `ect` feature view)."""
    return ect_image(w).ravel().astype(float)


def euler_characteristic(pixels):
    """chi of the cubical complex of a binary pixel set (no filtration)."""
    m = cell_masks(pixels)
    n = np.asarray(pixels).shape[0]
    _, sign = grid_cells(n)
    return int(round(float((m * sign).sum())))
