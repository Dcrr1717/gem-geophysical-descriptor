"""Authentic multiview descriptor views X1 / X2 / X3 (verbatim port).

Source: https://github.com/Dcrr1717/multiview-nn-descriptor, branch
`claude/practical-mccarthy-0tj8if` (commit 675269d), file
`topo-spectral-geofluids/tsg/descriptors.py`. That module is itself a direct
port of the neural-activation descriptor of Reinoso & Escudero (2026)
(`desc_x1`, `desc_x2`, `desc_x3`). The function bodies below are copied
unchanged; only this docstring and the comments marked "[GEM port]" are new.
The Vietoris-Rips variant of X2 (`x2_rips_pointcloud`, a sensitivity option
in the source that needs `ripser`) is not ported.

In the network study X1 was built from an activation matrix (samples x units).
Here the units are coarse regions of the mesh (8 x 8 = 64 patches, the
analogue of the 80-neuron cap). The samples are either

* SNAPSHOT mode (one field, the source's PRIMARY mode): the pixels of each
  patch in local coordinates, so an edge means two regions carry coherent
  spatial patterns (`patch_matrix` -> `graph_invariants`, wrapped as
  `x1_patch_coherence`);
* WINDOW mode (a T x H x W space-time window, the source's SECONDARY mode):
  the time instants, so an edge means two regions co-vary in time, a
  "climate network" (`region_series` -> `graph_invariants`, wrapped as
  `x1_functional_graph`).

[GEM port] The hard benchmark classifies single snapshots against
per-snapshot 2D IAAFT surrogates, so it uses SNAPSHOT mode. Window mode needs
a time series per sample, and stacking independent per-snapshot surrogates
into a window would destroy their temporal coherence (a trivial, non-spatial
cue). The window-mode functions are ported and tested but not used by the
benchmark.

* X1 (R^10): functional graph with edges |Pearson| > theta = 0.40; the ten
  invariants of the original `desc_x1` (density, average clustering,
  transitivity, degree mean / std / max-fraction, fraction of connected
  components, edges per node, betweenness mean / std).
* X2 (R^12): persistent homology of the sublevel-set filtration of the field
  (periodic cubical complex); the original `sig6` summary (mean, std, max,
  sum, count, p75 of the persistences) of H0 and H1.
* X3 (R^22): the label-free fusion of the original `desc_x3`, block-wise L2
  normalisation and concatenation, X3 = [X1/|X1|, X2/|X2|].

All views are meant for the amplitude-normalised field (zero mean, unit
standard deviation). [GEM port] The benchmark fields are already standardized
by `src.eval.hard_benchmark._standardize`, which equals `normalize_window` up
to round-off, so the views are applied to them directly.
"""
from __future__ import annotations

import gudhi
import networkx as nx
import numpy as np
from scipy.stats import spearmanr

X1_NAMES = ["density", "avg_clustering", "transitivity", "deg_mean", "deg_std",
            "deg_max_frac", "components_frac", "edges_per_node", "betw_mean", "betw_std"]
X2_NAMES = [f"{h}_{s}" for h in ("H0", "H1")
            for s in ("pers_mean", "pers_std", "pers_max", "pers_sum", "count", "pers_p75")]


def normalize_window(window):
    """Remove the window mean and divide by its space-time std."""
    w = np.asarray(window, dtype=np.float64)
    return (w - w.mean()) / (w.std() + 1e-30)


# ── X1: functional graph ─────────────────────────────────────────────────
def region_series(window, blocks=(8, 8)):
    """Block-average a (T, H, W) window into M = by*bx region time series (T, M)."""
    T, H, W = window.shape
    by, bx = blocks
    return window.reshape(T, by, H // by, bx, W // bx).mean(axis=(2, 4)).reshape(T, by * bx)


def graph_invariants(act, theta=0.40, max_n=80, corr="pearson", var_floor=1e-6):
    """`desc_x1` of the original repository applied to an activation-like matrix.

    act: (samples, units). Units whose temporal std is below `var_floor`
    (steady regions) carry no correlation information and get no edges;
    without this floor, round-off fluctuations of a converged steady state
    would produce a random graph.
    """
    N = min(act.shape[1], max_n)
    A = act[:, :N]
    live = A.std(axis=0) > var_floor
    C = np.zeros((N, N))
    if live.sum() >= 2:
        sub = A[:, live]
        Cs = np.corrcoef(sub.T) if corr == "pearson" else spearmanr(sub)[0]
        C[np.ix_(live, live)] = np.nan_to_num(np.atleast_2d(Cs))
    np.fill_diagonal(C, 0)
    G = nx.from_numpy_array((np.abs(C) > theta).astype(float))
    deg = np.array([d for _, d in G.degree()]) if G.number_of_nodes() else np.array([0.])
    try:
        bc = np.array(list(nx.betweenness_centrality(G).values()))
    except Exception:
        bc = np.zeros(len(deg))
    return np.array([nx.density(G), nx.average_clustering(G), nx.transitivity(G),
                     deg.mean(), deg.std(), deg.max() / max(1, len(deg)),
                     float(nx.number_connected_components(G)) / max(G.number_of_nodes(), 1),
                     G.number_of_edges() / max(G.number_of_nodes(), 1), bc.mean(), bc.std()],
                    dtype=np.float64)


def x1_functional_graph(window, theta=0.40, blocks=(8, 8), corr="pearson"):
    """Window mode: regions co-varying in time (climate network)."""
    return graph_invariants(region_series(window, blocks), theta=theta, corr=corr)


def patch_matrix(field, blocks=(8, 8)):
    """(pixels per patch, n patches): every column is one region's pattern."""
    H, W = field.shape
    by, bx = blocks
    P = field.reshape(by, H // by, bx, W // bx).transpose(1, 3, 0, 2)
    return P.reshape((H // by) * (W // bx), by * bx)


def x1_patch_coherence(field, theta=0.40, blocks=(8, 8), corr="pearson", max_n=80):
    """Snapshot mode: regions carrying coherent spatial patterns."""
    return graph_invariants(patch_matrix(field, blocks), theta=theta, corr=corr, max_n=max_n)


# ── X2: persistent homology of sublevel sets ─────────────────────────────
def sig6(pers):
    """Original `sig6` summary of a vector of persistences."""
    if len(pers) == 0:
        return np.zeros(6)
    p = np.asarray(pers)
    return np.array([p.mean(), p.std(), p.max(), p.sum(), float(len(p)), np.percentile(p, 75)])


def sublevel_persistence(field, periodic=(True, True), min_pers=0.01):
    """Finite H0/H1 persistences of the sublevel-set filtration of a 2D field.

    `min_pers` (in units of the window std) discards pairs created by
    round-off on plateaus of steady fields; it is far below any physical
    structure amplitude.
    """
    cc = gudhi.PeriodicCubicalComplex(top_dimensional_cells=field,
                                      periodic_dimensions=list(periodic))
    cc.compute_persistence(homology_coeff_field=2, min_persistence=min_pers)
    out = []
    for dim in (0, 1):
        iv = cc.persistence_intervals_in_dimension(dim)
        iv = iv[np.isfinite(iv[:, 1])] if len(iv) else iv
        out.append(iv[:, 1] - iv[:, 0] if len(iv) else np.zeros(0))
    return out


def x2_snapshot_ph(field, periodic=(True, True), min_pers=0.01):
    h0, h1 = sublevel_persistence(field, periodic, min_pers)
    return np.concatenate([sig6(h0), sig6(h1)])


def x2_sublevel_ph(window, periodic=(True, True), n_snap=8, min_pers=0.01):
    idx = np.linspace(0, window.shape[0] - 1, n_snap).round().astype(int)
    return np.mean([x2_snapshot_ph(window[t], periodic, min_pers) for t in idx], axis=0)


# ── X3: label-free fusion (original desc_x3) ─────────────────────────────
def desc_x3(x1, x2):
    n1 = x1 / (np.linalg.norm(x1) + 1e-8)
    n2 = x2 / (np.linalg.norm(x2) + 1e-8)
    return np.concatenate([n1, n2])
