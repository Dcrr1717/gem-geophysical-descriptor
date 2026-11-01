"""Authentic X1 / X2 / X3 port and its wiring into the hard benchmark."""
import numpy as np
import pytest

from src.descriptor.authentic import (X1_NAMES, X2_NAMES, desc_x3, graph_invariants,
                                      normalize_window, patch_matrix, region_series,
                                      sublevel_persistence, x1_functional_graph,
                                      x1_patch_coherence, x2_snapshot_ph, x2_sublevel_ph)
from src.eval.hard_benchmark import (CANDIDATES, FEATURE_SETS, _standardize, compute_features,
                                     union_summary, union_verdicts)
from src.eval.classify import evaluate_grouped


def _smooth_random(n=64, seed=0, width=6.0):
    rng = np.random.default_rng(seed)
    k = np.fft.fftfreq(n) * n
    kx, ky = np.meshgrid(k, k, indexing="ij")
    g = np.real(np.fft.ifft2(np.fft.fft2(rng.standard_normal((n, n)))
                             * np.exp(-(kx ** 2 + ky ** 2) / (2 * width ** 2))))
    return _standardize(g)


# --- the port (tests mirror the reference repository's own checks) -----------------
def test_graph_invariants_permutation_invariant():
    rng = np.random.default_rng(0)
    A = rng.standard_normal((40, 30))
    A[:, 1] = A[:, 0] + 0.1 * rng.standard_normal(40)
    p = rng.permutation(30)
    assert np.allclose(graph_invariants(A), graph_invariants(A[:, p]))
    assert graph_invariants(A).shape == (len(X1_NAMES),)


def test_x1_window_mode_steady_window_is_empty_graph():
    w = np.repeat(np.cos(4 * np.linspace(0, 2 * np.pi, 64, endpoint=False))[None, :, None], 16, 0)
    w = np.repeat(w, 64, axis=2)
    assert region_series(w, (8, 8)).shape == (16, 64)
    x1 = x1_functional_graph(normalize_window(w), blocks=(8, 8))
    assert x1[0] == 0 and x1[7] == 0


def test_patch_matrix_columns_are_patches():
    f = np.arange(16 * 32, dtype=float).reshape(16, 32)
    P = patch_matrix(f, blocks=(2, 4))
    assert P.shape == (64, 8)
    assert np.array_equal(P[:, 1], f[0:8, 8:16].ravel())
    assert np.array_equal(P[:, 4], f[8:16, 0:8].ravel())


def test_patch_coherence_orders_laminar_above_noise():
    x = np.linspace(0, 2 * np.pi, 128, endpoint=False)
    lam = np.repeat(np.cos(4 * x)[:, None], 128, axis=1) + 0.01 * np.cos(x)[None, :]
    noise = np.random.default_rng(0).standard_normal((128, 128))
    assert x1_patch_coherence(normalize_window(lam))[0] > 0.5
    assert x1_patch_coherence(normalize_window(noise))[0] < 0.05


def test_sublevel_ph_counts_extrema_on_torus():
    x = np.linspace(0, 2 * np.pi, 64, endpoint=False)
    X, Y = np.meshgrid(x, x)
    f = -np.exp(-((X - 2) ** 2 + (Y - 2) ** 2)) - 0.5 * np.exp(-((X - 4.5) ** 2 + (Y - 4.5) ** 2))
    h0, _ = sublevel_persistence(f, (True, True), min_pers=0.01)
    assert len(h0) == 1 and 0.40 < h0[0] < 0.47


def test_x2_and_x3_shapes():
    rng = np.random.default_rng(0)
    w = normalize_window(rng.standard_normal((8, 32, 32)))
    x2 = x2_sublevel_ph(w, n_snap=2)
    x1 = x1_functional_graph(w, blocks=(4, 4))
    x3 = desc_x3(x1, x2)
    assert x2.shape == (len(X2_NAMES),) and x3.shape == (22,)
    assert np.isclose(np.linalg.norm(x3[:10]), 1) and np.isclose(np.linalg.norm(x3[10:]), 1)
    assert x2_snapshot_ph(_smooth_random()).shape == (12,)


# --- benchmark wiring ------------------------------------------------------------------
def test_compute_features_with_authentic_views():
    fields = [_smooth_random(32, s) for s in range(2)]
    F = compute_features(fields, 32, workers=1, decimals=8, fullres=False, authentic={})
    dims = {"x1_graph": 10, "x2_ph": 12, "x1x2": 22, "x3_fusion": 22, "laplacian_view": 8,
            "x1x2_rich": 22 + F["topo_rich_all"].shape[1]}
    for k, d in dims.items():
        assert F[k].shape == (2, d), k
    np.testing.assert_allclose(F["laplacian_view"], F["spectral_view"][:, 3:])
    assert (F["baselines_union+x1x2_rich"].shape[1]
            == F["baselines_union"].shape[1] + dims["x1x2_rich"])
    for k in F:
        assert k in FEATURE_SETS, k
    # default (noise sweep) path: no authentic views
    assert "x1_graph" not in compute_features(fields, 32, workers=1, fullres=False)


def test_union_verdicts_flags_go_only_with_margin():
    rng = np.random.default_rng(0)
    groups = np.repeat(np.arange(12), 10)
    y = np.tile([0, 1], 60)
    weak = rng.standard_normal((120, 3))
    strong = np.c_[y + 0.05 * rng.standard_normal(120)]
    R = {"baselines_union": evaluate_grouped(weak, y, groups, n_splits=3, n_repeats=1),
         "x1x2_rich": evaluate_grouped(strong, y, groups, n_splits=3, n_repeats=1)}
    task = {"results": R}
    uv = union_verdicts(task, y, groups, 0.05, 200, 0)
    assert set(uv["configs"]) == {"x1x2_rich"} and "x1x2_rich" in CANDIDATES
    assert uv["verdict"] == "GO" and uv["go_arithmetically_possible"]
    assert uv["configs_reaching_go_margin_ci_above_0"] == ["x1x2_rich"]
    task["union_verdicts"] = uv
    assert union_summary({"t": task})["verdict"] == "GO"
    R2 = {"baselines_union": R["x1x2_rich"], "x1x2_rich": R["x1x2_rich"]}
    uv2 = union_verdicts({"results": R2}, y, groups, 0.05, 200, 0)
    assert uv2["verdict"] == "NO-GO" and uv2["best_margin"] == pytest.approx(0.0)
