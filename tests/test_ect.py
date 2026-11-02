"""ECT, DECT, H0/H1 counts and the vortex-event task (offline, small grids)."""
import numpy as np
import pytest
from scipy import ndimage

from src.descriptor.ect import (ECT_DIRECTIONS, ECT_LEVELS, ECT_THRESHOLDS, _indicator,
                                cell_masks, ect_features, ect_image, euler_characteristic,
                                level_masks)
from src.sims.kolmogorov import simulate
from src.sims.vortex_events import core_count, initial_condition, periodic_gaussian_pair


def _chi_reference(P):
    """chi of a planar pixel set = 8-connected components - 4-connected holes."""
    n_comp = ndimage.label(P, structure=np.ones((3, 3)))[1]
    n_bg = ndimage.label(np.pad(~P, 1, constant_values=True))[1]
    return n_comp - (n_bg - 1)


def test_euler_characteristic_matches_components_minus_holes():
    rng = np.random.default_rng(0)
    for _ in range(15):
        P = rng.random((24, 24)) < rng.uniform(0.2, 0.7)
        assert euler_characteristic(P) == _chi_reference(P)


def test_ect_shapes_and_limits():
    rng = np.random.default_rng(1)
    w = rng.standard_normal((32, 32))
    E = ect_image(w)
    assert E.shape == (len(ECT_LEVELS), ECT_DIRECTIONS, ECT_THRESHOLDS)
    assert ect_features(w).shape == (len(ECT_LEVELS) * ECT_DIRECTIONS * ECT_THRESHOLDS,)
    # all cells present at the largest threshold: chi of the whole sublevel set, any direction
    assert np.all(E[:, :, -1] == E[:, :1, -1])
    assert np.all(E[:, :, 0] == 0)                          # nothing below the lowest threshold


def test_ect_of_annulus_and_two_disks():
    y, x = np.mgrid[-16:16, -16:16] + 0.5
    r = np.hypot(x, y)
    assert euler_characteristic((r > 6) & (r < 12)) == 0     # annulus: 1 - 1
    disks = (np.hypot(x - 8, y) < 5) | (np.hypot(x + 8, y) < 5)
    assert euler_characteristic(disks) == 2


def test_ect_rotation_equivariance():
    """Rotating the image by 90 degrees shifts the direction axis by D/4."""
    rng = np.random.default_rng(2)
    P = rng.random((16, 16)) < 0.4
    n, D, T = 16, 16, 24
    S = _indicator(n, D, T).reshape(-1, D, T)
    e0 = np.einsum("c,cdt->dt", cell_masks(P).astype(np.float32), S)
    e1 = np.einsum("c,cdt->dt", cell_masks(np.rot90(P)).astype(np.float32), S)
    shifts = [np.allclose(np.roll(e0, k, axis=0), e1) for k in (D // 4, -D // 4)]
    assert any(shifts)


def test_level_masks_binary_and_monotone_in_level():
    w = np.random.default_rng(3).standard_normal((32, 32))
    M = level_masks(w)
    assert M.dtype == np.uint8 and set(np.unique(M)) <= {0, 1}
    assert np.all(np.diff(M.astype(int), axis=0) >= 0)       # sublevel sets are nested


def test_ph_counts_two_wells():
    from src.eval.ect_benchmark import ph_counts
    x = np.linspace(0, 2 * np.pi, 64, endpoint=False)
    X, Y = np.meshgrid(x, x, indexing="ij")
    w = -np.exp(-((X - 2) ** 2 + (Y - 2) ** 2) / 0.2) - np.exp(-((X - 4.5) ** 2 + (Y - 4.5) ** 2) / 0.2)
    w = (w - w.mean()) / w.std()
    c = ph_counts(w)
    assert c.shape == (6,) and c[0] == 1                      # one finite significant H0 pair


def test_vortex_pair_core_count():
    n = 128
    near = periodic_gaussian_pair(n, 0.8 * 0.3, 0.3, 10.0, (3.0, 3.0), 0.4)
    far = periodic_gaussian_pair(n, 4.5 * 0.3, 0.3, 10.0, (3.0, 3.0), 0.4)
    assert core_count(near) == 1 and core_count(far) == 2
    wrap = periodic_gaussian_pair(n, 4.5 * 0.3, 0.3, 10.0, (0.05, 0.05), 0.4)
    assert core_count(wrap) == 2                              # periodic boundaries handled


def test_simulate_with_initial_condition_is_unforced_and_finite():
    rng = np.random.default_rng(4)
    w0 = initial_condition(32, 4.0, 0.5, 5.0, 0.1, (2, 8), rng)
    snaps = simulate(n=32, nu=0.01, forcing_amp=0.0, dt=0.01, n_steps=50, spinup=0,
                     n_snapshots=2, scheme="rk4", w0=w0)
    assert np.all(np.isfinite(snaps[-1]))
    assert abs(snaps[-1].mean()) < 1e-10                       # zero circulation preserved
    with pytest.raises(ValueError):
        simulate(n=16, scheme="euler", w0=np.zeros((16, 16)))


def test_default_simulation_unchanged_by_w0_parameter():
    a = simulate(n=16, nu=0.05, dt=0.01, n_steps=20, spinup=10, n_snapshots=2, seed=3,
                 scheme="rk4")
    b = simulate(n=16, nu=0.05, dt=0.01, n_steps=20, spinup=10, n_snapshots=2, seed=3,
                 scheme="rk4", w0=None)
    np.testing.assert_array_equal(a[-1], b[-1])


# ---------------------------------------------------------------------------
# DECT (optional torch)
# ---------------------------------------------------------------------------
def test_dect_layer_reduces_to_exact_ect_when_sharp():
    torch = pytest.importorskip("torch")
    from src.descriptor.dect import DECTLayer
    w = np.random.default_rng(5).standard_normal((16, 16))
    M = level_masks(w, res=16)                                # (L, C)
    layer = DECTLayer(16, n_dirs=ECT_DIRECTIONS, n_t=ECT_THRESHOLDS, sharpness=5e4)
    with torch.no_grad():
        # nudge the thresholds off the exact cell heights (ties make the sigmoid 0.5)
        layer.t.add_(1e-4)
        out = layer(torch.from_numpy(M[None].astype(np.float32)))[0].numpy()
    exact = ect_image(w, res=16)
    assert np.abs(out - exact).max() < 1e-2


def test_dect_gradients_reach_directions_and_thresholds():
    torch = pytest.importorskip("torch")
    from src.descriptor.dect import DECTLayer
    M = level_masks(np.random.default_rng(6).standard_normal((16, 16)), res=16)
    layer = DECTLayer(16, n_dirs=8, n_t=8, sharpness=20.0)
    layer(torch.from_numpy(M[None].astype(np.float32))).pow(2).mean().backward()
    assert layer.theta.grad is not None and layer.theta.grad.abs().sum() > 0
    assert layer.t.grad is not None and layer.t.grad.abs().sum() > 0


def test_dect_grouped_training_runs_and_learns_a_trivial_task():
    pytest.importorskip("torch")
    from src.descriptor.dect import train_eval_grouped
    rng = np.random.default_rng(7)
    x = np.linspace(0, 2 * np.pi, 16, endpoint=False)
    X, Y = np.meshgrid(x, x, indexing="ij")
    fields, y, g = [], [], []
    for i in range(24):
        lab = i % 2
        k = 1 if lab == 0 else 3                          # 1 vs 9 blobs: trivially topological
        f = np.cos(k * X + rng.uniform(0, 6)) * np.cos(k * Y + rng.uniform(0, 6))
        fields.append(f + 0.05 * rng.standard_normal(f.shape))
        y.append(lab)
        g.append(i // 2)
    masks = np.stack([level_masks(f, res=16) for f in fields])
    hp = {"epochs": 15, "batch_size": 8, "lr": 0.01, "weight_decay": 0.0, "n_dirs": 8,
          "n_t": 8, "sharpness": 20.0}
    res = train_eval_grouped(masks, np.array(y), np.array(g), 16, hp, n_splits=3, n_repeats=1)
    assert res["best_classifier"] == "dect" and len(res["dect"]["oof"]) == 1
    assert res["best_accuracy"] >= 0.75
