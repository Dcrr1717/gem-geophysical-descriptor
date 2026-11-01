import numpy as np
from src.sims.kolmogorov import simulate
from src.descriptor.multiview import descriptor
from src.baselines.features import moment_features


def test_simulate_runs():
    snaps = simulate(n=32, nu=0.03, dt=0.002, n_steps=200, spinup=100, n_snapshots=3)
    assert len(snaps) == 3
    assert np.all(np.isfinite(snaps[0]))


def test_descriptor_finite():
    snaps = simulate(n=32, nu=0.03, dt=0.002, n_steps=200, spinup=100, n_snapshots=2)
    d = descriptor(snaps[0], topo_downsample=16)
    assert d.ndim == 1 and np.all(np.isfinite(d))


def test_baseline_shape():
    snaps = simulate(n=32, nu=0.03, dt=0.002, n_steps=150, spinup=80, n_snapshots=1)
    assert moment_features(snaps[0]).shape == (6,)
