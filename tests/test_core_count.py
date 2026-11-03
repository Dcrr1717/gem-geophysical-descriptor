"""Matched core-count task (task C): generator, design statistics, matching (offline)."""
import numpy as np

from src.descriptor.ect import roll_quiet_to_edge
from src.sims.core_count import (balance, balance_gate, design_stats, draw_cores,
                                 initial_condition, match_propensity, periodic_core)
from src.sims.vortex_events import core_count

CC = {"amplitude": 10.0, "steepness": [2.0, 8.0], "single_radius": [0.3, 0.75],
      "single_aspect": [1.5, 3.5], "single_steepness": [3.0, 8.0], "pair_radius": [0.3, 0.6],
      "pair_size_ratio": [0.35, 0.8], "pair_amp_ratio": [0.6, 1.0],
      "pair_steepness": [2.0, 5.0], "pair_separation": [4.0, 7.0], "max_separation": 3.0}


def test_elliptical_core_preserves_area():
    n = 256
    disk = periodic_core(n, (3.0, 3.0), 0.5, 6.0, 1.0, aspect=1.0)
    ell = periodic_core(n, (3.0, 3.0), 0.5, 6.0, 1.0, aspect=3.0, angle=0.7)
    a0, a1 = (disk > 0.5).sum(), (ell > 0.5).sum()
    assert abs(a1 - a0) / a0 < 0.05


def test_draw_cores_counts_and_separation():
    for label, k in ((0, 1), (1, 2)):
        for run in range(20):
            cores = draw_cores(label, CC, np.random.default_rng(run))
            assert len(cores) == k
            if k == 2:
                d = np.hypot(*(np.asarray(cores[0]["center"]) - np.asarray(cores[1]["center"])))
                rbar = 0.5 * (cores[0]["radius"] + cores[1]["radius"])
                assert d >= min(4.0 * rbar, CC["max_separation"]) - 1e-9
                assert max(c["amp"] for c in cores) == CC["amplitude"]


def test_initial_conditions_have_the_designed_core_count():
    for label, k in ((0, 1), (1, 2)):
        for run in range(10):
            rng = np.random.default_rng(100 + run)
            w = initial_condition(128, draw_cores(label, CC, rng), 0.05, (4, 24), rng)
            assert core_count(w) == k
            assert abs(w.mean()) < 1e-10


def test_design_stats_shape_and_area():
    w = periodic_core(128, (3.0, 3.0), 0.5, 2.0, 1.0)
    w = (w - w.mean()) / w.std()
    s = design_stats(w)
    assert s.shape == (4,) and np.all(np.isfinite(s))
    # half-max area of a Gaussian exp(-r^2/R^2): pi R^2 ln 2, over the (2 pi)^2 box
    assert abs(s[2] - np.pi * 0.25 * np.log(2) / (2 * np.pi) ** 2) < 0.002


def test_propensity_matching_balances_overlapping_classes():
    rng = np.random.default_rng(0)
    n = 400
    S0 = np.column_stack([rng.normal(6, 1, n), rng.normal(5, 1, n),
                          np.exp(rng.normal(-4.0, 0.3, n)), np.exp(rng.normal(2.2, 0.2, n))])
    S1 = np.column_stack([rng.normal(6.6, 1, n), rng.normal(5.3, 1, n),
                          np.exp(rng.normal(-4.1, 0.3, n)), np.exp(rng.normal(2.25, 0.2, n))])
    S, y = np.vstack([S0, S1]), np.r_[np.zeros(n, int), np.ones(n, int)]
    a, b, d = match_propensity(S, y, 0.2)
    assert len(a) == len(b) > 100
    assert len(set(a)) == len(a) and len(set(b)) == len(b)
    assert np.all(y[a] == 0) and np.all(y[b] == 1)
    before = max(abs(v["smd"]) for v in balance(S, y).values())
    idx = np.r_[a, b]
    after = max(abs(v["smd"]) for v in balance(S[idx], y[idx]).values())
    assert after < before and after < 0.1


def test_balance_gate_on_identical_distributions():
    rng = np.random.default_rng(1)
    S = np.column_stack([rng.normal(6, 1, 600), rng.normal(5, 1, 600),
                         np.exp(rng.normal(-4, 0.3, 600)), np.exp(rng.normal(2, 0.2, 600))])
    y = np.r_[np.zeros(300, int), np.ones(300, int)]
    g = np.arange(600) // 4
    gate = balance_gate(S, y, g, {"max_abs_smd": 0.25, "variance_ratio": [0.5, 2.0],
                                  "min_pairs": 100, "max_design_auc": 0.65})
    assert gate["passed"], gate


def test_roll_keeps_core_count_on_the_torus():
    w = periodic_core(128, (0.0, 0.0), 0.4, 2.0, 10.0) + periodic_core(128, (3.0, 3.0), 0.3,
                                                                           2.0, 9.0)
    assert core_count(w) == core_count(roll_quiet_to_edge(w)) == 2
