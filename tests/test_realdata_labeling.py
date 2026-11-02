"""Tibaldi-Molteni blocking labels on small synthetic 500 hPa height arrays.

Pure numpy: no network, no ERA5 file, no gudhi. Every pattern below has a
known answer (zonal flow, an omega-like high of known longitudinal extent,
profiles exactly at the thresholds, a block visible only for D = +4, ...).
"""
import numpy as np
import pytest

from src.realdata.label_blocking import (AMBIGUOUS, BLOCKING, G0, ZONAL, geopotential_to_height,
                                         height_from_raw, instantaneous_blocking, interp_lat,
                                         label_days, longest_run_span, persistent_runs,
                                         sector_blocking, tm_gradients)

LAT = np.arange(20.0, 90.01, 1.0)
LON = np.arange(-80.0, 80.01, 1.0)
SECTOR = {"west": -45.0, "east": 45.0}


def zonal_height(slope=-8.0, lat=LAT, lon=LON):
    """Westerly flow: height decreasing poleward everywhere (GHGS = slope < 0)."""
    return np.broadcast_to((5600.0 + slope * (lat - 40.0))[:, None], (lat.size, lon.size)).copy()


def high_profile(lat=LAT, amp=400.0, center=62.0, width=8.0):
    return amp * np.exp(-((lat - center) / width) ** 2)


def blocked_height(lon0, lon1, lat=LAT, lon=LON, **kw):
    """Zonal flow plus a strong high centred at 62N between lon0 and lon1
    (inclusive): GHGS ~ +10.8 and GHGN ~ -26.7 m/deg there, for every D."""
    box = ((lon >= lon0) & (lon <= lon1)).astype(float)
    return zonal_height(lat=lat, lon=lon) + high_profile(lat, **kw)[:, None] * box[None, :]


def profile_height(points, lat=LAT, nlon=3):
    """Height that is piecewise linear in latitude through `points` {lat: Z}."""
    p = sorted(points.items())
    z = np.interp(lat, [a for a, _ in p], [b for _, b in p])
    return np.repeat(z[:, None], nlon, axis=1)


def dates(n, start="2000-12-01"):
    return np.datetime64(start, "D") + np.arange(n)


# ---------------------------------------------------------------------------
# Instantaneous (per-longitude) index
# ---------------------------------------------------------------------------
def test_zonal_flow_is_never_blocked():
    Z = zonal_height()
    gs, gn = tm_gradients(Z, LAT)
    np.testing.assert_allclose(gs, -8.0)
    np.testing.assert_allclose(gn, -8.0)
    assert not instantaneous_blocking(Z, LAT).any()


def test_block_detected_exactly_on_its_longitudes():
    Z = blocked_height(-10, 20)
    ib = instantaneous_blocking(Z, LAT)
    np.testing.assert_array_equal(ib, (LON >= -10) & (LON <= 20))
    gs, gn = tm_gradients(Z, LAT)
    inside = (LON >= -10) & (LON <= 20)
    assert np.all(gs[:, inside] > 0) and np.all(gn[:, inside] < -10)


def test_reversal_south_is_required():
    # strong westerlies to the north (GHGN = -20) but no reversal (GHGS = -5)
    Z = profile_height({20: 5800, 40: 5600, 60: 5500, 80: 5100, 90: 5000})
    assert not instantaneous_blocking(Z, LAT, deltas=(0.0,)).any()


def test_westerlies_north_are_required():
    # reversal (GHGS = +5) but weak flow to the north (GHGN = -5): a cut-off
    # low to the south, not a block
    Z = profile_height({20: 5600, 40: 5500, 60: 5600, 80: 5500, 90: 5450})
    assert not instantaneous_blocking(Z, LAT, deltas=(0.0,)).any()
    Z = profile_height({20: 5600, 40: 5500, 60: 5600, 80: 5350, 90: 5300})   # GHGN = -12.5
    assert instantaneous_blocking(Z, LAT, deltas=(0.0,)).all()


def test_thresholds_are_strict_inequalities():
    at_ghgs = profile_height({20: 5600, 40: 5600, 60: 5600, 80: 5300, 90: 5200})  # GHGS = 0
    at_ghgn = profile_height({20: 5500, 40: 5500, 60: 5600, 80: 5400, 90: 5300})  # GHGN = -10
    assert not instantaneous_blocking(at_ghgs, LAT, deltas=(0.0,)).any()
    assert not instantaneous_blocking(at_ghgn, LAT, deltas=(0.0,)).any()
    assert instantaneous_blocking(at_ghgs, LAT, ghgs_min=-0.1, deltas=(0.0,)).all()
    assert instantaneous_blocking(at_ghgn, LAT, ghgn_max=-9.9, deltas=(0.0,)).all()


def test_delta_shift_catches_a_displaced_block():
    # only the latitudes shifted by D = +4 (44 / 64 / 84N) see the reversal
    Z = profile_height({20: 5800, 36: 5700, 40: 5600, 44: 5500, 56: 5500, 60: 5550,
                        64: 5650, 76: 5550, 80: 5450, 84: 5300, 90: 5250})
    gs, gn = tm_gradients(Z, LAT, deltas=(-4.0, 0.0, 4.0))
    np.testing.assert_allclose(gs[:, 0], [-10.0, -2.5, 7.5])
    np.testing.assert_allclose(gn[2, 0], -17.5)
    assert not instantaneous_blocking(Z, LAT, deltas=(-4.0, 0.0)).any()
    assert instantaneous_blocking(Z, LAT, deltas=(-4.0, 0.0, 4.0)).all()


def test_interp_lat_exact_off_grid_and_for_descending_latitudes():
    lat = np.arange(20.0, 90.01, 2.5)
    Z = zonal_height(slope=-7.0, lat=lat, lon=LON)
    np.testing.assert_allclose(interp_lat(Z, lat, 61.0), 5600.0 - 7.0 * 21.0)
    gs, gn = tm_gradients(Z, lat)                       # 36, 56, 64, 76, 84 are off-grid
    np.testing.assert_allclose(gs, -7.0)
    np.testing.assert_allclose(gn, -7.0)
    Zb = blocked_height(0, 30, lat=lat)
    np.testing.assert_array_equal(instantaneous_blocking(Zb, lat),
                                  instantaneous_blocking(Zb[::-1], lat[::-1]))


def test_missing_index_latitudes_raise():
    lat = np.arange(45.0, 90.01, 1.0)                   # phi_S + D down to 36N is missing
    with pytest.raises(ValueError, match="outside the data range"):
        instantaneous_blocking(zonal_height(lat=lat), lat)


# ---------------------------------------------------------------------------
# Sector extent
# ---------------------------------------------------------------------------
def test_longest_run_span_counts_adjacent_longitudes_only():
    rel = np.arange(10.0)
    flags = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                      [1, 1, 1, 0, 1, 1, 1, 1, 1, 0],
                      [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]], bool)
    np.testing.assert_array_equal(longest_run_span(flags, rel, 1.0), [-1, 0, 4, 9])


@pytest.mark.parametrize("width, blocked", [(14, False), (15, True), (30, True)])
def test_extent_threshold(width, blocked):
    Z = blocked_height(0, width)[None]                  # span = width degrees
    s = sector_blocking(instantaneous_blocking(Z, LAT), LON, SECTOR, 15.0)
    assert s["span"][0] == pytest.approx(width)
    assert s["sector_blocked"][0] == blocked
    assert s["any_ib"][0]


def test_two_narrow_blocks_do_not_add_up():
    Z = (blocked_height(-30, -20) + blocked_height(10, 20) - zonal_height())[None]
    ib = instantaneous_blocking(Z, LAT)
    assert ib.sum() == 22
    s = sector_blocking(ib, LON, SECTOR, 15.0)
    assert s["span"][0] == pytest.approx(10.0) and not s["sector_blocked"][0]


def test_only_the_sector_counts():
    outside = blocked_height(50, 75)[None]              # a wide block east of 45E
    s = sector_blocking(instantaneous_blocking(outside, LAT), LON, SECTOR, 15.0)
    assert not s["any_ib"][0] and not s["sector_blocked"][0]
    edge = blocked_height(35, 75)[None]                 # only 35..45E lies in the sector
    s = sector_blocking(instantaneous_blocking(edge, LAT), LON, SECTOR, 15.0)
    assert s["any_ib"][0] and s["span"][0] == pytest.approx(10.0) and not s["sector_blocked"][0]


@pytest.mark.parametrize("lon", [np.arange(0.0, 360.0, 2.5), np.arange(-180.0, 180.0, 2.5)])
def test_sector_across_the_dateline(lon):
    lon_n = (lon + 180.0) % 360.0 - 180.0
    box = (np.abs(lon_n) >= 170.0)                      # 170E .. 180 .. 170W (20 deg)
    Z = (zonal_height(lon=lon) + high_profile()[:, None] * box[None, :])[None]
    ib = instantaneous_blocking(Z, LAT)
    s = sector_blocking(ib, lon, {"west": 150.0, "east": -150.0}, 15.0)
    assert s["span"][0] == pytest.approx(20.0) and s["sector_blocked"][0]


# ---------------------------------------------------------------------------
# Day labels: persistence and buffer
# ---------------------------------------------------------------------------
def _series(n, blocks):
    """n days of zonal flow; blocks = {day index: (lon0, lon1)}."""
    return np.stack([blocked_height(*blocks[t]) if t in blocks else zonal_height()
                     for t in range(n)])


def test_persistent_block_buffer_and_short_block():
    blocks = {t: (-20, 20) for t in range(5, 11)}       # 6-day episode (days 5..10)
    blocks.update({t: (0, 30) for t in range(16, 19)})  # 3-day block (too short)
    lab = label_days(_series(24, blocks), LAT, LON, dates(24), sector=SECTOR,
                     min_extent_deg=15.0, min_duration_days=5, zonal_buffer_days=2)
    expect = np.full(24, ZONAL)
    expect[5:11] = BLOCKING
    expect[[3, 4, 11, 12]] = AMBIGUOUS                  # within 2 days of the episode
    expect[16:19] = AMBIGUOUS                           # blocked but not persistent
    np.testing.assert_array_equal(lab["labels"], expect)
    assert lab["episodes"] == [{"start": "2000-12-06", "end": "2000-12-11", "days": 6}]


def test_narrow_persistent_block_is_not_blocking():
    blocks = {t: (0, 10) for t in range(2, 10)}         # 8 days, but only 10 deg wide
    lab = label_days(_series(12, blocks), LAT, LON, dates(12), sector=SECTOR)
    assert not (lab["labels"] == BLOCKING).any()
    np.testing.assert_array_equal(lab["labels"][2:10], AMBIGUOUS)
    np.testing.assert_array_equal(lab["labels"][[0, 1, 10, 11]], ZONAL)


def test_persistence_breaks_at_calendar_gaps():
    d = np.r_[dates(3, "2001-02-26"), dates(3, "2001-12-01")]   # Feb 26-28, Dec 1-3
    mask, ep = persistent_runs(np.ones(6, bool), d, 5)
    assert not mask.any() and ep == []
    mask, ep = persistent_runs(np.ones(6, bool), d, 3)
    assert mask.all() and ep == [(0, 2), (3, 5)]
    with pytest.raises(ValueError):
        persistent_runs(np.ones(3, bool), d[[0, 2, 1]], 2)


def test_optional_strong_westerly_requirement_for_zonal():
    Z = np.stack([zonal_height(slope=-2.0), zonal_height(slope=-8.0)])
    lab = label_days(Z, LAT, LON, dates(2), sector=SECTOR)
    np.testing.assert_array_equal(lab["labels"], [ZONAL, ZONAL])
    lab = label_days(Z, LAT, LON, dates(2), sector=SECTOR, zonal_max_ghgs=-5.0)
    np.testing.assert_array_equal(lab["labels"], [AMBIGUOUS, ZONAL])
    np.testing.assert_allclose(lab["ghgs_sector_mean"], [-2.0, -8.0])


def test_label_days_rejects_unknown_parameters():
    with pytest.raises(TypeError, match="unknown"):
        label_days(zonal_height()[None], LAT, LON, dates(1), ghgs_threshold=0.0)


def test_geopotential_is_converted_to_height():
    Z = blocked_height(-10, 20)
    np.testing.assert_allclose(geopotential_to_height(Z * G0), Z)
    np.testing.assert_allclose(height_from_raw(Z * G0, "m**2 s**-2"), Z)
    np.testing.assert_allclose(height_from_raw(Z, "m"), Z)
    lab = label_days(height_from_raw(Z[None] * G0, "m**2 s**-2"), LAT, LON, dates(1),
                     min_duration_days=1)
    assert lab["labels"][0] == BLOCKING
