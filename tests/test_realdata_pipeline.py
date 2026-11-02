"""Real-data pipeline without network: config / seasons / CV groups, the CDS
request builder and credential check, preprocessing, the ERA5 netCDF reader,
and an end-to-end run on a SYNTHETIC STAND-IN for ERA5 (written to a pytest
temporary directory; it is test scaffolding, never data or a result)."""
import json
import re
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest
import yaml

from src.realdata import common
from src.realdata.common import (lon_range_covers, lon_window, months_by_year, season_of,
                                 spatial_key, time_groups, validate_config)
from src.realdata.download_era5 import CredentialsError, build_requests, check_credentials, \
    retrieve
from src.realdata import download_era5
from src.realdata.label_blocking import AMBIGUOUS, BLOCKING, G0, ZONAL
from src.realdata.run_realdata_benchmark import (balance_classes, crop_domain, periodize,
                                                 preprocess, resample_square, select_days)

REPO = Path(__file__).resolve().parents[1]


def _cfg(name="realdata.yaml"):
    return common.load_config(REPO / "config" / name)


# ---------------------------------------------------------------------------
# Config, seasons, longitudes, CV groups
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", ["realdata.yaml", "realdata_quick.yaml"])
def test_shipped_configs_are_valid_and_share_raw_files(name):
    cfg = _cfg(name)
    validate_config(cfg)
    assert spatial_key(cfg["download"]) == spatial_key(_cfg()["download"])


@pytest.mark.parametrize("path, value, msg", [
    (("download", "area", "south"), 40, "do not cover the blocking-index latitudes"),
    (("download", "area", "east"), 30, "do not cover labeling.sector"),
    (("benchmark", "grid"), 60, "multiple of 8"),
    (("benchmark", "periodize"), "fft", "periodize"),
    (("season", "months"), [12, 2], "consecutive"),
])
def test_validate_config_catches_mistakes(path, value, msg):
    cfg = _cfg()
    d = cfg
    for k in path[:-1]:
        d = d[k]
    d[path[-1]] = value
    with pytest.raises(ValueError, match=msg):
        validate_config(cfg)


def test_months_by_year_for_djf():
    assert months_by_year([12, 1, 2], 1979, 1981) == {1979: [12], 1980: [1, 2, 12],
                                                       1981: [1, 2, 12], 1982: [1, 2]}
    assert months_by_year([6, 7, 8], 2000, 2001) == {2000: [6, 7, 8], 2001: [6, 7, 8]}


def test_season_of():
    d = np.array(["1999-12-01", "2000-01-15", "2000-02-29", "2000-07-01", "2000-12-31"],
                 dtype="datetime64[D]")
    np.testing.assert_array_equal(season_of(d, [12, 1, 2]), [1999, 1999, 1999, -1, 2000])


def test_season_groups_never_split_a_winter():
    d = np.r_[np.datetime64("2000-12-01") + np.arange(90), np.datetime64("2001-12-01") +
              np.arange(90)]
    g, keep = time_groups(d, [12, 1, 2], "season")
    assert keep.all() and len(np.unique(g)) == 2
    assert (g[:90] == g[0]).all() and (g[90:] == g[90]).all() and g[0] != g[90]


def test_block_groups_are_separated_by_the_gap():
    d = np.datetime64("2000-12-01") + np.arange(90)          # Dec 1 2000 .. Feb 28 2001
    g, keep = time_groups(d, [12, 1, 2], "block", block_days=30, gap_days=5)
    assert len(np.unique(g)) == 3
    assert keep.sum() == 90 - 2 * 5                          # the last block is not purged
    dn = common.day_numbers(d)[keep]
    gk = g[keep]
    for a in np.unique(gk):
        for b in np.unique(gk):
            if a != b:
                gap = np.abs(dn[gk == a][:, None] - dn[gk == b][None, :]).min()
                assert gap > 5
    with pytest.raises(ValueError):
        time_groups(d, [12, 1, 2], "block", block_days=5, gap_days=5)
    with pytest.raises(ValueError):
        time_groups(np.array(["2000-07-01"], "datetime64[D]"), [12, 1, 2])


def test_lon_window_and_coverage_across_conventions():
    lon = np.arange(0.0, 360.0, 1.0)
    idx, rel = lon_window(lon, -45, 45)
    assert len(idx) == 91 and rel[0] == 0 and rel[-1] == 90
    np.testing.assert_allclose(common.normalize_lon(lon[idx])[[0, -1]], [-45, 45])
    assert lon_range_covers(-80, 80, -75, 75)
    assert not lon_range_covers(-80, 80, -90, 0)
    assert lon_range_covers(120, -120, 170, -170)            # across the dateline


# ---------------------------------------------------------------------------
# Download step (no network, no cdsapi needed)
# ---------------------------------------------------------------------------
def test_build_requests_one_per_year_in_cds_format():
    cfg = _cfg()
    jobs = build_requests(cfg)
    assert [j["year"] for j in jobs] == list(range(1979, 2025))
    first, mid, last = jobs[0], jobs[10], jobs[-1]
    assert first["months"] == [12] and mid["months"] == [1, 2, 12] and last["months"] == [1, 2]
    r = mid["request"]
    assert r["area"] == [90.0, -80.0, 25.0, 80.0]             # N, W, S, E
    assert r["variable"] == ["geopotential", "vorticity"]
    assert r["pressure_level"] == ["500"] and r["month"] == ["01", "02", "12"]
    assert r["data_format"] == "netcdf" and r["grid"] == [1.0, 1.0]
    assert len(r["day"]) == 31 and r["time"] == ["12:00"]
    key = spatial_key(cfg["download"])
    assert mid["target"].name == f"era5_500hPa_1989_m01-02-12_{key}.nc"
    cfg["download"]["area"]["west"] = -90
    assert spatial_key(cfg["download"]) != key


def test_check_credentials_sources(tmp_path):
    assert "environment" in check_credentials({"CDSAPI_URL": "u", "CDSAPI_KEY": "k"}, tmp_path)
    with pytest.raises(CredentialsError, match="Personal Access Token"):
        check_credentials({}, tmp_path)
    rc = tmp_path / ".cdsapirc"
    rc.write_text("url: https://cds.climate.copernicus.eu/api\n")
    with pytest.raises(CredentialsError, match="key:"):
        check_credentials({}, tmp_path)
    secret = "0123abcd-not-a-real-token"
    rc.write_text(f"url: https://cds.climate.copernicus.eu/api\nkey: {secret}\n")
    msg = check_credentials({}, tmp_path)
    assert str(rc) in msg and secret not in msg
    other = tmp_path / "elsewhere.rc"
    other.write_text(rc.read_text())
    assert str(other) in check_credentials({"CDSAPI_RC": str(other)}, tmp_path / "nohome")


class _FakeClient:
    def __init__(self, fail_times=0):
        self.calls, self.fail_times = [], fail_times

    def retrieve(self, name, request, target):
        self.calls.append((name, request, target))
        assert target.endswith(".part")
        if len(self.calls) <= self.fail_times:
            Path(target).write_bytes(b"partial")
            raise ConnectionError("simulated CDS outage")
        Path(target).write_bytes(b"CDF-netcdf-bytes")


def test_retrieve_writes_atomically_retries_and_skips(tmp_path):
    job = {"year": 2000, "months": [12], "dataset": "ds", "request": {"a": 1},
           "target": tmp_path / "raw" / "f.nc"}
    c = _FakeClient(fail_times=2)
    assert retrieve(c, job, retries=3, backoff=0, sleep=lambda s: None)
    assert len(c.calls) == 3 and job["target"].read_bytes() == b"CDF-netcdf-bytes"
    assert not list((tmp_path / "raw").glob("*.part"))
    assert not retrieve(c, job, retries=3, backoff=0, sleep=lambda s: None)   # exists
    assert len(c.calls) == 3
    job2 = dict(job, target=tmp_path / "raw" / "g.nc")
    with pytest.raises(ConnectionError):
        retrieve(_FakeClient(fail_times=5), job2, retries=2, backoff=0, sleep=lambda s: None)
    assert not job2["target"].exists() and not list((tmp_path / "raw").glob("*.part"))


def test_dry_run_contacts_no_server(tmp_path, monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "cdsapi", None)          # any import would fail
    monkeypatch.delenv("CDSAPI_URL", raising=False)
    monkeypatch.delenv("CDSAPI_KEY", raising=False)
    monkeypatch.setenv("CDSAPI_RC", str(tmp_path / "missing.rc"))
    jobs = download_era5.main(["--config", str(REPO / "config/realdata_quick.yaml"),
                               "--dry-run", "--only-years", "2011"])
    assert [j["year"] for j in jobs] == [2011]
    assert "No Copernicus CDS credentials" in capsys.readouterr().out


def test_no_credentials_in_the_repository():
    token = re.compile(r"key\s*:\s*['\"]?[0-9a-f]{8}-[0-9a-f]{4}-", re.I)
    for p in list((REPO / "src").rglob("*.py")) + list((REPO / "config").glob("*.yaml")) + \
            list((REPO / "docs").glob("*.md")):
        assert not token.search(p.read_text()), p


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------
def test_resample_square_shape_and_smooth_fields():
    h, w = 56, 151
    yy, xx = np.meshgrid(np.linspace(0, 1, h), np.linspace(0, 1, w), indexing="ij")
    out = resample_square(3.0 + 0 * xx, 64)
    assert out.shape == (64, 64)
    np.testing.assert_allclose(out, 3.0)
    ramp = resample_square(2 * xx - yy, 64)
    assert np.all(np.diff(ramp, axis=1) > 0) and np.all(np.diff(ramp, axis=0) < 0)


def test_periodize_taper_and_mirror_are_periodic():
    rng = np.random.default_rng(0)
    f = np.cumsum(np.cumsum(rng.standard_normal((32, 32)), 0), 1) + 50.0  # smooth, far from 0
    t = periodize(f, "taper", 0.25)
    assert np.abs(t[0]).max() == 0 and np.abs(t[:, -1]).max() == 0
    np.testing.assert_allclose(t[8:24, 8:24], (f - f.mean())[8:24, 8:24])
    m = periodize(f, "mirror")
    assert m.shape == (64, 64)
    np.testing.assert_array_equal(m[:, 0], m[:, -1])          # even reflection: wraps smoothly
    np.testing.assert_array_equal(m[0], m[-1])
    with pytest.raises(ValueError):
        periodize(f, "fft")


@pytest.mark.parametrize("method", ["taper", "mirror", "none"])
def test_preprocess_standardizes(method):
    rng = np.random.default_rng(1)
    lat, lon = np.arange(25.0, 90.1, 1.0), np.arange(-80.0, 80.1, 1.0)
    raw = rng.standard_normal((3, lat.size, lon.size)) * 1e-5
    bm = {"grid": 64, "periodize": method, "taper_alpha": 0.25,
          "domain": {"south": 30, "north": 85, "west": -75, "east": 75}}
    out, shape = preprocess(raw, lat, lon, bm)
    assert out.shape == (3, 64, 64) and shape == (56, 151)
    np.testing.assert_allclose(out.mean(axis=(1, 2)), 0, atol=1e-12)
    np.testing.assert_allclose(out.std(axis=(1, 2)), 1, atol=1e-12)


def test_crop_domain_orders_west_to_east():
    lat, lon = np.arange(25.0, 90.1, 1.0), np.arange(-180.0, 180.0, 1.0)
    f = np.broadcast_to(lon[None, None, :], (1, lat.size, lon.size))
    c, la, lo = crop_domain(f, lat, lon, {"south": 30, "north": 85, "west": 170, "east": -170})
    np.testing.assert_allclose(lo[[0, 10, -1]], [170, -180, -170])
    assert la[0] == 30 and la[-1] == 85 and c.shape == (1, 56, 21)


def test_balance_classes_and_select_days():
    y = np.array([0] * 30 + [1] * 10)
    b = balance_classes(y, seed=0)
    assert np.bincount(y[b]).tolist() == [10, 10] and np.all(np.diff(b) > 0)
    np.testing.assert_array_equal(b, balance_classes(y, seed=0))
    d = np.r_[np.datetime64("2000-12-01") + np.arange(20), np.datetime64("2001-12-01") +
              np.arange(20)]
    labels = np.array(([ZONAL] * 8 + [AMBIGUOUS] * 2 + [BLOCKING] * 6 + [AMBIGUOUS] * 4) * 2,
                      dtype=np.int8)
    cfg = {"seed": 0, "season": {"months": [12, 1, 2]},
           "benchmark": {"grouping": {"mode": "season"}, "balance": "undersample"}}
    idx, yy, g = select_days({"dates": d, "labels": labels}, cfg)
    assert not np.isin(labels[idx], [AMBIGUOUS]).any()
    assert yy.sum() == 12 and yy.size == 24
    np.testing.assert_array_equal(g, (idx >= 20).astype(int))


# ---------------------------------------------------------------------------
# Synthetic stand-in for ERA5 (test scaffolding only)
# ---------------------------------------------------------------------------
LAT_S = np.arange(25.0, 90.01, 1.0)
LON_S = np.arange(-80.0, 80.01, 1.0)


def _synthetic_heights(n_days, block_days, rng):
    """Westerly flow + travelling planetary waves; on `block_days` a strong high
    at 62N with a 30-deg flat top. Returns heights (n_days, lat, lon) in m."""
    lat, lon = LAT_S[:, None], LON_S[None, :]
    env = np.exp(-((lat - 50.0) / 15.0) ** 2)
    c = rng.uniform(-20, 20)
    dl = np.abs(lon - c)
    lon_shape = np.where(dl <= 15, 1.0, np.where(dl <= 25, np.cos(np.pi / 2 * (dl - 15) / 10) ** 2,
                                                  0.0))
    Z = []
    for t in range(n_days):
        z = 5600.0 - 8.0 * (lat - 40.0) + 0.0 * lon
        for k in (3, 4, 5, 6):
            z = z + 40.0 * env * np.cos(np.deg2rad(k * lon) + rng.uniform(0, 2 * np.pi))
        if t in block_days:
            z = z + 400.0 * np.exp(-((lat - 62.0) / 8.0) ** 2) * lon_shape
        Z.append(z)
    return np.array(Z)


def _geostrophic_vorticity(Z):
    a, f0 = 6.371e6, 1e-4
    phi = np.deg2rad(LAT_S)[:, None]
    dy = a * np.deg2rad(1.0)
    dx = a * np.cos(phi) * np.deg2rad(1.0)
    zy = np.gradient(Z, dy, axis=1)
    zx = np.gradient(Z, axis=2) / dx
    lap = np.gradient(zy, dy, axis=1) + np.gradient(zx, axis=2) / dx
    return (G0 / f0 * lap).astype(np.float32)


def _write_nc(path, days, Z, vo, style, xr):
    t = (np.asarray(days, "datetime64[D]") + np.timedelta64(12, "h")).astype("datetime64[ns]")
    lat = LAT_S[::-1]                                           # ERA5 order: north -> south
    z, v = (G0 * Z[:, None, ::-1, :]).astype(np.float32), vo[:, None, ::-1, :]
    lon = LON_S
    if style == "legacy":                                       # old CDS names, 0..360 lon
        lon = LON_S % 360.0
        order = np.argsort(lon)
        lon, z, v = lon[order], z[..., order], v[..., order]
    tn, ln = ("valid_time", "pressure_level") if style != "legacy" else ("time", "level")
    dims = (tn, ln, "latitude", "longitude")
    coords = {tn: t, ln: [500.0], "latitude": lat, "longitude": lon}
    dz = xr.Dataset({"z": (dims, z, {"units": "m**2 s**-2"})}, coords=coords)
    dv = xr.Dataset({"vo": (dims, v, {"units": "s**-1"})}, coords=coords)
    if style == "zip":                                          # CDS zip of one file per variable
        tmp = path.with_suffix(".tmpdir")
        tmp.mkdir()
        dz.to_netcdf(tmp / "z.nc")
        dv.to_netcdf(tmp / "vo.nc")
        with zipfile.ZipFile(path, "w") as zf:
            zf.write(tmp / "z.nc", "data_stream-oper_stepType-instant_z.nc")
            zf.write(tmp / "vo.nc", "data_stream-oper_stepType-instant_vo.nc")
    else:
        xr.merge([dz, dv]).to_netcdf(path)


def test_read_era5_handles_cds_formats_and_daily_mean(tmp_path):
    xr = pytest.importorskip("xarray")
    pytest.importorskip("netCDF4")
    from src.realdata.label_blocking import read_era5

    rng = np.random.default_rng(0)
    days = np.datetime64("2000-12-01") + np.arange(3)
    Z = _synthetic_heights(3, set(), rng)
    vo = _geostrophic_vorticity(Z)
    for style in ("new", "legacy", "zip"):
        _write_nc(tmp_path / f"{style}.nc", days, Z, vo, style, xr)
        d = read_era5([tmp_path / f"{style}.nc"])
        np.testing.assert_array_equal(d["dates"], days)
        np.testing.assert_array_equal(d["lat"], LAT_S)
        np.testing.assert_array_equal(d["lon"], LON_S)
        np.testing.assert_allclose(d["z"] / G0, Z, rtol=1e-6)
        np.testing.assert_allclose(d["vo"], vo, rtol=1e-6)
    d = read_era5([tmp_path / "new.nc", tmp_path / "legacy.nc"])   # same days twice
    assert d["dates"].size == 3                                      # duplicates dropped
    # two hours per day -> daily mean
    t2 = (np.datetime64("2000-12-01") + np.array([0, 12], "timedelta64[h]")).astype("datetime64[ns]")
    ds = xr.Dataset({"z": (("valid_time", "latitude", "longitude"),
                           np.stack([np.zeros((LAT_S.size, LON_S.size)),
                                     np.full((LAT_S.size, LON_S.size), 2.0)]).astype(np.float32)),
                     "vo": (("valid_time", "latitude", "longitude"),
                            np.zeros((2, LAT_S.size, LON_S.size), np.float32))},
                    coords={"valid_time": t2, "latitude": LAT_S, "longitude": LON_S})
    ds.to_netcdf(tmp_path / "hours.nc")
    d = read_era5([tmp_path / "hours.nc"])
    assert d["dates"].size == 1 and d["hours_per_day"][0] == 2
    np.testing.assert_allclose(d["z"], 1.0)


def test_end_to_end_on_synthetic_stand_in(tmp_path):
    """Label + benchmark on a synthetic stand-in (4 'winters' of 30 December
    days, one planted 7-day block each). Checks the plumbing, the label
    recovery and the report; the verdict on fake data means nothing."""
    xr = pytest.importorskip("xarray")
    pytest.importorskip("netCDF4")
    pytest.importorskip("gudhi")
    from src.realdata import label_blocking, run_realdata_benchmark

    cfg = _cfg("realdata_quick.yaml")
    cfg.update({"n_workers": 2, "out_dir": str(tmp_path / "results"),
                "season": {"months": [12, 1, 2], "first": 2000, "last": 2003}})
    cfg["download"]["out_dir"] = str(tmp_path / "raw")
    cfg["labeling"]["output"] = str(tmp_path / "labelled.npz")
    cfg["benchmark"].update({"cache_dir": str(tmp_path / "cache"),
                             "evaluation": {"n_splits": 2, "n_repeats": 1, "bootstrap": 50}})
    key = spatial_key(cfg["download"])
    (tmp_path / "raw").mkdir()
    rng = np.random.default_rng(42)
    planted = {}
    for i, year in enumerate(range(2000, 2004)):
        days = np.datetime64(f"{year}-12-01") + np.arange(30)
        start = int(rng.integers(6, 17))
        planted[year] = (days, set(range(start, start + 7)))
        Z = _synthetic_heights(30, planted[year][1], rng)
        _write_nc(tmp_path / "raw" / f"era5_500hPa_{year}_m12_{key}.nc", days, Z,
                  _geostrophic_vorticity(Z), ("new", "legacy", "zip", "new")[i], xr)

    out, summ = label_blocking.run(cfg)
    lab = np.load(out)
    labels = lab["labels"]
    expect = []
    for year in range(2000, 2004):
        b = planted[year][1]
        for t in range(30):
            dist = min(abs(t - s) for s in b)
            expect.append(BLOCKING if t in b else (AMBIGUOUS if dist <= 2 else ZONAL))
    np.testing.assert_array_equal(labels, expect)
    assert summ["n_episodes"] == 4 and summ["counts"]["BLOCKING"] == 28
    assert lab["vorticity"].shape == (120, LAT_S.size, LON_S.size)
    assert (tmp_path / "results" / "realdata_labels_quick.md").exists()

    res = run_realdata_benchmark.run(cfg)
    assert res["data"]["n_blocking"] == res["data"]["n_zonal"] == 28
    assert res["data"]["n_groups"] == 4
    assert res["verdict_overall"] in ("GO", "NO-GO")
    task = res["tasks"]["blocking_vs_zonal"]
    for k in ("descriptor", "baselines_union", "x1x2", "topo_rich_all", "raw_pixels"):
        assert 0.0 <= task["results"][k]["best_accuracy"] <= 1.0
    md = (tmp_path / "results" / "realdata_benchmark_quick.md").read_text()
    for s in ("## Overall verdict", "Leakage check", "baselines_union", "Tibaldi & Molteni"):
        assert s in md
    js = json.loads((tmp_path / "results" / "realdata_benchmark_quick.json").read_text())
    assert '"oof"' not in json.dumps(js)
    for f in ("realdata_benchmark_quick_fields.png", "realdata_benchmark_quick_accuracy.png",
              "realdata_benchmark_quick_margin_vs_union.png", "realdata_ib_frequency_quick.png"):
        assert (tmp_path / "results" / "figures" / f).exists()
    # the report can be re-rendered from the cached full results
    (tmp_path / "results" / "realdata_benchmark_quick.md").unlink()
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    run_realdata_benchmark.main(["--config", str(cfg_path), "--report-only"])
    assert (tmp_path / "results" / "realdata_benchmark_quick.md").read_text() == md
