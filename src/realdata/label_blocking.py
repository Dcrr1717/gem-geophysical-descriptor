"""Step 2: label every winter day BLOCKING / ZONAL / AMBIGUOUS.

Index: Tibaldi & Molteni (1990), "On the operational predictability of
blocking", Tellus 42A, 343-365 (TM90). For every longitude lambda and day,
with Z the 500 hPa geopotential height (m) and latitudes in degrees,

    GHGS(lambda) = (Z(phi_0) - Z(phi_S)) / (phi_0 - phi_S)
    GHGN(lambda) = (Z(phi_N) - Z(phi_0)) / (phi_N - phi_0)
    phi_N = 80N + D,  phi_0 = 60N + D,  phi_S = 40N + D,  D in {-4, 0, +4} deg

and the longitude is INSTANTANEOUSLY BLOCKED if, for at least one D,

    GHGS > 0   and   GHGN < -10 m per degree latitude

(reversal of the meridional height gradient south of 60N, i.e. easterly
geostrophic flow, with strong westerlies to the north). Z is read off the
grid by linear interpolation in latitude, so off-grid latitudes are exact for
any grid spacing.

Day labels for one longitude sector (default Euro-Atlantic, 45W-45E):

* the sector is BLOCKED on a day if its longest run of adjacent blocked grid
  longitudes spans >= `min_extent_deg` (15 deg; the large-scale extent
  criterion of Davini et al. 2012, J. Climate 25, 6496-6509), where the span
  of a run is the longitude difference between its first and last point;
* BLOCKING (1): the day belongs to a run of >= `min_duration_days` (5)
  consecutive calendar days with the sector blocked (persistence criterion,
  as for the blocking events of Davini et al. 2012);
* ZONAL (0): no longitude of the sector is instantaneously blocked that day
  (the TM90 reversal is absent at every longitude and every D) and the day is
  more than `zonal_buffer_days` (2) days away from any BLOCKING day;
  optionally (`zonal_max_ghgs`, off by default) the sector-mean GHGS at D = 0
  must also be <= the given value (strong westerlies);
* AMBIGUOUS (-1): everything else (short or narrow blocks, transition days);
  excluded from the benchmark.

All thresholds are config parameters (`labeling:` in config/realdata.yaml)
and are fixed before looking at any classification result.

Usage
  python -m src.realdata.label_blocking --config config/realdata.yaml
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np

from .common import (day_numbers, in_seasons, load_config, lon_window, normalize_lon,
                     season_of, spatial_key, validate_config)

G0 = 9.80665                       # standard gravity (m s^-2), WMO
BLOCKING, ZONAL, AMBIGUOUS = 1, 0, -1
LABEL_NAMES = {BLOCKING: "BLOCKING", ZONAL: "ZONAL", AMBIGUOUS: "AMBIGUOUS"}

DEFAULTS = {
    "phi_n": 80.0, "phi_0": 60.0, "phi_s": 40.0, "deltas": (-4.0, 0.0, 4.0),
    "ghgs_min": 0.0, "ghgn_max": -10.0,
    "sector": {"west": -45.0, "east": 45.0},
    "min_extent_deg": 15.0, "min_duration_days": 5, "zonal_buffer_days": 2,
    "zonal_max_ghgs": None,
}


# ---------------------------------------------------------------------------
# Tibaldi-Molteni index (pure numpy)
# ---------------------------------------------------------------------------
def _as_float(a):
    a = np.asarray(a)
    return a if np.issubdtype(a.dtype, np.floating) else a.astype(float)


def geopotential_to_height(z):
    """Geopotential (m^2 s^-2) -> geopotential height (m); float32 stays float32
    (ample precision for gradients of ~10 m/deg, half the memory of float64)."""
    return _as_float(z) / G0


def interp_lat(field, lat, phi):
    """Linear interpolation of `field` (..., nlat, nlon) at latitude `phi`.
    `lat` may be ascending or descending; `phi` must lie inside its range."""
    lat = np.asarray(lat, dtype=float)
    order = np.argsort(lat)
    ls = lat[order]
    if len(ls) < 2 or np.any(np.diff(ls) <= 0):
        raise ValueError("latitudes must be distinct (at least two)")
    if not ls[0] - 1e-9 <= phi <= ls[-1] + 1e-9:
        raise ValueError(f"latitude {phi} outside the data range [{ls[0]}, {ls[-1]}]; "
                         "enlarge download.area")
    j = int(np.clip(np.searchsorted(ls, phi), 1, len(ls) - 1))
    i = j - 1
    w = (phi - ls[i]) / (ls[j] - ls[i])
    a = np.take(field, order[i], axis=-2)
    b = np.take(field, order[j], axis=-2)
    return (1.0 - w) * a + w * b


def tm_gradients(Z, lat, phi_n=80.0, phi_0=60.0, phi_s=40.0, deltas=(-4.0, 0.0, 4.0)):
    """GHGS, GHGN (m per degree latitude), each shaped (len(deltas), ..., nlon)."""
    gs, gn = [], []
    for d in deltas:
        zn, z0, zs = (interp_lat(Z, lat, p + d) for p in (phi_n, phi_0, phi_s))
        gs.append((z0 - zs) / (phi_0 - phi_s))
        gn.append((zn - z0) / (phi_n - phi_0))
    return np.stack(gs), np.stack(gn)


def instantaneous_blocking(Z, lat, ghgs_min=0.0, ghgn_max=-10.0, phi_n=80.0, phi_0=60.0,
                           phi_s=40.0, deltas=(-4.0, 0.0, 4.0)):
    """TM90 instantaneous blocking per longitude: (..., nlon) bool. Strict
    inequalities: GHGS > ghgs_min and GHGN < ghgn_max for at least one delta."""
    gs, gn = tm_gradients(Z, lat, phi_n, phi_0, phi_s, deltas)
    return np.any((gs > ghgs_min) & (gn < ghgn_max), axis=0)


def grid_spacing(lon):
    """Longitude spacing of a regular grid (degrees)."""
    u = np.unique(np.round(normalize_lon(lon), 6))
    d = np.diff(np.r_[u, u[0] + 360.0])
    return float(np.median(d[d > 0]))


def longest_run_span(flags, rel, dlon):
    """Span (deg) of the longest run of adjacent True longitudes per row.

    flags: (T, m) bool, columns ordered west -> east with eastward offsets
    `rel` (m,). Two columns are adjacent if their offsets differ by at most
    1.5 grid spacings. The span of a run is rel[last] - rel[first] (a single
    blocked point has span 0). Rows without any True get -1."""
    flags = np.atleast_2d(np.asarray(flags, dtype=bool))
    rel = np.asarray(rel, dtype=float)
    out = np.full(flags.shape[0], -1.0)
    for t, row in enumerate(flags):
        idx = np.flatnonzero(row)
        if idx.size == 0:
            continue
        r = rel[idx]
        brk = np.flatnonzero((np.diff(idx) != 1) | (np.diff(r) > 1.5 * dlon))
        starts = np.r_[0, brk + 1]
        ends = np.r_[brk, idx.size - 1]
        out[t] = float(np.max(r[ends] - r[starts]))
    return out


def sector_blocking(ib, lon, sector, min_extent_deg=15.0):
    """Sector-level diagnostics from per-longitude blocking `ib` (T, nlon)."""
    idx, rel = lon_window(lon, sector["west"], sector["east"])
    if idx.size == 0:
        raise ValueError(f"no grid longitude inside the sector {sector}")
    sub = np.asarray(ib, dtype=bool)[:, idx]
    span = longest_run_span(sub, rel, grid_spacing(lon))
    return {"sector_idx": idx, "any_ib": sub.any(axis=1), "span": span,
            "sector_blocked": span >= float(min_extent_deg)}


def persistent_runs(flag, dates, min_days):
    """Days belonging to runs of >= `min_days` consecutive CALENDAR days with
    `flag` True (a gap in the dates breaks a run). Returns (mask, episodes)
    with episodes = [(first index, last index), ...]. Dates must be sorted."""
    flag = np.asarray(flag, dtype=bool)
    d = day_numbers(dates)
    if np.any(np.diff(d) <= 0):
        raise ValueError("dates must be strictly increasing")
    mask = np.zeros(flag.size, bool)
    episodes = []
    i = 0
    while i < flag.size:
        if not flag[i]:
            i += 1
            continue
        j = i
        while j + 1 < flag.size and flag[j + 1] and d[j + 1] - d[j] == 1:
            j += 1
        if j - i + 1 >= int(min_days):
            mask[i:j + 1] = True
            episodes.append((i, j))
        i = j + 1
    return mask, episodes


def _distance_to(mask, d):
    """Calendar-day distance from every day to the nearest day with mask True."""
    tgt = np.sort(d[mask])
    if tgt.size == 0:
        return np.full(d.size, np.iinfo(np.int64).max)
    k = np.searchsorted(tgt, d)
    lo = np.abs(d - tgt[np.clip(k - 1, 0, tgt.size - 1)])
    hi = np.abs(tgt[np.clip(k, 0, tgt.size - 1)] - d)
    return np.minimum(lo, hi)


def label_days(Z, lat, lon, dates, **params):
    """Label days from 500 hPa geopotential HEIGHT Z (T, nlat, nlon) in metres.

    Returns a dict with `labels` (T,) int8 in {1 BLOCKING, 0 ZONAL,
    -1 AMBIGUOUS} and the diagnostics behind them."""
    unknown = set(params) - set(DEFAULTS)
    if unknown:
        raise TypeError(f"unknown labeling parameters: {sorted(unknown)}")
    p = {**DEFAULTS, **params}
    Z = _as_float(Z)
    if Z.ndim != 3:
        raise ValueError("Z must be (time, lat, lon)")
    deltas = tuple(float(x) for x in p["deltas"])
    geo = dict(phi_n=float(p["phi_n"]), phi_0=float(p["phi_0"]), phi_s=float(p["phi_s"]))
    ib = instantaneous_blocking(Z, lat, float(p["ghgs_min"]), float(p["ghgn_max"]),
                                deltas=deltas, **geo)
    sec = sector_blocking(ib, lon, p["sector"], float(p["min_extent_deg"]))
    blocking, episodes = persistent_runs(sec["sector_blocked"], dates, int(p["min_duration_days"]))
    d = day_numbers(dates)
    dist = _distance_to(blocking, d)
    gs0, _ = tm_gradients(Z, lat, deltas=(0.0,), **geo)
    ghgs_mean = gs0[0][:, sec["sector_idx"]].mean(axis=1)
    zonal = ~sec["any_ib"] & ~blocking & (dist > int(p["zonal_buffer_days"]))
    if p["zonal_max_ghgs"] is not None:
        zonal &= ghgs_mean <= float(p["zonal_max_ghgs"])
    labels = np.full(len(d), AMBIGUOUS, dtype=np.int8)
    labels[zonal] = ZONAL
    labels[blocking] = BLOCKING
    ds = np.asarray(dates, dtype="datetime64[D]")
    return {"labels": labels, "ib": ib, "any_ib": sec["any_ib"], "span": sec["span"],
            "sector_blocked": sec["sector_blocked"], "ghgs_sector_mean": ghgs_mean,
            "episodes": [{"start": str(ds[i]), "end": str(ds[j]), "days": int(j - i + 1)}
                         for i, j in episodes],
            "params": {**p, "deltas": list(deltas)}}


# ---------------------------------------------------------------------------
# ERA5 netCDF input (xarray, imported lazily)
# ---------------------------------------------------------------------------
_TIME = ("valid_time", "time")
_LEVEL = ("pressure_level", "level", "isobaricInhPa")
_LAT = ("latitude", "lat")
_LON = ("longitude", "lon")
_VARS = {"z": ("z", "geopotential"), "vo": ("vo", "vorticity")}


def _first(names, pool, what):
    for n in names:
        if n in pool:
            return n
    raise KeyError(f"no {what} coordinate/variable among {names}; found {sorted(pool)}")


def _nc_paths(path, tmpdir):
    """A downloaded file may be a netCDF or a zip of netCDFs (CDS packs
    multi-file results); return the netCDF paths."""
    path = Path(path)
    if zipfile.is_zipfile(path):
        out = Path(tmpdir) / path.stem
        with zipfile.ZipFile(path) as zf:
            names = [n for n in zf.namelist() if n.endswith(".nc")]
            for n in names:
                if Path(n).is_absolute() or ".." in Path(n).parts:
                    raise ValueError(f"unsafe member {n!r} in {path}")
            zf.extractall(out, members=names)
        return [out / n for n in names]
    return [path]


def _read_one(ds, level):
    tname = _first(_TIME, set(ds.dims) | set(ds.coords), "time")
    if tname not in ds.dims:
        ds = ds.expand_dims(tname)
    for lv in _LEVEL:
        if lv in ds.dims:
            ds = ds.sel({lv: level}) if ds.sizes[lv] > 1 else ds.isel({lv: 0})
        elif lv in ds.coords:
            ds = ds.drop_vars(lv)
    if "expver" in ds.dims:
        # legacy CDS: ERA5 (1) and preliminary ERA5T (5) halves, NaN where absent
        ds = ds.max("expver", skipna=True)
    lat_n = _first(_LAT, set(ds.coords) | set(ds.dims), "latitude")
    lon_n = _first(_LON, set(ds.coords) | set(ds.dims), "longitude")
    lat = ds[lat_n].values.astype(float)
    lon = normalize_lon(ds[lon_n].values.astype(float))
    lo, oo = np.argsort(lat), np.argsort(lon)       # canonical: lat ascending, lon [-180, 180)
    out = {"time": ds[tname].values.astype("datetime64[s]"), "lat": lat[lo], "lon": lon[oo]}
    for key, names in _VARS.items():
        v = _first(names, set(ds.data_vars), key)
        a = ds[v].transpose(tname, lat_n, lon_n).values.astype(np.float32)
        out[key] = a[:, lo][:, :, oo]
        out[key + "_units"] = str(ds[v].attrs.get("units", ""))
    return out


def read_era5(paths, level=500):
    """Read raw ERA5 files -> dict(dates (D,), lat asc, lon [-180,180) asc,
    z (D, lat, lon), vo (D, lat, lon), z_units). Several hours of the same day
    are averaged (daily mean); duplicate timestamps across files are dropped."""
    import xarray as xr

    parts = []
    with tempfile.TemporaryDirectory() as tmp:
        for p in paths:
            dss = []
            for nc in _nc_paths(p, tmp):
                with xr.open_dataset(nc) as ds:
                    dss.append(ds.load())
            # a zip from CDS may hold z and vo in separate files: merge them
            parts.append(_read_one(dss[0] if len(dss) == 1 else xr.merge(dss), level))
    if not parts:
        raise ValueError("no ERA5 files to read")
    lat0, lon0 = parts[0]["lat"], parts[0]["lon"]
    for q in parts[1:]:
        if q["lat"].shape != lat0.shape or q["lon"].shape != lon0.shape or \
                not (np.allclose(q["lat"], lat0) and np.allclose(q["lon"], lon0)):
            raise ValueError("raw files have different grids; use one download config "
                             "(same area and grid) per output")
    t = np.concatenate([q["time"] for q in parts])
    z = np.concatenate([q["z"] for q in parts])
    vo = np.concatenate([q["vo"] for q in parts])
    t, first = np.unique(t, return_index=True)          # drop duplicate timestamps, sort
    z, vo = z[first], vo[first]
    days = t.astype("datetime64[D]")                    # sorted, so each day is contiguous
    dates, starts, cnt = np.unique(days, return_index=True, return_counts=True)

    def daily(a):
        if np.all(cnt == 1):
            return a
        s = np.add.reduceat(a, starts, axis=0, dtype=np.float64)
        return (s / cnt[:, None, None]).astype(np.float32)
    return {"dates": dates, "lat": lat0, "lon": lon0, "z": daily(z), "vo": daily(vo),
            "z_units": parts[0]["z_units"], "hours_per_day": cnt}


def height_from_raw(z, units):
    """ERA5 `z` is geopotential (m^2 s^-2); accept height (m) too."""
    u = units.replace(" ", "").lower()
    Z = _as_float(z) if u in ("m", "gpm") else geopotential_to_height(z)
    med = float(np.median(Z))
    if not 4000.0 < med < 6500.0:
        print(f"[label] WARNING: median 500 hPa height {med:.0f} m is implausible "
              f"(units '{units}'); check the input", file=sys.stderr)
    return Z


# ---------------------------------------------------------------------------
# Summary report
# ---------------------------------------------------------------------------
def label_summary(lab, dates, lon, months):
    labels = lab["labels"]
    seasons = season_of(dates, months)
    n = int(labels.size)
    durs = np.array([e["days"] for e in lab["episodes"]], dtype=float)
    ep_season = season_of(np.array([e["start"] for e in lab["episodes"]], "datetime64[D]"),
                          months)
    per_season = []
    for s in np.unique(seasons):
        m = seasons == s
        per_season.append({"season": int(s), "days": int(m.sum()),
                           **{LABEL_NAMES[k]: int((labels[m] == k).sum()) for k in LABEL_NAMES},
                           "episodes": int((ep_season == s).sum())})
    return {"n_days": n,
            "counts": {LABEL_NAMES[k]: int((labels == k).sum()) for k in LABEL_NAMES},
            "fractions": {LABEL_NAMES[k]: float((labels == k).mean()) if n else 0.0
                          for k in LABEL_NAMES},
            "sector_blocked_day_fraction": float(lab["sector_blocked"].mean()) if n else 0.0,
            "any_ib_in_sector_day_fraction": float(lab["any_ib"].mean()) if n else 0.0,
            "n_episodes": int(durs.size),
            "episode_days_mean": float(durs.mean()) if durs.size else 0.0,
            "episode_days_median": float(np.median(durs)) if durs.size else 0.0,
            "episode_days_max": int(durs.max()) if durs.size else 0,
            "ib_frequency_by_lon": {"lon": [float(x) for x in lon],
                                    "freq": [float(x) for x in lab["ib"].mean(axis=0)]},
            "per_season": per_season,
            "episodes": lab["episodes"]}


def write_label_report(path, summ, cfg, meta):
    p = summ
    lb = {**DEFAULTS, **cfg["labeling"]}
    sec = lb["sector"]
    c, f = p["counts"], p["fractions"]
    L = ["# ERA5 blocking labels (Tibaldi-Molteni 1990)", "",
         f"Source: {meta['n_files']} raw ERA5 file(s), {p['n_days']} days in seasons "
         f"{cfg['season']['first']}-{cfg['season']['last']} (months {cfg['season']['months']}), "
         f"500 hPa, grid {meta['grid_deg']:.2f} deg, {meta['hours_per_day']} hour(s) per day "
         "(daily mean if > 1).", "",
         "## Definition", "",
         f"- GHGS = (Z({lb['phi_0']}+D) - Z({lb['phi_s']}+D)) / {lb['phi_0'] - lb['phi_s']}, "
         f"GHGN = (Z({lb['phi_n']}+D) - Z({lb['phi_0']}+D)) / {lb['phi_n'] - lb['phi_0']}, "
         f"D in {list(lb['deltas'])} deg; longitude blocked iff GHGS > {lb['ghgs_min']} and "
         f"GHGN < {lb['ghgn_max']} m/deg for some D (Tibaldi & Molteni 1990).",
         f"- Sector {sec['west']} to {sec['east']} deg E blocked iff a run of adjacent blocked "
         f"longitudes spans >= {lb['min_extent_deg']} deg.",
         f"- BLOCKING: >= {lb['min_duration_days']} consecutive sector-blocked days. ZONAL: no "
         f"blocked longitude in the sector and > {lb['zonal_buffer_days']} days from any "
         f"BLOCKING day" + (f", sector-mean GHGS(D=0) <= {lb['zonal_max_ghgs']} m/deg"
                            if lb.get("zonal_max_ghgs") is not None else "")
         + ". AMBIGUOUS: the rest (excluded from the benchmark).", "",
         "## Counts", "",
         "| label | days | fraction |", "|---|---|---|"]
    for k in ("BLOCKING", "ZONAL", "AMBIGUOUS"):
        L.append(f"| {k} | {c[k]} | {f[k]:.3f} |")
    L += ["", f"- Blocking episodes: {p['n_episodes']} (duration mean "
              f"{p['episode_days_mean']:.1f} d, median {p['episode_days_median']:.1f} d, max "
              f"{p['episode_days_max']} d).",
          f"- Days with the sector blocked (no persistence filter): "
          f"{p['sector_blocked_day_fraction']:.3f}; days with at least one blocked longitude "
          f"in the sector: {p['any_ib_in_sector_day_fraction']:.3f}.",
          f"- Sanity check: `figures/{meta['figure']}` shows the instantaneous-blocking "
          "frequency per longitude. Compare its shape with published winter climatologies of "
          "the TM90 index before trusting the labels.", "",
          "## Per season", "", "| season | days | BLOCKING | ZONAL | AMBIGUOUS | episodes |",
          "|---|---|---|---|---|---|"]
    for r in p["per_season"]:
        L.append(f"| {r['season']} | {r['days']} | {r['BLOCKING']} | {r['ZONAL']} | "
                 f"{r['AMBIGUOUS']} | {r['episodes']} |")
    Path(path).write_text("\n".join(L) + "\n")


def ib_frequency_figure(summ, sector, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from ..eval.hard_figures import SURFACE, INK, INK2, GRID, _style

    lon = np.array(summ["ib_frequency_by_lon"]["lon"])
    fr = np.array(summ["ib_frequency_by_lon"]["freq"])
    fig, ax = plt.subplots(figsize=(8, 3.2), facecolor=SURFACE)
    _style(ax)
    idx, _ = lon_window(lon, sector["west"], sector["east"])
    ax.axvspan(lon[idx].min(), lon[idx].max(), color=GRID, alpha=0.7, lw=0,
               label="labelling sector")
    ax.plot(lon, fr, color="#2a78d6", lw=1.8)
    ax.set_xlabel("longitude (deg E)", fontsize=9, color=INK2)
    ax.set_ylabel("fraction of days blocked", fontsize=9, color=INK2)
    ax.set_title("Instantaneous blocking frequency (Tibaldi-Molteni 1990), ERA5 500 hPa",
                 fontsize=10, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def labeling_params(cfg):
    lb = cfg.get("labeling", {})
    return {k: lb[k] for k in DEFAULTS if k in lb}


def run(cfg):
    validate_config(cfg)
    dl, season, lb = cfg["download"], cfg["season"], cfg["labeling"]
    raw = Path(dl.get("out_dir", "data/era5/raw"))
    key = spatial_key(dl)
    paths = sorted(raw.glob(f"era5_*_{key}.nc"))
    if not paths:
        raise SystemExit(f"[label] no raw files era5_*_{key}.nc in {raw}; run "
                         "`python -m src.realdata.download_era5` with the same config first")
    print(f"[label] reading {len(paths)} raw file(s) from {raw}")
    data = read_era5(paths, level=int(dl["pressure_level"]))
    keep = in_seasons(data["dates"], season["months"], season["first"], season["last"])
    dates = data["dates"][keep]
    if dates.size == 0:
        raise SystemExit("[label] no day of the configured seasons in the raw files")
    Z = height_from_raw(data["z"][keep], data["z_units"])
    lab = label_days(Z, data["lat"], data["lon"], dates, **labeling_params(cfg))
    summ = label_summary(lab, dates, data["lon"], season["months"])
    out = Path(lb.get("output", "data/era5/labelled_blocking.npz"))
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out, dates=dates, lat=data["lat"], lon=data["lon"],
        vorticity=data["vo"][keep].astype(np.float32), labels=lab["labels"],
        season=season_of(dates, season["months"]), ib=lab["ib"], span=lab["span"],
        any_ib=lab["any_ib"], sector_blocked=lab["sector_blocked"],
        ghgs_sector_mean=lab["ghgs_sector_mean"],
        params=np.array(json.dumps(lab["params"])), spatial_key=np.array(key),
        source_files=np.array([p.name for p in paths]))
    print(f"[label] {dates.size} days -> {out}: " + ", ".join(
        f"{k} {v}" for k, v in summ["counts"].items()))
    rep_dir = Path(cfg.get("out_dir", "results"))
    rep_dir.mkdir(parents=True, exist_ok=True)
    stem = lb.get("report_stem", "realdata_labels")
    fig_name = f"{stem.replace('labels', 'ib_frequency')}.png"
    meta = {"n_files": len(paths), "grid_deg": float(np.median(np.diff(data["lat"]))),
            "hours_per_day": int(np.median(data["hours_per_day"])), "figure": fig_name}
    (rep_dir / f"{stem}.json").write_text(json.dumps(
        {"config": {"season": season, "download": dl, "labeling": lab["params"]},
         **meta, **summ}, indent=2, default=str))
    write_label_report(rep_dir / f"{stem}.md", summ, cfg, meta)
    if cfg.get("figures", True):
        (rep_dir / "figures").mkdir(parents=True, exist_ok=True)
        ib_frequency_figure(summ, {**DEFAULTS, **lb}["sector"], rep_dir / "figures" / fig_name)
    return out, summ


def main(argv=None):
    ap = argparse.ArgumentParser(description="Label ERA5 days BLOCKING / ZONAL (TM90).")
    ap.add_argument("--config", default="config/realdata.yaml")
    args = ap.parse_args(argv)
    run(load_config(args.config))


if __name__ == "__main__":
    main()
