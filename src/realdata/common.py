"""Bookkeeping shared by the real-data steps (no network, no heavy dependencies).

* config loading and validation (catches an inconsistent config before a long
  download starts);
* winter seasons: which calendar months to request per year, which season a
  day belongs to;
* longitude windows that may cross the dateline;
* time-block groups for grouped cross-validation, so that consecutive
  (autocorrelated) days never straddle the train / test split.

Season convention: a season is identified by the calendar year of its FIRST
month. With `months: [12, 1, 2]` (DJF), season 1999 is Dec 1999 + Jan-Feb 2000.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import yaml

PERIODIZE_METHODS = ("taper", "mirror", "none")
GROUPING_MODES = ("season", "block")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
def load_config(path) -> dict:
    cfg = yaml.safe_load(Path(path).read_text())
    if not isinstance(cfg, dict):
        raise ValueError(f"{path}: not a YAML mapping")
    return cfg


def spatial_key(dl: dict) -> str:
    """Short hash of everything that fixes the CONTENT of a raw file except the
    dates (dataset, variables, level, area, grid, times). Raw files from two
    configs are interchangeable iff their keys match."""
    keep = {k: dl.get(k) for k in ("dataset", "variables", "pressure_level", "area", "grid",
                                   "times")}
    return hashlib.sha1(json.dumps(keep, sort_keys=True).encode()).hexdigest()[:8]


def validate_config(cfg: dict) -> None:
    """Raise ValueError listing every inconsistency in a real-data config."""
    problems = []
    try:
        validate_months(cfg["season"]["months"])
    except (KeyError, ValueError, TypeError) as e:
        problems.append(f"season.months: {e}")
    s = cfg.get("season", {})
    if not int(s.get("first", 0)) <= int(s.get("last", -1)):
        problems.append("season.first must be <= season.last")

    dl, lab, bm = cfg.get("download", {}), cfg.get("labeling", {}), cfg.get("benchmark", {})
    area = dl.get("area", {})
    try:
        N, S, W, E = (float(area[k]) for k in ("north", "south", "west", "east"))
    except (KeyError, TypeError, ValueError):
        problems.append("download.area needs numeric north / south / west / east")
        N = S = W = E = None
    if N is not None:
        if not -90 <= S < N <= 90:
            problems.append(f"download.area: need -90 <= south < north <= 90, got {S}, {N}")
        deltas = [float(d) for d in lab.get("deltas", [0.0])]
        lo = float(lab.get("phi_s", 40.0)) + min(deltas)
        hi = float(lab.get("phi_n", 80.0)) + max(deltas)
        if S > lo or N < hi:
            problems.append(f"download.area latitudes [{S}, {N}] do not cover the blocking-index "
                            f"latitudes [{lo}, {hi}] (phi_s + min(deltas), phi_n + max(deltas))")
        sec = lab.get("sector", {})
        if sec and not lon_range_covers(W, E, sec["west"], sec["east"]):
            problems.append(f"download.area longitudes [{W}, {E}] do not cover labeling.sector "
                            f"[{sec['west']}, {sec['east']}]")
        dom = bm.get("domain", {})
        if dom:
            if not lon_range_covers(W, E, dom["west"], dom["east"]):
                problems.append(f"download.area longitudes [{W}, {E}] do not cover "
                                f"benchmark.domain [{dom['west']}, {dom['east']}]")
            if not (S <= float(dom["south"]) < float(dom["north"]) <= N):
                problems.append(f"benchmark.domain latitudes [{dom['south']}, {dom['north']}] "
                                f"not inside download.area [{S}, {N}]")
    if not float(lab.get("phi_s", 40)) < float(lab.get("phi_0", 60)) < float(lab.get("phi_n", 80)):
        problems.append("labeling: need phi_s < phi_0 < phi_n")
    if int(lab.get("min_duration_days", 1)) < 1:
        problems.append("labeling.min_duration_days must be >= 1")
    if float(lab.get("min_extent_deg", 0)) < 0:
        problems.append("labeling.min_extent_deg must be >= 0")
    if int(lab.get("zonal_buffer_days", 0)) < 0:
        problems.append("labeling.zonal_buffer_days must be >= 0")
    if bm:
        n = int(bm.get("grid", 64))
        if n < 32 or n % 8:
            problems.append(f"benchmark.grid must be a multiple of 8 and >= 32 (got {n}); the "
                            "X1 view splits the field into 8 x 8 regions")
        if bm.get("periodize", "taper") not in PERIODIZE_METHODS:
            problems.append(f"benchmark.periodize must be one of {PERIODIZE_METHODS}")
        if bm.get("periodize") == "mirror" and n % 16:
            problems.append("benchmark.periodize = mirror needs grid to be a multiple of 16")
        g = bm.get("grouping", {})
        if g.get("mode", "season") not in GROUPING_MODES:
            problems.append(f"benchmark.grouping.mode must be one of {GROUPING_MODES}")
        if g.get("mode") == "block" and not (
                0 <= int(g.get("gap_days", 0)) < int(g.get("block_days", 30))):
            problems.append("benchmark.grouping: need 0 <= gap_days < block_days")
        if bm.get("balance", "undersample") not in ("undersample", "none"):
            problems.append("benchmark.balance must be 'undersample' or 'none'")
    if problems:
        raise ValueError("invalid real-data config:\n  - " + "\n  - ".join(problems))


# ---------------------------------------------------------------------------
# Seasons
# ---------------------------------------------------------------------------
def validate_months(months) -> list[int]:
    m = [int(x) for x in months]
    if not m or len(m) > 12 or any(not 1 <= x <= 12 for x in m) or len(set(m)) != len(m):
        raise ValueError(f"months must be distinct integers in 1..12, got {months}")
    for a, b in zip(m, m[1:]):
        if b != a % 12 + 1:
            raise ValueError(f"months must be consecutive (e.g. [12, 1, 2]), got {months}")
    return m


def months_by_year(months, first_season, last_season) -> dict[int, list[int]]:
    """{calendar year: months to download} covering seasons first..last (inclusive)."""
    m = validate_months(months)
    out: dict[int, set] = {}
    for s in range(int(first_season), int(last_season) + 1):
        for mm in m:
            out.setdefault(s if mm >= m[0] else s + 1, set()).add(mm)
    return {y: sorted(v) for y, v in sorted(out.items())}


def as_days(dates) -> np.ndarray:
    return np.asarray(dates, dtype="datetime64[D]")


def day_numbers(dates) -> np.ndarray:
    """Integer day numbers (days since 1970-01-01)."""
    return as_days(dates).astype(np.int64)


def season_of(dates, months) -> np.ndarray:
    """Season id (year of the season's first month) per date; -1 outside the season."""
    m0 = validate_months(months)
    d = as_days(dates)
    y = d.astype("datetime64[Y]").astype(np.int64) + 1970
    mo = d.astype("datetime64[M]").astype(np.int64) % 12 + 1
    s = np.where(mo >= m0[0], y, y - 1)
    return np.where(np.isin(mo, m0), s, -1)


def in_seasons(dates, months, first, last) -> np.ndarray:
    s = season_of(dates, months)
    return (s >= int(first)) & (s <= int(last))


# ---------------------------------------------------------------------------
# Longitudes
# ---------------------------------------------------------------------------
def normalize_lon(lon):
    """Map longitudes to [-180, 180)."""
    return (np.asarray(lon, dtype=float) + 180.0) % 360.0 - 180.0


def lon_width(west, east) -> float:
    """Eastward width of [west, east] in degrees, in (0, 360]."""
    w = (float(east) - float(west)) % 360.0
    return 360.0 if w == 0 else w


def lon_window(lon, west, east, tol=1e-6):
    """Indices of the longitudes inside [west, east] (eastward, may cross the
    dateline), ordered west -> east, and their eastward offsets from `west`."""
    rel = (normalize_lon(lon) - float(west)) % 360.0
    rel = np.where(rel > 360.0 - tol, 0.0, rel)
    sel = np.flatnonzero(rel <= lon_width(west, east) + tol)
    order = np.argsort(rel[sel], kind="stable")
    return sel[order], rel[sel][order]


def lon_range_covers(outer_w, outer_e, inner_w, inner_e, tol=1e-6) -> bool:
    W = lon_width(outer_w, outer_e)
    a = (float(inner_w) - float(outer_w)) % 360.0
    a = 0.0 if a > 360.0 - tol else a
    return a + lon_width(inner_w, inner_e) <= W + tol


# ---------------------------------------------------------------------------
# Grouped-CV time blocks
# ---------------------------------------------------------------------------
def time_groups(dates, months, mode="season", block_days=30, gap_days=5):
    """Group ids for grouped cross-validation, plus a mask of the days to keep.

    mode = "season": one group per season. Two DJF seasons are separated by
        nine months, far beyond the ~1-2 week decorrelation time of 500 hPa
        flow, so no autocorrelated pair of days can straddle train and test.
    mode = "block": consecutive blocks of `block_days` calendar days inside
        each season, counted from the season's first nominal day (e.g. 1 Dec).
        The last `gap_days` days of every block that is followed by another
        block of the same season are dropped (purged), so two days in
        different groups are always more than `gap_days` days apart.

    Returns (groups: int array, keep: bool array), aligned with `dates`.
    """
    m = validate_months(months)
    d = day_numbers(dates)
    season = season_of(dates, m)
    if (season < 0).any():
        raise ValueError("time_groups: some dates are outside the configured season months")
    if mode == "season":
        keys, keep = season, np.ones(len(d), bool)
    elif mode == "block":
        block_days, gap_days = int(block_days), int(gap_days)
        if not 0 <= gap_days < block_days:
            raise ValueError("time_groups: need 0 <= gap_days < block_days")
        start = np.array([np.datetime64(f"{s:04d}-{m[0]:02d}-01", "D").astype(np.int64)
                          for s in season], dtype=np.int64)
        off = d - start
        block = off // block_days
        pos = off % block_days
        last = {s: block[season == s].max() for s in np.unique(season)}
        is_last = np.array([b == last[s] for s, b in zip(season, block)], dtype=bool)
        keep = (pos < block_days - gap_days) | is_last
        keys = season.astype(np.int64) * 1000 + block
    else:
        raise ValueError(f"time_groups: mode must be one of {GROUPING_MODES}, got {mode!r}")
    groups = np.unique(keys, return_inverse=True)[1].astype(int)
    return groups, keep
