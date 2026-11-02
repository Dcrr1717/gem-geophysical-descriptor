"""Step 1: download ERA5 500 hPa geopotential and relative vorticity (Copernicus CDS).

Data: ERA5 hourly data on pressure levels (Hersbach et al. 2020, QJRMS 146,
1999-2049; doi:10.1002/qj.3803), CDS dataset `reanalysis-era5-pressure-levels`,
variables `geopotential` (netCDF name `z`, m^2 s^-2) and `vorticity` (relative
vorticity, `vo`, s^-1) at 500 hPa, over the area, grid, hours and winter
seasons set in the config. One request (= one netCDF file) per calendar year,
containing only the season months of that year; existing files are skipped,
so an interrupted run can simply be restarted.

Credentials are NEVER stored in this repository. `cdsapi` reads them from

* the environment variables CDSAPI_URL and CDSAPI_KEY, or
* the file ~/.cdsapirc (or the file named by CDSAPI_RC):

      url: https://cds.climate.copernicus.eu/api
      key: <your personal access token>

See docs/REALDATA.md for the free registration and the licence acceptance.

Usage
  python -m src.realdata.download_era5 --config config/realdata.yaml --dry-run
  python -m src.realdata.download_era5 --config config/realdata.yaml
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from .common import load_config, months_by_year, spatial_key, validate_config

CDS_HELP = """\
No Copernicus CDS credentials found. Create a free account at
https://cds.climate.copernicus.eu, accept the licence of the dataset
'ERA5 hourly data on pressure levels from 1940 to present', copy your
Personal Access Token from your CDS profile page, and EITHER create the file
~/.cdsapirc (Windows: %USERPROFILE%\\.cdsapirc) containing

    url: https://cds.climate.copernicus.eu/api
    key: <your personal access token>

OR set the environment variables CDSAPI_URL and CDSAPI_KEY.
Never commit the token. Full steps: docs/REALDATA.md."""


class CredentialsError(RuntimeError):
    pass


def check_credentials(environ=None, home=None) -> str:
    """Return a description of where cdsapi will find credentials (never the
    token itself); raise CredentialsError if there are none."""
    env = os.environ if environ is None else environ
    if env.get("CDSAPI_URL") and env.get("CDSAPI_KEY"):
        return "environment variables CDSAPI_URL / CDSAPI_KEY"
    rc = env.get("CDSAPI_RC") or str(Path(home or Path.home()) / ".cdsapirc")
    if not Path(rc).is_file():
        raise CredentialsError(CDS_HELP)
    fields = {}
    for line in Path(rc).read_text().splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            fields[k.strip().lower()] = v.strip()
    if not fields.get("url") or not fields.get("key"):
        raise CredentialsError(f"{rc} must contain a 'url:' line and a 'key:' line.\n\n"
                               + CDS_HELP)
    if ":" in fields["key"]:
        print(f"[download] WARNING: the key in {rc} looks like the legacy 'UID:KEY' format. "
              "The current CDS expects only the Personal Access Token.", file=sys.stderr)
    return f"file {rc}"


def build_requests(cfg) -> list[dict]:
    """One CDS request per calendar year: {year, months, target, dataset, request}."""
    dl, season = cfg["download"], cfg["season"]
    key = spatial_key(dl)
    out_dir = Path(dl.get("out_dir", "data/era5/raw"))
    a = dl["area"]
    level = int(dl["pressure_level"])
    jobs = []
    for year, months in months_by_year(season["months"], season["first"],
                                       season["last"]).items():
        req = {
            "product_type": ["reanalysis"],
            "variable": list(dl["variables"]),
            "pressure_level": [str(level)],
            "year": [str(year)],
            "month": [f"{m:02d}" for m in months],
            "day": [f"{d:02d}" for d in range(1, 32)],   # invalid dates are ignored by CDS
            "time": list(dl["times"]),
            # CDS order: [North, West, South, East]
            "area": [float(a["north"]), float(a["west"]), float(a["south"]), float(a["east"])],
            "data_format": "netcdf",
            "download_format": "unarchived",
        }
        if dl.get("grid"):
            req["grid"] = [float(g) for g in dl["grid"]]
        mtag = "-".join(f"{m:02d}" for m in months)
        jobs.append({"year": year, "months": months, "dataset": dl["dataset"], "request": req,
                     "target": out_dir / f"era5_{level}hPa_{year}_m{mtag}_{key}.nc"})
    return jobs


def _sha256(path, chunk=1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def retrieve(client, job, retries=3, backoff=60.0, sleep=time.sleep) -> bool:
    """Download one job to `<target>.part`, then rename. Returns True if a
    download happened, False if the target already existed."""
    target = Path(job["target"])
    if target.exists():
        print(f"[download] exists, skipping {target.name}")
        return False
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_name(target.name + ".part")
    for attempt in range(1, retries + 1):
        try:
            print(f"[download] {target.name}: year {job['year']}, months {job['months']} "
                  f"(attempt {attempt}/{retries})", flush=True)
            client.retrieve(job["dataset"], job["request"], str(part))
            os.replace(part, target)
            return True
        except Exception as e:  # network / queue / server errors: retry with backoff
            print(f"[download] {target.name}: {type(e).__name__}: {e}", file=sys.stderr)
            if part.exists():
                part.unlink()
            if attempt == retries:
                raise
            sleep(backoff * 2 ** (attempt - 1))
    return False


def update_manifest(out_dir, job):
    """Provenance: request and checksum of every raw file."""
    path = Path(out_dir) / "manifest.json"
    man = json.loads(path.read_text()) if path.exists() else {}
    t = Path(job["target"])
    man[t.name] = {"dataset": job["dataset"], "request": job["request"],
                   "bytes": t.stat().st_size, "sha256": _sha256(t),
                   "downloaded_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    path.write_text(json.dumps(man, indent=2, sort_keys=True))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", default="config/realdata.yaml")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the requests and check credentials; contact no server")
    ap.add_argument("--only-years", type=int, nargs="+", default=None,
                    help="restrict to these calendar years (e.g. for a first test)")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    validate_config(cfg)
    jobs = build_requests(cfg)
    if args.only_years:
        jobs = [j for j in jobs if j["year"] in set(args.only_years)]
    todo = [j for j in jobs if not Path(j["target"]).exists()]
    print(f"[download] {len(jobs)} yearly requests, {len(todo)} still to download -> "
          f"{cfg['download'].get('out_dir', 'data/era5/raw')}")
    try:
        src = check_credentials()
        print(f"[download] CDS credentials: {src}")
    except CredentialsError as e:
        if not args.dry_run:
            raise SystemExit(str(e))
        print(f"[download] (dry run) {e}")
    if args.dry_run:
        for j in jobs:
            print(f"  {j['target'].name}\n    {json.dumps(j['request'])}")
        return jobs
    import cdsapi  # imported lazily: tests and --dry-run need no cdsapi install

    client = cdsapi.Client()
    dl = cfg["download"]
    for j in todo:
        if retrieve(client, j, retries=int(dl.get("retries", 3)),
                    backoff=float(dl.get("retry_backoff_s", 60))):
            update_manifest(dl.get("out_dir", "data/era5/raw"), j)
    print("[download] done")
    return jobs


if __name__ == "__main__":
    main()
