# Real-data benchmark: ERA5 winter blocking vs zonal flow

This is a step-by-step guide to running the real-data Go/No-Go benchmark on your
own machine. You need a free Copernicus account, about 0.5 GB of disk, and a
few hours (most of it waiting in the Climate Data Store queue).

No ERA5 data is in this repository, and none was downloaded to build this
pipeline: the cloud session that wrote it cannot reach the CDS. The code is
unit-tested on small synthetic arrays only (`tests/test_realdata_*.py`). The
first real numbers will come from your run.

## Why this benchmark

The synthetic hard benchmark was **NO-GO**: on coherent 2D turbulence vs
IAAFT surrogates, local physical diagnostics (gradient statistics,
Okubo–Weiss, increment flatness) matched or beat every topological
configuration (`docs/RESULTS.md`). Those classes differ in small-scale
structure, and local statistics see that well.

Atmospheric **blocking** is a large-scale regime. On a blocked day the
500 hPa flow reverses: a quasi-stationary high sits poleward of a low, the jet
splits, and closed contours appear. On a zonal day the contours run roughly
west to east. The difference is the connectivity of the large-scale pattern
rather than small-scale statistics. This is the setting where a topological
descriptor could still beat classical local diagnostics. The pipeline tests
that hypothesis with the same rules, baselines and statistics as the synthetic
benchmark.

## What the pipeline does

| Step | Command | Reads | Writes |
|---|---|---|---|
| 1. Download | `python -m src.realdata.download_era5` | CDS (network) | `data/era5/raw/era5_500hPa_<year>_m<months>_<key>.nc`, `manifest.json` |
| 2. Label | `python -m src.realdata.label_blocking` | raw files | `data/era5/labelled_blocking.npz`, `results/realdata_labels.{md,json}`, `results/figures/realdata_ib_frequency.png` |
| 3. Benchmark | `python -m src.realdata.run_realdata_benchmark` | labelled file | `results/realdata_benchmark.{md,json}`, `results/figures/realdata_benchmark_*.png` |

All three steps read one config file, `config/realdata.yaml`. Every parameter
in it is fixed before any result is seen. `config/realdata_quick.yaml` is a
10-winter version for checking the setup.

- **Data.** ERA5 reanalysis, 500 hPa geopotential (`z`) and relative vorticity
  (`vo`). One field per day at 12 UTC, on a 1° grid (regridded by the CDS).
  Area 25–90°N, 80°W–80°E. Winters (DJF) 1979/80 to 2023/24: 45 seasons,
  about 4060 days.
- **Labels.** The labels come from the geopotential height. The index is
  Tibaldi & Molteni (1990):
  - GHGS = (Z(φ0) − Z(φS)) / (φ0 − φS) and GHGN = (Z(φN) − Z(φ0)) / (φN − φ0).
  - φN = 80°N + Δ, φ0 = 60°N + Δ, φS = 40°N + Δ, with Δ ∈ {−4°, 0°, +4°}.
  - A longitude is blocked if **GHGS > 0 and GHGN < −10 m per degree
    latitude** for at least one Δ.
  - In the Euro-Atlantic sector (45°W–45°E):
    - **BLOCKING**: a run of adjacent blocked longitudes spans at least 15°,
      on at least 5 consecutive days. These are the large-scale extent and
      persistence criteria of Davini et al. (2012).
    - **ZONAL**: no blocked longitude in the sector, and more than 2 days away
      from any BLOCKING day.
    - **AMBIGUOUS**: all other days (short or narrow blocks, transition days).
      They are excluded.
  - The module docstring of `src/realdata/label_blocking.py` states every
    detail.
- **Classifier input.** Relative vorticity only. The labels come from
  geopotential, so the classifier never sees the variable the index is
  computed from. The two are still dynamically linked
  (quasi-geostrophically ζ ≈ (g/f)∇²Z), so the label is close to a function
  of the input. The real question is which compact representation captures
  it best. The `raw_pixels` baseline (LogReg / RF on the 16 × 16 field)
  guards against a trivially linear solution. Each field goes through four
  steps:
  1. Crop to 30–85°N, 75°W–75°E.
  2. Resample to 64 × 64 (anti-aliased bilinear). At 57.5°N the box is about
     1.4 : 1 in physical distance, so the square grid distorts little.
  3. Make the field periodic with a Tukey taper. The descriptor and the
     baselines use FFTs and periodic complexes.
  4. Standardize to zero mean and unit variance, with the same function as
     the synthetic fields.
- **Evaluation.** The unchanged feature registry of the synthetic benchmark:
  the descriptor and its views, the authentic X1/X2, the richer topology, and
  every baseline (moments, spectral slope, raw pixels, power spectrum,
  histogram, Okubo–Weiss, gradient statistics, increment flatness, and
  `baselines_union`). The protocol:
  - Classes are balanced by undersampling, so chance = 0.5.
  - Repeated StratifiedGroupKFold (5 folds × 5 repeats), LogReg +
    RandomForest.
  - **Group = one winter season**, so consecutive, autocorrelated days are
    never split between train and test.
  - Margin CIs come from a paired bootstrap over seasons.
  - The report also gives shuffled-day CV accuracies, to show how much
    leakage the grouping prevents.

## Step 0 — Free CDS account and API token (once)

1. **Register.** Create a free account at
   <https://cds.climate.copernicus.eu> ("Login – Register").
2. **Accept the ERA5 licence.** Open the dataset page *ERA5 hourly data on
   pressure levels from 1940 to present*
   (<https://cds.climate.copernicus.eu/datasets/reanalysis-era5-pressure-levels>).
   Go to the *Download* tab, scroll to *Terms of use* and accept the
   *Licence to use Copernicus Products*. Without this, every request is
   refused.
3. **Copy your Personal Access Token.** It is on your CDS profile page (click
   your name, top right). <https://cds.climate.copernicus.eu/how-to-api> also
   shows a ready-to-paste snippet.
4. **Store the token outside the repository.** The scripts never contain
   credentials. `cdsapi` reads them from `~/.cdsapirc` or from environment
   variables.

   Linux / macOS:

   ```bash
   printf 'url: https://cds.climate.copernicus.eu/api\nkey: %s\n' 'PASTE-YOUR-TOKEN-HERE' > ~/.cdsapirc
   ```

   ```bash
   chmod 600 ~/.cdsapirc
   ```

   Windows (PowerShell):

   ```powershell
   "url: https://cds.climate.copernicus.eu/api`nkey: PASTE-YOUR-TOKEN-HERE" | Set-Content -Encoding ascii "$env:USERPROFILE\.cdsapirc"
   ```

   Alternative: set the environment variables `CDSAPI_URL` and `CDSAPI_KEY`.
   `CDSAPI_RC` can point to an rc file somewhere else.

   The key is the token alone. The old `UID:KEY` format of the legacy CDS no
   longer works, and the download script warns if it finds it. Never commit
   the token: `.cdsapirc` is in `.gitignore`, and a test scans `src/`,
   `config/` and `docs/` for anything that looks like a token.

## Step 1 — Install

From the repository root:

```bash
python -m pip install -r requirements.txt -r requirements-realdata.txt
```

Check the installation. Neither command contacts a server: the dry run prints
the 46 yearly requests and where the credentials were found.

```bash
python -m pytest -q tests/test_realdata_labeling.py tests/test_realdata_pipeline.py
```

```bash
python -m src.realdata.download_era5 --config config/realdata.yaml --dry-run
```

## Step 2 — Quick trial (10 winters, recommended first)

```bash
python -m src.realdata.download_era5 --config config/realdata_quick.yaml
```

```bash
python -m src.realdata.label_blocking --config config/realdata_quick.yaml
```

```bash
python -m src.realdata.run_realdata_benchmark --config config/realdata_quick.yaml
```

The outputs go to `results/quick/`, which is not committed. The quick trial
checks the setup end to end, but its numbers are **not** the result. Its raw
files use the same area, grid and hours as the full run, so the full run
reuses them.

To test the download with a single request first, add `--only-years 2011` to
the download command.

## Step 3 — Full run (45 winters)

```bash
python -m src.realdata.download_era5 --config config/realdata.yaml
```

```bash
python -m src.realdata.label_blocking --config config/realdata.yaml
```

```bash
python -m src.realdata.run_realdata_benchmark --config config/realdata.yaml
```

The same three steps are wrapped in `scripts/run_realdata.sh` and
`scripts/run_realdata.ps1` (Windows).

- **Download.** 46 small requests, one per calendar year, each holding only
  that year's winter months. Each file is a few MB, under 0.5 GB in total.
  Waiting time depends on the CDS queue: from minutes to hours per request.
  The download is resumable. Interrupt it with Ctrl-C and run it again:
  complete files are skipped and partial `.part` files are discarded.
  `manifest.json` records every request and the SHA-256 checksum of every
  file.
- **Labelling.** Takes seconds to minutes.
- **Benchmark.** Dominated by the random forests: roughly half an hour to an
  hour on a 4-core laptop. Set `n_workers` in the config to change the number
  of processes used for feature extraction. If you only need to re-render the
  Markdown report from cached results, use `--report-only`.

## Step 4 — Check the labels before looking at the verdict

Open `results/realdata_labels.md` and `results/figures/realdata_ib_frequency.png`.

- **Counts.** The report gives BLOCKING / ZONAL / AMBIGUOUS days, the number
  of blocking episodes and their durations, and a per-season table.
  Blocking should be a minority of days. Episodes last from 5 days up to a
  few weeks.
- **Frequency curve.** The figure shows the instantaneous blocking frequency
  at every longitude of the download area. Compare its shape with published
  winter climatologies of the Tibaldi–Molteni index (e.g. Tibaldi & Molteni
  1990; Davini et al. 2012). If the curve is flat, near zero everywhere, or
  peaks far from the Atlantic–European region, something is wrong (units,
  latitude order, area). Stop and investigate before reading the benchmark.
- **Too few blocking seasons.** If BLOCKING days occur in fewer seasons than
  there are CV folds, the benchmark stops with an explanation.

## Step 5 — Read the verdict

`results/realdata_benchmark.md` has the same structure as
`results/hard_benchmark.md`:

1. **Overall verdict.** This is the strict verdict: descriptor accuracy minus
   the best of *all* baselines. It is **GO iff margin ≥ +0.05**, with grouped
   CV.
2. **Verdict vs the baselines named in the brief.** Only moments, spectral
   slope and raw pixels. Reported for continuity; it is weaker evidence.
3. **Any configuration vs `baselines_union`.** Does any topological
   configuration beat all non-topological baselines concatenated by
   ≥ +0.05? Three numbers help you read it:
   - **Headroom** = 1 − accuracy(`baselines_union`). If the headroom is below
     0.05, GO is arithmetically impossible: the baselines already saturate
     the task.
   - **CI.** Margins come with a 95% paired bootstrap CI over seasons. A
     margin ≥ 0.05 whose CI includes 0 is fragile.
   - **"configs ≥ +0.05 (of which CI > 0)".** Counts how many configurations
     reach the threshold. The maximum over ~20 configurations is optimistic
     (multiple comparisons).
4. **Leakage check.** Compares shuffled-day CV with grouped CV. A large gap
   means day-level splits leak through autocorrelation. Only the grouped
   numbers enter the verdict.

How to act on the outcome:

| Outcome | Meaning | Next step |
|---|---|---|
| Strict **GO**, margin CI above 0 | The descriptor beats every classical diagnostic on a real regime-classification task | Main result of the GEM paper. The synthetic NO-GO becomes the honest negative control (topology adds nothing for small-scale structure, but does for large-scale regimes) |
| Strict NO-GO, but some configuration ≥ +0.05 over `baselines_union` with CI > 0 | A topological view adds value on top of all baselines | Promising but post hoc. Report it as such, run the sensitivity checks below, and report all of them |
| NO-GO, headroom < 0.05 | Baselines saturate the task; nothing can win | The task is too easy for a GO. Consider the SPMAC extended abstract (Plan B) |
| NO-GO with headroom ≥ 0.05 | Topology does not help on real regimes either | SPMAC extended abstract (Plan B) |

**Pre-registration discipline.** Do not change thresholds, sector, domain or
descriptor settings after seeing the verdict. If you run sensitivity checks,
decide them in advance and report all of them. Good candidates: a copy of the
config with `grouping.mode: block`, `periodize: mirror`, or
`zonal_max_ghgs: -5`. Give each copy its own `report_stem` and
`labeling.output`.

## What to commit

- **Commit:** `results/realdata_labels.{md,json}`,
  `results/realdata_benchmark.{md,json}` and `results/figures/realdata_*.png`
  (whitelisted in `.gitignore`).
- **Never commit:** `data/` (raw and labelled ERA5, git-ignored) or your
  `.cdsapirc`.
- **Publications** using ERA5 must cite Hersbach et al. (2020) and the CDS
  dataset, and acknowledge the Copernicus Climate Change Service, as required
  by the licence.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `No Copernicus CDS credentials found` | Step 0.4: create `~/.cdsapirc` or set `CDSAPI_URL` / `CDSAPI_KEY` |
| 401 / authentication error | Wrong token, or the legacy `UID:KEY` format; paste the Personal Access Token only |
| 403 / "required licences not accepted" | Step 0.2: accept the dataset licence on the CDS website |
| Request stays queued for a long time | CDS load. Leave it running, or stop it and re-run later (resumable) |
| The CDS rejects the `grid` keyword | Set `download.grid: null` (native 0.25°, about 16× more data). Preprocessing resamples anyway. Changing download settings changes the file key, so the files are downloaded again |
| `latitude ... outside the data range` | `download.area` must cover 36–84°N (index latitudes ± 4°) |
| `class ... occurs in only N groups` | Download more seasons, use `grouping.mode: block`, or lower `n_splits` |
| `raw files have different grids` | Raw files with the same key came from different areas or grids; use a clean `download.out_dir` |
| netCDF read errors | `python -m pip install -r requirements-realdata.txt` (needs `netCDF4`) |

## References

- Tibaldi, S. & Molteni, F. (1990). On the operational predictability of
  blocking. *Tellus A* 42(3), 343–365. doi:10.3402/tellusa.v42i3.11882
- Davini, P., Cagnazzo, C., Gualdi, S. & Navarra, A. (2012). Bidimensional
  diagnostics, variability, and trends of Northern Hemisphere blocking.
  *J. Climate* 25, 6496–6509. doi:10.1175/JCLI-D-12-00032.1
- Hersbach, H. et al. (2020). The ERA5 global reanalysis. *Q. J. R. Meteorol.
  Soc.* 146, 1999–2049. doi:10.1002/qj.3803
- Hersbach, H. et al. (2023). ERA5 hourly data on pressure levels from 1940 to
  present. Copernicus Climate Change Service (C3S) Climate Data Store.
  doi:10.24381/cds.bd0915c6
