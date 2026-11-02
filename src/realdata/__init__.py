"""Real-data pipeline: ERA5 500 hPa winter blocking vs zonal flow.

Three steps, each a module with a command-line entry point, all driven by one
YAML config (`config/realdata.yaml`):

1. `download_era5`          fetch ERA5 500 hPa geopotential + relative vorticity
                            from the Copernicus Climate Data Store (cdsapi);
2. `label_blocking`         label every day BLOCKING / ZONAL / AMBIGUOUS with the
                            Tibaldi-Molteni (1990) 500 hPa gradient-reversal index;
3. `run_realdata_benchmark` feed the labelled vorticity fields to the existing
                            descriptor, baselines and grouped-CV Go/No-Go evaluation.

`common` holds the bookkeeping shared by the three steps (config validation,
winter seasons, longitude handling, time-block CV groups). See
`docs/REALDATA.md` for the step-by-step instructions.
"""
