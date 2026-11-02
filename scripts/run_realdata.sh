#!/usr/bin/env bash
# ERA5 blocking benchmark, all three steps. Usage: scripts/run_realdata.sh [config]
# Needs CDS credentials in ~/.cdsapirc or CDSAPI_URL / CDSAPI_KEY (docs/REALDATA.md).
set -euo pipefail
cd "$(dirname "$0")/.."
CFG="${1:-config/realdata.yaml}"
python -m src.realdata.download_era5 --config "$CFG"
python -m src.realdata.label_blocking --config "$CFG"
python -m src.realdata.run_realdata_benchmark --config "$CFG"
