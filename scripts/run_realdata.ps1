# ERA5 blocking benchmark, all three steps. Usage: scripts\run_realdata.ps1 [-Config path]
# Needs CDS credentials in %USERPROFILE%\.cdsapirc or CDSAPI_URL / CDSAPI_KEY (docs/REALDATA.md).
param([string]$Config = "config/realdata.yaml")
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
python -m src.realdata.download_era5 --config $Config
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
python -m src.realdata.label_blocking --config $Config
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
python -m src.realdata.run_realdata_benchmark --config $Config
exit $LASTEXITCODE
