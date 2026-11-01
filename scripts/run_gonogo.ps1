param([string]$Config = "config/experiment.yaml")
Set-Location (Join-Path $PSScriptRoot "..")
python -m src.eval.gonogo --config $Config
