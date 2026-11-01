#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python -m src.eval.gonogo --config "${1:-config/experiment.yaml}"
