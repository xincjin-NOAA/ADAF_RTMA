#!/bin/bash
#
# Submit many batch-inference jobs (predict.py) defined in one YAML file (defaults + predictions).
# Thin wrapper around tools/submit_predict.py -- see docs/PREDICTING.md.
#
# Usage (from anywhere; paths are resolved against the repo root):
#   ./submit_predictions_from_yaml.sh [config_file|-] [filter] [--dry-run] [--debug] [--version V] [--force] [--set KEY=VALUE ...]
#
# Examples:
#   ./submit_predictions_from_yaml.sh                              # every run in predict_configs.yaml
#   ./submit_predictions_from_yaml.sh - ocelot3 --dry-run          # write job scripts for "*ocelot3*" without submitting
#   ./submit_predictions_from_yaml.sh - goes_lowres_test --version june   # outputs under predictions/goes_lowres_test/june
#   ./submit_predictions_from_yaml.sh - ocelot3 --set end_time=2023-02-15T23:00 --set time=00:30:00
#
# Needs a python3 with ruamel.yaml or PyYAML (the ADAF environment has ruamel.yaml);
# set PYTHON=/path/to/python to choose one.

set -e
REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
exec "${PYTHON:-python3}" "$REPO_ROOT/tools/submit_predict.py" "$@"
