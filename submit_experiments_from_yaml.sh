#!/bin/bash
#
# Submit many training experiments defined in one YAML file (defaults + experiments).
# Thin wrapper around tools/submit_experiments.py -- see docs/SUBMITTING_JOBS.md.
#
# Usage (from anywhere; paths are resolved against the repo root):
#   ./submit_experiments_from_yaml.sh [config_file|-] [filter] [--dry-run] [--debug] [--version V] [--force] [--set KEY=VALUE ...]
#
# Examples:
#   ./submit_experiments_from_yaml.sh                          # every experiment in experiment_configs.yaml
#   ./submit_experiments_from_yaml.sh - smoke --dry-run        # write job scripts for "*smoke*" without submitting
#   ./submit_experiments_from_yaml.sh - goes_base --version v2 # outputs under training_runs/goes_base/v2
#   ./submit_experiments_from_yaml.sh - ocelot3 --set max_epochs=1 --set batch_size=1
#
# Needs a python3 with ruamel.yaml or PyYAML (the ADAF environment has ruamel.yaml);
# set PYTHON=/path/to/python to choose one.

set -e
REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
exec "${PYTHON:-python3}" "$REPO_ROOT/tools/submit_experiments.py" "$@"
