#!/bin/bash
#
# Submit one batch-inference job (predict.py) from a predict YAML, with optional key=value overrides.
# Thin wrapper around tools/submit_predict.py -- see docs/PREDICTING.md.
#
# Usage:
#   ./submit_predict.sh <predict.yaml> [key=value ...] [--dry-run] [--debug] [--force]
#
# Examples:
#   ./submit_predict.sh configs/predict_example.yaml
#   ./submit_predict.sh configs/predict_example.yaml end_time=2023-06-30T23:00 slurm.time=08:00:00
#   ./submit_predict.sh configs/predict_ocelot3_example.yaml checkpoint=training_runs/ocelot3_lowres/ckpt.tar --dry-run
#
# key=value pairs override the file (values are parsed as YAML; dotted keys reach nested blocks).

set -e
if [ -z "$1" ] || [[ "$1" == -* ]]; then
    echo "Usage: $0 <predict.yaml> [key=value ...] [--dry-run] [--debug] [--force]" >&2
    exit 1
fi
CONFIG_FILE="$1"
shift

ARGS=()
while [ $# -gt 0 ]; do
    case "$1" in
        -*)  ARGS+=("$1"); shift ;;
        *=*) ARGS+=(--set "$1"); shift ;;
        *)   echo "Unexpected argument '$1' (expected key=value or a --flag)" >&2; exit 1 ;;
    esac
done

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
exec "${PYTHON:-python3}" "$REPO_ROOT/tools/submit_predict.py" "$CONFIG_FILE" "${ARGS[@]}"
