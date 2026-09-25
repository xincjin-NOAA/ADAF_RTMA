#!/bin/bash
#
# Submit one training run from a flat YAML config, with optional key=value overrides.
# Thin wrapper around tools/submit_experiments.py -- see docs/SUBMITTING_JOBS.md.
#
# Usage:
#   ./submit_train.sh <config.yaml> [key=value ...] [--dry-run] [--debug] [--version V] [--force]
#
# Examples:
#   ./submit_train.sh configs/train_example.yaml
#   ./submit_train.sh configs/train_example.yaml max_epochs=1 time=00:20:00 --dry-run
#   ./submit_train.sh configs/train_ocelot3_example.yaml resume=true max_epochs=800
#
# key=value pairs override the file (values are parsed as YAML: 1, 0.5, true, "[4, 4]").

set -e
if [ -z "$1" ] || [[ "$1" == -* ]]; then
    echo "Usage: $0 <config.yaml> [key=value ...] [--dry-run] [--debug] [--version V] [--force]" >&2
    exit 1
fi
CONFIG_FILE="$1"
shift

ARGS=()
while [ $# -gt 0 ]; do
    case "$1" in
        --version) ARGS+=("$1" "$2"); shift 2 ;;
        -*)        ARGS+=("$1"); shift ;;
        *=*)       ARGS+=(--set "$1"); shift ;;
        *)         echo "Unexpected argument '$1' (expected key=value or a --flag)" >&2; exit 1 ;;
    esac
done

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
exec "${PYTHON:-python3}" "$REPO_ROOT/tools/submit_experiments.py" "$CONFIG_FILE" "" "${ARGS[@]}"
