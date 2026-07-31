#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: ./export-splat.sh CONFIG_YML NAME" >&2
  exit 2
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="$(realpath -- "$1")"
NAME="$2"
OUTPUT_DIR="$SCRIPT_DIR/exports/$NAME-splat"

[[ -f "$CONFIG" ]] || { echo "Config not found: $CONFIG" >&2; exit 1; }
[[ ! -e "$OUTPUT_DIR" ]] || { echo "Output exists: $OUTPUT_DIR" >&2; exit 1; }

mkdir -p "$SCRIPT_DIR/exports"
cd "$SCRIPT_DIR"
pixi run ns-export gaussian-splat \
  --load-config "$CONFIG" \
  --output-dir "$OUTPUT_DIR"

echo "Gaussian splat written to: $OUTPUT_DIR"
