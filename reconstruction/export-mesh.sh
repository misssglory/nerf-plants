#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'USAGE'
Usage: ./export-mesh.sh CONFIG_YML NAME [poisson|tsdf]

Example:
  ./export-mesh.sh outputs/plant_001/nerfacto/.../config.yml plant_001 poisson

Poisson normally gives the best Nerfstudio mesh, but requires a nerfacto model
trained with --pipeline.model.predict-normals True. TSDF works with all models.
USAGE
  exit 2
}

[[ $# -ge 2 && $# -le 3 ]] || usage

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="$(realpath -- "$1")"
NAME="$2"
MODE="${3:-poisson}"
OUTPUT_DIR="$SCRIPT_DIR/exports/$NAME-$MODE"

[[ -f "$CONFIG" ]] || { echo "Config not found: $CONFIG" >&2; exit 1; }
case "$MODE" in
  poisson|tsdf) ;;
  *) echo "MODE must be poisson or tsdf." >&2; exit 1 ;;
esac

mkdir -p "$SCRIPT_DIR/exports"
[[ ! -e "$OUTPUT_DIR" ]] || {
  echo "Export directory already exists: $OUTPUT_DIR" >&2
  exit 1
}

cd "$SCRIPT_DIR"
pixi run ns-export "$MODE" \
  --load-config "$CONFIG" \
  --output-dir "$OUTPUT_DIR"

echo "Mesh export written to: $OUTPUT_DIR"
echo "Remember: Nerfstudio scale is arbitrary until you calibrate it from a known marker."
