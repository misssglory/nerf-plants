#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: ./gaussian/view-brush.sh NAME_OR_PLY" >&2
  exit 2
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RECON_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
TOOLS_DIR="${PLANT_TOOLS_DIR:-$RECON_DIR/.tools}"
BRUSH_VIEWER="${BRUSH_VIEWER:-$TOOLS_DIR/brush/bin/brush}"
ARG="$1"

[[ -x "$BRUSH_VIEWER" ]] || {
  echo "Brush viewer is missing: $BRUSH_VIEWER" >&2
  echo "Run gaussian/setup-brush.sh with PLANT_BRUSH_BUILD_VIEWER=1." >&2
  exit 1
}

if [[ -f "$ARG" ]]; then
  TARGET="$(realpath -- "$ARG")"
elif [[ -d "$ARG" ]]; then
  TARGET="$(realpath -- "$ARG")"
else
  TARGET="$(find "$RECON_DIR/outputs-gaussian/$ARG/brush" -type f -name '*.ply' -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -n1 | cut -d' ' -f2-)"
  [[ -n "$TARGET" ]] || {
    echo "No Brush PLY found for dataset: $ARG" >&2
    exit 1
  }
fi

export WGPU_BACKEND="${WGPU_BACKEND:-vulkan}"
exec "$BRUSH_VIEWER" "$TARGET" --with-viewer
