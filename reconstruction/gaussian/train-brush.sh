#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'USAGE'
Usage: ./gaussian/train-brush.sh NAME [STEPS] [EXTRA_BRUSH_ARGS...]

Environment defaults for an AMD 780M-class integrated GPU:
  PLANT_GS_STEPS=30000
  PLANT_GS_MAX_RESOLUTION=1280
  PLANT_GS_MAX_SPLATS=750000
  PLANT_GS_EVAL_EVERY=500
  PLANT_GS_EVAL_SPLIT_EVERY=8
  PLANT_GS_EXPORT_EVERY=5000
  PLANT_GS_CACHE_SIZE=2GiB
  PLANT_GS_WITH_VIEWER=0
  WGPU_BACKEND=vulkan

Examples:
  ./gaussian/train-brush.sh plant_003
  PLANT_GS_WITH_VIEWER=1 ./gaussian/train-brush.sh plant_003 20000
USAGE
  exit 2
}

[[ $# -ge 1 ]] || usage
NAME="$1"
shift

if [[ $# -ge 1 && "$1" =~ ^[0-9]+$ ]]; then
  STEPS="$1"
  shift
else
  STEPS="${PLANT_GS_STEPS:-30000}"
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RECON_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
DATA_DIR="$RECON_DIR/data/processed/$NAME"
TOOLS_DIR="${PLANT_TOOLS_DIR:-$RECON_DIR/.tools}"
BRUSH_CLI="${BRUSH_CLI:-$TOOLS_DIR/brush/bin/brush-cli}"
BRUSH_VIEWER="${BRUSH_VIEWER:-$TOOLS_DIR/brush/bin/brush}"

[[ -f "$DATA_DIR/transforms.json" ]] || {
  echo "Nerfstudio dataset not found: $DATA_DIR/transforms.json" >&2
  echo "Run process-video.sh first." >&2
  exit 1
}
[[ -x "$BRUSH_CLI" ]] || {
  echo "Brush is not installed at: $BRUSH_CLI" >&2
  echo "Enter 'nix develop .#gaussian', then run gaussian/setup-brush.sh" >&2
  exit 1
}

MAX_RESOLUTION="${PLANT_GS_MAX_RESOLUTION:-1280}"
MAX_SPLATS="${PLANT_GS_MAX_SPLATS:-750000}"
EVAL_EVERY="${PLANT_GS_EVAL_EVERY:-500}"
EVAL_SPLIT_EVERY="${PLANT_GS_EVAL_SPLIT_EVERY:-8}"
EXPORT_EVERY="${PLANT_GS_EXPORT_EVERY:-5000}"
CACHE_SIZE="${PLANT_GS_CACHE_SIZE:-2GiB}"
WITH_VIEWER="${PLANT_GS_WITH_VIEWER:-0}"
GROWTH_STOP="${PLANT_GS_GROWTH_STOP:-15000}"

for pair in \
  "STEPS:$STEPS" \
  "PLANT_GS_MAX_RESOLUTION:$MAX_RESOLUTION" \
  "PLANT_GS_MAX_SPLATS:$MAX_SPLATS" \
  "PLANT_GS_EVAL_EVERY:$EVAL_EVERY" \
  "PLANT_GS_EVAL_SPLIT_EVERY:$EVAL_SPLIT_EVERY" \
  "PLANT_GS_EXPORT_EVERY:$EXPORT_EVERY" \
  "PLANT_GS_GROWTH_STOP:$GROWTH_STOP"; do
  key="${pair%%:*}"
  value="${pair#*:}"
  [[ "$value" =~ ^[1-9][0-9]*$ ]] || {
    echo "$key must be a positive integer, got: $value" >&2
    exit 2
  }
done

case "$WITH_VIEWER" in 0|1) ;; *) echo "PLANT_GS_WITH_VIEWER must be 0 or 1" >&2; exit 2 ;; esac

TIMESTAMP="$(date +%Y-%m-%d_%H%M%S)"
RUN_DIR="$RECON_DIR/outputs-gaussian/$NAME/brush/$TIMESTAMP"
mkdir -p "$RUN_DIR"

export WGPU_BACKEND="${WGPU_BACKEND:-vulkan}"
export RUST_BACKTRACE="${RUST_BACKTRACE:-1}"
export RUST_LOG="${RUST_LOG:-info}"

if [[ "$WITH_VIEWER" == "1" ]]; then
  [[ -x "$BRUSH_VIEWER" ]] || {
    echo "Brush viewer binary is missing: $BRUSH_VIEWER" >&2
    echo "Re-run setup with PLANT_BRUSH_BUILD_VIEWER=1." >&2
    exit 1
  }
  EXECUTABLE="$BRUSH_VIEWER"
  viewer_args=(--with-viewer)
else
  EXECUTABLE="$BRUSH_CLI"
  viewer_args=()
fi

command=(
  "$EXECUTABLE" "$DATA_DIR"
  "${viewer_args[@]}"
  --total-train-iters "$STEPS"
  --max-resolution "$MAX_RESOLUTION"
  --max-splats "$MAX_SPLATS"
  --growth-stop-iter "$GROWTH_STOP"
  --eval-split-every "$EVAL_SPLIT_EVERY"
  --eval-every "$EVAL_EVERY"
  --export-every "$EXPORT_EVERY"
  --export-path "$RUN_DIR"
  --export-name "${NAME}_{iter}.ply"
  --max-scene-batch-cache-size "$CACHE_SIZE"
  "$@"
)

cat <<INFO
Brush Gaussian training:
  dataset:          $DATA_DIR
  run directory:    $RUN_DIR
  backend:          WebGPU ($WGPU_BACKEND)
  steps:            $STEPS
  max resolution:   $MAX_RESOLUTION
  max splats:       $MAX_SPLATS
  eval views:       every ${EVAL_SPLIT_EVERY}th image
  evaluate every:   $EVAL_EVERY steps
  export every:     $EXPORT_EVERY steps
  viewer:           $WITH_VIEWER

Brush prints evaluation PSNR/SSIM and current splat count during training.
INFO
printf '%q ' "${command[@]}" > "$RUN_DIR/command.sh"
printf '\n' >> "$RUN_DIR/command.sh"
chmod +x "$RUN_DIR/command.sh"

set +e
"${command[@]}" 2>&1 | tee "$RUN_DIR/train.log"
status=${PIPESTATUS[0]}
set -e

if (( status != 0 )); then
  echo "Brush exited with status $status. Log: $RUN_DIR/train.log" >&2
  exit "$status"
fi

LATEST_PLY="$(find "$RUN_DIR" -maxdepth 1 -type f -name '*.ply' -printf '%T@ %p\n' | sort -nr | head -n1 | cut -d' ' -f2-)"
if [[ -n "$LATEST_PLY" ]]; then
  ln -sfn "$(basename -- "$LATEST_PLY")" "$RUN_DIR/latest.ply"
  echo "Latest Gaussian splat: $LATEST_PLY"
else
  echo "Training completed, but no PLY was found. Ensure STEPS is at least EXPORT_EVERY." >&2
fi
