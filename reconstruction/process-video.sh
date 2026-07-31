#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'USAGE'
Usage: ./process-video.sh VIDEO NAME [FRAME_COUNT]

Example:
  ./process-video.sh "$HOME/PlantCaptures/plant_20260730_221500.mp4" plant_001 120

FRAME_COUNT defaults to 120. For one slow 45-90 second orbit, 80-160 is a
sensible starting range; too many nearly-identical video frames can hurt COLMAP.
USAGE
  exit 2
}

[[ $# -ge 2 && $# -le 3 ]] || usage

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
VIDEO="$(realpath -- "$1")"
NAME="$2"
FRAME_COUNT="${3:-120}"
OUTPUT_DIR="$SCRIPT_DIR/data/processed/$NAME"
TRANSFORMS="$OUTPUT_DIR/transforms.json"

[[ -f "$VIDEO" ]] || { echo "Video not found: $VIDEO" >&2; exit 1; }
[[ "$NAME" =~ ^[A-Za-z0-9._-]+$ ]] || {
  echo "NAME may contain only letters, digits, dot, underscore and dash." >&2
  exit 1
}
[[ "$FRAME_COUNT" =~ ^[0-9]+$ ]] || { echo "FRAME_COUNT must be an integer." >&2; exit 1; }

mkdir -p "$SCRIPT_DIR/data/processed"

if [[ -f "$TRANSFORMS" ]]; then
  echo "Completed output already exists: $OUTPUT_DIR" >&2
  echo "Choose another NAME or remove the directory intentionally." >&2
  exit 1
elif [[ -e "$OUTPUT_DIR" ]]; then
  echo "Removing incomplete previous output (no transforms.json): $OUTPUT_DIR" >&2
  rm -rf -- "$OUTPUT_DIR"
fi

cd "$SCRIPT_DIR"

echo "Video:  $VIDEO"
echo "Output: $OUTPUT_DIR"
echo "Frames: $FRAME_COUNT"

# ns-process-data imports Open3D and PyTorch before running FFmpeg/COLMAP.
# Validate both binary-runtime layers before doing the expensive work.
"$SCRIPT_DIR/check-open3d-runtime.sh"
"$SCRIPT_DIR/check-python-runtime.sh"

# The video subcommand extracts frames with FFmpeg and estimates camera poses
# with COLMAP. --matching-method sequential is appropriate for ordered video.
pixi run ns-process-data video \
  --data "$VIDEO" \
  --output-dir "$OUTPUT_DIR" \
  --num-frames-target "$FRAME_COUNT" \
  --matching-method sequential \
  --no-gpu

if [[ ! -s "$TRANSFORMS" ]]; then
  echo >&2
  echo "ERROR: ns-process-data finished without creating: $TRANSFORMS" >&2
  echo "The dataset is incomplete and cannot be trained." >&2
  exit 1
fi

pixi run python - "$TRANSFORMS" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
with path.open("r", encoding="utf-8") as handle:
    data = json.load(handle)
frames = data.get("frames", [])
if not frames:
    raise SystemExit(f"ERROR: {path} contains no registered frames")
print(f"Registered frames in transforms.json: {len(frames)}")
PY

cat <<MSG

Processed dataset created successfully at:
  $OUTPUT_DIR

Inspect these before training:
  $OUTPUT_DIR/images
  $TRANSFORMS

Then run:
  ./train.sh "$NAME" nerfacto
MSG
