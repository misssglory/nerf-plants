#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'USAGE'
Usage: ./train.sh NAME [nerfacto|splatfacto] [EXTRA_NS_TRAIN_ARGS...]

Environment:
  PLANT_TRAIN_DEVICE=auto|cuda|cpu   Device selection (default: auto)
  PLANT_CPU_ITERATIONS=N             CPU default iterations (default: 1000)
  PLANT_CPU_RAYS=N                   CPU rays per batch (default: 256)
  PLANT_CPU_IMAGE_SCALE=FLOAT        CPU image scale (default: 0.5)

Examples:
  ./train.sh plant_001 nerfacto
  PLANT_TRAIN_DEVICE=cpu ./train.sh plant_001 nerfacto
  PLANT_CPU_ITERATIONS=2000 ./train.sh plant_001 nerfacto
  ./train.sh plant_001 nerfacto --max-num-iterations 5000
  ./train.sh plant_001 splatfacto
USAGE
  exit 2
}

[[ $# -ge 1 ]] || usage
NAME="$1"
METHOD="${2:-nerfacto}"
if [[ $# -ge 2 ]]; then
  shift 2
else
  shift 1
fi

case "$METHOD" in
  nerfacto|splatfacto) ;;
  *) echo "Unsupported method: $METHOD" >&2; usage ;;
esac

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="$SCRIPT_DIR/data/processed/$NAME"
OUTPUT_DIR="$SCRIPT_DIR/outputs"

[[ -f "$DATA_DIR/transforms.json" ]] || {
  echo "Incomplete processed dataset: $DATA_DIR" >&2
  echo "Missing: $DATA_DIR/transforms.json" >&2
  echo "Re-run process-video.sh after repairing the environment." >&2
  exit 1
}

mkdir -p "$OUTPUT_DIR"
cd "$SCRIPT_DIR"

requested_device="${PLANT_TRAIN_DEVICE:-auto}"
case "$requested_device" in
  auto|cuda|cpu) ;;
  *)
    echo "Invalid PLANT_TRAIN_DEVICE=$requested_device; expected auto, cuda, or cpu." >&2
    exit 2
    ;;
esac

cuda_available="$({ pixi run python - <<'PY'
import torch
print("yes" if torch.cuda.is_available() else "no")
PY
} | tail -n1)"

if [[ "$requested_device" == "auto" ]]; then
  if [[ "$cuda_available" == "yes" ]]; then
    device="cuda"
  else
    device="cpu"
  fi
else
  device="$requested_device"
fi

if [[ "$device" == "cuda" && "$cuda_available" != "yes" ]]; then
  echo "CUDA was requested, but PyTorch reports torch.cuda.is_available() == False." >&2
  echo "This Nerfstudio environment cannot use the Radeon 780M as CUDA." >&2
  exit 1
fi

has_arg_prefix() {
  local prefix="$1"
  shift
  local arg
  for arg in "$@"; do
    [[ "$arg" == "$prefix" || "$arg" == "$prefix="* ]] && return 0
  done
  return 1
}

if [[ "$METHOD" == "splatfacto" && "$device" != "cuda" ]]; then
  cat >&2 <<'ERROR'
Splatfacto is not supported by this project without a CUDA-capable NVIDIA GPU.
Its gsplat rasterizer is CUDA-oriented in this pinned Nerfstudio environment.
Use Nerfacto in CPU smoke-test mode, or use Brush/OpenSplat for AMD graphics.
ERROR
  exit 1
fi

if [[ "$METHOD" == "nerfacto" ]]; then
  common_args=(
    --data "$DATA_DIR"
    --output-dir "$OUTPUT_DIR"
    --pipeline.model.predict-normals True
  )

  if [[ "$device" == "cuda" ]]; then
    echo "Training Nerfacto with CUDA."
    exec pixi run ns-train nerfacto \
      --machine.device-type cuda \
      "${common_args[@]}" \
      "$@"
  fi

  cpu_iterations="${PLANT_CPU_ITERATIONS:-1000}"
  cpu_rays="${PLANT_CPU_RAYS:-256}"
  cpu_image_scale="${PLANT_CPU_IMAGE_SCALE:-0.5}"

  cpu_args=(
    --machine.device-type cpu
    --mixed-precision False
    --pipeline.model.implementation torch
    --pipeline.datamanager.train-num-rays-per-batch "$cpu_rays"
    --pipeline.datamanager.eval-num-rays-per-batch "$cpu_rays"
    --pipeline.datamanager.camera-res-scale-factor "$cpu_image_scale"
    --pipeline.model.eval-num-rays-per-chunk 2048
    --viewer.num-rays-per-chunk 2048
  )

  if ! has_arg_prefix --max-num-iterations "$@"; then
    cpu_args+=(--max-num-iterations "$cpu_iterations")
  fi

  cat <<INFO
No CUDA-capable NVIDIA GPU was detected.
Starting Nerfacto in CPU compatibility mode:
  implementation: torch
  mixed precision: disabled
  rays per batch: $cpu_rays
  image scale: $cpu_image_scale
  default iterations: $cpu_iterations

This is a functional smoke-test path, not a practical production path.
On a Ryzen 780M system it can be extremely slow because this mode uses CPU,
not the AMD iGPU.
INFO

  exec pixi run ns-train nerfacto \
    "${cpu_args[@]}" \
    "${common_args[@]}" \
    "$@"
else
  echo "Training Splatfacto with CUDA."
  exec pixi run ns-train splatfacto \
    --machine.device-type cuda \
    --data "$DATA_DIR" \
    --output-dir "$OUTPUT_DIR" \
    "$@"
fi
