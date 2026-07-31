#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'USAGE'
Usage: ./train.sh NAME [nerfacto|splatfacto] [EXTRA_NS_TRAIN_ARGS...]

Early stopping is enabled by default. Nerfstudio performs full validation at a
fixed interval; training stops after rendering quality no longer improves.

Environment:
  PLANT_TRAIN_DEVICE=auto|cuda|cpu   Device selection (default: auto)
  PLANT_EARLY_STOP=0|1              Enable validation early stopping (default: 1)
  PLANT_EVAL_INTERVAL=N             Full-validation/save interval (default: 500)
  PLANT_EVAL_FRAME_INTERVAL=N       Hold out every Nth video frame (default: 8)
  PLANT_EARLY_STOP_MIN_STEPS=N      Do not stop before this step (default: 1000)
  PLANT_EARLY_STOP_PATIENCE=N       Stale full validations before stop (default: 4)
  PLANT_EARLY_STOP_PSNR_DELTA=FLOAT Minimum meaningful PSNR gain (default: 0.03)
  PLANT_EARLY_STOP_LPIPS_DELTA=FLOAT Minimum meaningful LPIPS drop (default: 0.002)
  PLANT_EARLY_STOP_POLL=SECONDS     TensorBoard polling interval (default: 5)
  PLANT_TRAIN_VIS=VALUE             tensorboard or viewer+tensorboard
                                    (default: tensorboard on CPU,
                                     viewer+tensorboard on CUDA)
  PLANT_MAX_ITERATIONS=N            Hard safety ceiling
                                    (default: 10000 CPU, 30000 CUDA)
  PLANT_CPU_RAYS=N                   CPU rays per batch (default: 256)
  PLANT_CPU_IMAGE_SCALE=FLOAT        CPU image scale (default: 0.5)

Examples:
  ./train.sh plant_001 nerfacto
  PLANT_EARLY_STOP_PATIENCE=6 ./train.sh plant_001 nerfacto
  PLANT_TRAIN_VIS=viewer+tensorboard ./train.sh plant_001 nerfacto
  PLANT_EARLY_STOP=0 PLANT_MAX_ITERATIONS=200 ./train.sh plant_001 nerfacto
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

require_positive_int() {
  local name="$1"
  local value="$2"
  [[ "$value" =~ ^[1-9][0-9]*$ ]] || {
    echo "$name must be a positive integer; got: $value" >&2
    exit 2
  }
}

require_nonnegative_int() {
  local name="$1"
  local value="$2"
  [[ "$value" =~ ^[0-9]+$ ]] || {
    echo "$name must be a non-negative integer; got: $value" >&2
    exit 2
  }
}

if [[ "$METHOD" == "splatfacto" && "$device" != "cuda" ]]; then
  cat >&2 <<'ERROR'
Splatfacto is not supported by this project without a CUDA-capable NVIDIA GPU.
Its gsplat rasterizer is CUDA-oriented in this pinned Nerfstudio environment.
Use Nerfacto in CPU mode, or use Brush/OpenSplat for AMD graphics.
ERROR
  exit 1
fi

early_stop="${PLANT_EARLY_STOP:-1}"
case "$early_stop" in
  0|1) ;;
  *) echo "PLANT_EARLY_STOP must be 0 or 1; got: $early_stop" >&2; exit 2 ;;
esac

eval_interval="${PLANT_EVAL_INTERVAL:-500}"
eval_frame_interval="${PLANT_EVAL_FRAME_INTERVAL:-8}"
min_steps="${PLANT_EARLY_STOP_MIN_STEPS:-1000}"
patience="${PLANT_EARLY_STOP_PATIENCE:-4}"
psnr_delta="${PLANT_EARLY_STOP_PSNR_DELTA:-0.03}"
lpips_delta="${PLANT_EARLY_STOP_LPIPS_DELTA:-0.002}"
poll_seconds="${PLANT_EARLY_STOP_POLL:-5}"

require_positive_int PLANT_EVAL_INTERVAL "$eval_interval"
require_positive_int PLANT_EVAL_FRAME_INTERVAL "$eval_frame_interval"
require_nonnegative_int PLANT_EARLY_STOP_MIN_STEPS "$min_steps"
require_positive_int PLANT_EARLY_STOP_PATIENCE "$patience"

if [[ "$device" == "cuda" ]]; then
  default_max_iterations=30000
  default_vis="viewer+tensorboard"
else
  default_max_iterations=10000
  default_vis="tensorboard"
fi
max_iterations="${PLANT_MAX_ITERATIONS:-$default_max_iterations}"
train_vis="${PLANT_TRAIN_VIS:-$default_vis}"
require_positive_int PLANT_MAX_ITERATIONS "$max_iterations"
if [[ "$early_stop" == "1" ]]; then
  case "$train_vis" in
    tensorboard|viewer+tensorboard) ;;
    *)
      echo "PLANT_TRAIN_VIS must be tensorboard or viewer+tensorboard when early stopping is used." >&2
      exit 2
      ;;
  esac
fi

run_timestamp="$(date +%Y-%m-%d_%H%M%S)"
run_dir="$OUTPUT_DIR/$NAME/$METHOD/$run_timestamp"

model_args=()
if [[ "$METHOD" == "nerfacto" ]]; then
  model_args+=(--pipeline.model.predict-normals True)
fi

# Trainer/model arguments must appear before the dataparser subcommand in
# Nerfstudio's Tyro CLI. The dataparser and its options are appended last.
enforced_run_args=(
  --output-dir "$OUTPUT_DIR"
  --experiment-name "$NAME"
  --timestamp "$run_timestamp"
)

dataparser_args=(
  nerfstudio-data
  --data "$DATA_DIR"
  --eval-mode interval
  --eval-interval "$eval_frame_interval"
)

if [[ "$device" == "cuda" ]]; then
  device_args=(--machine.device-type cuda)
else
  cpu_rays="${PLANT_CPU_RAYS:-256}"
  cpu_image_scale="${PLANT_CPU_IMAGE_SCALE:-0.5}"
  require_positive_int PLANT_CPU_RAYS "$cpu_rays"
  device_args=(
    --machine.device-type cpu
    --mixed-precision False
    --pipeline.model.implementation torch
    --pipeline.datamanager.train-num-rays-per-batch "$cpu_rays"
    --pipeline.datamanager.eval-num-rays-per-batch "$cpu_rays"
    --pipeline.datamanager.camera-res-scale-factor "$cpu_image_scale"
    --pipeline.model.eval-num-rays-per-chunk 2048
    --viewer.num-rays-per-chunk 2048
  )
fi

max_args=()
if ! has_arg_prefix --max-num-iterations "$@"; then
  max_args=(--max-num-iterations "$max_iterations")
fi

# These are enforced after user arguments because the supervisor depends on a
# known output directory, TensorBoard scalars, synchronized validation/saves,
# and retained checkpoints.
monitor_args=(
  --vis "$train_vis"
  --steps-per-eval-batch "$eval_interval"
  --steps-per-eval-image 1000000000
  --steps-per-eval-all-images "$eval_interval"
  --steps-per-save "$eval_interval"
  --save-only-latest-checkpoint False
)

train_command=(
  ns-train "$METHOD"
  "${device_args[@]}"
  "${model_args[@]}"
  "${max_args[@]}"
  "$@"
  "${enforced_run_args[@]}"
)

if [[ "$early_stop" == "1" ]]; then
  train_command+=("${monitor_args[@]}")
  train_command+=("${dataparser_args[@]}")

  cat <<INFO
Training with validation early stopping:
  device:                  $device
  run directory:           $run_dir
  held-out frames:         every ${eval_frame_interval}th registered frame
  full validation:         every $eval_interval steps
  earliest stop:           step $min_steps
  patience:                $patience validations
  meaningful PSNR gain:    $psnr_delta dB
  meaningful LPIPS drop:   $lpips_delta
  hard iteration ceiling:  $max_iterations
  visualizer:              $train_vis

Progress reports include training loss, validation loss, PSNR, SSIM, LPIPS,
changes since the previous validation, and remaining patience.
INFO

  exec pixi run python "$SCRIPT_DIR/early_stop_train.py" \
    --run-dir "$run_dir" \
    --min-steps "$min_steps" \
    --patience "$patience" \
    --min-psnr-delta "$psnr_delta" \
    --min-lpips-delta "$lpips_delta" \
    --poll-seconds "$poll_seconds" \
    -- "${train_command[@]}"
fi

train_command+=("${dataparser_args[@]}")

cat <<INFO
Early stopping is disabled.
Training until the configured hard iteration ceiling.
  device:         $device
  run directory:  $run_dir
INFO

exec pixi run "${train_command[@]}"
