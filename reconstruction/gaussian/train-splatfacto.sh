#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: ./gaussian/train-splatfacto.sh NAME [EXTRA_NS_TRAIN_ARGS...]" >&2
  exit 2
fi

NAME="$1"
shift
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RECON_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"

cuda_available="$({
  cd "$RECON_DIR"
  pixi run python - <<'PY'
import torch
print("yes" if torch.cuda.is_available() else "no")
PY
} | tail -n1)"

if [[ "$cuda_available" != "yes" ]]; then
  cat >&2 <<'ERROR'
Splatfacto in this project requires CUDA and a supported NVIDIA GPU.
For AMD 780M, use:
  ./train-gaussian.sh NAME brush
ERROR
  exit 1
fi

export PLANT_TRAIN_DEVICE=cuda
export PLANT_TRAIN_VIS="${PLANT_TRAIN_VIS:-viewer+tensorboard}"
exec "$RECON_DIR/train.sh" "$NAME" splatfacto "$@"
