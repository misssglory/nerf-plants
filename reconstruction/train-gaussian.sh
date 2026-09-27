#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'USAGE'
Usage: ./train-gaussian.sh NAME [auto|brush|splatfacto] [BACKEND_ARGS...]

Backends:
  auto        Use Splatfacto when CUDA is available; otherwise use Brush.
  brush       Cross-platform WebGPU Gaussian training (recommended for AMD 780M).
  splatfacto  Nerfstudio/gsplat CUDA Gaussian training (NVIDIA only here).

Environment:
  PLANT_GAUSSIAN_BACKEND=auto|brush|splatfacto

Examples:
  ./train-gaussian.sh plant_003
  ./train-gaussian.sh plant_003 brush
  ./train-gaussian.sh plant_003 brush 20000
  ./train-gaussian.sh plant_003 splatfacto
USAGE
  exit 2
}

[[ $# -ge 1 ]] || usage
NAME="$1"
shift
BACKEND="${1:-${PLANT_GAUSSIAN_BACKEND:-auto}}"
if [[ $# -ge 1 ]]; then
  shift
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

cuda_available=no
if command -v pixi >/dev/null 2>&1 && [[ -f "$SCRIPT_DIR/pixi.toml" ]]; then
  cuda_available="$({
    cd "$SCRIPT_DIR"
    pixi run python - <<'PY' 2>/dev/null || true
try:
    import torch
    print("yes" if torch.cuda.is_available() else "no")
except Exception:
    print("no")
PY
  } | tail -n1)"
fi

if [[ "$BACKEND" == "auto" ]]; then
  if [[ "$cuda_available" == "yes" ]]; then
    BACKEND=splatfacto
  else
    BACKEND=brush
  fi
fi

case "$BACKEND" in
  brush)
    echo "Gaussian backend: Brush (WebGPU)"
    exec "$SCRIPT_DIR/gaussian/train-brush.sh" "$NAME" "$@"
    ;;
  splatfacto)
    echo "Gaussian backend: Nerfstudio Splatfacto (CUDA)"
    exec "$SCRIPT_DIR/gaussian/train-splatfacto.sh" "$NAME" "$@"
    ;;
  *)
    echo "Unknown Gaussian backend: $BACKEND" >&2
    usage
    ;;
esac
