#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if ! command -v pixi >/dev/null 2>&1; then
  echo "pixi is missing. Enter the Nix shell first:" >&2
  echo "  nix develop .#nerfstudio" >&2
  exit 1
fi

printf '\nInstalling the pinned Pixi environment...\n'
pixi install

printf '\nChecking binary and Python runtimes...\n'
"$SCRIPT_DIR/check-open3d-runtime.sh"
"$SCRIPT_DIR/check-python-runtime.sh"
pixi run python verify_environment.py

printf '\nVerifying Nerfstudio commands...\n'
pixi run ns-process-data --help >/dev/null
pixi run ns-train --help >/dev/null
pixi run ns-export --help >/dev/null

CUDA_AVAILABLE="$({ pixi run python - <<'PY'
import torch
print("yes" if torch.cuda.is_available() else "no")
PY
} | tail -n1)"

if [[ "$CUDA_AVAILABLE" == "yes" ]]; then
  echo "CUDA is available; preparing the fast Nerfacto backend."
  if pixi run python -c 'import tinycudann' >/dev/null 2>&1; then
    echo "tiny-cuda-nn is already installed."
  else
    echo "Building tiny-cuda-nn against the pinned CUDA/PyTorch stack..."
    pixi run python -m pip install --no-build-isolation \
      'git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch'
  fi
else
  cat <<'MSG'

WARNING: CUDA is unavailable.
Video extraction and COLMAP preprocessing are ready, but the included
Nerfacto/Splatfacto training scripts are CUDA-oriented and are not expected to
train practically on this machine. This is normal on an AMD Radeon 780M.
MSG
fi

cat <<'MSG'

Plant Capture reconstruction environment is ready.

Process a capture:
  ./process-video.sh ../captures/plant.mp4 plant_001 120

CUDA machine only — train a mesh-capable model:
  ./train.sh plant_001 nerfacto
MSG
