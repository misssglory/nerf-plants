#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

pixi run python - <<'PY'
import torch

print(f"PyTorch: {torch.__version__}")
print(f"CUDA build: {torch.version.cuda}")
print(f"HIP build: {torch.version.hip}")
print(f"CUDA API available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Accelerator: {torch.cuda.get_device_name(0)}")
else:
    print("Accelerator: none usable by this PyTorch environment")
    print("Nerfacto will use CPU compatibility mode.")
PY

if command -v rocminfo >/dev/null 2>&1; then
  echo
  echo "ROCm agents visible to the host:"
  rocminfo 2>/dev/null | grep -E '^[[:space:]]*Name:[[:space:]]+gfx' | sort -u || true
fi
