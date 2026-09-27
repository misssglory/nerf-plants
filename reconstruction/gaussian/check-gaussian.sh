#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RECON_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
BRUSH_CLI="${BRUSH_CLI:-$RECON_DIR/.tools/brush/bin/brush-cli}"

printf '%s\n' '=== Vulkan / WebGPU path ==='
if command -v vulkaninfo >/dev/null 2>&1; then
  vulkaninfo --summary 2>&1 | sed -n '1,100p' || true
else
  echo "vulkaninfo not found; enter nix develop .#gaussian"
fi

printf '\n%s\n' '=== Brush ==='
if [[ -x "$BRUSH_CLI" ]]; then
  "$BRUSH_CLI" --version || true
  echo "Brush CLI: $BRUSH_CLI"
else
  echo "Brush is not built. Run: reconstruction/gaussian/setup-brush.sh"
fi

printf '\n%s\n' '=== Nerfstudio Splatfacto / CUDA ==='
if command -v pixi >/dev/null 2>&1 && [[ -f "$RECON_DIR/pixi.toml" ]]; then
  (
    cd "$RECON_DIR"
    pixi run python - <<'PY' 2>/dev/null || true
try:
    import torch
    print("torch:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("device:", torch.cuda.get_device_name(0))
except Exception as exc:
    print("PyTorch check failed:", exc)
PY
  )
else
  echo "Pixi/Nerfstudio environment not available in this shell."
fi
