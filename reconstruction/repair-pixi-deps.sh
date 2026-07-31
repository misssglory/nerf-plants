#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if ! command -v pixi >/dev/null 2>&1; then
  echo "pixi is missing. Enter the Nix shell first:" >&2
  echo "  nix develop .#nerfstudio" >&2
  exit 1
fi

if [[ -f pixi.lock ]]; then
  backup="pixi.lock.before-protobuf-fix.$(date +%Y%m%d_%H%M%S)"
  mv pixi.lock "$backup"
  echo "Backed up stale lock file to: $backup"
fi

pixi install
pixi run python verify_environment.py

echo
echo "Pixi environment repaired. You can now run:"
echo "  ./train.sh plant_002 nerfacto"
