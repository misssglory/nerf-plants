#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: ./gaussian/view-splatfacto.sh NAME [CONFIG_YML]" >&2
  exit 2
fi

NAME="$1"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RECON_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"

if [[ $# -eq 2 ]]; then
  CONFIG="$(realpath -- "$2")"
else
  CONFIG="$(find "$RECON_DIR/outputs/$NAME/splatfacto" -name config.yml -type f -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -n1 | cut -d' ' -f2-)"
fi

[[ -n "$CONFIG" && -f "$CONFIG" ]] || {
  echo "No Splatfacto config found for: $NAME" >&2
  exit 1
}

cd "$RECON_DIR"
exec pixi run ns-viewer --load-config "$CONFIG" --viewer.websocket-port 7007
