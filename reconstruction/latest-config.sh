#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
NAME="${1:-}"

if [[ -n "$NAME" ]]; then
  find "$SCRIPT_DIR/outputs" -path "*/$NAME/*/config.yml" -type f -printf '%T@ %p\n' 2>/dev/null \
    | sort -nr | head -n1 | cut -d' ' -f2-
else
  find "$SCRIPT_DIR/outputs" -name config.yml -type f -printf '%T@ %p\n' 2>/dev/null \
    | sort -nr | head -n1 | cut -d' ' -f2-
fi
