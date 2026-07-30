#!/usr/bin/env bash
set -euo pipefail
: "${PLANT_CAPTURE_TOKEN:=replace-with-a-long-random-token}"
exec python3 "$(dirname "$0")/server.py" \
  --output "${PLANT_CAPTURE_OUTPUT:-$HOME/PlantCaptures}" \
  --port "${PLANT_CAPTURE_PORT:-8765}" \
  --token "$PLANT_CAPTURE_TOKEN"
