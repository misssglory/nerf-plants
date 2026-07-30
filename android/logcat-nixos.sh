#!/usr/bin/env bash
set -euo pipefail

# Show only messages associated with the app process when supported.
pid="$(adb -d shell pidof com.example.plantcapture 2>/dev/null | tr -d '\r' || true)"
if [[ -n "$pid" ]]; then
  exec adb -d logcat --pid="$pid"
else
  echo "The app is not running; showing filtered logcat." >&2
  exec adb -d logcat | grep --line-buffered -E 'PlantCapture|plantcapture|AndroidRuntime'
fi
