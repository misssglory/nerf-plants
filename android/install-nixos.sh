#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"
./build-nixos.sh

apk="app/build/outputs/apk/debug/app-debug.apk"

echo
echo "Connected Android devices:"
adb devices -l

if ! adb -d get-state >/dev/null 2>&1; then
  cat >&2 <<'MSG'
No authorized physical Android device was found.

On the phone:
  1. Enable Developer options.
  2. Enable USB debugging.
  3. Connect USB in data-transfer mode.
  4. Accept the RSA authorization popup.

Then run this script again.
MSG
  exit 1
fi

adb -d install -r "$apk"
adb -d shell am start -n com.example.plantcapture/.MainActivity
