#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if ! command -v pixi >/dev/null 2>&1; then
  echo "pixi is missing. Enter the shell first:" >&2
  echo "  nix develop .#nerfstudio" >&2
  exit 1
fi

printf 'LD_LIBRARY_PATH entries:\n'
printf '%s\n' "${LD_LIBRARY_PATH:-}" | tr ':' '\n' | sed 's/^/  /'

printf '\nTesting Open3D import...\n'
if pixi run python - <<'PY'
import open3d
print("Open3D:", open3d.__version__)
print("module:", open3d.__file__)
PY
then
  echo "Open3D runtime is ready."
  exit 0
fi

printf '\nOpen3D import failed. Missing shared libraries, if any:\n' >&2
pybind="$(find .pixi/envs/default/lib -path '*/site-packages/open3d/cpu/pybind*.so' -print -quit 2>/dev/null || true)"
if [[ -n "$pybind" ]]; then
  echo "  $pybind" >&2
  ldd "$pybind" | grep -F 'not found' || true
else
  echo "Open3D pybind library was not found under .pixi/envs/default." >&2
fi
exit 1
