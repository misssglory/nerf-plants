#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

pixi run python - <<'PY'
import json
import sys

import numpy as np
import torch

print(f"Python:  {sys.version.split()[0]}")
print(f"NumPy:   {np.__version__}")
print(f"PyTorch: {torch.__version__}")

major = int(np.__version__.split('.', 1)[0])
if major >= 2:
    raise SystemExit(
        "ERROR: this Nerfstudio/PyTorch environment requires NumPy 1.x; "
        f"found {np.__version__}. Run: pixi update numpy"
    )

# Exercise the exact torch<->NumPy bridge that failed during ns-process-data.
x = torch.tensor([1.0, 2.0])
y = x.numpy()
assert y.tolist() == [1.0, 2.0]
print("Torch/NumPy bridge is ready.")
PY
