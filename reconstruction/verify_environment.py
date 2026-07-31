from __future__ import annotations

import shutil
import subprocess
import sys


def command_version(command: list[str]) -> str:
    executable = shutil.which(command[0])
    if executable is None:
        return "missing"
    try:
        result = subprocess.run(
            command,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
    except OSError as exc:
        return f"error: {exc}"
    first_line = result.stdout.strip().splitlines()
    return first_line[0] if first_line else f"exit {result.returncode}"


print("python:", sys.version.split()[0])
print("ffmpeg:", command_version(["ffmpeg", "-version"]))
print("colmap:", command_version(["colmap", "-h"]))

try:
    import torch
except ImportError as exc:
    raise SystemExit(f"torch import failed: {exc}") from exc

print("torch:", torch.__version__)
print("torch CUDA build:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("compute capability:", torch.cuda.get_device_capability(0))

try:
    import nerfstudio
except ImportError as exc:
    raise SystemExit(f"nerfstudio import failed: {exc}") from exc

print("nerfstudio:", getattr(nerfstudio, "__version__", "installed"))

try:
    import tinycudann  # noqa: F401
except ImportError:
    print("tiny-cuda-nn: missing (nerfacto will not use its fast CUDA backend)")
else:
    print("tiny-cuda-nn: available")

try:
    import google.protobuf
except ImportError as exc:
    raise SystemExit(f"protobuf import failed: {exc}") from exc

protobuf_version = google.protobuf.__version__
print("protobuf:", protobuf_version)
if protobuf_version != "3.20.3":
    raise SystemExit(
        "expected protobuf 3.20.3 for Nerfstudio 1.1.5, got "
        f"{protobuf_version}; remove pixi.lock and run pixi install"
    )

try:
    import tensorboard
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator  # noqa: F401
except ImportError as exc:
    raise SystemExit(f"tensorboard event reader import failed: {exc}") from exc
else:
    print("tensorboard:", tensorboard.__version__)
    print("tensorboard event reader: available")
