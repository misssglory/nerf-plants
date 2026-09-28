# rust-edge-gui 0.6.1

Rust/egui plant-mask viewer for top-view cameras. It supports disk images, HTTP camera snapshots, continuous capture, image-sequence history/timeline, pixel inspection, fast color masks, and native offline YOLO segmentation through ONNX Runtime.

## What changed in 0.6

The detector is no longer limited to `G - max(R,B)`. The GUI now has four mask sources:

1. **RG/B plant index** — default fast CPU heuristic for cameras where plant pixels have high red + green and low blue.
2. **YOLO segmentation** — uses an Ultralytics `*-seg.onnx` instance-segmentation model or `*-sem.onnx` semantic-segmentation model.
3. **Hybrid: YOLO ∩ color** — keeps YOLO pixels that also overlap an expanded RG/B color gate.
4. **Legacy green excess** — the original detector, retained for compatibility.

The runtime uses `ultralytics-inference` directly from Rust. Python is not required to run an exported ONNX model.

## RG/B plant index

For each pixel:

```text
RG mean      = (R + G) / 2
Blue deficit = RG mean - B
Plant index  = (R + G - 2B) / (R + G + 2B + 1)
G/R          = G / (R + 1)
```

The mask accepts the pixel when all four GUI limits pass. A good starting point for a camera with strong red + green and weak blue is:

```text
Blue deficit       28
Plant index         0.28
Min G/R             0.72
Min RG brightness   35
Min shape area      80
Final mask grow     1
```

Use the original-image pixel inspector to sample leaves, soil, pot and background and tune the thresholds from the real RGB values.

## YOLO segmentation

Open **Plant detection / mask source** and select **YOLO segmentation** or **Hybrid**.

Controls:

- **model** — local Ultralytics segmentation `.onnx` path;
- **Plant class IDs** — comma-separated class IDs to retain;
- **Confidence** — detection threshold for instance segmentation;
- **NMS IoU** — non-maximum suppression threshold;
- **Mask threshold** — probability threshold for instance masks;
- **Input size** — 320, 512, 640, 768, 1024 or 1280;
- **Device** — `Auto` or forced `CPU` in this portable build;
- **Fallback to RG/B** — if the model cannot load or inference fails, processing continues with the color detector;
- **Reload model + run** — invalidates the cached model and immediately runs it again;
- **Save AI mask…** — writes the raw combined YOLO mask.

YOLO/Hybrid slider edits intentionally wait for **Apply parameters**. This prevents a heavy inference run on every mouse movement.

### Instance segmentation

A model such as:

```text
yolo26n-seg.onnx
```

returns one mask per detected object. If **Plant class IDs** is empty, all detected instance classes are merged into the AI mask. For the stock COCO model, `potted plant` is class `58`, but a custom top-view leaf model is strongly recommended.

For a custom single-class instance model, plant/leaf is commonly class `0`.

### Semantic segmentation

A model such as:

```text
yolo26n-sem.onnx
```

returns one class ID per pixel. For semantic models **Plant class IDs must not be empty**, otherwise the whole semantic scene could be treated as foreground. Use the class IDs from the custom model's dataset.

The stock semantic model is intended for its original training classes, not top-view leaf phenotyping. Train/export a custom model for reliable plant/background segmentation.

### Export a model to ONNX

Runtime inference is Rust-only. Training/export can be done separately with Ultralytics Python:

```bash
# Example: official instance segmentation model
yolo export model=yolo26n-seg.pt format=onnx imgsz=640

# Your trained model
yolo export model=/path/to/best.pt format=onnx imgsz=640
```

Copy the resulting `.onnx` anywhere on the target machine and select it in the GUI. Once the local model exists, inference does not require Python.

See `MODEL_SETUP.md` for a compact stock/custom-model setup checklist.

## Hybrid mode

Hybrid mode computes:

```text
final source mask = YOLO mask ∩ expanded RG/B mask
```

**Hybrid color gate expand** dilates the color candidate mask before the intersection. This lets YOLO provide semantic/object context while the camera-specific color signal cleans up soil/background false positives without requiring perfect pixel color at every leaf edge.

After the source mask is built, the normal **Min shape area** and **Final mask grow** stages are applied.

## AI-mask preview

There are three floating views:

- **Original image** — includes x/y + RGBA + HEX hover inspection;
- **Processed — final plant mask**;
- **YOLO raw plant mask** — available after a YOLO/Hybrid inference.

Pan/zoom can be linked across all three.

## Camera loading

Enter an IP or address in **Camera IP / address**, for example:

```text
10.87.121.137
```

Then press **Load camera + process**. A plain IP is normalized to `http://IP/`. If the root response is not a decodable image, `/capture` is tried as a fallback.

A complete snapshot endpoint also works:

```text
http://10.87.121.137/snapshot.jpg
```

Every loaded frame is automatically processed with the currently selected detector.

## Saving and continuous capture

- **Save original…** writes the current source image.
- **Save processed…** writes the current final result.
- Continuous capture requests a camera frame every configurable `n` seconds.
- A session can save **Original**, **Processed**, or both.
- Sessions are written as `capture_YYYYMMDD_HHMMSS/original/` and `processed/`.
- Matching original/processed frame names make later analysis easy.
- The capture loop waits for the previous load + YOLO/color processing to finish, so frames do not pile up.

## Image sequences and history

The GUI persists recent single-image/URL sources and recent image-sequence folders.

Opening a sequence adds an editor-style timeline with:

- frame number;
- **ABS** wall-clock/file timestamp;
- **VIDEO** time relative to the first frame;
- a scrub slider that loads and reprocesses the selected frame.

A capture-session root can be opened directly. The sequence loader prefers `original/`, then `processed/`, then images in the selected folder.

## Build on NixOS

`ultralytics-inference 0.0.49` requires Rust 1.89 or newer. The included flake tracks `nixos-unstable` and supplies Rust/Cargo plus the desktop runtime libraries.

```bash
nix develop
cargo run --release
```

Disk image:

```bash
cargo run --release -- /path/to/image.jpg
```

Camera:

```bash
cargo run --release -- 10.87.121.137
```

## GPU note

The ZIP keeps `ultralytics-inference` with `default-features = false` for a small, portable CPU/Auto build. The library also supports optional accelerator features such as CUDA, TensorRT, CoreML, OpenVINO and others, but those require matching system/runtime dependencies. CPU is the safest baseline for the project.

## License note

This project now depends on `ultralytics-inference`, which is published under **AGPL-3.0**. If you plan to redistribute this application commercially or under a different licensing model, review the Ultralytics licensing terms before distribution.
