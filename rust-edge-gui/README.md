# rust-edge-gui 0.7.5

Rust/egui plant-mask viewer for top-view cameras. It supports disk images, HTTP camera snapshots, continuous capture, image-sequence history/timeline, pixel inspection, fast color masks, native offline YOLO segmentation, per-window navigation, sequence playback, temporal false-positive filtering, and tracked mask-area plots.




## What changed in 0.7.5

- **Smaller plot window** — the tracked-area plot can now be resized down to a compact height; help text and the legend disappear adaptively instead of imposing a content-driven minimum.
- **Automatic pivots** — when frame 1 of a sequence is processed, every closed connected mask component receives a pivot automatically at its centroid. Mask blobs touching the image border are not auto-seeded.
- **Collision groups** — if two or more tracked shapes merge into one connected mask component in any frame, they permanently become one logical group. The plot uses the union of their pixels in every frame, so a collision is not double-counted and the relationship is applied retroactively.
- **Grouped overlays** — collided members render as one group pivot/label. Hovering that marker still highlights it yellow; `P` removes the whole group.
- **Continue an existing recording** — open a sequence and use **Continue active sequence** under Continuous capture. New frames are appended to that sequence with numbering continued from the largest existing `frame_XXXXXX` index; existing pivots, groups and history stay intact.

## What changed in 0.7.4

- Plot window height is stable and resizable; adding tracks no longer changes its height recursively.
- Hover a tracked pivot/marker to highlight it yellow; press `P` while hovered to remove it.
- `Space` globally toggles sequence play/pause unless a text field is being edited.
- Click the plot X-axis to jump to the nearest frame by absolute timestamp. This can be disabled with `Click plot X-axis to seek frame` in Sequence mask tracking.
- The click-to-seek setting persists across app relaunches.

## What changed in 0.7.3

- **Immediate historical pivot plot** — adding a pivot retroactively matches that anchored component against every compatible mask already present in the sequence cache. Old points no longer require replaying the sequence just to appear.
- **Anchor-stable identity** — a track permanently remembers the frame, clicked pixel and original component where `P` was pressed. Matching propagates outward from that anchor through the nearest confirmed frame instead of overwriting one global reference and drifting.
- **Growth/shrink tolerance** — same-shape matching uses the stronger of IoU and overlap relative to the smaller component, with centroid distance as a tie-breaker.
- **Absolute-time plot** — plot X coordinates now use actual frame timestamps, including irregular capture gaps, with formatted local-time ticks.
- **Current-frame marker** — the plot renders a dashed vertical line and `F# time` label for the frame currently shown by the sequence viewer.
- **Resizable plot height** — plot canvas sizing follows the persisted/user-resized window rather than `available_height()`, which previously expanded the plot to screen height.
- **Window-position restore** — floating controls, plot and image windows opt out of egui screen clamping so persisted coordinates are reapplied exactly on relaunch.

## What changed in 0.7.2

- **No more live ghost frames** — a recording frame enters the active sequence only after the backing file has been written successfully. `sequence.json` is updated atomically at the same point.
- **Missing-frame self-healing** — if an older/stale timeline entry points to a file that disappeared, navigation prunes it instead of sending a nonexistent path to the decoder.
- **Correct live tracking index** — capture processing starts after the raw frame is committed, so tracking and symmetric temporal filtering are associated with the new frame rather than the previously selected frame.
- **Capture timing is no longer tied to YOLO speed** — once a raw frame is written, the next timed acquisition is free to proceed even if the previous mask job is still running. This keeps original-frame recording close to the requested cadence while analysis continues independently.
- **Resizable 2D-scrolling controls** — the floating Controls / sequence window can be resized in both axes and now has horizontal as well as vertical scrolling.
- **Automatic chronological glue** — add sequence folders in any order; all frames are merged by timestamp from oldest to newest. The glued sequence is written under the configured **Save directory** and opened automatically.
- **Timestamp-first sequence ordering** — sequence loading uses timestamp order with path only as a tie-breaker.


## What changed in 0.7.1

- **Live recording stays a sequence** — continuous capture no longer clears `active_sequence`. The sequence/timeline and temporal controls remain visible while frames arrive, and the live timeline grows frame-by-frame.
- **Sequence glue UI** — add multiple sequence folders and glue them into a new sequence. In 0.7.2 the input-list order is ignored and frames are sorted automatically by timestamp. Frames are copied into `original/`; `sequence.json` preserves the original timestamps.
- **Symmetric temporal filter** — temporal persistence now uses both previous and future cached masks. For a configured window `n`, support radius is `floor(n/2) - 1` frames backward and the same number forward.
- **Persistent GUI state** — detector parameters, colors, YOLO controls, temporal/tracking/playback controls, capture options, histories, glue list, image-window pan/zoom/open state, floating-window positions/sizes, and egui scroll/collapse state are restored after relaunch.
- **Native window persistence** — eframe persistence is enabled with a stable `rust-edge-gui` app ID, so the main OS window position/size is restored too.


## What changed in 0.7

- **Independent image navigation** — Original, Processed and YOLO windows no longer share pan/zoom state. Trackpad pan/zoom is accepted only by the visible image viewport under the pointer.
- **Monospace UI** — all normal GUI text styles use a constant-width font.
- **Floating scrollable controls** — controls, status and sequence transport live in one floating window instead of competing bottom/side panels.
- **Sequence playback** — Play/Pause/Prev/Next, configurable playback FPS, loop mode and optional wait-for-processing mode.
- **Tracked mask pivots** — while a sequence is open, hover a closed component in the Processed window and press `P`. The component becomes a tracked shape; its marker follows the matched component centroid on subsequent processed frames.
- **Cross-frame shape matching** — tracked components are associated frame-to-frame by configurable mask overlap.
- **Floating size plot** — tracked component area (pixels) is plotted against sequence frame as masks are processed.
- **Temporal false-positive filter** — optionally require a component to overlap masks in surrounding cached frames before it is rendered in the final processed image. Configure temporal window `n`, required confirmations and overlap threshold.

### Sequence tracking workflow

1. Open an image sequence. Frame 1 automatically seeds pivots for all closed mask components once processing finishes.
2. Leave **Wait for processing** enabled and press **Play** to process frames sequentially.
3. You can still add/remove pivots manually in the **Processed** window with `P`; collided tracks are permanently grouped.
4. Adjust **Same-shape overlap** if the shape grows or moves enough to break matching.
5. Open **Tracked mask size** to see the area curve.
6. For temporal cleanup, enable **Filter transient components using surrounding frames**. For temporal window `n`, the app uses `floor(n/2)-1` cached frames before and after the current frame. Play/analyze the sequence to populate the cache, then scrub/replay for the full symmetric filter.

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

Each image window owns its own pan/zoom state; trackpad gestures affect only the image viewport under the pointer.

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

The GUI persists recent single-image/URL sources and recent image-sequence folders. It also persists detector/UI settings and floating-window geometry. Continuous recording is intentionally **not** auto-resumed after relaunch.

Opening a sequence adds an editor-style timeline/transport with:

- frame number;
- **ABS** wall-clock/file timestamp;
- **VIDEO** time relative to the first frame;
- a scrub slider that loads and reprocesses the selected frame;
- Play/Pause/Prev/Next and configurable playback FPS;
- optional loop and wait-for-processing playback;
- `P` hotkey tracking for a closed processed-mask component;
- cross-frame IoU matching and a floating mask-area plot;
- optional symmetric previous/next-frame persistence filtering for transient false positives;
- live-growing timelines during continuous capture;
- sequence glue/concatenation with timestamp-preserving `sequence.json`.

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
