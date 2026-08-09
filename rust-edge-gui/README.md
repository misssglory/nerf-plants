# Rust Green Shape + Edge Composer — async wgpu build

Native Rust/egui application for selecting a closed green plant shape, detecting holes, and compositing multiple adaptive edge layers.

## What changed in 0.2

- `eframe` now uses the `wgpu` renderer (`Vulkan` is preferred on Linux).
- Heavy image processing runs on a background worker thread.
- Slider changes use **latest-request-wins** cancellation, so stale results never replace newer settings.
- Processing progress and stage are shown in the bottom bar.
- CPU-heavy loops and independent edge layers use `rayon` parallelism.
- Every extra edge layer has independent opacity.
- Layer 0 outline has independent opacity.
- A special green-area overlay is generated from Layer 0's selected shape.
- The green-area overlay has color and opacity controls.
- The selected green shape can become a real alpha mask in the saved PNG.
- The dimmed original stays underneath the colored overlay.
- Hovered-image mouse-wheel zoom remains available.

## Run with Nix

```bash
nix develop
cargo run --release -- /path/to/image.jpg
```

Check Vulkan availability:

```bash
vulkaninfo --summary
```

The flake exports:

```bash
WGPU_BACKEND=vulkan
```

To test another wgpu backend, override it before launch, for example:

```bash
WGPU_BACKEND=gl cargo run --release -- image.jpg
```

The app still uses CPU processing for segmentation and edge detection. `wgpu` accelerates GUI and texture rendering; `rayon` accelerates the actual image-processing loops.

## Important controls

### Layer 0 — closed green shape

- Green excess threshold
- Green ratio threshold
- Mask grow radius
- Hole detection
- Outline color and opacity
- Weighted-center marker

Layer 0 always selects the connected green component containing, or nearest to, the weighted center of all green evidence.

### Special green-area transparency layer

- overlay color
- overlay opacity
- optional output alpha mask
- shape output alpha (`0` fully transparent, `255` opaque)

The alpha mask is preserved when saving PNG. Detected holes remain outside the selected green mask.

### Extra edge layers

Each layer has:

- low/high edge thresholds
- threshold reduction near the green shape
- reduction radius
- color
- opacity
- independent enable/delete controls

## Performance notes

- Interactive changes are queued asynchronously.
- Older jobs are cancelled or ignored when newer slider values arrive.
- Edge layers are computed in parallel.
- You can limit or increase CPU parallelism with:

```bash
RAYON_NUM_THREADS=8 cargo run --release -- image.jpg
```
