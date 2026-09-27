# rust-egui-viewer

Rust/egui image viewer and green-shape detector.

## Features

- Load an image from disk, drag-and-drop, CLI path, or an HTTP/HTTPS URL.
- URL history is stored in the platform config directory.
- Green-shape detection by green-excess + green-ratio thresholds.
- Connected components with configurable minimum shape area.
- Optional mask growth and Sobel edge overlay.
- Background processing with latest-request-wins cancellation.
- No repaint polling loop: after loading/processing ends, worker threads block on channels and the UI can idle normally.
- Separate floating, movable, resizable windows for Original and Processed images.
- Fit-to-window follows window borders while preserving aspect ratio.
- Pan by dragging or touchpad two-finger translation.
- Zoom with pinch gesture or Ctrl+mouse-wheel, anchored to cursor position.
- 1:1, Fit, Center, +/- and logarithmic zoom slider.
- Optional linked pan/zoom between the two image windows.

## Build

```bash
cargo run --release
```

Open a file from the command line:

```bash
cargo run --release -- /path/to/image.jpg
```

Open a network image directly:

```bash
cargo run --release -- http://192.168.1.100/capture
```

For ESP32 camera sources, enter the exact endpoint that returns image bytes. If the base URL returns an HTML camera page, use its JPEG/capture endpoint instead.
