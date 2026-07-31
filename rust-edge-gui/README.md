# Rust Multi-layer Edge Viewer

Native `egui`/`eframe` application for interactively composing several colored Canny edge layers.

## Features

- Load an image from a CLI path, native file dialog, or drag-and-drop.
- Side-by-side original and layered-edge previews.
- Independent preview scaling for original and edge images (`0.1×` to `8×`).
- Add and delete any number of edge layers.
- Enable/disable each layer independently.
- Separate Canny low/high thresholds and color for each layer.
- Shared optional Gaussian pre-blur.
- Black or white output background.
- Alpha-aware color compositing when layer edges overlap.
- Live recalculation while sliders are dragged, with an option to disable it.
- Save the rendered layered edge image as PNG.

## Run with Nix

```bash
nix develop
cargo run --release -- /path/to/image.jpg
```

Or start without an image:

```bash
cargo run --release
```

## Controls

- **Original / Edges scale**: each preview is scaled independently. `1×` fits large images to their column; values above `1×` can be explored with scrollbars.
- **Pre-blur σ**: Gaussian blur applied once before all Canny layers.
- **White background**: switches the edge composite from black to white.
- **Add layer**: creates another independently configured Canny layer.
- **Layer checkbox**: enables or disables that layer.
- **Color button**: changes the edge color, including alpha when supported by the picker.
- **Low / High**: Canny hysteresis thresholds for the selected layer.
- **Delete**: removes the layer.

Layers are rendered from top to bottom in the controls. Later layers are composited over earlier layers.

## Notes

`imageproc::edges::canny` accepts floating-point thresholds. Its useful range can exceed the familiar OpenCV `0..255` range, so the UI allows values up to `1140`.
