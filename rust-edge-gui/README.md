# Rust Green Shape + Edge Composer

A small native Rust + egui tool for interactively finding a closed green shape (for example a leaf), then layering additional colored edge detections on top of a dimmed original image.

## Features

- open image from CLI, file dialog, or drag-and-drop
- black UI theme
- original and overlay preview side by side
- mouse-wheel zoom on the hovered preview
- separate stored scale for original and overlay preview
- dimmed original image underneath the overlay preview
- locked **Layer 0** that finds the closed green shape around the weighted center of green pixels
- hole detection inside the selected green shape
- area statistics for the shape and its holes
- multiple extra edge layers with separate thresholds, colors, and threshold reduction near the green shape
- save the composite overlay as PNG

## Build and run

```bash
nix develop
cargo run --release -- /path/to/image.jpg
```

or without an image:

```bash
nix develop
cargo run --release
```

## Notes

The first layer is special:

- it computes a weighted green center from green pixels
- it finds the connected green component that contains or is closest to that center
- it outlines that selected closed green shape
- it can also find and outline holes

The additional edge layers use a simple adaptive Sobel + hysteresis edge detector. Their thresholds can be reduced near the green shape so weak leaf boundaries are easier to keep.
