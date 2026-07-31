# Rust Canny Edge Viewer

Interactive Canny edge detection with Rust, `egui`, `image`, and `imageproc`.

## Run in the Nix development shell

```bash
nix develop
cargo run --release -- /path/to/image.jpg
```

The image argument is optional. You can open or drag an image into the window.

## Controls

- Low and high Canny hysteresis thresholds
- Optional Gaussian pre-blur
- Inverted output
- Live update while dragging sliders
- Open, drag-and-drop, and save as PNG

`imageproc::edges::canny` uses a fixed internal Gaussian blur. The extra blur slider applies an additional blur before Canny, which is useful for suppressing fine texture and noise.

## egui 0.35 API note

This project targets `eframe = 0.35.0`. In this release, applications implement
`App::ui(&mut Ui, &mut Frame)`, and side/top/bottom layouts use `egui::Panel`.
