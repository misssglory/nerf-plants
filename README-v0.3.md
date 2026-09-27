# rust-edge-gui v0.3 — idle CPU + floating image viewers

This targets the **async/wgpu 0.2.0** source (`EdgeApp` + background `ProcessingWorker`).

## Changes

- GUI repaint timer is active **only while processing**, at 10 Hz instead of 30 Hz; after the final result it schedules nothing, so the app can sleep when idle.
- Original and processed images live in independent **movable/resizable egui floating windows**.
- `Fit` automatically fits the whole image to the current floating-window borders.
- `1:1`, `Center`, `−`, `+`, and logarithmic zoom sliders.
- Drag image to pan.
- Two-finger touchpad scroll pans.
- Touchpad pinch zooms through `egui::InputState::zoom_delta()` when exposed by winit/Wayland/X11.
- Ctrl/Command + wheel is a zoom fallback.
- Optional linked zoom/pan for the original and processed views.

## Apply to your current project

Copy `apply_v0_3.py` into the project root, then:

```bash
python3 apply_v0_3.py
cargo fmt
cargo check
cargo run --release -- /path/to/image.jpg
```

The script creates `src/main.rs.v0.2-backup` before writing the modified source.

## Revert

```bash
mv src/main.rs.v0.2-backup src/main.rs
```

## Touchpad note

Pinch gesture availability ultimately depends on the platform backend/driver. egui's `zoom_delta()` is used, so Wayland/libinput touchpad pinch has the best chance of arriving as a native zoom gesture. Two-finger scrolling still works as pan even when pinch is not exposed.
