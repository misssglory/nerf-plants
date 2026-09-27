# rust-edge-gui

Rust/egui green-shape + edge viewer with disk and camera-IP loading.

## Camera loading

Enter an IP or address in **Camera IP / address**, for example:

```text
10.87.121.137
```

Then press **Load camera + process**. The app normalizes a plain IP to:

```text
http://10.87.121.137/
```

The exact root address is tried first, matching the earlier controller workflow. If the root returns HTML or is not a decodable image, the app also tries `/capture` as a compatibility fallback. A successfully loaded frame is immediately passed to green-shape processing.

You may also enter a full endpoint such as:

```text
http://10.87.121.137/snapshot.jpg
```

In that case the exact endpoint is used.

The HTTP response is fully drained before JPEG/PNG decoding so high-resolution images are not intentionally decoded from a partial response.

## Other features

- Open images from disk or drag-and-drop.
- Input history persists across restarts.
- Green-shape detection and optional edge overlay run on a background worker.
- Loading automatically triggers processing.
- Worker threads block while idle and do not keep a repaint loop alive after work completes.
- Separate floating Original and Processed windows.
- Resizable image windows with Fit / 1:1 / Center / +/- controls.
- Drag to pan.
- Touchpad two-finger pan.
- Pinch or Ctrl+wheel zoom around the cursor.
- Optional linked pan/zoom between Original and Processed.

## Build on NixOS

```bash
nix develop
cargo run --release
```

Or load a disk image from the command line:

```bash
cargo run --release -- /path/to/image.jpg
```

Or a camera address:

```bash
cargo run --release -- 10.87.121.137
```
