# rust-edge-gui

Rust/egui green-shape + edge viewer with disk images, camera snapshots, continuous capture, sequence history, and timeline playback/scrubbing.

## Camera loading

Enter an IP or address in **Camera IP / address**, for example:

```text
10.87.121.137
```

Then press **Load camera + process**. The app normalizes a plain IP to `http://IP/`. The exact root address is tried first; if it returns HTML or is not a decodable image, `/capture` is tried as a compatibility fallback. A successfully loaded frame is immediately passed to green-shape processing.

You may also enter a complete snapshot endpoint such as:

```text
http://10.87.121.137/snapshot.jpg
```

The HTTP response is fully drained before JPEG/PNG decoding so a high-resolution frame is not intentionally decoded from a partial response.

## Saving and continuous capture

- **Save original…** writes the currently loaded original image to disk.
- **Save processed…** writes the current processed result to disk.
- **Continuous capture** requests a new camera frame every configurable `n` seconds.
- Choose whether a capture session saves **Original**, **Processed**, or both.
- Each recording creates a session folder such as `capture_20260928_021530/` with `original/` and/or `processed/` subfolders.
- Original and processed frames use matching frame names so corresponding results are easy to pair.
- The capture loop waits for the previous frame to finish loading and processing instead of piling up overlapping requests.

## Image and sequence history

The left panel has separate fields for:

- a single image path or URL;
- an image-sequence folder.

Both recent source history and recent sequence history are persisted in the app state. A capture-session root folder can be opened directly; the sequence loader prefers `original/`, falls back to `processed/`, and also supports ordinary folders containing images.

When a sequence is open, the central UI shows an editor-style time row with:

- frame position;
- an **ABS** absolute timestamp based on the frame file timestamp;
- **VIDEO** time relative to the first frame;
- a scrub slider that loads the selected frame and automatically reprocesses it.

Dropping a directory onto the app opens it as an image sequence. Dropping a file opens it as a single image.

## Pixel inspection

Hover the original image to display the exact image coordinate plus RGBA and hexadecimal pixel values. The lookup respects current pan, zoom, and Fit scaling.

## Other features

- Green-shape detection and optional edge overlay run on a background worker.
- Separate floating Original and Processed windows.
- Resizable image windows with Fit / 1:1 / Center / +/- controls.
- Drag to pan.
- Touchpad two-finger pan.
- Pinch or Ctrl+wheel zoom around the cursor.
- Optional linked pan/zoom between Original and Processed.
- Worker threads block while idle; continuous capture only schedules repainting while recording.

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
