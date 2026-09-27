# Rust Green Shape + Edge Composer — ESP32 capture, profiles, async wgpu

Native Rust/egui application for selecting a closed green plant shape, detecting holes, and compositing multiple adaptive edge layers.

## What changed in 0.4

### Camera resolution selector

The **Camera resolution** dropdown is shown next to the image-source controls.

For a standard Espressif `CameraWebServer`, when a concrete resolution is selected the app first calls:

```text
http://<controller>/control?var=framesize&val=<N>
```

and then fetches the still frame from:

```text
http://<controller>/capture
```

`Keep camera setting` skips the control request and captures using whatever resolution is already active on the controller.

The dropdown includes the ESP camera frame sizes through UXGA (1600×1200). The application always displays the **actual decoded image size** after loading, so unsupported or ignored settings are visible immediately.

Local image files are never resized by this control.

### Persistent profiles

Profiles are now first-class UI state. Each profile remembers independently:

- selected camera resolution;
- last successful image/controller source;
- the last 40 source-history entries;
- shared blur and original dimness;
- black preview mode;
- update-while-dragging mode;
- preview scales;
- all Layer 0 green-shape settings;
- green-area overlay/alpha settings;
- every extra edge layer, including enabled state, thresholds, reduction settings, radius, color and opacity.

Use the **Profile** selector to switch profiles.

- **Save now** writes the current profile immediately.
- Enter a name and use **Save as / copy** to create a new profile from the current settings (or overwrite that named profile).
- **Delete** removes the current profile when at least one other profile exists.
- Changes are automatically saved after a short debounce while editing.
- Pending changes are also flushed when the app exits.
- The last active profile is restored on the next launch.

Profiles are stored under:

```text
$XDG_CONFIG_HOME/rust-edge-gui/profiles/
```

or, when `XDG_CONFIG_HOME` is not set:

```text
~/.config/rust-edge-gui/profiles/
```

The active profile name is stored in:

```text
~/.config/rust-edge-gui/active-profile.txt
```

The old v0.3.x global `source-history.txt` is imported into the first profile when no profile exists yet.

## ESP32 source input

Enter any of:

```text
192.168.1.42
esp32cam.local
http://192.168.1.42/capture
/path/to/image.jpg
```

A bare controller host/IP is resolved to `/capture`. A complete `http://` URL is used exactly as typed.

Remote loading and image decoding run on a background worker, so the GUI stays responsive.

`/stream` is intentionally rejected because the processing pipeline consumes one still frame at a time.

## Run with Nix

```bash
nix develop
cargo run --release -- /path/to/image.jpg
```

The flake prefers Vulkan for wgpu on Linux:

```bash
WGPU_BACKEND=vulkan cargo run --release
```

The segmentation/edge algorithms remain CPU-side and use Rayon; wgpu accelerates GUI/texture rendering.
