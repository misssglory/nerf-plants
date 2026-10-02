# v0.9.2

- Add MQTT 3.1.1 QoS1 pump publisher with GUI broker credentials and `FORWARD` / `OFF` control.
- Add timed pump runs with automatic OFF and fail-safe OFF retries after ambiguous/failed publishes.
- Add Home Assistant moisture-sensor selection plus ON/OFF hysteresis thresholds in the 0–100 range.
- Evaluate automatic watering only on newly received HA snapshot samples; HA history backfill never actuates the pump.
- Add an automatic `Pump status` timeseries (`0 = OFF`, `1 = ON`) integrated with existing rename/group/plot assignment controls.
- Persist pump status points and their plot metadata with `sequence_state.json`; persist MQTT/automation settings in app state, with password persistence opt-in.
- Bump package and HTTP user-agent version to 0.9.2.

# v0.9.1

- Add an in-memory LZ4 cache for decoded sequence frames. Cached payloads contain RGBA plus grayscale pixels, so playback avoids repeated PNG/JPEG decoding and disk reads.
- Warm the whole active sequence in a background prefetch thread; opening another sequence cancels the old prefetch generation. Cache misses still load from disk and immediately populate RAM.
- Add sequence UI statistics for cached frame count, compressed RAM usage, decoded/raw size, compression ratio, percentage saved, and cache hit rate.
- Keep live-captured original frames in the same LZ4 RAM cache and remove deleted frames from it safely.
- Omit transient per-frame `Loading image…` / `Loaded …; processing…` entries from Status History, including old persisted transient entries.
- Render formatted status/history timestamps and sequence absolute time in dark green.
- Bump package and HTTP user-agent version to 0.9.1.

# v0.9.0

- Add `Original` and `Processed` visibility toggles to **Timeseries / plot windows**.
- Re-opening either image window from that manager resets its window position to the default and resets image pan/zoom to Fit; closing it does not destroy the saved transform. Normal relaunch restores visibility, position, size, pan and zoom from persisted state.
- Stop rebuilding merged shape-group timeseries from every mask pixel on every egui repaint. Group union areas/centroids are now cached and invalidated only when tracking/masks actually change.
- Skip pivot-overlay construction entirely while the Processed window is hidden.
- Decimate very dense plot rendering to at most roughly four rendered samples per horizontal pixel while retaining the final point, reducing draw cost for long sequences without changing stored data.
- Bump package/user-agent version to 0.9.0.

# v0.8.9

- Fixed egui 0.35 compilation for transparent plot windows: use `Context::theme()` + `Context::style_of()` instead of removed `Context::style()`.
- Plot frame now derives its fill from the active dark/light theme before applying per-window alpha.

# v0.8.8

- Add per-plot `Background opacity` control with a `0.00..=1.00` range.
- Default every new and migrated plot window to `0.00` background opacity (fully transparent).
- Apply opacity to the floating plot-window background while keeping plotted series, pivots/points, axes, grid and text opaque.
- Persist opacity in both app preferences and sequence-level `sequence_state.json`.
- Bump package/user-agent version to 0.8.8.

# v0.8.7

- Move scheduled-capture retry cooldown out of `config.toml` and into the Continuous capture interface.
- Add persisted HTTP image request timeout to the Continuous capture interface.
- Keep Wi-Fi switch/readiness timeout as a separate persisted control.
- `config.toml` now contains Telegram credentials/settings only; capture timing is app UI state.

# v0.8.6

- Capture Wi-Fi readiness gate: after NetworkManager activation, wait for the target UUID to be active, its Wi-Fi device to have IPv4, and the camera host:port to accept TCP connections before issuing the HTTP image request.
- If capture Wi-Fi is already active, network switching is skipped completely, but host readiness is still verified.
- The configured timeout is a shared budget for profile activation plus readiness polling.
- On readiness failure after a switch, restore the previous Wi-Fi before the normal retry cooldown/Telegram path.
- Readiness polling supports systems with more than one simultaneously active Wi-Fi profile by matching the target UUID directly.

## v0.8.5 — Wi-Fi profile dropdown

- Replace the manual NetworkManager capture-profile text input with a dropdown populated from saved Wi-Fi connection profiles.
- Mark the active saved profile with `●` and inactive saved profiles with `○`; display a short UUID beside each name.
- Add `Refresh` to reload saved/active NetworkManager profiles on demand and keep `Use current` as a quick selector.
- Persist the chosen capture network as a canonical UUID.
- Migrate older v0.8.4 persisted connection names to UUIDs at startup when the profile still exists.
- Keep the existing no-op behavior when capture/current UUIDs are equal and previous-network restoration during retry cooldown.
- Bump package version to 0.8.5.

## 0.8.3

- Persist sequence-specific analysis data in `<sequence>/sequence_state.json`.
- Persist tracked pivots, collision groups, per-frame mask geometry/areas, centroids, and tracking observations with the sequence.
- Compress tracked mask geometry into contiguous pixel-index runs in JSON to keep sequence state substantially smaller than raw pixel arrays.
- Persist Home Assistant / FlowerCare sensor configuration and collected numeric samples with the sequence. Credentials remain app-local and are never copied into sequence state.
- Persist timeseries metadata (`name`, logical `group`, visibility, target `plot_id`) with the sequence.
- Persist plot windows, titles, positions/sizes, open state, and plot viewport state with the sequence.
- Restore sequence analysis state automatically when a sequence is opened.
- New sequences start with fresh numeric sensor history while keeping the app-level HA entity list as a convenient template.
- Debounced persistence now also runs after automatic shape-tracking updates and Home Assistant samples arrive, not only after direct UI interaction.
- Bump package/user-agent version to 0.8.3.

# Changes

## 0.8.2

- Add formatted local timestamps to current Status and a persistent 250-entry status/error history.
- Make processed-image pivot colors identical to the corresponding shape-area timeseries colors.
- Add `config.toml` Telegram notification configuration (historically included retry timing; v0.8.7 moves capture timing into the GUI).
- Send Telegram notification when image loading fails.
- For continuous capture, retry a failed scheduled frame after a cooldown only until the next scheduled frame is due; never let one bad slot freeze subsequent capture.
- Send one recovery notification when a retried capture succeeds.
- Bump package/user-agent version to 0.8.2.

## 0.8.1

- Fix Home Assistant compilation by enabling reqwest's `json` feature, making `blocking::Response::json()` available.
- Add `Delete current frame` to sequence transport. Deletion removes the current backing image plus the matching original/processed counterpart when present and rewrites `sequence.json`.
- Reindex temporal raw/final mask caches and tracked-shape observations after frame deletion so subsequent frame numbers remain consistent.
- If a deleted frame was a pivot anchor, re-anchor the track to the nearest surviving confirmed observation; drop the track only when no surviving observation exists.
- Reload the nearest remaining frame after deletion and keep empty active sequences usable for continued capture.
- Bump package version to 0.8.1.

## 0.8.0

- Add raw pre-temporal mask caching so later frames can both remove false positives and restore newly confirmed sprouts.
- Automatically retire temporal-filtered automatic pivots and automatically create pivots for newly confirmed closed shapes during live sequences.
- Add multiple independent plot windows with per-series target assignment.
- Add generic timeseries metadata: editable name, logical group, visibility and plot target.
- Add whole-group plot moves and Shift+drag rectangle grouping of Processed-image shape series.
- Add Home Assistant REST integration for numeric sensor states/history, including FlowerCare sensor entities, manual fetch, historical backfill and periodic polling.
- Persist plot windows, timeseries metadata and Home Assistant configuration; tokens are only stored when explicitly requested.
- Bump package version to 0.8.0.

## 0.7.6

- Add independent two-axis trackpad navigation to the tracked-size plot: horizontal scroll pans time, vertical scroll pans area.
- Add independent Ctrl+scroll plot zoom: Ctrl+horizontal zooms X/time and Ctrl+vertical zooms Y/area, anchored under the pointer.
- Persist plot pan/zoom state with the rest of the UI preferences (state schema v4) and add an in-window `Reset plot view` action.
- Show plot data-point values on hover: track/group ID, frame, mask area and absolute timestamp.
- Show Processed-image pivot values on hover: track/group identity, current mask area, frame, pivot coordinates and absolute timestamp; hovered pivots remain yellow.
- Scale Processed-image pivot markers logarithmically from the current frame/group mask area, with bounded screen-space radii.
- Suppress the pixel-inspector tooltip while hovering a pivot so the two value cards do not overlap.
- Bump package version to 0.7.6.

## 0.7.5

- Lower the tracked-size plot hard minimum and make help/legend responsive so the plot window can be resized much smaller vertically without content forcing it open again.
- Automatically add one pivot at the centroid of every closed connected mask component in sequence frame 1. Components touching the image border are excluded from auto-seeding.
- Add permanent collision grouping: if independent tracks ever resolve to the same connected component, they are merged into one logical group for every frame.
- Plot merged groups as the union of member pixels per frame, avoiding double-counting during collisions while preserving combined area before/after the collision.
- Render one grouped pivot/overlay for collided shapes; pressing `P` on that grouped pivot removes the whole merged group.
- Add `Continue active sequence` capture mode. New frames are appended to the currently open sequence, frame numbering continues from existing files, and existing tracking/history is preserved.
- Bump package version to 0.7.5.

## 0.7.4

- Fix tracked-size plot height feedback: plot canvas now follows the current visible window body instead of the previous persisted outer size, and the track legend no longer wraps vertically.
- Hovered pivot/track markers are highlighted yellow.
- Pressing `P` while hovering an existing pivot removes that track instead of stacking another pivot on it.
- Add global `Space` play/pause hotkey for image sequences (suppressed while editing text).
- Add click-to-seek on the plot X-axis; clicking chooses the nearest frame by absolute timestamp.
- Add a persisted `Click plot X-axis to seek frame` interface toggle.
- Persist the new plot interaction setting and bump state schema to v3.
- Bump package version to 0.7.4.


## 0.7.3

- Backfill new shape tracks immediately from all compatible cached sequence masks; playback is no longer required to populate already-cached historical points.
- Keep pivot frame, pivot pixel and anchor component immutable for every track.
- Match outward from the nearest confirmed frame and use IoU/containment overlap plus centroid tie-breaking to tolerate mask growth/shrink without reference drift.
- Render tracked-area plots against real absolute timestamps instead of uniform frame indices.
- Add a dashed vertical current-frame marker and formatted local absolute-time X-axis ticks.
- Fix plot-window vertical resizing by removing screen-sized `available_height()` canvas allocation.
- Disable egui position clamping for persisted floating image/control/plot windows so saved positions restore exactly.
- Bump package version to 0.7.3.

## 0.7.2

- Fixed live-recording ghost frames: a frame is added to the active sequence only after its backing image has been saved successfully.
- Capture sessions now update `sequence.json` atomically after committed frames, preventing timeline entries that point to files which do not exist.
- Sequence navigation validates paths and prunes stale/missing frames instead of attempting to decode them.
- Capture processing is scheduled after original-file commit so temporal filtering/tracking use the correct live-sequence frame index.
- Decoupled timed image acquisition from mask processing: slow YOLO/temporal jobs no longer block the next raw capture interval; interactive sequence navigation is deferred only while a raw capture is actually being fetched/saved.
- Controls window now supports both horizontal and vertical scrolling and remains freely resizable; controls keep a stable minimum content width so horizontal scrolling is actually usable.
- Sequence gluing is now chronological automatically: all frames from all selected sequences are merged by timestamp (oldest to newest), regardless of the order of the input list.
- Glue output is created directly under the configured Save directory, removing the extra destination dialog that could appear to do nothing behind floating windows.
- Sequence loading now sorts frames by timestamp (with path as a deterministic tie-breaker).
- Bumped package version to 0.7.2.

## 0.7.1

- Fixed continuous capture clearing `active_sequence`; sequence/timeline and temporal controls now stay visible while recording.
- Continuous capture now appends each incoming frame to the active live sequence so the timeline grows during recording.
- Added sequence glue UI with ordered inputs, ↑/↓ reordering, frame copying, and timestamp-preserving `sequence.json` manifests.
- Sequence loader now understands `sequence.json` manifests before falling back to `original/`, `processed/`, or root-folder scans.
- Temporal filtering is now symmetric: support radius is `floor(n/2)-1` frames backward and forward instead of future-only look-ahead.
- Persisted mask/color settings, YOLO settings, temporal/tracking/playback settings, capture options, history fields, glue list, independent image-view transforms, and floating-window geometry.
- Enabled eframe persistence and stable app ID so the main native window plus egui memory (scroll/collapse state) survive relaunches.
- Bumped package version to 0.7.1.

## 0.7.0

- Removed linked image transforms; Original, Processed and YOLO windows now keep completely independent pan/zoom state.
- Trackpad gestures are applied only when the pointer is over the unobscured image viewport.
- Switched all normal GUI text styles to monospace/constant-width fonts.
- Moved controls, notifications/status and sequence transport into one floating vertically scrollable window.
- Added sequence Play/Pause/Prev/Next transport, configurable FPS, loop mode and optional wait-for-processing playback.
- Added `P` hotkey on the Processed sequence view to create a pivot on the closed mask component under the cursor.
- Added tracked-shape association between frames using configurable component IoU; the visible pivot marker follows the matched component centroid.
- Added floating tracked-mask-area plot with one curve per pivot/shape.
- Added per-sequence mask cache used for tracking and temporal filtering.
- Added temporal false-positive filter using configurable future-frame look-ahead, confirmation count and overlap threshold.
- Bumped package version to 0.7.0.

## 0.6.1

- Fixed compile error caused by stale `yolo_last_mask_pixels` and `yolo_last_instances` field names.
- Reset logic now uses the actual `yolo_mask_pixels` and `yolo_instance_count` fields.
- Removed the unused `Rgba` import warning.
- Added a static consistency pass for `GreenViewerApp` field references.

## 0.6.0

- Added detection-mode selector: RG/B Plant Index, YOLO segmentation, Hybrid YOLO/color, and Legacy Green Excess.
- Added RG/B color detector for cameras where plant pixels have high R + G and low B.
- Added Blue deficit, Plant index, Min G/R, Min RG brightness and Hybrid color-gate expansion controls.
- Added native Rust YOLO ONNX inference with `ultralytics-inference 0.0.49`.
- Added support for both YOLO instance segmentation (`*-seg.onnx`) and semantic segmentation (`*-sem.onnx`).
- Added YOLO model path, class filter, confidence, IoU, mask threshold, input-size and device controls.
- Added persistent YOLO UI preferences.
- Added background YOLO worker with cached model/session so sequences and continuous capture reuse the loaded model.
- Added stale-job dropping so rapid frame/settings changes do not queue obsolete YOLO work.
- Added raw YOLO mask preview and Save AI mask action.
- Added Hybrid mode: YOLO mask intersected with an expanded camera-specific RG/B mask.
- Added optional automatic fallback to the RG/B detector when YOLO loading/inference fails.
- Linked pan/zoom can now synchronize Original, Processed and AI Mask windows.
- Updated final status from green-shape terminology to generic plant-mask terminology.
- Bumped package version to 0.6.0.

## 0.5.0

- Added separate **Save original** and **Save processed** actions.
- Added continuous camera capture with a user-configurable interval in seconds.
- Added capture-session directories with independent original/processed saving.
- Added persistent image/source history and persistent image-sequence history fields.
- Added image-sequence folder loading, including capture-session roots.
- Added editor-style sequence timeline/scrubbing with frame number, absolute timestamp, and relative video time.
- Added directory drag-and-drop as sequence loading.
- Added original-image pixel inspection on hover (x/y, RGBA, hex).
- Added capture-state handling so a slow camera/processor does not enqueue overlapping capture requests.
- Updated package version to 0.5.0.

## v0.8.4 — Capture Wi-Fi switching

- Added optional NetworkManager Wi-Fi switching for continuous capture.
- Capture Wi-Fi is configured by an existing NetworkManager connection name or UUID; Wi-Fi credentials remain in NetworkManager and are not stored by the app.
- The active Wi-Fi profile is detected automatically before every scheduled capture/retry.
- Target/current profiles are compared by canonical NetworkManager UUID; if they match, no disconnect/reconnect is performed.
- When profiles differ, the image-source worker switches to the capture profile, fetches the frame, and restores the previously active Wi-Fi before returning success/failure to the GUI.
- Retry cooldown therefore runs on the previous network automatically; the next retry switches to the capture profile again.
- Added capture-network timeout and a `Use current` helper in Continuous capture settings.
- Network switching runs inside the source worker so `nmcli --wait` does not block the egui UI thread.
