# Changes

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
