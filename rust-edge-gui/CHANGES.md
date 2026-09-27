# Changes

1. Replaces the old two-column `ScrollArea` preview with two floating image windows.
2. Adds persistent `ImageViewState` for original and processed textures.
3. Adds Fit, 1:1, Center, logarithmic zoom, +/− zoom, pan, linked views.
4. Uses egui `InputState::zoom_delta()` for pinch/Ctrl-wheel zoom.
5. Uses egui `InputState::translation_delta()` for touchpad pan.
6. Keeps the existing `detect_green_shape` and image-processing pipeline untouched.
7. Reduces UI polling during processing from 33 ms to 125 ms.
8. Wakes egui immediately when processing completes or fails.
9. Stops scheduling processing repaint timers after completion.
