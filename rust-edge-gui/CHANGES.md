# Changes

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
