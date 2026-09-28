# Offline plant segmentation model setup

The GUI does not bundle model weights. Put an Ultralytics segmentation ONNX model on disk and select it from the **YOLO segmentation (offline ONNX)** panel.

## Quick test with a stock instance-seg model

Export/download `yolo26n-seg.onnx` on a machine with Ultralytics installed, copy the ONNX file to the target machine, select it in the GUI, and use class ID `58` for the COCO `potted plant` class.

Stock COCO weights are only a smoke test for a top-view growing setup. They are not trained to segment every leaf from above.

## Recommended custom model

For reliable top-view masks, train a one-class dataset (`plant` or `leaf`) from your own camera. A one-class Ultralytics instance-seg model normally uses class ID `0`.

Example export after training:

```bash
yolo export model=/path/to/best.pt format=onnx imgsz=640
```

Then in the GUI:

```text
Detection mode: YOLO segmentation   (or Hybrid)
Model: /path/to/best.onnx
Plant class IDs: 0
Confidence: 0.25
Mask threshold: 0.50
Input size: 640
Device: Auto
```

## Hybrid starting point for the current camera

If plant pixels have high R + G and low B, start with:

```text
Blue deficit:       28
Plant index:         0.28
Min G/R:             0.72
Min RG brightness:   35
Hybrid color expand: 3
Min shape area:      80
Final mask grow:     1
```

Use the Original window's hover pixel inspector to tune these values against leaf, soil, pot, and background samples.

## Semantic segmentation

YOLO26 semantic ONNX models are also accepted. For semantic models, **Plant class IDs cannot be empty**; enter the class ID(s) assigned to plant/leaf in the training dataset.
