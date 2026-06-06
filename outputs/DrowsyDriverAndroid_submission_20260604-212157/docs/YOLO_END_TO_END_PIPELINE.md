# YOLO End-To-End Pipeline

## Goal

Use YOLO object detection to answer both practical questions at once:

```text
Where are the relevant fatigue cues?
What state are they in?
```

Instead of relying on MediaPipe landmarks, YOLO directly detects objects/classes
such as `open_eye`, `closed_eye`, and `yawning`.

## Pipeline

```text
Camera frame
-> YOLO detector
-> bounding boxes for open_eye / closed_eye / yawning
-> confidence filtering
-> temporal smoothing
-> sound/vibration alert
-> minimal event log
```

## Recommended Classes

Use the smallest useful class set:

```text
open_eye
closed_eye
yawning
```

Optional classes if the dataset supports them:

```text
face
normal
drowsy
```

Do not mix classification folders with YOLO object detection unless labels are
converted to bounding boxes.

## Dataset Requirements

YOLO requires bounding-box labels. A valid YOLO label file contains:

```text
class_id x_center y_center width height
```

Values are normalized from 0 to 1.

Expected Ultralytics-style structure:

```text
yolo_dataset/
  images/
    train/
    val/
    test/
  labels/
    train/
    val/
    test/
  data.yaml
```

`data.yaml` should define train/val/test paths and class names. Ultralytics
documents this format for object detection datasets.

## Dataset Candidates

| Priority | Dataset | Use |
|---|---|---|
| 1 | `aryansharma8911/open-closed-eyes-and-yawning-labelled` | Best starting point because it reports YOLO-format annotations for open eyes, closed eyes, and yawning |
| 2 | Roboflow Drowsiness/Fatigue Detection datasets | Good if free export in YOLO format is available |
| 3 | Simuletic driver monitoring sample | Useful synthetic reference; JSON labels may require conversion |
| 4 | MRL or classification-only datasets | Not direct YOLO data; requires manual or automatic bounding-box annotation |

## Training Commands

Typical Ultralytics workflow:

```bash
pip install ultralytics
yolo detect train model=yolo11n.pt data=yolo_dataset/data.yaml imgsz=640 epochs=50
yolo detect val model=runs/detect/train/weights/best.pt data=yolo_dataset/data.yaml
yolo export model=runs/detect/train/weights/best.pt format=tflite
```

If using the latest Ultralytics family available in your environment, replace
`yolo11n.pt` with the smallest current nano model supported by your installed
Ultralytics version.

## Android Deployment Considerations

YOLO-to-Android is possible through TFLite export, but it is more complex than
the current MediaPipe + CNN path:

- TFLite detection output requires correct post-processing or embedded NMS.
- Object detection inference is heavier than classifying a small cropped eye ROI.
- Android integration must draw/use bounding boxes and class confidences.
- The current app would need a new YOLO inference path, not just the existing
  `TfliteDrowsinessClassifier`.

## Evaluation Metrics

Report both detection and safety metrics:

- mAP50 or mAP50-95 for detection quality;
- precision/recall per class;
- recall for `closed_eye` and `yawning`;
- FPS on device or laptop;
- alert-level false positives and missed alerts after temporal smoothing.

## Strengths

- Solves localization and state detection in one model.
- Better conceptual fit if the project wants object detection.
- Can detect multiple cues in one frame, such as eyes and yawning.
- Easier to extend to other driver-monitoring objects later.

## Risks

- Needs bounding-box labels.
- Dataset quality matters more; poor boxes lead to poor detector behavior.
- Training/export/deploy is heavier than CNN.
- Android TFLite post-processing can be a deadline risk.
- If only classification datasets are available, annotation work can dominate
  the project time.

## Defense Statement

> YOLO is selected when the project goal is end-to-end object detection: locating
> and classifying open eyes, closed eyes, and yawning regions directly from the
> camera frame. This is more general than ROI classification, but it requires
> YOLO-format bounding-box datasets and a more complex Android deployment path.

## Useful Sources

- Ultralytics object detection dataset format:
  https://docs.ultralytics.com/datasets/detect/
- Ultralytics object detection task:
  https://docs.ultralytics.com/tasks/detect/
- Ultralytics TFLite export:
  https://docs.ultralytics.com/integrations/tflite/
