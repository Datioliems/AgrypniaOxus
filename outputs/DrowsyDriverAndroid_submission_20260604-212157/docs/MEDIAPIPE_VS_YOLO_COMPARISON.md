# MediaPipe vs YOLO Comparison

## Short Recommendation

For the current deadline and Android prototype, use **MediaPipe + CNN** as the
main implementation.

Use **YOLO end-to-end** only if you have a YOLO-format dataset ready and accept
the extra Android deployment risk.

Use **Hybrid YOLO + CNN** only as an advanced extension, not the default 4-day
plan.

## Comparison Table

| Criterion | MediaPipe + CNN | YOLO end-to-end | Hybrid YOLO + CNN |
|---|---|---|---|
| Main idea | MediaPipe locates face/eyes; CNN classifies cropped eye ROI | YOLO detects `open_eye`, `closed_eye`, `yawning` boxes directly | YOLO locates eyes/face; CNN classifies ROI |
| Dataset needed | Classification images for eye states | YOLO bounding-box labels | Bounding boxes plus classification labels |
| Current project fit | Very high | Medium | Medium-low |
| Android complexity | Low-medium | High | High |
| Deadline risk | Low | High | High |
| Explainability | Strong with EAR/MAR | Medium, depends on boxes/confidence | Medium |
| Best use | Course demo and robust report | Object detection research direction | Future extension |
| Main failure mode | Bad landmarks under occlusion/lighting | Bad boxes or post-processing | Both detector and classifier errors |

## Dataset Decision

| Dataset | MediaPipe + CNN | YOLO end-to-end |
|---|---|---|
| `prasadvpatil/mrl-dataset` | Good for `eyes_open` / `eyes_closed` classification | Not direct unless bounding boxes are added |
| `dheerajperumandla/drowsiness-dataset` | Useful for classification or report | Not direct unless it includes YOLO labels |
| `aryansharma8911/open-closed-eyes-and-yawning-labelled` | Can be converted to crops, but not necessary | Best YOLO candidate because it reports YOLO-format labels |
| Roboflow drowsiness datasets | Optional | Good if YOLO export is available |
| Android phone self-recorded data | Good for demo/testing | Needs annotation before YOLO training |

## Rubric Mapping

| Rubric part | MediaPipe + CNN | YOLO end-to-end |
|---|---|---|
| Chapter 1 | Strong practical scope, clear pretrained + custom model split | Strong if framed as object detection |
| Chapter 2 | Easier data preparation and labeling | Must explain bounding boxes and annotation |
| Chapter 3 | Clear pipeline: landmarks, EAR/MAR, CNN/TFLite | Clear but more complex: detector, labels, post-processing |
| Chapter 4 | Classification metrics + FPS + demo | Detection mAP + class recall + FPS + demo |
| Product | Existing Android app already matches | Requires new inference path |
| Presentation | Easier to defend in 8 minutes | More impressive but harder to defend if unfinished |

## Android Deployment Impact

MediaPipe + CNN:

```text
CameraX
-> MediaPipe landmarks
-> crop eye ROI
-> small TFLite classifier
-> smoothing
-> alert
```

YOLO end-to-end:

```text
CameraX
-> YOLO TFLite detector
-> parse boxes/classes/confidence
-> filter closed_eye/yawning
-> smoothing
-> alert
```

The YOLO path needs more Android work because object detection outputs must be
decoded, filtered, and possibly passed through NMS. If export includes embedded
NMS, Android integration is easier, but it still needs validation.

## Streamlit Fallback

Both methods can have a Streamlit fallback:

| Method | Streamlit fallback |
|---|---|
| MediaPipe + CNN | OpenCV webcam + MediaPipe + Keras/TFLite classifier |
| YOLO | OpenCV webcam + Ultralytics YOLO `.pt` on laptop |
| Hybrid | OpenCV webcam + YOLO boxes + CNN classification |

Streamlit is easier than Android for YOLO because Ultralytics can run directly
in Python. However, Android native should remain the main product if it runs.

## Final Recommendation By Scenario

| Scenario | Choose |
|---|---|
| Need highest chance of finishing in 4 days | MediaPipe + CNN |
| Need object detection research and have YOLO labels ready | YOLO end-to-end |
| Need a strong report with low implementation risk | MediaPipe + CNN, mention YOLO as extension |
| Need a more ambitious demo and can accept risk | YOLO end-to-end on Streamlit first, Android later |
| Need Android native app as final product | MediaPipe + CNN |

## Suggested Defense Answer

> MediaPipe + CNN is the main implementation because MediaPipe solves the
> localization problem: where the face, eyes, and mouth are. The project-built
> CNN solves the classification problem: whether the cropped eye region is open
> or closed. YOLO end-to-end can solve both localization and state detection, but
> it requires bounding-box labels and a more complex Android deployment path, so
> it is treated as an alternative or future extension unless YOLO labels are
> already available.
