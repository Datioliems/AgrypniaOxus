# MediaPipe Landmark Pipeline

## Goal

Use MediaPipe to answer the practical question: **where are the driver's face,
eyes, and mouth?** Then use lightweight rules and CNN classification to decide
whether the driver shows drowsiness signals.

This is the safest main path for the current 4-day project because the Android
prototype already contains CameraX, MediaPipe Face Landmarker, EAR/MAR,
eye-region crop, TFLite classifier loading, alerting, and event logging.

## Pipeline

```text
Android CameraX frame
-> MediaPipe Face Landmarker
-> face + eye + mouth landmarks
-> EAR/MAR baseline
-> crop eye ROI from landmarks
-> CNN/TFLite eyes_open vs eyes_closed
-> temporal smoothing
-> sound/vibration alert
-> minimal event log
```

## What Is Pretrained And What Is Built By The Project

| Component | Source | Role |
|---|---|---|
| MediaPipe Face Landmarker | Pretrained MediaPipe model | Detect face landmarks in realtime |
| EAR/MAR baseline | Rule-based project logic | Explainable eye/mouth openness signal |
| Eye CNN | Project-trained model | Classify cropped eye ROI as `eyes_open` or `eyes_closed` |
| Temporal smoothing | Project logic | Prevent false alert from normal blinking |
| Android alert/logging | Project logic | Sound/vibration and event-only CSV log |

## Dataset Strategy

Use classification datasets for the CNN because MediaPipe already gives the eye
location.

Recommended:

- `prasadvpatil/mrl-dataset`: main dataset for open/closed eye classification.
- `dheerajperumandla/drowsiness-dataset`: optional supplementary drowsiness or
  yawn data.
- Self-collected Android phone clips: small test/demo set to match deployment
  camera conditions.

Expected structure:

```text
dataset/
  train/
    eyes_open/
    eyes_closed/
  val/
    eyes_open/
    eyes_closed/
  test/
    eyes_open/
    eyes_closed/
```

## Training And Evaluation

```bash
python tools/summarize_dataset.py --data dataset --out outputs/dataset_summary
python tools/train_eye_classifier.py --data dataset --out app/src/main/assets/drowsiness_model.tflite --epochs 12
python tools/evaluate_eye_classifier.py --data dataset --tflite app/src/main/assets/drowsiness_model.tflite
```

Report:

- dataset counts by split/class;
- accuracy;
- precision/recall/F1 for `eyes_closed`;
- confusion matrix;
- FPS on Android phone if available.

## Android Deployment

The Android app flow stays compact:

- CameraX provides frames.
- MediaPipe runs in `LIVE_STREAM` mode.
- The app computes EAR/MAR and crops the eye band.
- If `drowsiness_model.tflite` exists, the app shows CNN label/confidence.
- Final alert remains based on temporal smoothing for demo stability.

## Strengths

- Best fit for the current project title: MediaPipe + CNN.
- Avoids bounding-box annotation work.
- Easy to explain: MediaPipe finds eyes, CNN classifies eye state.
- Smaller model and simpler Android integration than YOLO.
- Strong for 4-day deadline.
- Works well with privacy-by-design because only event logs are stored.

## Limitations

- Landmark quality can drop with sunglasses, extreme head pose, occlusion, or
  weak lighting.
- CNN is trained only on eye ROI, so the final behavior depends on MediaPipe
  crop quality.
- Needs Android phone testing for real FPS and camera stability.
- Does not directly detect other objects such as phone usage, cigarette, or seat
  belt.

## Defense Statement

> MediaPipe solves the localization problem: face, eyes, and mouth positions.
> The project-built CNN solves the classification problem: whether the cropped
> eye region is open or closed. This keeps the system lightweight, explainable,
> and feasible for Android deployment within the deadline.

## Useful Sources

- MediaPipe Face Landmarker Android:
  https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker/android
- Android CameraX ImageAnalysis:
  https://developer.android.com/media/camera/camerax/analyze
- TensorFlow Lite delegates:
  https://www.tensorflow.org/lite/performance/delegates
