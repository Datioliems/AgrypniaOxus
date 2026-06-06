# Android Native Deploy Steps

## 1. Prepare Assets

- `face_landmarker.task` is already included in `app/src/main/assets`.
- Train or obtain a CNN model and export it as `drowsiness_model.tflite`.
- Copy it to `app/src/main/assets/drowsiness_model.tflite`.

## 2. Run The Prototype

- Open `D:\2026.AI\DrowsyDriverAndroid` in Android Studio.
- Connect Android phone with USB debugging enabled.
- Select the device and press Run.
- The app opens camera preview, shows status, EAR, MAR, FPS, and alert state.
- If `drowsiness_model.tflite` is present, the app crops the eye band from
  MediaPipe landmarks and shows CNN label/confidence in the overlay.

## 3. Tune Thresholds

In `DrowsinessAnalyzer.kt` tune:

- `eyeClosedThreshold`
- `yawnThreshold`
- `drowsyDurationMs`

Use values that work on your phone and dataset. Report these values in Chapter 3.

## 4. Integrate CNN/TFLite

Recommended CNN input:

- eye crop: `64x64x3` or `96x96x3`;
- classes: `eyes_open`, `eyes_closed`;
- optional mouth crop classes: `not_yawning`, `yawning`.

Recommended realtime logic:

```text
MediaPipe Face Landmarker
-> crop eye band from landmarks
-> CNN/TFLite predicts eyes_closed or eyes_open
-> if CNN says eyes_closed with high confidence for > 1.7 seconds:
    DROWSY_ALERT
else if MAR says yawning for > 1.0 second:
    YAWNING
else:
    AWAKE
```

## 5. Demo Checklist

- Start app before presentation.
- Keep phone fixed at eye level.
- Show awake state first.
- Close eyes for 2 seconds to trigger alert.
- Yawn/open mouth to show MAR response.
- Show FPS and confidence on screen.
- Keep a recorded video backup.
