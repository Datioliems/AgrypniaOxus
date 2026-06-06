# Demo Day Checklist

## Before Entering The Room

- Charge Android phone.
- Enable USB debugging.
- Keep USB cable ready.
- Install app on phone if possible.
- Record final demo video backup.
- Keep report printed or open.
- Keep source code open.
- Keep dataset/metrics table open.

## Windows To Open

1. Android Studio project.
2. `MainActivity.kt`.
3. `DrowsinessAnalyzer.kt`.
4. `TfliteDrowsinessClassifier.kt`.
5. `EventLogger.kt`.
6. Word report.
7. Demo video backup.
8. Metrics/confusion matrix.

## Live Demo Steps

1. Show app camera preview.
2. Point to overlay: state, EAR, MAR, FPS, CNN.
3. Show awake state.
4. Close eyes for about 2 seconds.
5. Wait for `Drowsy alert`.
6. Open mouth/yawn to show MAR/yawning.
7. Mention that final alert is smoothed to avoid false alarm from normal blink.

## If The Phone Camera Is Unstable

Use the recorded video backup immediately. Do not spend presentation time fixing
USB/camera issues.

## Questions To Be Ready For

- Why MediaPipe?
- Where is CNN used?
- Why Android phone instead of laptop webcam?
- Why recall/F1 instead of only accuracy?
- What are limitations?
- How would this be deployed in a real vehicle?
- What would you improve with more time?

## One-Sentence Answer For Deployment

> In real deployment, the Android phone or Android box is mounted in the cabin,
> runs CameraX, MediaPipe and TFLite on-device, then triggers sound/vibration
> alerts without depending on network connectivity.
