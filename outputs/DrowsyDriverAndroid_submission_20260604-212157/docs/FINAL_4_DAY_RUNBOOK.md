# Final 4-Day Runbook

## Priority Order

1. Android app runs on phone.
2. Demo video exists.
3. CNN model and metrics exist.
4. Report has real numbers and screenshots.
5. Submission zip is packaged.

## Do Not Over-Scope

Avoid adding:

- multi-face support;
- cloud backend;
- complex UI navigation;
- LSTM/Transformer;
- fleet dashboard;
- night-driving hardware experiments.

These are good future work, not 4-day requirements.

## Minimum Viable Defense

If time gets tight, defend this:

- Android native CameraX app.
- MediaPipe Face Landmarker.
- EAR/MAR + smoothing + alert.
- CNN training/evaluation as separate but connected TFLite path.
- Research gaps: realtime, deployment, safety metrics, explainability.

## Strong Defense

If everything runs:

- Android app shows camera and status.
- MediaPipe landmark baseline triggers alert.
- CNN overlay shows `eyes_closed` or `eyes_open`.
- Metrics table has recall/F1.
- Demo video is smooth.

## What To Say About Limitations

Be honest:

- This is a prototype, not a certified vehicle safety system.
- It can fail with sunglasses, poor lighting, partial occlusion, and extreme head pose.
- Dataset coverage is limited.
- Real deployment needs controlled road testing and privacy/safety review.

Honesty here helps the project look more mature, not weaker.
