# FAQ: Dataset, Streamlit, Android Run, And Required Knowledge

## 1. Dataset For This Project

Recommended dataset strategy:

- **NTHU Driver Drowsiness Detection Dataset**: main driver drowsiness reference
  dataset. Good for report/research positioning because it is driver-specific
  and includes day/night, glasses, sunglasses, yawning, slow blinking, and
  drowsy behavior. Access usually requires request/license.
- **YawDD**: useful for yawning detection. Good if the project includes
  `yawning` as a separate signal.
- **DROZY**: research-grade drowsiness dataset with more modalities. Good for
  related-work discussion, but more complex for a 4-day implementation.
- **EyeState/YawnDrowsiness Kaggle variants**: fastest path for training a
  simple CNN for `eyes_open` vs `eyes_closed`.
- **Self-collected Android phone videos**: important for demo and deployment
  validation because the final app runs on Android phone camera.

Practical 4-day recommendation:

1. Use an easy eye-state dataset for CNN training.
2. Use NTHU/YawDD/DROZY in related work and dataset discussion.
3. Record small Android-phone clips for demo/domain validation.

## 2. Can This Be Deployed On Web With Streamlit?

Yes. A Streamlit version is possible and useful as a fallback or presentation
demo.

Recommended Streamlit architecture:

```text
Webcam / uploaded video
-> OpenCV frame loop
-> MediaPipe Face Landmarker or Face Mesh
-> EAR/MAR baseline
-> optional Keras/TFLite CNN
-> Streamlit UI: status, FPS, confidence, alert
```

What needs to change:

- Android `CameraX` becomes OpenCV/webcam input.
- Android UI overlay becomes Streamlit components.
- Android sound/vibration becomes Streamlit visual alert or local audio.
- Android internal event log becomes CSV saved in the Streamlit project folder.
- TFLite model can still be used, but Keras `.h5/.keras` may be easier on web.

What does not change:

- MediaPipe landmark logic.
- EAR/MAR logic.
- CNN concept.
- Evaluation metrics.
- Dataset/training pipeline.

For the final report, Android native should remain the main deployment target if
you can run it. Streamlit can be described as a fallback demo or web prototype.

## 3. How To Run The Current Android Project

From project root:

```powershell
powershell -ExecutionPolicy Bypass -File tools/check_android_env.ps1
```

Then:

1. Install/open Android Studio.
2. Open `D:\2026.AI\DrowsyDriverAndroid`.
3. Let Android Studio sync Gradle and install missing SDK components.
4. Connect Android phone.
5. Enable Developer Options and USB debugging.
6. Press Run in Android Studio.

The required MediaPipe model is already included:

```text
app/src/main/assets/face_landmarker.task
```

The CNN model is optional until trained:

```text
app/src/main/assets/drowsiness_model.tflite
```

If this file is missing, the app can still run MediaPipe + EAR/MAR baseline and
will show `CNN model pending`.

## 4. How To Install The Native App On A Phone

Option A: Android Studio direct run

1. Connect phone with USB debugging.
2. Select the phone in Android Studio.
3. Press Run.
4. Android Studio installs the debug app automatically.

Option B: Build APK and install

After Java/SDK/Gradle are ready:

```powershell
powershell -ExecutionPolicy Bypass -File tools/build_android.ps1
```

Expected APK:

```text
app/build/outputs/apk/debug/app-debug.apk
```

Install by:

- dragging/copying APK to phone and opening it, or
- using `adb install app-debug.apk` if ADB is available.

For a course demo, Option A is usually fastest.

## 5. Required Knowledge

Minimum knowledge:

- Python basics for dataset preparation, training, and evaluation scripts.
- Basic machine learning metrics: accuracy, precision, recall, F1-score,
  confusion matrix.
- CNN basics: image input, convolution, pooling, softmax classification.
- MediaPipe landmarks: face landmarks, eye/mouth points, EAR/MAR.
- Android Studio basics: open project, Gradle sync, run on device.
- Kotlin basics: classes, functions, Android activity lifecycle.
- CameraX basics: camera preview and image analysis.
- TensorFlow Lite basics: converting/loading a model for mobile inference.

Useful but not mandatory:

- OpenCV frame processing.
- Streamlit for fallback web demo.
- Git/GitHub for commit history.
- Privacy-by-design for camera-based AI.

## 6. Best 4-Day Priority

1. Run Android baseline on phone.
2. Record demo video.
3. Prepare simple eye-state dataset.
4. Train/evaluate CNN and export TFLite.
5. Update report numbers and screenshots.
6. Practice 8-minute presentation.
