# Sources For Report

Use these sources in Chapter 1.3, Chapter 3, Chapter 4, and references.

## Official Technical Documentation

1. MediaPipe Face Landmarker for Android

   URL: https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker/android

   Use for:

   - Face Landmarker supports image, video, and live stream modes.
   - `LIVE_STREAM` receives results asynchronously.
   - Video/live stream modes use tracking to reduce latency.
   - Android example runs on a physical Android camera stream.

2. Android CameraX ImageAnalysis

   URL: https://developer.android.com/media/camera/camerax/analyze?hl=en

   Use for:

   - CameraX `ImageAnalysis` provides frames for realtime analysis.
   - `STRATEGY_KEEP_ONLY_LATEST` helps avoid frame backlog when analysis is
     slower than camera capture.

3. TensorFlow Lite Delegates

   URL: https://www.tensorflow.org/lite/performance/delegates

   Use for:

   - TFLite delegates can accelerate inference on mobile/edge hardware.
   - Android devices may use GPU/NNAPI/DSP/NPU depending on availability.

## Datasets

4. NTHU Driver Drowsiness Detection Dataset

   URL: https://cv.cs.nthu.edu.tw/php/callforpaper/datasets/DDD/

   Use for:

   - Driver-specific drowsiness video dataset.
   - Includes subjects with/without glasses and sunglasses.
   - Includes day/night scenarios, yawning, slow blinking, falling asleep.
   - Dataset access requires license agreement/email request.

5. DROZY: ULg Multimodality Drowsiness Database

   URL: https://www.drozy.uliege.be/

   Use for:

   - Multimodal drowsiness monitoring database.
   - Contains drowsiness-related signals/images/videos for algorithm research.

6. YawDD: Yawning Detection Dataset

   URL: https://ieee-dataport.org/open-access/yawdd-yawning-detection-dataset

   Use for:

   - In-car yawning detection.
   - Useful for training/evaluating yawning vs non-yawning.

## Related Research

7. DDD TinyML: A TinyML-Based Driver Drowsiness Detection Model Using Deep Learning

   URL: https://www.mdpi.com/1424-8220/23/12/5696

   Use for:

   - Shows motivation for lightweight/on-device drowsiness detection.
   - Discusses deep learning, TensorFlow Lite, and resource-constrained
     deployment.

8. A Real-Time Embedded System for Driver Drowsiness Detection Based on Visual
   Analysis of the Eyes and Mouth Using CNN and MAR

   URL: https://pmc.ncbi.nlm.nih.gov/articles/PMC11479241/

   Use for:

   - Strongly related hybrid approach.
   - Uses MediaPipe Face Mesh landmarks, CNN eye state identification, and MAR
     for mouth/yawning cues.

9. Real-Time Machine Learning-Based Driver Drowsiness Detection Using Visual
   Features

   URL: https://pmc.ncbi.nlm.nih.gov/articles/PMC10219078/

   Use for:

   - Visual-feature drowsiness detection.
   - Uses MediaPipe/Dlib landmarks, EAR, MAR, and head-position features.

## How To Position Your Contribution

Your project should not claim to beat all previous research. Claim a realistic
student-project contribution:

- Android phone as a low-cost on-device deployment target.
- Hybrid interpretable landmark features and CNN/TFLite classifier.
- Realtime-oriented design with FPS/latency consideration.
- Safety-oriented evaluation using recall/F1 for drowsy state.
- Practical demo and documented limitations.
