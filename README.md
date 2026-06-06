# Drowsy Driver Android Prototype

Android native prototype for a driver drowsiness warning system using CameraX,
MediaPipe Face Landmarker, landmark-based temporal rules, and an optional
TensorFlow Lite CNN classifier.

## Project Goal

Build a 4-day feasible final project that satisfies the AI course rubric:

- clear AI problem definition with input/output and practical scope;
- dataset collection, preprocessing, and labeling plan;
- feature extraction using MediaPipe facial landmarks;
- model component using CNN/TensorFlow Lite;
- measurable evaluation and realtime demo;
- practical deployment path using Android phones as edge devices.

## Current Prototype

The current app can be opened in Android Studio as a native Android project.
It contains:

- CameraX realtime camera preview.
- MediaPipe Face Landmarker integration in `LIVE_STREAM` mode.
- EAR/MAR landmark-based drowsiness baseline.
- Temporal smoothing: drowsy alert after closed eyes persist.
- Sound/vibration alert.
- Minimal local event logging for `DROWSY` and `YAWNING` states.
- Eye ROI crop from MediaPipe landmarks and TFLite classifier inference for
  `drowsiness_model.tflite`.

Required assets before full run:

- `app/src/main/assets/face_landmarker.task` is included.
- optional: `app/src/main/assets/drowsiness_model.tflite`

## Recommended 4-Day Scope

Use this Android app as the final demo target. If the Android build setup causes
delay, keep a short Python/OpenCV demo as a backup, but present the Android
pipeline as the main deployment architecture.

Day 1:

- Prepare datasets and labels.
- Run Android camera preview and MediaPipe landmark baseline.
- Record short self-collected videos using the target Android phone.

Day 2:

- Train a small CNN for eyes open/closed or yawning/not-yawning.
- Evaluate with accuracy, precision, recall, F1-score, and confusion matrix.
- Export to TensorFlow Lite.

Day 3:

- Put the `.tflite` model into Android assets.
- Connect CNN output with the temporal warning logic.
- Tune thresholds and smoothing duration.
- Record final demo video.

Day 4:

- Finish report using the provided template.
- Prepare an 8-minute explanation and demo script.
- Add limitations, deployment plan, and future work.

## Android Studio Setup

1. Open this folder in Android Studio:
   `D:\2026.AI\DrowsyDriverAndroid`
2. Let Android Studio install/sync Gradle, Android SDK, and Kotlin plugin if prompted.
3. Run on a physical Android phone. Use the front camera for quick testing.
4. Train/export `drowsiness_model.tflite` to enable the CNN line in the overlay.

This workspace does not currently include a Gradle wrapper. Android Studio can
still open and sync the project using its managed Gradle/JDK setup. For terminal
builds, add a Gradle wrapper after Android Studio/JDK are installed.

Environment check:

```powershell
powershell -ExecutionPolicy Bypass -File tools/check_android_env.ps1
```

Terminal build after Java/SDK/Gradle are ready:

```powershell
powershell -ExecutionPolicy Bypass -File tools/build_android.ps1
```

Package current submission artifacts:

```powershell
powershell -ExecutionPolicy Bypass -File tools/package_submission.ps1
```

Verify required artifacts:

```bash
python tools/verify_project.py
```

## Report Positioning

Suggested Vietnamese project title:

> He thong phat hien va canh bao dau hieu buon ngu cua tai xe tren thiet bi
> Android su dung MediaPipe va mang no-ron tich chap

Use Android phones as both camera and edge computing device. This makes the
system more realistic than a laptop-only webcam prototype and avoids network
latency.

## Student Project Documents

- `docs/PLAN_4_NGAY.md`: 4-day execution plan.
- `docs/FINAL_4_DAY_RUNBOOK.md`: final runbook and scope guardrails.
- `docs/PROJECT_STATUS_AND_SUBMISSION_CHECKLIST.md`: evidence and submission checklist.
- `docs/RUBRIC_MAPPING.md`: grading-rubric mapping to current artifacts.
- `docs/DEPLOYMENT_ARCHITECTURE_AND_VALUE.md`: deployment architecture and practical value.
- `docs/PRIVACY_SAFETY_ETHICS.md`: privacy, safety, and ethics notes.
- `docs/PRESENTATION_8_MIN_SCRIPT.md`: 8-minute presentation script.
- `docs/DEMO_DAY_CHECKLIST.md`: live demo preparation checklist.
- `docs/NGAY_1_CHAY_DEMO_ANDROID.md`: first-day Android baseline demo checklist.
- `docs/NGAY_2_TRAIN_CNN.md`: second-day CNN training and TFLite checklist.
- `docs/ANDROID_4_DAY_EXECUTION_CHECKLIST.md`: Android-native checklist.
- `docs/ANDROID_DEPLOY_STEPS.md`: deployment steps.
- `docs/ANDROID_BUILD_TROUBLESHOOTING.md`: build/setup troubleshooting.
- `docs/FAQ_RUN_DEPLOY_KNOWLEDGE.md`: dataset, Streamlit, run/install, and knowledge FAQ.
- `docs/AI_MODEL_AND_DEPLOYMENT_FULL_GUIDE.md`: end-to-end guide for training, Android, Streamlit, and other deployment paths.
- `docs/IS54A_REQUIREMENTS_COVERAGE.md`: direct mapping from the IS54A grading file to project artifacts.
- `docs/STREAMLIT_AND_OTHER_DEPLOYMENT.md`: Streamlit fallback and alternative deployment systems.
- `docs/IS54A_VISUAL_AI_DEPLOYMENT_ROADMAP.html`: visual HTML roadmap for the IS54A rubric, AI model, Android, and Streamlit.
- `docs/MEDIAPIPE_LANDMARK_PIPELINE.md`: MediaPipe + EAR/MAR + CNN/TFLite method.
- `docs/YOLO_END_TO_END_PIPELINE.md`: YOLO object-detection alternative.
- `docs/MEDIAPIPE_VS_YOLO_COMPARISON.md`: decision table comparing both methods.
- `docs/DATA_COLLECTION_ANDROID_PHONE.md`: Android-phone data collection guide.
- `docs/DE_CUONG_BAO_CAO.md`: report outline aligned with the course template.
- `docs/BAO_CAO_NHAP.md`: Markdown draft report content.
- `outputs/Bao_cao_nhap_DrowsyDriverAndroid.docx`: editable Word draft report.
  Structural check: 70 paragraphs, 4 tables, 19 headings. Visual render QA was
  not available on this machine because LibreOffice/soffice is not installed.
- `outputs/Bao_cao_IS54A_DrowsyDriverAndroid_theo_khung.docx`: structured Word
  report following the official IS54A frame with cover page, contribution table,
  table-of-contents placeholder, abbreviations, chapters 1-4, conclusion, and
  references.
- `docs/THI_NGHIEM_VA_BANG_KET_QUA.md`: experiment design and result tables.
- `docs/EDA_PREPROCESS_FEATURE_TRAINING_RESULTS.md`: actual EDA, preprocessing,
  feature engineering, training, and evaluation results.
- `docs/Q_AND_A_THUYET_TRINH.md`: likely presentation questions and answers.
- `research/DATASETS_AND_GAPS.md`: datasets, research gaps, and practical value.
- `research/SOURCES_FOR_REPORT.md`: sources and how to cite them.
- `tools/DATASET_PREP.md`: dataset preparation format.
- `tools/prepare_eye_dataset.py`: maps `rawdata/data` awake/sleepy folders to the training labels `eyes_open`/`eyes_closed`.
- `tools/eda_eye_dataset.py`: creates EDA tables and plots for the prepared eye-state dataset.
- `tools/summarize_dataset.py`: creates dataset summary and manifest CSV.
- `tools/train_eye_classifier.py`: small CNN training script template.
- `tools/evaluate_eye_classifier.py`: metrics and confusion-matrix script.
- `tools/plot_model_results.py`: creates training-curve and confusion-matrix images.
- `tools/build_is54a_report_docx.py`: builds the IS54A-structured Word report.
- `tools/check_android_env.ps1`: Windows environment check script.
- `tools/build_android.ps1`: Windows Android build helper.
- `tools/package_submission.ps1`: creates a zip with project/report/docs artifacts.
- `tools/verify_project.py`: verifies required artifacts and writes project verification reports.
- `streamlit_app.py`: Streamlit fallback demo for uploaded images or webcam snapshots.
- `requirements-streamlit.txt`: Streamlit fallback dependencies.
- `outputs/project_verification.md`: latest required/optional artifact verification report.
- `outputs/templates/dataset_summary_template.csv`: table template for Chapter 2.
- `outputs/templates/experiment_results_template.csv`: table template for Chapter 4.
