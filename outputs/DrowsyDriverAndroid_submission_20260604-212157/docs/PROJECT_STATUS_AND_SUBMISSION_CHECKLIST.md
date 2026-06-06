# Project Status And Submission Checklist

## Current Evidence

| Requirement | Current evidence | Status |
|---|---|---|
| 4-day execution plan | `docs/PLAN_4_NGAY.md`, `docs/ANDROID_4_DAY_EXECUTION_CHECKLIST.md` | Ready |
| Playwright plan artifact | `docs/PROJECT_PLAN.html`, `output/playwright/project-plan.png` | Verified by Playwright snapshot/screenshot |
| Android native project | `settings.gradle`, `build.gradle`, `app/build.gradle`, Android source files | Created |
| MediaPipe model asset | `app/src/main/assets/face_landmarker.task` | Included |
| CameraX + MediaPipe pipeline | `MainActivity.kt` | Implemented, needs Android device build/run |
| EAR/MAR + smoothing alert | `DrowsinessAnalyzer.kt`, `AlertController.kt` | Implemented, needs device test |
| Privacy-preserving event logging | `EventLogger.kt`, `docs/PRIVACY_SAFETY_ETHICS.md` | Implemented, needs device test |
| CNN/TFLite path | `TfliteDrowsinessClassifier.kt`, eye ROI crop in `MainActivity.kt` | Implemented, needs trained `drowsiness_model.tflite` |
| Dataset/training path | `tools/DATASET_PREP.md`, `tools/train_eye_classifier.py` | Ready |
| Dataset statistics path | `docs/DATA_COLLECTION_ANDROID_PHONE.md`, `tools/summarize_dataset.py`, `outputs/templates/dataset_summary_template.csv` | Ready |
| Evaluation metrics path | `tools/evaluate_eye_classifier.py`, `docs/THI_NGHIEM_VA_BANG_KET_QUA.md` | Ready |
| Research gaps and sources | `research/DATASETS_AND_GAPS.md`, `research/SOURCES_FOR_REPORT.md` | Ready |
| Draft report | `docs/BAO_CAO_NHAP.md`, `outputs/Bao_cao_nhap_DrowsyDriverAndroid.docx` | Created |
| Build troubleshooting | `tools/check_android_env.ps1`, `docs/ANDROID_BUILD_TROUBLESHOOTING.md` | Ready |
| Rubric mapping | `docs/RUBRIC_MAPPING.md` | Ready |
| Deployment/value/privacy framing | `docs/DEPLOYMENT_ARCHITECTURE_AND_VALUE.md`, `docs/PRIVACY_SAFETY_ETHICS.md` | Ready |
| Artifact verification | `tools/verify_project.py`, `outputs/project_verification.md` | Required artifacts pass |

## Known Missing Evidence

These are not finished until run on a machine with Android Studio/JDK/SDK:

- Gradle sync/build result.
- APK output.
- Physical Android phone demo.
- FPS measurement on phone.
- Screenshot/video of the running Android app.
- Trained `drowsiness_model.tflite`.
- Metrics from `outputs/evaluation/metrics.json`.

## Day-By-Day Submission Targets

### Day 1

- Run `tools/check_android_env.ps1`.
- Open project in Android Studio.
- Run baseline app on phone.
- Record video of `Awake`, `Eyes closed`, `Yawning`, `Drowsy alert`.
- Fill initial FPS and threshold notes.

### Day 2

- Prepare eye dataset.
- Run `tools/summarize_dataset.py` to create dataset counts and manifest.
- Train CNN.
- Export `app/src/main/assets/drowsiness_model.tflite`.
- Run evaluation script.
- Fill accuracy, precision, recall, F1, confusion matrix.

### Day 3

- Re-run Android app with CNN model present.
- Confirm overlay shows `CNN eyes_closed` or `CNN eyes_open`.
- Tune thresholds and smoothing.
- Record final demo video.

### Day 4

- Update Word report values marked `TBD`.
- Add screenshots/demo images.
- Prepare Q&A.
- Package final submission zip.

## Final Submission Contents

Recommended:

- Android project folder.
- `outputs/Bao_cao_nhap_DrowsyDriverAndroid.docx`, edited with real values.
- Demo video.
- Dataset statistics table.
- Confusion matrix and metrics.
- Source references.
- Short README with run instructions.

## Presentation Message

Use this framing:

> Nhóm xây dựng prototype Android on-device. MediaPipe Face Landmarker trích
> xuất landmark realtime, EAR/MAR làm baseline có thể giải thích, CNN/TFLite
> phân loại vùng mắt, và temporal smoothing quyết định cảnh báo. Hướng này giải
> quyết khoảng trống giữa demo offline/laptop và triển khai thực tế chi phí thấp
> bằng Android phone.
