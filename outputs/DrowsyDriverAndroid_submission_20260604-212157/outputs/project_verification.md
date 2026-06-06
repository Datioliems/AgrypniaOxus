# Project Verification

Required artifacts: PASS
Optional runtime evidence: 0/5

## Required Artifacts

| Artifact | Path | Status | Size |
|---|---|---|---:|
| Android Gradle settings | `settings.gradle` | Present | 338 |
| Android root build file | `build.gradle` | Present | 140 |
| Android app build file | `app/build.gradle` | Present | 982 |
| Android manifest | `app/src/main/AndroidManifest.xml` | Present | 876 |
| MainActivity | `app/src/main/java/com/ai2026/drowsydriver/MainActivity.kt` | Present | 9488 |
| DrowsinessAnalyzer | `app/src/main/java/com/ai2026/drowsydriver/DrowsinessAnalyzer.kt` | Present | 3775 |
| TFLite classifier | `app/src/main/java/com/ai2026/drowsydriver/TfliteDrowsinessClassifier.kt` | Present | 2159 |
| Event logger | `app/src/main/java/com/ai2026/drowsydriver/EventLogger.kt` | Present | 1654 |
| MediaPipe model asset | `app/src/main/assets/face_landmarker.task` | Present | 3758596 |
| README | `README.md` | Present | 7091 |
| 4-day plan | `docs/PLAN_4_NGAY.md` | Present | 2254 |
| Rubric mapping | `docs/RUBRIC_MAPPING.md` | Present | 4109 |
| Project status checklist | `docs/PROJECT_STATUS_AND_SUBMISSION_CHECKLIST.md` | Present | 3888 |
| Deployment/value doc | `docs/DEPLOYMENT_ARCHITECTURE_AND_VALUE.md` | Present | 4361 |
| Privacy/safety doc | `docs/PRIVACY_SAFETY_ETHICS.md` | Present | 2277 |
| AI model and deployment full guide | `docs/AI_MODEL_AND_DEPLOYMENT_FULL_GUIDE.md` | Present | 9330 |
| IS54A requirements coverage | `docs/IS54A_REQUIREMENTS_COVERAGE.md` | Present | 5762 |
| Streamlit and other deployment guide | `docs/STREAMLIT_AND_OTHER_DEPLOYMENT.md` | Present | 4341 |
| IS54A visual AI deployment roadmap | `docs/IS54A_VISUAL_AI_DEPLOYMENT_ROADMAP.html` | Present | 16534 |
| MediaPipe pipeline doc | `docs/MEDIAPIPE_LANDMARK_PIPELINE.md` | Present | 3992 |
| YOLO pipeline doc | `docs/YOLO_END_TO_END_PIPELINE.md` | Present | 4323 |
| MediaPipe vs YOLO comparison doc | `docs/MEDIAPIPE_VS_YOLO_COMPARISON.md` | Present | 4642 |
| Pipeline docs HTML | `docs/PIPELINE_DOCS_COMPARISON.html` | Present | 3490 |
| Pipeline docs screenshot | `output/playwright/pipeline-docs-comparison.png` | Present | 62539 |
| IS54A visual roadmap screenshot | `output/playwright/is54a-visual-roadmap.png` | Present | 266363 |
| Presentation script | `docs/PRESENTATION_8_MIN_SCRIPT.md` | Present | 5267 |
| Demo day checklist | `docs/DEMO_DAY_CHECKLIST.md` | Present | 1464 |
| Draft Word report | `outputs/Bao_cao_nhap_DrowsyDriverAndroid.docx` | Present | 12132 |
| Playwright latest screenshot | `output/playwright/project-plan-latest.png` | Present | 114780 |
| Dataset summary script | `tools/summarize_dataset.py` | Present | 3375 |
| Training script | `tools/train_eye_classifier.py` | Present | 2702 |
| Evaluation script | `tools/evaluate_eye_classifier.py` | Present | 6127 |
| Android env check script | `tools/check_android_env.ps1` | Present | 2287 |
| Android build helper | `tools/build_android.ps1` | Present | 660 |
| Submission package script | `tools/package_submission.ps1` | Present | 2162 |
| Streamlit fallback app | `streamlit_app.py` | Present | 3512 |
| Streamlit requirements | `requirements-streamlit.txt` | Present | 41 |

## Optional Runtime Evidence

These require Android Studio/device run or trained model outputs.

| Artifact | Path | Status | Size |
|---|---|---|---:|
| Trained CNN model | `app/src/main/assets/drowsiness_model.tflite` | Missing |  |
| Dataset summary JSON | `outputs/dataset_summary/dataset_summary.json` | Missing |  |
| Evaluation metrics JSON | `outputs/evaluation/metrics.json` | Missing |  |
| Evaluation confusion matrix | `outputs/evaluation/confusion_matrix.csv` | Missing |  |
| Android debug APK | `app/build/outputs/apk/debug/app-debug.apk` | Missing |  |
