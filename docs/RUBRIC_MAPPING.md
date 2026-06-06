# Rubric Mapping

This file maps the course grading guide to the current project artifacts.

## 1. Report Book - 4.5 Points

| Rubric item | Max | Current evidence | Status | Next action |
|---|---:|---|---|---|
| Chapter 1: overview, AI problem, input/output, scope, related work | 0.5 | `docs/BAO_CAO_NHAP.md`, `docs/DE_CUONG_BAO_CAO.md`, `research/SOURCES_FOR_REPORT.md`, `docs/DEPLOYMENT_ARCHITECTURE_AND_VALUE.md` | Draft ready | Add screenshots and final wording after demo |
| Chapter 2: data source, storage, labeling, observation, preprocessing, citations | 1.0 | `docs/DATA_COLLECTION_ANDROID_PHONE.md`, `tools/summarize_dataset.py`, `outputs/templates/dataset_summary_template.csv` | Path ready | Collect real data and fill counts |
| Chapter 3: features, new methods, split, pipeline, model/pretrained/framework | 1.5 | `MainActivity.kt`, `DrowsinessAnalyzer.kt`, `TfliteDrowsinessClassifier.kt`, `docs/BAO_CAO_NHAP.md` | Strong draft | Add actual training config and model result |
| Chapter 4: metrics, evaluation results, demo technology, screenshots/functions | 1.0 | `tools/evaluate_eye_classifier.py`, `docs/THI_NGHIEM_VA_BANG_KET_QUA.md`, `outputs/templates/experiment_results_template.csv` | Path ready | Train/evaluate and add demo screenshots |
| Conclusion and formatting | 0.5 | `outputs/Bao_cao_nhap_DrowsyDriverAndroid.docx` | Draft ready | Replace `TBD`, update table of contents, visual polish |
| Privacy/safety maturity | Bonus support | `docs/PRIVACY_SAFETY_ETHICS.md`, Word report section "Khía cạnh riêng tư và an toàn" | Ready | Mention briefly in Q&A/limitations |

## 2. Product - 1.5 Points

| Rubric item | Max | Current evidence | Status | Next action |
|---|---:|---|---|---|
| Product demo matches problem, UI/input/output flow, code matches report | 1.0 | Android source in `app/`, MediaPipe asset, overlay, alert logic, local event logger | Implemented prototype | Build/run on Android phone and record video |
| Project organization, commit/code quality/comments | 0.5 | Structured folders: `app`, `docs`, `research`, `tools`, `outputs`; README and scripts | Organized | Create Git repo/commit history if needed |

## 3. Presentation - 1.0 Point

| Rubric item | Max | Current evidence | Status | Next action |
|---|---:|---|---|---|
| Finish within 8 minutes | 0.2 | `docs/PRESENTATION_8_MIN_SCRIPT.md` | Ready | Practice once with timer |
| Present highlights and full demo | 0.8 | `docs/DEMO_DAY_CHECKLIST.md`, `output/playwright/project-plan-latest.png` | Ready | Record final Android demo video |

## 4. Individual - 3.0 Points

| Rubric item | Max | Current evidence | Status | Next action |
|---|---:|---|---|---|
| Present and answer questions about assigned work | 2.0 | `docs/Q_AND_A_THUYET_TRINH.md`, source code docs | Ready | Each member reads their code/report section |
| Answer project-wide questions | 0.5 | `docs/PRESENTATION_8_MIN_SCRIPT.md`, `docs/PROJECT_STATUS_AND_SUBMISSION_CHECKLIST.md` | Ready | Practice Q&A |
| Contribution ratio | 0.5 | Word report assignment table | Placeholder | Fill real names, IDs, percentages, signatures |

## Highest-Risk Missing Items

1. **Android build/run evidence**

   Need:

   - Android Studio sync.
   - App running on phone.
   - Screenshot/video.

2. **CNN model evidence**

   Need:

   - `app/src/main/assets/drowsiness_model.tflite`.
   - Training/evaluation output.
   - Confusion matrix and recall/F1.

3. **Report real values**

   Need:

   - Replace `TBD`.
   - Add dataset counts.
   - Add screenshots.
   - Add metric values.

## Minimum Passing Strategy

If time is tight:

- Run Android baseline MediaPipe + EAR/MAR + alert.
- Record a clean demo video.
- Train/evaluate a small eye classifier, even if CNN is only partially tuned.
- Fill Chapter 2 and Chapter 4 with real counts/metrics.
- Be honest in limitations.

## Strong Scoring Strategy

If time allows:

- Android app shows `CNN eyes_open/eyes_closed` in overlay.
- Final report includes confusion matrix and FPS.
- Presentation explains research gaps clearly.
- Demo video is ready before entering the presentation room.
