"""
Verify required project artifacts before submission.

Usage:
  python tools/verify_project.py
"""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs"


REQUIRED_FILES = {
    "Android Gradle settings": "settings.gradle",
    "Android root build file": "build.gradle",
    "Android app build file": "app/build.gradle",
    "Android manifest": "app/src/main/AndroidManifest.xml",
    "MainActivity": "app/src/main/java/com/ai2026/drowsydriver/MainActivity.kt",
    "DrowsinessAnalyzer": "app/src/main/java/com/ai2026/drowsydriver/DrowsinessAnalyzer.kt",
    "TFLite classifier": "app/src/main/java/com/ai2026/drowsydriver/TfliteDrowsinessClassifier.kt",
    "Event logger": "app/src/main/java/com/ai2026/drowsydriver/EventLogger.kt",
    "MediaPipe model asset": "app/src/main/assets/face_landmarker.task",
    "README": "README.md",
    "4-day plan": "docs/PLAN_4_NGAY.md",
    "Rubric mapping": "docs/RUBRIC_MAPPING.md",
    "Project status checklist": "docs/PROJECT_STATUS_AND_SUBMISSION_CHECKLIST.md",
    "Deployment/value doc": "docs/DEPLOYMENT_ARCHITECTURE_AND_VALUE.md",
    "Privacy/safety doc": "docs/PRIVACY_SAFETY_ETHICS.md",
    "AI model and deployment full guide": "docs/AI_MODEL_AND_DEPLOYMENT_FULL_GUIDE.md",
    "IS54A requirements coverage": "docs/IS54A_REQUIREMENTS_COVERAGE.md",
    "Streamlit and other deployment guide": "docs/STREAMLIT_AND_OTHER_DEPLOYMENT.md",
    "IS54A visual AI deployment roadmap": "docs/IS54A_VISUAL_AI_DEPLOYMENT_ROADMAP.html",
    "MediaPipe pipeline doc": "docs/MEDIAPIPE_LANDMARK_PIPELINE.md",
    "YOLO pipeline doc": "docs/YOLO_END_TO_END_PIPELINE.md",
    "MediaPipe vs YOLO comparison doc": "docs/MEDIAPIPE_VS_YOLO_COMPARISON.md",
    "Pipeline docs HTML": "docs/PIPELINE_DOCS_COMPARISON.html",
    "Pipeline docs screenshot": "output/playwright/pipeline-docs-comparison.png",
    "IS54A visual roadmap screenshot": "output/playwright/is54a-visual-roadmap.png",
    "Presentation script": "docs/PRESENTATION_8_MIN_SCRIPT.md",
    "Demo day checklist": "docs/DEMO_DAY_CHECKLIST.md",
    "EDA/preprocess/feature/training results": "docs/EDA_PREPROCESS_FEATURE_TRAINING_RESULTS.md",
    "Draft Word report": "outputs/Bao_cao_nhap_DrowsyDriverAndroid.docx",
    "IS54A structured Word report": "outputs/Bao_cao_IS54A_DrowsyDriverAndroid_theo_khung.docx",
    "Playwright latest screenshot": "output/playwright/project-plan-latest.png",
    "Dataset summary script": "tools/summarize_dataset.py",
    "Dataset preparation script": "tools/prepare_eye_dataset.py",
    "Dataset EDA script": "tools/eda_eye_dataset.py",
    "Training script": "tools/train_eye_classifier.py",
    "Evaluation script": "tools/evaluate_eye_classifier.py",
    "Model result plotting script": "tools/plot_model_results.py",
    "Android env check script": "tools/check_android_env.ps1",
    "Android build helper": "tools/build_android.ps1",
    "IS54A report builder": "tools/build_is54a_report_docx.py",
    "Submission package script": "tools/package_submission.ps1",
    "Streamlit fallback app": "streamlit_app.py",
    "Streamlit requirements": "requirements-streamlit.txt",
}


OPTIONAL_RUNTIME_EVIDENCE = {
    "Trained CNN model": "app/src/main/assets/drowsiness_model.tflite",
    "Dataset summary JSON": "outputs/dataset_summary/dataset_summary.json",
    "EDA summary JSON": "outputs/eda/eda_summary.json",
    "Training history JSON": "outputs/training/training_history.json",
    "Evaluation metrics JSON": "outputs/evaluation/metrics.json",
    "Evaluation confusion matrix": "outputs/evaluation/confusion_matrix.csv",
    "Android debug APK": "app/build/outputs/apk/debug/app-debug.apk",
}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    required = check_files(REQUIRED_FILES)
    optional = check_files(OPTIONAL_RUNTIME_EVIDENCE)

    result = {
        "required": required,
        "optional_runtime_evidence": optional,
        "required_passed": all(item["exists"] for item in required),
        "optional_present_count": sum(1 for item in optional if item["exists"]),
        "optional_total": len(optional),
    }

    (OUT_DIR / "project_verification.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (OUT_DIR / "project_verification.md").write_text(render_markdown(result), encoding="utf-8")

    print("Required artifacts:", "PASS" if result["required_passed"] else "MISSING")
    print(
        "Optional runtime evidence:",
        f"{result['optional_present_count']}/{result['optional_total']}",
    )
    print(f"Saved {OUT_DIR / 'project_verification.md'}")

    if not result["required_passed"]:
        raise SystemExit(1)


def check_files(mapping: dict[str, str]) -> list[dict[str, object]]:
    rows = []
    for label, relative in mapping.items():
        path = ROOT / relative
        rows.append(
            {
                "label": label,
                "path": relative.replace("\\", "/"),
                "exists": path.exists(),
                "size": path.stat().st_size if path.exists() and path.is_file() else None,
            }
        )
    return rows


def render_markdown(result: dict[str, object]) -> str:
    lines = [
        "# Project Verification",
        "",
        f"Required artifacts: {'PASS' if result['required_passed'] else 'MISSING'}",
        f"Optional runtime evidence: {result['optional_present_count']}/{result['optional_total']}",
        "",
        "## Required Artifacts",
        "",
        "| Artifact | Path | Status | Size |",
        "|---|---|---|---:|",
    ]
    for item in result["required"]:
        lines.append(format_row(item))

    lines += [
        "",
        "## Optional Runtime Evidence",
        "",
        "These require Android Studio/device run or trained model outputs.",
        "",
        "| Artifact | Path | Status | Size |",
        "|---|---|---|---:|",
    ]
    for item in result["optional_runtime_evidence"]:
        lines.append(format_row(item))
    lines.append("")
    return "\n".join(lines)


def format_row(item: dict[str, object]) -> str:
    status = "Present" if item["exists"] else "Missing"
    size = "" if item["size"] is None else str(item["size"])
    return f"| {item['label']} | `{item['path']}` | {status} | {size} |"


if __name__ == "__main__":
    main()
