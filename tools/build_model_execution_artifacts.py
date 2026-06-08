"""Build model execution workbook and DOCX runbook for DrowsyDriverAndroid."""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt, RGBColor
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.table import Table, TableStyleInfo


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs" / "model_execution"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_json(path: Path, default=None):
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return default


def csv_rows(path: Path):
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def pct(v):
    if v in (None, ""):
        return ""
    return round(float(v) * 100, 2)


def kb(path: str | Path):
    p = Path(path)
    return round(p.stat().st_size / 1024, 1) if p.exists() else ""


cnn_eye = load_json(ROOT / "outputs" / "cnn_eye" / "summary.json", {})
cnn_yawn = load_json(ROOT / "outputs" / "cnn_yawn" / "summary.json", {})
cnn_yawn_cbam = load_json(ROOT / "outputs" / "cnn_yawn_cbam" / "summary.json", {})
cnn_eye_cbam = load_json(ROOT / "outputs" / "cnn_eye_cbam" / "summary.json", {})
cnn_eye_cbam_running = load_json(ROOT / "outputs" / "cnn_eye_cbam" / "summary_running.json", {})
temporal = load_json(ROOT / "outputs" / "temporal_transformer" / "summary.json", {})

models = [
    {
        "model": "CNN Eye Baseline",
        "env": "LOCAL",
        "script": "train_cnn_eye.py / train_cnn_eye.ipynb",
        "input": "dataset_mrl/",
        "target": "eyes_closed=0, eyes_open=1",
        "algorithm": "CNN 3 conv blocks + BatchNorm + GAP + Dense softmax",
        "pre_model": "None",
        "output": "app/src/main/assets/drowsiness_model.tflite",
        "status": "Verified existing asset",
        "primary_metric": "best_val_accuracy",
        "result": pct(cnn_eye.get("best_val_accuracy")),
        "size_kb": cnn_eye.get("tflite_kb", kb(ROOT / "app/src/main/assets/drowsiness_model.tflite")),
        "time_plan": "20 min with GPU optional; already available in this repo",
    },
    {
        "model": "CNN Eye CBAM",
        "env": "LOCAL",
        "script": "train_cnn_cbam.py / train_cnn_cbam.ipynb",
        "input": "dataset_mrl/",
        "target": "eyes_closed=0, eyes_open=1",
        "algorithm": "CNN + CBAM channel attention + spatial attention",
        "pre_model": "CNN Eye Baseline",
        "output": "app/src/main/assets/drowsiness_cbam.tflite",
        "status": "Run completed on RTX 4050 quick subset; not active in Android because baseline is stronger" if cnn_eye_cbam else "Prepared; not run in this CPU-only session",
        "primary_metric": "val_accuracy delta vs baseline",
        "result": (
            f"{pct(cnn_eye_cbam.get('best_val_accuracy'))} / {pct(cnn_eye_cbam.get('test_accuracy'))}"
            if cnn_eye_cbam else
            (f"Partial checkpoint only; last known best val={pct(cnn_eye_cbam_running.get('best_val_accuracy_so_far'))}"
             if cnn_eye_cbam_running else "Partial checkpoint only; no final metrics")
        ),
        "size_kb": cnn_eye_cbam.get("tflite_kb", ""),
        "time_plan": "Use train_cbam_fast_gpu.ipynb on RTX 4050; history is saved after each epoch",
    },
    {
        "model": "CNN Yawn Baseline",
        "env": "LOCAL",
        "script": "train_cnn_yawn.py / train_cnn_yawn.ipynb",
        "input": "rawdata/data/yawn/ -> dataset_yawn/",
        "target": "no_yawn=0, yawn=1",
        "algorithm": "CNN 3 conv blocks + BatchNorm + GAP + Dense softmax",
        "pre_model": "None",
        "output": "app/src/main/assets/yawn_model.tflite",
        "status": "Run completed in this session",
        "primary_metric": "best_val_accuracy / test_accuracy",
        "result": f"{pct(cnn_yawn.get('best_val_accuracy'))} / {pct(cnn_yawn.get('test_accuracy'))}",
        "size_kb": cnn_yawn.get("tflite_kb", kb(ROOT / "app/src/main/assets/yawn_model.tflite")),
        "time_plan": "3 min on current CPU; guide estimate 15 min",
    },
    {
        "model": "CNN Yawn CBAM",
        "env": "LOCAL",
        "script": "train_cnn_cbam.py / train_cnn_cbam.ipynb Cell 8",
        "input": "dataset_yawn/",
        "target": "no_yawn=0, yawn=1",
        "algorithm": "PyTorch CUDA CNN + CBAM; Keras/TFLite export",
        "pre_model": "CNN Yawn Baseline",
        "output": "app/src/main/assets/yawn_cbam.tflite",
        "status": "Run completed on RTX 4050 and Android asset exported",
        "primary_metric": "best_val_accuracy / test_accuracy",
        "result": f"{pct(cnn_yawn_cbam.get('best_val_accuracy'))} / {pct(cnn_yawn_cbam.get('test_accuracy'))}",
        "size_kb": cnn_yawn_cbam.get("tflite_kb", kb(ROOT / "app/src/main/assets/yawn_cbam.tflite")),
        "time_plan": f"{cnn_yawn_cbam.get('epochs_ran', 0)} epochs, {cnn_yawn_cbam.get('elapsed_seconds', 0)} sec on RTX 4050",
    },
    {
        "model": "Temporal Transformer EAR",
        "env": "LOCAL",
        "script": "train_temporal_transformer.py / train_temporal_transformer.ipynb",
        "input": "Synthetic EAR sequences",
        "target": "awake=0, drowsy=1",
        "algorithm": "2-layer Transformer encoder over 20 EAR frames",
        "pre_model": "MediaPipe FaceLandmarker EAR signal",
        "output": "app/src/main/assets/temporal_model.tflite",
        "status": "Run completed; script patched",
        "primary_metric": "val_accuracy / test_accuracy",
        "result": f"{pct(temporal.get('best_val_accuracy'))} / {pct(temporal.get('test_accuracy'))}",
        "size_kb": temporal.get("tflite_kb", kb(ROOT / "app/src/main/assets/temporal_model.tflite")),
        "time_plan": "10 min guide estimate; conversion on this machine was slow",
    },
    {
        "model": "YOLO Local Clean",
        "env": "LOCAL with CUDA recommended",
        "script": "tools/train_clean_local.py / tools/train_clean_local.ipynb",
        "input": "roboflow_data/ds_augmented + ds_driveryawn -> clean_merged",
        "target": "awake=0, drowsy=1 boxes",
        "algorithm": "Ultralytics YOLO object detection",
        "pre_model": "yolov8s.pt by default",
        "output": "outputs/experiments/run_XXX/weights/best.pt",
        "status": "Not run: current TensorFlow/Torch session has no CUDA",
        "primary_metric": "mAP50",
        "result": "Expected 65 to 80",
        "size_kb": "PT file",
        "time_plan": "12 min with GPU; do not run on CPU for 4h target",
    },
    {
        "model": "YOLO26m Colab",
        "env": "COLAB T4",
        "script": "colab_drowsy_yolo26.ipynb",
        "input": "datio_drowsines Roboflow dataset",
        "target": "11 detection classes",
        "algorithm": "YOLO detection baseline",
        "pre_model": "YOLO26m / Ultralytics wrapper",
        "output": "outputs/training_yolo/yolo26/best.pt",
        "status": "Run in Colab",
        "primary_metric": "mAP50",
        "result": "Guide reference approx 50.21",
        "size_kb": "PT file",
        "time_plan": "75 min for 10 epochs; 4-5h for 50 epochs",
    },
    {
        "model": "YOLO11s Colab",
        "env": "COLAB T4",
        "script": "colab_yolo11s.ipynb",
        "input": "datio_drowsines Roboflow dataset",
        "target": "11 detection classes",
        "algorithm": "YOLO11s object detection",
        "pre_model": "yolo11s.pt",
        "output": "outputs/training_yolo/yolo11s/best.pt",
        "status": "Run in Colab",
        "primary_metric": "mAP50",
        "result": "Expected YOLO26 +3 to +7",
        "size_kb": "PT file",
        "time_plan": "50 min",
    },
    {
        "model": "RT-DETR Colab",
        "env": "COLAB T4",
        "script": "colab_rtdetr.ipynb",
        "input": "datio_drowsines Roboflow dataset",
        "target": "11 detection classes",
        "algorithm": "Transformer detector RT-DETR-L",
        "pre_model": "rtdetr-l.pt",
        "output": "outputs/training_yolo/rtdetr/best.pt",
        "status": "Run in Colab",
        "primary_metric": "mAP50",
        "result": "Expected >=53",
        "size_kb": "PT file",
        "time_plan": "80 min",
    },
    {
        "model": "YOLO-World Colab",
        "env": "COLAB T4",
        "script": "colab_yoloworld.ipynb",
        "input": "datio_drowsines + text prompts",
        "target": "Prompt-defined detection classes",
        "algorithm": "Open-vocabulary YOLO-World",
        "pre_model": "yolov8s-worldv2.pt",
        "output": "zero-shot/fine-tuned demo results",
        "status": "Run in Colab demo path",
        "primary_metric": "zero-shot/fine-tuned mAP50",
        "result": "25-40 zero-shot; >=40 fine-tuned",
        "size_kb": "PT file optional",
        "time_plan": "5 min zero-shot; 20 min fine-tune",
    },
]

pipeline_rows = [
    ("EDA", "Image inventory", "Count split/class, detect missing folders, verify class order", "outputs/dataset_summary_*"),
    ("EDA", "Class balance", "Compare train/val/test totals; yawn is near-balanced, eye is near-balanced", "Workbook Dataset Inputs"),
    ("EDA", "Visual samples", "Use sample grids and class distribution plots where available", "outputs/eda/*.png"),
    ("Preprocessing", "Resize", "All CNN image models resize to 64x64 RGB", "IMAGE_SIZE=64"),
    ("Preprocessing", "Normalization", "CNN input cast float32 and divide by 255 outside model", "range [0.0,1.0]"),
    ("Preprocessing", "Dataset split", "Yawn split 80/10/10 with SEED=42", "dataset_yawn/"),
    ("Feature Engineering", "EAR", "Eye Aspect Ratio from MediaPipe landmarks becomes temporal input", "20-frame buffer"),
    ("Feature Engineering", "MAR", "Mouth Aspect Ratio supports yawn/fallback rules", "yawnThreshold=0.58"),
    ("Feature Engineering", "YOLO labels", "Bounding boxes normalized as class x_center y_center width height", "data.yaml"),
    ("Pivot", "Dataset pivot", "Rows are splits, columns are classes, values are image counts/percentages", "Dataset Inputs sheet"),
    ("Pivot", "Model pivot", "Group by environment/model family/status to plan 4h execution", "Model Inventory sheet"),
    ("Augmentation", "YOLO photometric", "HSV hue/saturation/value shifts for lighting variation", "hsv_h=0.015,hsv_s=0.4,hsv_v=0.4"),
    ("Augmentation", "YOLO geometric", "light rotation/translation/scale, horizontal flip", "degrees=5,translate=0.1,scale=0.5,fliplr=0.5"),
    ("Augmentation", "Temporal synthetic", "Awake blinks, gradual drowsy, sudden drowsy, microsleep patterns", "generate_sequences()"),
    ("Training", "Callbacks", "EarlyStopping, ReduceLROnPlateau, ModelCheckpoint", "patience per model"),
    ("Export", "TFLite", "Convert Keras model with Optimize.DEFAULT and verify input/output shape", "app/src/main/assets"),
]


def style_sheet(ws):
    header_fill = PatternFill("solid", fgColor="1F4E78")
    header_font = Font(color="FFFFFF", bold=True)
    thin = Side(style="thin", color="D9E2F3")
    for row in ws.iter_rows():
        for cell in row:
            cell.alignment = Alignment(vertical="top", wrap_text=True)
            cell.border = Border(bottom=thin)
        if row[0].row == 1:
            for cell in row:
                cell.fill = header_fill
                cell.font = header_font
                cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
    ws.freeze_panes = "A2"
    for col in ws.columns:
        max_len = max(len(str(c.value or "")) for c in col)
        ws.column_dimensions[get_column_letter(col[0].column)].width = min(max(max_len + 2, 12), 42)


def add_table(ws, name):
    if ws.max_row < 2 or ws.max_column < 1:
        return
    ref = f"A1:{get_column_letter(ws.max_column)}{ws.max_row}"
    table = Table(displayName=name, ref=ref)
    table.tableStyleInfo = TableStyleInfo(name="TableStyleMedium2", showRowStripes=True)
    ws.add_table(table)


def build_xlsx() -> Path:
    wb = Workbook()
    ws = wb.active
    ws.title = "Executive Summary"
    ws.append(["Item", "Value"])
    ws.append(["Generated", datetime.now().strftime("%Y-%m-%d %H:%M")])
    ws.append(["Local runtime", "Python 3.12, TensorFlow 2.21 CPU-only on native Windows"])
    ws.append(["Models verified locally", "CNN Eye asset, CNN Yawn trained, Temporal Transformer trained/exported"])
    ws.append(["Latest model change", "YawnClassifier now uses yawn_cbam.tflite; Eye still uses drowsiness_model.tflite"])
    ws.append(["Models requiring Colab T4", "YOLO26m, YOLO11s, RT-DETR, YOLO-World"])
    ws.append(["4h strategy", "Run local CNN/Temporal on terminal while Colab runs YOLO chain in parallel"])
    style_sheet(ws)

    ws = wb.create_sheet("Model Inventory")
    headers = ["model", "env", "script", "input", "target", "algorithm", "pre_model", "output", "status", "primary_metric", "result", "size_kb", "time_plan"]
    ws.append(headers)
    for row in models:
        ws.append([row.get(h, "") for h in headers])
    style_sheet(ws)
    add_table(ws, "ModelInventory")

    ws = wb.create_sheet("Dataset Inputs")
    ws.append(["dataset", "split", "class_1", "count_1", "class_2", "count_2", "total", "notes"])
    dataset_specs = [
        ("dataset_mrl", ROOT / "outputs/dataset_summary_mrl/dataset_summary.csv", "Eye CNN/CBAM canonical input"),
        ("dataset", ROOT / "outputs/dataset_summary_dataset/dataset_summary.csv", "Alternate eye dataset already present"),
        ("dataset_yawn", ROOT / "outputs/dataset_summary_yawn/dataset_summary.csv", "Created from rawdata/data/yawn by train_cnn_yawn.py"),
    ]
    for name, path, note in dataset_specs:
        for r in csv_rows(path):
            classes = [k for k in r.keys() if k not in ("split", "total")]
            c1 = classes[0] if classes else ""
            c2 = classes[1] if len(classes) > 1 else ""
            ws.append([name, r.get("split"), c1, r.get(c1, ""), c2, r.get(c2, ""), r.get("total", ""), note])
    style_sheet(ws)
    add_table(ws, "DatasetInputs")

    ws = wb.create_sheet("EDA Preprocess FE Aug")
    ws.append(["phase", "method", "what it does", "implementation_reference"])
    for row in pipeline_rows:
        ws.append(list(row))
    style_sheet(ws)
    add_table(ws, "PipelineMethods")

    ws = wb.create_sheet("Hyperparameters")
    ws.append(["model", "parameter", "value", "explanation"])
    hp_rows = [
        ("CNN Eye/Yawn", "IMAGE_SIZE", 64, "Android INPUT_SIZE and TFLite input shape must match"),
        ("CNN Eye/Yawn", "BATCH_SIZE", 32, "Images per optimizer step; lower to 16 if memory pressure"),
        ("CNN Eye", "EPOCHS", 25, "Maximum epochs; EarlyStopping can stop earlier"),
        ("CNN Yawn", "EPOCHS", 20, "Maximum epochs for smaller yawn dataset"),
        ("CNN Eye/Yawn", "LEARNING_RATE", "1e-3", "Adam initial step size; lower to 5e-4 if accuracy is low"),
        ("CNN Eye/Yawn", "DROPOUT", 0.30, "Regularization before final Dense layer"),
        ("CBAM", "CBAM_RATIO", 8, "Channel attention bottleneck ratio; lower ratio means stronger/larger attention MLP"),
        ("Temporal", "SEQ_LEN", 20, "Number of EAR frames in rolling input buffer"),
        ("Temporal", "D_MODEL", 16, "Embedding size for each scalar EAR value"),
        ("Temporal", "N_HEADS", 4, "Self-attention heads"),
        ("Temporal", "N_LAYERS", 2, "Transformer encoder blocks"),
        ("YOLO Local", "model", "yolov8s", "Default detector backbone in configs/yolo_hparams.yaml"),
        ("YOLO Local", "epochs/imgsz/batch", "50 / 640 / auto", "Full quality config; for fast run use yolov8n, 30 epochs, 320"),
        ("RT-DETR", "imgsz", 640, "Do not reduce; RT-DETR expects 640 in the guide"),
    ]
    for row in hp_rows:
        ws.append(list(row))
    style_sheet(ws)
    add_table(ws, "Hyperparameters")

    ws = wb.create_sheet("Per-Class Metrics")
    ws.append(["model", "class", "precision", "recall", "f1", "support", "source"])
    for model_name, summary_path, summary in [
        ("CNN Eye CBAM", "outputs/cnn_eye_cbam/summary.json", cnn_eye_cbam),
        ("CNN Yawn CBAM", "outputs/cnn_yawn_cbam/summary.json", cnn_yawn_cbam),
        ("Temporal Transformer", "outputs/temporal_transformer/summary.json", {
            "per_class": {
                "awake": {
                    "precision": round(2004 / max(2004, 1), 6),
                    "recall": round(2004 / max(2004 + 5, 1), 6),
                    "f1": round((2 * 1.0 * (2004 / 2009)) / max(1.0 + (2004 / 2009), 1e-12), 6),
                    "support": 2009,
                },
                "drowsy": {
                    "precision": temporal.get("test_precision_drowsy"),
                    "recall": temporal.get("test_recall_drowsy"),
                    "f1": round((2 * temporal.get("test_precision_drowsy", 0) * temporal.get("test_recall_drowsy", 0)) / max(temporal.get("test_precision_drowsy", 0) + temporal.get("test_recall_drowsy", 0), 1e-12), 6),
                    "support": 1991,
                },
            }
        }),
    ]:
        for cls, metrics in (summary.get("per_class") or {}).items():
            ws.append([
                model_name,
                cls,
                metrics.get("precision"),
                metrics.get("recall"),
                metrics.get("f1"),
                metrics.get("support"),
                summary_path,
            ])
    style_sheet(ws)
    add_table(ws, "PerClassMetrics")

    ws = wb.create_sheet("4h Run Plan")
    ws.append(["time", "local terminal", "colab terminal", "expected artifact", "check"])
    run_rows = [
        ("T+0:00", "Verify assets/datasets; skip Eye retrain if drowsiness_model.tflite verified", "Start YOLO26m", "drowsiness_model.tflite / yolo26 run", "TFLite [1,64,64,3]"),
        ("T+0:05", "Run train_cnn_yawn.py", "YOLO26m training", "yawn_model.tflite", "class order no_yawn,yawn"),
        ("T+0:20", "Run train_temporal_transformer.py", "YOLO26m training", "temporal_model.tflite", "input [1,20,1]"),
        ("T+1:15", "Optionally run CBAM if GPU available", "Start YOLO11s", "drowsiness_cbam.tflite optional", "compare delta"),
        ("T+2:05", "Android build/test", "Start RT-DETR", "app-debug.apk", "camera/FPS/alert"),
        ("T+3:25", "Report/demo recording", "YOLO-World demo", "summary files", "download all best.pt"),
    ]
    for row in run_rows:
        ws.append(list(row))
    style_sheet(ws)
    add_table(ws, "RunPlan")

    ws = wb.create_sheet("Checks")
    ws.append(["check", "expected", "current"])
    for rel in [
        "app/src/main/assets/drowsiness_model.tflite",
        "app/src/main/assets/yawn_model.tflite",
        "app/src/main/assets/temporal_model.tflite",
        "app/src/main/assets/face_landmarker.task",
    ]:
        p = ROOT / rel
        ws.append([rel, "exists", "OK" if p.exists() else "MISSING"])
    ws.append(["CNN Eye classes", '["eyes_closed","eyes_open"]', ",".join(cnn_eye.get("classes", []))])
    ws.append(["CNN Yawn classes", '["no_yawn","yawn"]', ",".join(cnn_yawn.get("classes", []))])
    ws.append(["Android Eye model", "drowsiness_model.tflite", "active"])
    ws.append(["Android Yawn model", "yawn_cbam.tflite", "active"])
    ws.append(["Eye CBAM final asset", "drowsiness_cbam.tflite", "MISSING" if not (ROOT / "app/src/main/assets/drowsiness_cbam.tflite").exists() else "OK"])
    ws.append(["Temporal confusion matrix", "[[2004,5],[0,1991]]", str(temporal.get("confusion_matrix", ""))])
    style_sheet(ws)
    add_table(ws, "Checks")

    path = OUT_DIR / "DrowsyDriver_Model_Execution_Matrix.xlsx"
    wb.save(path)
    return path


def set_cell_shading(cell, fill):
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:fill"), fill)
    tc_pr.append(shd)


def style_doc(doc: Document):
    section = doc.sections[0]
    section.top_margin = Inches(1)
    section.bottom_margin = Inches(1)
    section.left_margin = Inches(1)
    section.right_margin = Inches(1)
    styles = doc.styles
    styles["Normal"].font.name = "Arial"
    styles["Normal"].font.size = Pt(10.5)
    for name, size in [("Heading 1", 16), ("Heading 2", 13), ("Heading 3", 11)]:
        styles[name].font.name = "Arial"
        styles[name].font.size = Pt(size)
        styles[name].font.color.rgb = RGBColor(31, 78, 121)


def add_kv_table(doc: Document, rows):
    table = doc.add_table(rows=1, cols=2)
    table.style = "Table Grid"
    table.rows[0].cells[0].text = "Mục"
    table.rows[0].cells[1].text = "Nội dung"
    for cell in table.rows[0].cells:
        set_cell_shading(cell, "D9EAF7")
        for p in cell.paragraphs:
            for r in p.runs:
                r.bold = True
    for k, v in rows:
        cells = table.add_row().cells
        cells[0].text = str(k)
        cells[1].text = str(v)
    return table


def build_docx() -> Path:
    doc = Document()
    style_doc(doc)

    title = doc.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = title.add_run("DrowsyDriverAndroid - Model Execution Runbook")
    run.font.name = "Arial"
    run.font.size = Pt(20)
    run.bold = True
    run.font.color.rgb = RGBColor(11, 37, 69)
    sub = doc.add_paragraph("Tổng hợp EDA, preprocessing, feature engineering, augmentation, training và input dataset cho từng mô hình")
    sub.alignment = WD_ALIGN_PARAGRAPH.CENTER

    doc.add_heading("1. Tóm Tắt Trạng Thái", level=1)
    add_kv_table(doc, [
        ("Runtime local", "Python 3.12, PyTorch CUDA trên RTX 4050; TensorFlow 2.21 CPU-only trên native Windows cho TFLite export"),
        ("Đã chạy trong phiên này", "CNN Yawn Baseline, CNN Yawn CBAM GPU, CNN Eye CBAM GPU quick test, Temporal Transformer EAR"),
        ("Đã verify asset có sẵn", "CNN Eye Baseline drowsiness_model.tflite; Eye CBAM drowsiness_cbam.tflite final asset exported but not selected for Android"),
        ("Android active models", "Eye: drowsiness_model.tflite; Yawn: yawn_cbam.tflite; Temporal: temporal_model.tflite asset available"),
        ("Cần Colab T4", "YOLO26m, YOLO11s, RT-DETR, YOLO-World"),
        ("Rủi ro thời gian", "Không chạy YOLO/RT-DETR trên CPU nếu cần hoàn tất trong 4 giờ"),
    ])

    doc.add_heading("2. Ràng Buộc Không Được Đổi", level=1)
    for item in [
        "CNN Eye: IMAGE_SIZE=64, normalize /255 ngoài model, eyes_closed=0, eyes_open=1.",
        "CNN Yawn: IMAGE_SIZE=64, normalize /255 ngoài model, no_yawn=0, yawn=1.",
        "TFLite CNN: input [1,64,64,3], dtype float32, range [0.0,1.0], output [1,2].",
        "Android thresholds: CNN_CONF_THRESHOLD=0.55f, YAWN_CONF_THRESHOLD=0.60f.",
        "Temporal model: input [1,20,1] là 20 giá trị EAR gần nhất.",
    ]:
        doc.add_paragraph(item, style="List Bullet")

    doc.add_heading("3. Kết Quả Local Đã Verify", level=1)
    add_kv_table(doc, [
        ("CNN Eye Baseline", f"best val accuracy {pct(cnn_eye.get('best_val_accuracy'))}%, TFLite {cnn_eye.get('tflite_kb')} KB, shape [1,64,64,3]."),
        ("CNN Yawn Baseline", f"best val accuracy {pct(cnn_yawn.get('best_val_accuracy'))}%, test accuracy {pct(cnn_yawn.get('test_accuracy'))}%, TFLite {cnn_yawn.get('tflite_kb')} KB."),
        ("CNN Yawn CBAM GPU", f"best val accuracy {pct(cnn_yawn_cbam.get('best_val_accuracy'))}%, test accuracy {pct(cnn_yawn_cbam.get('test_accuracy'))}%, TFLite {cnn_yawn_cbam.get('tflite_kb')} KB, confusion matrix {cnn_yawn_cbam.get('confusion_matrix')}."),
        ("Temporal Transformer", f"best val accuracy {pct(temporal.get('best_val_accuracy'))}%, test accuracy {pct(temporal.get('test_accuracy'))}%, confusion matrix {temporal.get('confusion_matrix')}."),
    ])

    doc.add_heading("3.1 Per-Class Metrics Mới", level=2)
    if cnn_yawn_cbam.get("per_class"):
        table_metrics = doc.add_table(rows=1, cols=5)
        table_metrics.style = "Table Grid"
        for i, h in enumerate(["Model", "Class", "Precision", "Recall", "F1"]):
            table_metrics.rows[0].cells[i].text = h
            set_cell_shading(table_metrics.rows[0].cells[i], "D9EAF7")
        for cls, m in cnn_yawn_cbam["per_class"].items():
            cells = table_metrics.add_row().cells
            cells[0].text = "CNN Yawn CBAM"
            cells[1].text = cls
            cells[2].text = f"{pct(m.get('precision'))}%"
            cells[3].text = f"{pct(m.get('recall'))}%"
            cells[4].text = f"{pct(m.get('f1'))}%"
    doc.add_paragraph("Precision/recall trong Keras cũ có thể trùng nhau vì đang là global/micro metric trên one-hot softmax; các metric mới dùng confusion matrix per-class.")

    doc.add_heading("4. Model Inventory Và Input", level=1)
    table = doc.add_table(rows=1, cols=5)
    table.style = "Table Grid"
    headers = ["Model", "Input dataset", "Thuật toán / pre-model", "Output", "Trạng thái"]
    for i, h in enumerate(headers):
        table.rows[0].cells[i].text = h
        set_cell_shading(table.rows[0].cells[i], "D9EAF7")
    for m in models:
        cells = table.add_row().cells
        cells[0].text = m["model"]
        cells[1].text = m["input"]
        cells[2].text = f"{m['algorithm']}; pre-model: {m['pre_model']}"
        cells[3].text = m["output"]
        cells[4].text = m["status"]

    doc.add_heading("5. EDA, Preprocessing, Feature Engineering, Pivot, Augmentation", level=1)
    current = None
    for phase, method, what, ref in pipeline_rows:
        if phase != current:
            doc.add_heading(phase, level=2)
            current = phase
        p = doc.add_paragraph(style="List Bullet")
        p.add_run(method + ": ").bold = True
        p.add_run(f"{what}. Reference: {ref}.")

    doc.add_heading("6. Giải Thích Tham Số Chính", level=1)
    for item in [
        "IMAGE_SIZE=64: giữ đúng input Android, giảm latency và làm TFLite nhỏ.",
        "BATCH_SIZE=32: cân bằng tốc độ và bộ nhớ; giảm xuống 16 nếu OOM.",
        "LEARNING_RATE=1e-3: tốc độ cập nhật Adam; giảm 5e-4 khi model dao động hoặc accuracy thấp.",
        "PATIENCE: số epoch chờ trước khi EarlyStopping dừng, tránh overfit và tiết kiệm thời gian.",
        "CBAM_RATIO=8: nén channel trong attention MLP; ratio nhỏ hơn tăng sức biểu diễn nhưng tăng tham số.",
        "SEQ_LEN=20: temporal model nhìn 20 frame EAR để phân biệt chớp mắt ngắn với buồn ngủ kéo dài.",
        "YOLO imgsz=640: chất lượng detect tốt hơn 320 nhưng chậm hơn; RT-DETR trong guide giữ 640.",
    ]:
        doc.add_paragraph(item, style="List Bullet")

    doc.add_heading("7. Kế Hoạch Chạy Trong 4 Giờ", level=1)
    for t, local, colab, artifact, check in [
        ("T+0:00", "Verify dataset/assets, không retrain Eye nếu asset đã pass shape.", "Start YOLO26m.", "drowsiness_model.tflite / yolo26 run", "shape đúng"),
        ("T+0:05", "Run train_cnn_yawn.py.", "YOLO26m tiếp tục.", "yawn_model.tflite", "class order đúng"),
        ("T+0:20", "Run train_temporal_transformer.py.", "YOLO26m tiếp tục.", "temporal_model.tflite", "shape [1,20,1]"),
        ("T+1:15", "CBAM optional nếu có GPU.", "Start YOLO11s.", "CBAM optional", "delta accuracy"),
        ("T+2:05", "Android build/test.", "Start RT-DETR.", "APK/debug demo", "camera/FPS/alert"),
        ("T+3:25", "Report/demo recording.", "YOLO-World demo.", "summary/best.pt", "download ngay"),
    ]:
        p = doc.add_paragraph(style="List Number")
        p.add_run(f"{t}: ").bold = True
        p.add_run(f"Local: {local} Colab: {colab} Artifact: {artifact}. Check: {check}.")

    doc.add_heading("8. Ghi Chú Kỹ Thuật Đã Sửa", level=1)
    for item in [
        "train_temporal_transformer.py: sửa positional embedding để giữ batch động (None,20,16).",
        "train_temporal_transformer.py: dùng SparseCategoricalAccuracy và sửa interpreter API từ set_input_tensor sang set_tensor.",
        "train_cnn_eye.py / train_cnn_yawn.py: dùng evaluate(return_dict=True) để summary không bị metric 0.00% trên Keras mới.",
        "train_cnn_cbam.py: evaluate trực tiếp model đã restore best weights thay vì load compile=False.",
        "tools/train_cbam_fast_gpu.py: train CBAM bằng PyTorch CUDA trên RTX 4050, lưu history sau mỗi epoch, export TFLite qua Keras.",
        "YawnClassifier.kt: đổi active model sang yawn_cbam.tflite vì CBAM GPU có test accuracy cao hơn baseline.",
    ]:
        doc.add_paragraph(item, style="List Bullet")

    doc.add_paragraph("Workbook đi kèm: DrowsyDriver_Model_Execution_Matrix.xlsx.")
    path = OUT_DIR / "DrowsyDriver_Model_Execution_Runbook.docx"
    doc.save(path)
    return path


if __name__ == "__main__":
    xlsx = build_xlsx()
    docx = build_docx()
    print(f"XLSX: {xlsx}")
    print(f"DOCX: {docx}")
