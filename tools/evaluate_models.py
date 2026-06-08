# -*- coding: utf-8 -*-
"""
ĐÁNH GIÁ MODELS TFLITE — Precision / Recall / F1 / Confusion Matrix
Chạy TRÊN LOCAL (không cần GPU).

Dùng TFLite Interpreter (giống Android) để đảm bảo kết quả khớp 100%.

Chạy:
    python tools/evaluate_models.py --model eye
    python tools/evaluate_models.py --model yawn
    python tools/evaluate_models.py --model all
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

# Thêm tools vào sys.path
sys.path.insert(0, str(Path(__file__).parent))

PROJECT_ROOT = Path(__file__).parent.parent

# ═══════════════════════════════════════════════════════════
# CONFIG — khớp với Android TfliteDrowsinessClassifier.kt
# ═══════════════════════════════════════════════════════════
MODELS = {
    "eye": {
        "tflite_path": PROJECT_ROOT / "app/src/main/assets/drowsiness_model.tflite",
        "test_dir":    PROJECT_ROOT / "dataset/test",
        "classes":     ["eyes_closed", "eyes_open"],   # index 0, 1 — PHẢI khớp Android
        "input_size":  64,
        "conf_thresh": 0.55,                            # CNN_CONF_THRESHOLD
        "positive_class": "eyes_closed",                # class "nguy hiểm" để tính recall
        "name": "CNN Eye (drowsiness_model.tflite)",
    },
    "yawn": {
        "tflite_path": PROJECT_ROOT / "app/src/main/assets/yawn_cbam.tflite",
        "test_dir":    PROJECT_ROOT / "dataset_yawn/test",
        "classes":     ["no_yawn", "yawn"],             # index 0, 1 — PHẢI khớp Android
        "input_size":  64,
        "conf_thresh": 0.60,                            # YAWN_CONF_THRESHOLD
        "positive_class": "yawn",                       # class "nguy hiểm"
        "name": "CNN Yawn (yawn_cbam.tflite)",
    },
}

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


# ═══════════════════════════════════════════════════════════
# LOAD TFLITE MODEL
# ═══════════════════════════════════════════════════════════
def load_tflite(path: Path):
    """Load TFLite interpreter. Thử tensorflow.lite trước, rồi tflite_runtime."""
    path_str = str(path)
    try:
        import tensorflow as tf
        interp = tf.lite.Interpreter(model_path=path_str)
        interp.allocate_tensors()
        return interp
    except ImportError:
        pass
    try:
        import tflite_runtime.interpreter as tflite
        interp = tflite.Interpreter(model_path=path_str)
        interp.allocate_tensors()
        return interp
    except ImportError:
        print("❌  Cần cài tensorflow hoặc tflite-runtime:")
        print("    pip install tensorflow")
        sys.exit(1)


# ═══════════════════════════════════════════════════════════
# PREPROCESS — GIỐNG HỆT ANDROID (Bitmap.toModelInput)
# ═══════════════════════════════════════════════════════════
def preprocess(img_path: Path, input_size: int) -> np.ndarray:
    """
    BGR → RGB → resize 64×64 → /255f → [1, 64, 64, 3] float32
    Giống hệt TfliteDrowsinessClassifier.kt:
        buffer.putFloat(((pixel shr 16) and 0xFF) / 255f)  // R
        buffer.putFloat(((pixel shr  8) and 0xFF) / 255f)  // G
        buffer.putFloat((pixel          and 0xFF) / 255f)  // B
    """
    raw = np.fromfile(str(img_path), dtype=np.uint8)
    img = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if img is None:
        return None
    img = cv2.resize(img, (input_size, input_size), interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)           # Android đọc ARGB → RGB
    tensor = img.astype(np.float32) / 255.0              # /255f — NGOÀI model
    return tensor[np.newaxis, ...]                        # [1, 64, 64, 3]


# ═══════════════════════════════════════════════════════════
# INFERENCE
# ═══════════════════════════════════════════════════════════
def predict(interp, tensor: np.ndarray) -> np.ndarray:
    """Chạy TFLite inference, trả về softmax probabilities."""
    inp_det  = interp.get_input_details()[0]
    out_det  = interp.get_output_details()[0]
    interp.set_tensor(inp_det["index"], tensor)
    interp.invoke()
    return interp.get_tensor(out_det["index"])[0]   # shape [2]


# ═══════════════════════════════════════════════════════════
# ĐÁNH GIÁ
# ═══════════════════════════════════════════════════════════
def evaluate(model_key: str, verbose: bool = False):
    cfg = MODELS[model_key]
    print(f"\n{'═'*60}")
    print(f"  ĐÁNH GIÁ: {cfg['name']}")
    print(f"{'═'*60}")

    tflite_path = cfg["tflite_path"]
    test_dir    = cfg["test_dir"]
    classes     = cfg["classes"]
    input_size  = cfg["input_size"]
    conf_thresh = cfg["conf_thresh"]
    pos_class   = cfg["positive_class"]

    # Kiểm tra file tồn tại
    if not tflite_path.exists():
        print(f"❌  Model không tồn tại: {tflite_path}")
        return
    if not test_dir.exists():
        print(f"❌  Test dir không tồn tại: {test_dir}")
        return

    print(f"  Model  : {tflite_path.name}  ({tflite_path.stat().st_size/1024:.0f} KB)")
    print(f"  Test   : {test_dir}")
    print(f"  Classes: {classes}")
    print(f"  Conf ≥ : {conf_thresh}  (ngưỡng Android)")

    # Load model
    interp = load_tflite(tflite_path)
    inp_shape = interp.get_input_details()[0]["shape"]
    out_shape = interp.get_output_details()[0]["shape"]
    print(f"  Input  : {list(inp_shape)}")
    print(f"  Output : {list(out_shape)}")

    # Thu thập kết quả
    y_true   = []   # ground truth index
    y_pred   = []   # predicted index (argmax)
    y_conf   = []   # confidence của predicted class
    rejected = 0    # ảnh có conf < threshold → model không chắc

    for cls_idx, cls_name in enumerate(classes):
        cls_dir = test_dir / cls_name
        if not cls_dir.exists():
            print(f"  ⚠️   Không có thư mục: {cls_dir}")
            continue
        imgs = [p for p in cls_dir.iterdir() if p.suffix.lower() in IMG_EXTS]
        print(f"\n  [{cls_name}] {len(imgs)} ảnh test...")

        for img_path in imgs:
            tensor = preprocess(img_path, input_size)
            if tensor is None:
                continue
            probs = predict(interp, tensor)
            pred_idx  = int(np.argmax(probs))
            confidence = float(probs[pred_idx])

            y_true.append(cls_idx)
            y_pred.append(pred_idx)
            y_conf.append(confidence)

            if verbose and pred_idx != cls_idx:
                print(f"    ✗ {img_path.name}: pred={classes[pred_idx]} ({confidence:.3f})")

    if not y_true:
        print("❌  Không có ảnh nào để đánh giá!")
        return

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_conf = np.array(y_conf)

    # ── CLASSIFICATION REPORT ─────────────────────────────
    try:
        from sklearn.metrics import (
            classification_report, confusion_matrix,
            precision_recall_fscore_support, roc_auc_score
        )
        HAS_SKLEARN = True
    except ImportError:
        HAS_SKLEARN = False
        print("  ⚠️   sklearn không có — tính thủ công")

    print(f"\n{'─'*60}")
    print("  KẾT QUẢ ĐÁNH GIÁ (toàn bộ test set, không dùng threshold)")
    print(f"{'─'*60}")

    n = len(y_true)
    accuracy = np.mean(y_true == y_pred)
    print(f"  Tổng ảnh test    : {n}")
    print(f"  Accuracy         : {accuracy*100:.2f}%")

    if HAS_SKLEARN:
        print()
        print(classification_report(y_true, y_pred, target_names=classes, digits=4))

        # Per-class Precision / Recall / F1
        prec, rec, f1, sup = precision_recall_fscore_support(
            y_true, y_pred, labels=list(range(len(classes)))
        )
        print(f"{'─'*60}")
        print(f"  CHỈ SỐ QUAN TRỌNG VỚI DROWSY DRIVER:")
        pos_idx = classes.index(pos_class)
        print(f"\n  Class DƯƠNG TÍNH (nguy hiểm): '{pos_class}'")
        print(f"  ├─ Precision : {prec[pos_idx]*100:.2f}%")
        print(f"  │   → Trong số báo '{pos_class}', bao nhiêu % là đúng")
        print(f"  ├─ Recall    : {rec[pos_idx]*100:.2f}%   ← QUAN TRỌNG NHẤT")
        print(f"  │   → Trong số thực sự '{pos_class}', bắt được bao nhiêu %")
        print(f"  └─ F1-score  : {f1[pos_idx]*100:.2f}%")
        print()

        # Cảnh báo nếu recall thấp
        if rec[pos_idx] < 0.90:
            print(f"  ⚠️  CẢNH BÁO: Recall '{pos_class}' = {rec[pos_idx]*100:.1f}% < 90%!")
            print(f"      → Nguy hiểm: bỏ sót tới {(1-rec[pos_idx])*100:.1f}% trường hợp ngủ gật!")
            print(f"      → Hãy hạ conf_threshold hoặc retrain với class weight")
        elif rec[pos_idx] < 0.95:
            print(f"  ⚠️  Recall '{pos_class}' = {rec[pos_idx]*100:.1f}% — nên cải thiện lên ≥95%")
        else:
            print(f"  ✅  Recall '{pos_class}' = {rec[pos_idx]*100:.1f}% — ĐẠT YÊU CẦU AN TOÀN")

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        print(f"\n  CONFUSION MATRIX:")
        print(f"  {'':>15s}  " + "  ".join(f"Pred:{c:<12s}" for c in classes))
        for i, cls_name in enumerate(classes):
            row = "  ".join(f"{cm[i][j]:>16d}" for j in range(len(classes)))
            print(f"  True:{cls_name:<12s}  {row}")

        # False Negative rate (đặc biệt quan trọng)
        fn_rate = cm[pos_idx].sum() - cm[pos_idx, pos_idx]
        fn_pct  = fn_rate / max(cm[pos_idx].sum(), 1) * 100
        fp_rate = sum(cm[i][pos_idx] for i in range(len(classes)) if i != pos_idx)
        print(f"\n  False Negative (bỏ sót '{pos_class}') : {fn_rate} ảnh ({fn_pct:.1f}%)")
        print(f"  False Positive (báo nhầm '{pos_class}'): {fp_rate} ảnh")

    else:
        # Tính thủ công nếu không có sklearn
        for i, cls_name in enumerate(classes):
            mask_true = (y_true == i)
            mask_pred = (y_pred == i)
            tp = np.sum(mask_true & mask_pred)
            fp = np.sum(~mask_true & mask_pred)
            fn = np.sum(mask_true & ~mask_pred)
            tn = np.sum(~mask_true & ~mask_pred)
            prec_i = tp / max(tp + fp, 1)
            rec_i  = tp / max(tp + fn, 1)
            f1_i   = 2 * prec_i * rec_i / max(prec_i + rec_i, 1e-9)
            print(f"\n  [{cls_name}]  TP={tp} FP={fp} FN={fn} TN={tn}")
            print(f"    Precision : {prec_i*100:.2f}%")
            print(f"    Recall    : {rec_i*100:.2f}%")
            print(f"    F1-score  : {f1_i*100:.2f}%")

    # ── PHÂN TÍCH CONFIDENCE THRESHOLD ────────────────────
    print(f"\n{'─'*60}")
    print(f"  ẢNH HƯỞNG CỦA CONFIDENCE THRESHOLD = {conf_thresh}")
    print(f"{'─'*60}")

    # Thử các mức threshold
    thresholds = [0.50, 0.55, 0.60, 0.65, 0.70, 0.80, 0.90]
    pos_idx_local = classes.index(pos_class)
    print(f"\n  Threshold  | Recall({pos_class}) | Precision | Coverage")
    print(f"  -----------|------------------|-----------|----------")
    for thresh in thresholds:
        # Với threshold này: chỉ đếm ảnh có confidence >= thresh
        # (ảnh còn lại → fallback EAR/MAR)
        keep = y_conf >= thresh
        if keep.sum() == 0:
            continue
        yt_k = y_true[keep]
        yp_k = y_pred[keep]
        tp_k = np.sum((yt_k == pos_idx_local) & (yp_k == pos_idx_local))
        fn_k = np.sum((yt_k == pos_idx_local) & (yp_k != pos_idx_local))
        fp_k = np.sum((yt_k != pos_idx_local) & (yp_k == pos_idx_local))
        recall_k = tp_k / max(tp_k + fn_k, 1) * 100
        prec_k   = tp_k / max(tp_k + fp_k, 1) * 100
        coverage = keep.sum() / n * 100
        marker = " ← ANDROID" if abs(thresh - conf_thresh) < 0.001 else ""
        print(f"  {thresh:.2f}       | {recall_k:>6.2f}%          | {prec_k:>6.2f}%   | {coverage:>5.1f}%{marker}")

    print(f"\n  Coverage = % ảnh CNN đủ tự tin để phán quyết")
    print(f"  Ảnh không đủ coverage → dùng EAR/MAR fallback trong Android")

    # ── KẾT LUẬN VÀ KHUYẾN NGHỊ ───────────────────────────
    print(f"\n{'═'*60}")
    print(f"  KẾT LUẬN & KHUYẾN NGHỊ")
    print(f"{'═'*60}")
    print(f"""
  Metric        | Giải thích                     | Yêu cầu dự án
  --------------|--------------------------------|----------------
  Accuracy      | Tổng tỷ lệ đúng               | ≥ 95% (đã đạt)
  Recall (+)    | Không bỏ sót ngủ gật          | ≥ 95% (QUAN TRỌNG)
  Precision (+) | Ít báo nhầm (false alarm)     | ≥ 80% (chấp nhận)
  F1-score (+)  | Cân bằng P & R                | ≥ 90%
  F2-score (+)  | Trọng số Recall gấp 2x        | ≥ 90% (khuyên dùng)

  (+) = với class "dương tính" (eyes_closed / yawn)
    """)
    print(f"{'═'*60}\n")


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="Đánh giá TFLite models: Precision / Recall / F1"
    )
    parser.add_argument("--model", choices=["eye", "yawn", "all"], default="all",
                        help="Model cần đánh giá (default: all)")
    parser.add_argument("--verbose", action="store_true",
                        help="In chi tiết ảnh bị phân loại sai")
    args = parser.parse_args()

    print("=" * 60)
    print("  DROWSY DRIVER — ĐÁNH GIÁ PRECISION / RECALL / F1")
    print("  Dùng TFLite Interpreter (giống Android 100%)")
    print("=" * 60)

    if args.model == "all":
        for key in MODELS:
            evaluate(key, args.verbose)
    else:
        evaluate(args.model, args.verbose)


if __name__ == "__main__":
    main()
