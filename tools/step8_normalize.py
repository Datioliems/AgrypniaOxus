# -*- coding: utf-8 -*-
"""
BƯỚC 8 — VERIFY NORMALIZATION & PREPROCESSING CONFIG
Xác nhận pipeline normalize /255.0 NGOÀI model hoạt động đúng.
Tạo preprocessing_config.json cho training notebooks sử dụng.

KHÔNG lưu tất cả ảnh thành .npy (tốn ổ đĩa) — chỉ verify sample.
Training notebook sẽ đọc từng batch và normalize on-the-fly.

Chạy:
    python step8_normalize.py --dataset cnn_eye
    python step8_normalize.py --dataset cnn_eye --save-sample  # Lưu 100 tensor mẫu
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    SPLITS_DIR, REPORTS_DIR, LOGS_DIR, TENSORS_DIR, DATASETS,
    CNN_INPUT_SIZE, TFLITE_INPUT_SHAPE, TFLITE_INPUT_DTYPE, TFLITE_VALUE_RANGE
)
from utils import (
    setup_logging, save_json, safe_imread,
    print_banner, print_stat, print_ok, print_warn, print_fail,
    build_report, create_dirs, list_images
)

STEP = 8
STEP_NAME = "normalize"


# ─────────────────────────────────────────────
# NORMALIZE FUNCTION (giống hệt code Android)
# ─────────────────────────────────────────────

def prepare_cnn_tensor(img_bgr: np.ndarray, target_size: int = CNN_INPUT_SIZE) -> np.ndarray:
    """
    Chuẩn bị tensor cho CNN — logic PHẢI giống hệt Android CameraAnalyzer.kt:
    1. BGR → RGB
    2. Resize về target_size×target_size
    3. float32 / 255.0  ← NGOÀI MODEL, không trong Keras
    """
    # 1. BGR → RGB
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    # 2. Resize
    h, w = img_rgb.shape[:2]
    if min(h, w) < target_size:
        interp = cv2.INTER_CUBIC   # Upscale
    else:
        interp = cv2.INTER_AREA    # Downscale
    img_resized = cv2.resize(img_rgb, (target_size, target_size), interpolation=interp)
    # 3. Normalize
    tensor = img_resized.astype(np.float32) / 255.0
    return tensor


def verify_tensor(tensor: np.ndarray, target_size: int) -> tuple[bool, list[str]]:
    """Kiểm tra tensor theo TFLite spec."""
    errors = []
    if tensor.shape != (target_size, target_size, 3):
        errors.append(f"Shape sai: {tensor.shape} ≠ ({target_size}, {target_size}, 3)")
    if tensor.dtype != np.float32:
        errors.append(f"Dtype sai: {tensor.dtype} ≠ float32")
    if tensor.min() < TFLITE_VALUE_RANGE[0] - 1e-6:
        errors.append(f"Min value âm: {tensor.min():.6f}")
    if tensor.max() > TFLITE_VALUE_RANGE[1] + 1e-6:
        errors.append(f"Max value > 1: {tensor.max():.6f}")
    return len(errors) == 0, errors


def verify_tflite_model(model_path: Path, target_size: int) -> dict:
    """Nếu TFLite model đã có → verify input/output spec."""
    result = {
        "model_path": str(model_path),
        "exists": model_path.exists(),
        "compatible": None,
        "issues": [],
    }
    if not model_path.exists():
        result["issues"].append("Model chưa tồn tại — sẽ kiểm tra sau khi train xong")
        return result

    try:
        import tensorflow as tf
        interp  = tf.lite.Interpreter(model_path=str(model_path))
        interp.allocate_tensors()
        in_det  = interp.get_input_details()[0]
        out_det = interp.get_output_details()[0]

        expected_in  = [1, target_size, target_size, 3]
        expected_out = [1, 2]

        if list(in_det["shape"]) != expected_in:
            result["issues"].append(f"Input shape sai: {list(in_det['shape'])} ≠ {expected_in}")
        if in_det["dtype"] != np.float32:
            result["issues"].append(f"Input dtype sai: {in_det['dtype']}")
        if list(out_det["shape"]) != expected_out:
            result["issues"].append(f"Output shape sai: {list(out_det['shape'])} ≠ {expected_out}")

        result["compatible"] = len(result["issues"]) == 0
        result["input_shape"]  = list(in_det["shape"])
        result["output_shape"] = list(out_det["shape"])
    except ImportError:
        result["issues"].append("TensorFlow không cài — bỏ qua TFLite check")
    except Exception as e:
        result["issues"].append(f"TFLite load error: {e}")

    return result


# ─────────────────────────────────────────────
# MAIN RUNNER
# ─────────────────────────────────────────────

def run_normalize(dataset_name: str, save_sample: bool = False) -> dict:
    print_banner("VERIFY NORMALIZE & PREPROCESSING CONFIG", STEP)
    logger = setup_logging(f"step{STEP}_{dataset_name}", LOGS_DIR)

    cfg         = DATASETS[dataset_name]
    target_size = cfg["target_size"]
    classes     = cfg["classes"]
    split_path  = SPLITS_DIR / dataset_name

    create_dirs(REPORTS_DIR)

    if not split_path.exists():
        print_fail(f"Split không tồn tại: {split_path} → Chạy step7 trước")
        sys.exit(1)

    print_stat("Dataset", dataset_name)
    print_stat("Target size", target_size)
    print_stat("Classes", classes)
    print_stat("Normalize method", "img.astype(float32) / 255.0  [NGOÀI model]")

    # ─── Sample verify ────────────────────────
    SAMPLE_N    = 50   # Số ảnh test mỗi class
    all_pass    = True
    verify_errs = []
    sample_stats = {}

    for split in ["train", "val", "test"]:
        for class_dir in sorted((split_path / split).iterdir()):
            if not class_dir.is_dir():
                continue
            class_name = class_dir.name
            img_paths  = list_images(class_dir)[:SAMPLE_N]

            class_tensors = []
            class_errors  = []

            for img_path in img_paths:
                img = safe_imread(img_path)
                if img is None:
                    continue
                tensor = prepare_cnn_tensor(img, target_size)
                ok, errs = verify_tensor(tensor, target_size)
                if not ok:
                    class_errors.extend(errs)
                    all_pass = False
                else:
                    class_tensors.append(tensor)

            if class_tensors:
                stacked = np.stack(class_tensors)
                sample_stats[f"{split}/{class_name}"] = {
                    "n_verified":   len(class_tensors),
                    "n_errors":     len(class_errors),
                    "mean":  round(float(stacked.mean()), 4),
                    "std":   round(float(stacked.std()), 4),
                    "min":   round(float(stacked.min()), 4),
                    "max":   round(float(stacked.max()), 4),
                }
            verify_errs.extend(class_errors)

    # ─── Lưu tensor mẫu (tuỳ chọn) ───────────
    if save_sample:
        sample_dir = TENSORS_DIR / dataset_name
        create_dirs(sample_dir)
        for split in ["train"]:
            for class_dir in sorted((split_path / split).iterdir()):
                if not class_dir.is_dir():
                    continue
                class_name = class_dir.name
                imgs    = list_images(class_dir)[:100]
                tensors = []
                for p in imgs:
                    img = safe_imread(p)
                    if img is not None:
                        tensors.append(prepare_cnn_tensor(img, target_size))
                if tensors:
                    arr  = np.stack(tensors).astype(np.float32)
                    dest = sample_dir / f"{split}_{class_name}_sample.npy"
                    np.save(str(dest), arr)
                    print_ok(f"Saved sample tensor: {dest} {arr.shape}")

    # ─── In kết quả ───────────────────────────
    print()
    for key, stats in sample_stats.items():
        print_stat(f"  {key}", f"mean={stats['mean']}, std={stats['std']}, range=[{stats['min']}, {stats['max']}]")

    if all_pass:
        print_ok("Tất cả tensor verify PASS — normalize đúng spec TFLite")
    else:
        print_fail(f"Có {len(verify_errs)} lỗi verify:")
        for e in verify_errs[:5]:
            print(f"    - {e}")

    # ─── Tạo preprocessing config ─────────────
    preproc_config = {
        "dataset": dataset_name,
        "target_size": target_size,
        "input_channels": 3,
        "color_order": "RGB",          # Sau khi chuyển từ BGR
        "normalize": "divide_by_255",  # NGOÀI model
        "value_range": list(TFLITE_VALUE_RANGE),
        "tflite_input_spec": {
            "shape": TFLITE_INPUT_SHAPE,
            "dtype": TFLITE_INPUT_DTYPE,
        },
        "class_to_index": {c: i for i, c in enumerate(classes)},
        "android_code_snippet": (
            "// Kotlin — CameraAnalyzer.kt\n"
            "val normalized = bitmap.pixels / 255.0f\n"
            "// Input tensor shape: [1, 64, 64, 3] float32\n"
            "// KHÔNG có normalize layer trong model"
        ),
        "python_code_snippet": (
            "img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)\n"
            "img = cv2.resize(img, (64, 64))\n"
            "tensor = img.astype(np.float32) / 255.0\n"
            "batch = np.expand_dims(tensor, 0)  # shape (1,64,64,3)"
        ),
        "verify_stats_sample": sample_stats,
    }
    config_path = SPLITS_DIR / dataset_name / "preprocessing_config.json"
    save_json(preproc_config, config_path)
    print(f"\n  Preprocessing config: {config_path}")

    status = "PASS" if all_pass else "FAIL"
    report = build_report(
        step=STEP, name=STEP_NAME, dataset=dataset_name,
        input_count=sum(s["n_verified"] for s in sample_stats.values()),
        output_count=sum(s["n_verified"] for s in sample_stats.values()),
        status=status,
        errors=verify_errs[:20],
        metrics=preproc_config,
    )
    save_json(report, REPORTS_DIR / f"step{STEP}_{dataset_name}_{STEP_NAME}.json")

    if status == "FAIL":
        sys.exit(1)
    return report


def main():
    parser = argparse.ArgumentParser(description="Bước 8: Normalize Verification")
    parser.add_argument("--dataset", default="cnn_eye",
                        choices=list(DATASETS.keys()) + ["all"])
    parser.add_argument("--save-sample", action="store_true",
                        help="Lưu 100 tensor mẫu dưới dạng .npy")
    args = parser.parse_args()

    if args.dataset == "all":
        for ds in DATASETS:
            run_normalize(ds, args.save_sample)
    else:
        run_normalize(args.dataset, args.save_sample)


if __name__ == "__main__":
    main()
