# -*- coding: utf-8 -*-
"""
BƯỚC 4 — CHUẨN HÓA KÍCH THƯỚC (Resize & Standardize)
Đưa tất cả ảnh về kích thước cố định mà KHÔNG làm méo tỷ lệ.

Quy tắc chọn phương pháp:
  - Aspect ratio 0.7–1.4  → resize_direct_square (gần vuông, ổn)
  - Ngoài range đó        → resize_with_padding  (thêm viền đen)
  - Scale up              → INTER_CUBIC
  - Scale down            → INTER_AREA

Chạy:
    python step4_resize.py --dataset cnn_eye
    python step4_resize.py --dataset cnn_yawn
    python step4_resize.py --dataset yolo --size 640
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    PROCESSED_DIR, REPORTS_DIR, LOGS_DIR, REJECTED_DIR, DATASETS,
    CNN_INPUT_SIZE, YOLO_INPUT_SIZE
)
from utils import (
    setup_logging, save_json, safe_imread, save_image, copy_image,
    print_banner, print_stat, print_ok, print_warn, print_fail,
    build_report, create_dirs, list_images
)

STEP = 4
STEP_NAME = "resize"

# Aspect ratio threshold: nếu w/h trong khoảng này → direct resize
AR_DIRECT_MIN = 0.7
AR_DIRECT_MAX = 1.4


# ─────────────────────────────────────────────
# RESIZE FUNCTIONS
# ─────────────────────────────────────────────

def choose_interpolation(src_size: tuple[int, int], target_size: int) -> int:
    """Chọn interpolation dựa trên scale direction."""
    min_dim = min(src_size)
    if min_dim < target_size:
        return cv2.INTER_CUBIC    # Upscale → cubic mịn hơn
    else:
        return cv2.INTER_AREA     # Downscale → area tránh aliasing


def resize_direct_square(img: np.ndarray, target: int) -> np.ndarray:
    """
    Resize trực tiếp thành target×target.
    Dùng khi ảnh đã gần vuông (aspect ratio 0.7–1.4).
    """
    interp = choose_interpolation(img.shape[:2], target)
    return cv2.resize(img, (target, target), interpolation=interp)


def resize_with_padding(img: np.ndarray, target: int,
                         pad_color: tuple = (0, 0, 0)) -> tuple[np.ndarray, dict]:
    """
    Resize giữ nguyên aspect ratio + padding màu đen đến target×target.
    Trả về (ảnh đã resize, metadata transform).
    """
    h, w     = img.shape[:2]
    scale    = target / max(h, w)
    new_w    = int(w * scale)
    new_h    = int(h * scale)

    interp   = choose_interpolation((h, w), target)
    resized  = cv2.resize(img, (new_w, new_h), interpolation=interp)

    # Tính padding
    pad_top    = (target - new_h) // 2
    pad_bottom = target - new_h - pad_top
    pad_left   = (target - new_w) // 2
    pad_right  = target - new_w - pad_left

    padded = cv2.copyMakeBorder(
        resized, pad_top, pad_bottom, pad_left, pad_right,
        cv2.BORDER_CONSTANT, value=pad_color
    )

    meta = {
        "original_size": [h, w],
        "scale": round(scale, 4),
        "padding": {"top": pad_top, "bottom": pad_bottom,
                    "left": pad_left, "right": pad_right},
        "method": "padding",
    }
    return padded, meta


def smart_resize(img: np.ndarray, target: int) -> tuple[np.ndarray, str]:
    """
    Chọn phương pháp resize phù hợp dựa trên aspect ratio.
    Trả về (resized_img, method_used).
    """
    h, w = img.shape[:2]
    ar   = w / h if h > 0 else 1.0

    if AR_DIRECT_MIN <= ar <= AR_DIRECT_MAX:
        return resize_direct_square(img, target), "direct"
    else:
        resized, _ = resize_with_padding(img, target)
        return resized, "padding"


def verify_output(img: np.ndarray, target: int) -> tuple[bool, str]:
    """Kiểm tra output đúng spec."""
    if img is None:
        return False, "img is None"
    if img.shape != (target, target, 3):
        return False, f"Shape sai: {img.shape} ≠ ({target}, {target}, 3)"
    if img.dtype != np.uint8:
        return False, f"dtype sai: {img.dtype} ≠ uint8"
    return True, ""


# ─────────────────────────────────────────────
# MAIN RUNNER
# ─────────────────────────────────────────────

def run_resize(dataset_name: str, target_size: int = None) -> dict:
    print_banner("CHUẨN HÓA KÍCH THƯỚC (Resize)", STEP)
    logger = setup_logging(f"step{STEP}_{dataset_name}", LOGS_DIR)

    cfg         = DATASETS[dataset_name]
    if target_size is None:
        target_size = cfg["target_size"]

    src_path  = PROCESSED_DIR / cfg["raw_subdir"] / "roi_extracted"
    out_path  = PROCESSED_DIR / cfg["raw_subdir"] / f"resized_{target_size}"
    rej_path  = REJECTED_DIR / dataset_name / "step4_resize"

    # Fallback: nếu roi_extracted không tồn tại, dùng quality_filtered
    if not src_path.exists():
        src_path = PROCESSED_DIR / cfg["raw_subdir"] / "quality_filtered"
        print_warn(f"roi_extracted không có, dùng quality_filtered: {src_path}")

    if not src_path.exists():
        print_fail(f"Không tìm thấy nguồn ảnh: {src_path}")
        sys.exit(1)

    create_dirs(out_path, rej_path, REPORTS_DIR)

    print_stat("Dataset", dataset_name)
    print_stat("Target size", f"{target_size}×{target_size}")
    print_stat("Nguồn", src_path)

    total   = 0
    saved   = 0
    failed  = 0
    by_method = {"direct": 0, "padding": 0}

    for split in ["train", "val", "test"]:
        split_src = src_path / split
        if not split_src.exists():
            continue

        for class_dir in sorted(split_src.iterdir()):
            if not class_dir.is_dir():
                continue
            class_name = class_dir.name
            split_saved = 0

            for img_path in list_images(class_dir):
                total += 1
                img = safe_imread(img_path)
                if img is None:
                    failed += 1
                    continue

                # Đảm bảo 3 kênh BGR
                if len(img.shape) == 2:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                elif img.shape[2] == 4:
                    img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

                # Resize
                resized, method = smart_resize(img, target_size)
                by_method[method] = by_method.get(method, 0) + 1

                # Verify
                ok, msg = verify_output(resized, target_size)
                if not ok:
                    failed += 1
                    logger.warning(f"VERIFY FAIL {img_path.name}: {msg}")
                    copy_image(img_path, rej_path / split / class_name / img_path.name)
                    continue

                # Lưu
                dst = out_path / split / class_name / img_path.name
                if save_image(resized, dst):
                    saved += 1
                    split_saved += 1
                else:
                    failed += 1

        print(f"  [{split}] saved: {split_saved}")

    fail_pct = failed / max(total, 1) * 100
    status = "PASS"
    warnings, errors = [], []

    if fail_pct > 5:
        errors.append(f"Resize fail rate cao: {fail_pct:.1f}%")
        status = "FAIL"

    print()
    print_stat("Tổng ảnh", total)
    print_stat("Đã resize", saved)
    print_stat("Thất bại", f"{failed} ({fail_pct:.1f}%)")
    print_stat("Direct resize", by_method.get("direct", 0))
    print_stat("Padding resize", by_method.get("padding", 0))

    if status == "PASS":
        print_ok(f"Bước 4 hoàn thành — {saved} ảnh {target_size}×{target_size}")

    for w in warnings:
        print_warn(w)
    for e in errors:
        print_fail(e)

    report = build_report(
        step=STEP, name=STEP_NAME, dataset=dataset_name,
        input_count=total, output_count=saved,
        status=status, warnings=warnings, errors=errors,
        metrics={
            "target_size": target_size,
            "fail_pct": round(fail_pct, 2),
            "by_method": by_method,
            "output_dir": str(out_path),
        }
    )
    save_json(report, REPORTS_DIR / f"step{STEP}_{dataset_name}_{STEP_NAME}.json")

    if status == "FAIL":
        sys.exit(1)
    return report


def main():
    parser = argparse.ArgumentParser(description="Bước 4: Resize")
    parser.add_argument("--dataset", default="cnn_eye",
                        choices=list(DATASETS.keys()) + ["all"])
    parser.add_argument("--size", type=int, default=None,
                        help="Target size (mặc định theo config dataset)")
    args = parser.parse_args()

    if args.dataset == "all":
        for ds in DATASETS:
            run_resize(ds, args.size)
    else:
        run_resize(args.dataset, args.size)


if __name__ == "__main__":
    main()
