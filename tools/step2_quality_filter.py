# -*- coding: utf-8 -*-
"""
BƯỚC 2 — LỌC CHẤT LƯỢNG (Quality Filtering)
Loại bỏ ảnh mờ (Laplacian), quá tối/sáng, thiếu tương phản.
Ảnh bị loại được copy sang rejected/ kèm lý do.

Chạy:
    python step2_quality_filter.py --dataset cnn_eye
    python step2_quality_filter.py --dataset cnn_yawn --blur-min 80
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    PROCESSED_DIR, REPORTS_DIR, LOGS_DIR, REJECTED_DIR,
    DATASETS, BLUR_LAPLACIAN_MIN, BRIGHTNESS_MIN, BRIGHTNESS_MAX, CONTRAST_STD_MIN
)
from utils import (
    setup_logging, save_json, safe_imread, save_image, copy_image,
    print_banner, print_stat, print_ok, print_warn, print_fail,
    build_report, create_dirs, list_images
)

STEP = 2
STEP_NAME = "quality_filter"


# ─────────────────────────────────────────────
# KIỂM TRA TỪNG TIÊU CHÍ
# ─────────────────────────────────────────────

def check_blur(gray: np.ndarray, min_val: float) -> tuple[bool, float]:
    """Laplacian variance — ảnh sắc nét có variance cao."""
    score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    return score >= min_val, round(score, 2)


def check_brightness(gray: np.ndarray) -> tuple[bool, float]:
    """Độ sáng trung bình pixel."""
    val = float(np.mean(gray))
    ok  = BRIGHTNESS_MIN <= val <= BRIGHTNESS_MAX
    return ok, round(val, 2)


def check_contrast(gray: np.ndarray) -> tuple[bool, float]:
    """Độ tương phản = std deviation pixel values."""
    val = float(np.std(gray))
    return val >= CONTRAST_STD_MIN, round(val, 2)


def check_not_uniform(gray: np.ndarray, threshold: float = 5.0) -> tuple[bool, str]:
    """Phát hiện ảnh gần như toàn một màu (pure black/white artifact)."""
    val = float(np.std(gray))
    if val < threshold:
        return False, f"Quá đồng nhất (std={val:.1f})"
    return True, ""


# ─────────────────────────────────────────────
# FILTER MỘT ẢNH
# ─────────────────────────────────────────────

def filter_one(img_path: Path, blur_min: float) -> dict:
    """
    Chạy tất cả checks. Trả về dict với passed=True/False và metrics.
    """
    result = {
        "path": str(img_path),
        "passed": True,
        "reasons": [],
        "metrics": {},
    }

    img = safe_imread(img_path)
    if img is None:
        result["passed"] = False
        result["reasons"].append("CANNOT_READ: File bị hỏng hoặc không đọc được")
        return result

    # Đảm bảo có 3 kênh
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Check 1: Kích thước tối thiểu
    h, w = img.shape[:2]
    if min(h, w) < 10:
        result["passed"] = False
        result["reasons"].append(f"TOO_SMALL: {h}×{w} < 10px")
        return result

    # Check 2: Blur
    blur_ok, blur_score = check_blur(gray, blur_min)
    result["metrics"]["blur"] = blur_score
    if not blur_ok:
        result["passed"] = False
        result["reasons"].append(f"BLUR: {blur_score} < {blur_min}")

    # Check 3: Brightness
    bright_ok, bright_val = check_brightness(gray)
    result["metrics"]["brightness"] = bright_val
    if not bright_ok:
        result["passed"] = False
        result["reasons"].append(
            f"BRIGHTNESS: {bright_val} out of [{BRIGHTNESS_MIN}, {BRIGHTNESS_MAX}]"
        )

    # Check 4: Contrast
    contrast_ok, contrast_val = check_contrast(gray)
    result["metrics"]["contrast"] = contrast_val
    if not contrast_ok:
        result["passed"] = False
        result["reasons"].append(f"LOW_CONTRAST: std={contrast_val} < {CONTRAST_STD_MIN}")

    # Check 5: Uniform (pure black/white)
    uniform_ok, uniform_msg = check_not_uniform(gray)
    if not uniform_ok:
        result["passed"] = False
        result["reasons"].append(f"UNIFORM_IMAGE: {uniform_msg}")

    return result


# ─────────────────────────────────────────────
# MAIN RUNNER
# ─────────────────────────────────────────────

def run_quality_filter(dataset_name: str, blur_min: float = None) -> dict:
    print_banner("LỌC CHẤT LƯỢNG (Quality Filter)", STEP)
    logger = setup_logging(f"step{STEP}_{dataset_name}", LOGS_DIR)

    cfg      = DATASETS[dataset_name]
    raw_path = cfg.get("raw_path", None)
    if raw_path is None:
        print_fail(f"raw_path chưa được cấu hình cho '{dataset_name}'")
        sys.exit(1)
    out_path = PROCESSED_DIR / cfg["raw_subdir"] / "quality_filtered"
    rej_path = REJECTED_DIR / dataset_name / "step2_quality"

    create_dirs(out_path, rej_path, REPORTS_DIR)

    if blur_min is None:
        blur_min = BLUR_LAPLACIAN_MIN

    logger.info(f"Dataset: {dataset_name} | Blur min: {blur_min}")
    print_stat("Dataset", dataset_name)
    print_stat("Nguồn", raw_path)
    print_stat("Blur threshold (Laplacian)", blur_min)

    total   = 0
    passed  = 0
    failed  = 0
    by_reason = {}
    rejection_log = []

    for split in ["train", "val", "test"]:
        split_src = raw_path / split
        if not split_src.exists():
            continue

        for class_dir in sorted(split_src.iterdir()):
            if not class_dir.is_dir():
                continue
            class_name = class_dir.name
            print(f"\n  [{split}/{class_name}]")

            img_list = list_images(class_dir)
            split_pass = 0
            split_fail = 0

            for img_path in img_list:
                total += 1
                result = filter_one(img_path, blur_min)

                if result["passed"]:
                    # Copy sang processed dir với cùng cấu trúc
                    dst = out_path / split / class_name / img_path.name
                    copy_image(img_path, dst)
                    passed += 1
                    split_pass += 1
                else:
                    failed += 1
                    split_fail += 1
                    # Copy sang rejected với tên file = original + reason code
                    reason_code = result["reasons"][0].split(":")[0]
                    rej_dst = rej_path / split / class_name / f"{reason_code}_{img_path.name}"
                    copy_image(img_path, rej_dst)
                    # Ghi log
                    rejection_log.append({
                        "path": str(img_path),
                        "reasons": result["reasons"],
                        "metrics": result["metrics"],
                    })
                    for r in result["reasons"]:
                        by_reason[r.split(":")[0]] = by_reason.get(r.split(":")[0], 0) + 1
                    logger.debug(f"REJECT {img_path.name}: {result['reasons']}")

            print_stat(f"    Pass", split_pass)
            print_stat(f"    Reject", split_fail)

    reject_pct = failed / max(total, 1) * 100

    # Lưu rejection log
    save_json({"rejection_log": rejection_log}, rej_path / "rejection_log.json")

    # Status
    status = "PASS"
    warnings, errors = [], []
    if reject_pct > 30:
        errors.append(f"Quá nhiều ảnh bị loại: {reject_pct:.1f}% → Kiểm tra lại nguồn dữ liệu")
        status = "FAIL"
    elif reject_pct > 15:
        warnings.append(f"Tỷ lệ loại cao: {reject_pct:.1f}%")
        status = "WARNING"

    print()
    print_stat("Tổng ảnh đầu vào", total)
    print_stat("Qua lọc (pass)", f"{passed} ({100 - reject_pct:.1f}%)")
    print_stat("Bị loại (reject)", f"{failed} ({reject_pct:.1f}%)")
    print()
    print("  Lý do loại:")
    for reason, cnt in sorted(by_reason.items(), key=lambda x: -x[1]):
        print_stat(f"    {reason}", cnt)

    for w in warnings:
        print_warn(w)
    for e in errors:
        print_fail(e)
    if status == "PASS":
        print_ok(f"Bước 2 hoàn thành — {passed}/{total} ảnh qua lọc")
    print(f"\n  Rejection log: {rej_path / 'rejection_log.json'}")

    report = build_report(
        step=STEP, name=STEP_NAME, dataset=dataset_name,
        input_count=total, output_count=passed,
        status=status, warnings=warnings, errors=errors,
        metrics={
            "blur_min_used": blur_min,
            "reject_pct": round(reject_pct, 2),
            "by_reason": by_reason,
            "output_dir": str(out_path),
        }
    )
    save_json(report, REPORTS_DIR / f"step{STEP}_{dataset_name}_{STEP_NAME}.json")

    if status == "FAIL":
        logger.error("Bước 2 FAIL")
        sys.exit(1)
    return report


def main():
    parser = argparse.ArgumentParser(description="Bước 2: Quality Filter")
    parser.add_argument("--dataset", default="cnn_eye",
                        choices=list(DATASETS.keys()) + ["all"])
    parser.add_argument("--blur-min", type=float, default=None,
                        help=f"Ngưỡng blur Laplacian (mặc định: {BLUR_LAPLACIAN_MIN})")
    args = parser.parse_args()

    if args.dataset == "all":
        for ds in DATASETS:
            run_quality_filter(ds, args.blur_min)
    else:
        run_quality_filter(args.dataset, args.blur_min)


if __name__ == "__main__":
    main()
