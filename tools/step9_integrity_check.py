# -*- coding: utf-8 -*-
"""
BƯỚC 9 — KIỂM TRA TOÀN VẸN DỮ LIỆU (Integrity Check)
Chạy TRƯỚC MỖI LẦN TRAIN. Nếu fail → DỪNG NGAY, không train.

Kiểm tra:
  1. Tất cả ảnh đọc được không bị hỏng
  2. Đúng shape sau resize (target×target×3)
  3. Dtype uint8
  4. Class order nhất quán với Android
  5. Phân bố class không quá lệch (train)
  6. Val/Test KHÔNG có ảnh augmented (kiểm tra tên file aug_*)

Chạy:
    python step9_integrity_check.py --dataset cnn_eye
    python step9_integrity_check.py --dataset all  ← Chạy trước khi train
"""
import argparse
import sys
from pathlib import Path
from collections import Counter

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    SPLITS_DIR, REPORTS_DIR, LOGS_DIR, DATASETS, CNN_INPUT_SIZE
)
from utils import (
    setup_logging, save_json, safe_imread,
    print_banner, print_stat, print_ok, print_warn, print_fail,
    build_report, create_dirs, list_images
)

STEP = 9
STEP_NAME = "integrity_check"


def check_dataset_split(split_dir: Path, class_names: list,
                         expected_size: int, split_name: str,
                         logger) -> tuple[list, list]:
    """
    Kiểm tra toàn bộ một split (train/val/test).
    Trả về (warnings, errors).
    """
    warnings = []
    errors   = []
    class_counts = {}

    if not split_dir.exists():
        warnings.append(f"Split '{split_name}' không tồn tại: {split_dir}")
        return warnings, errors

    for class_dir in sorted(split_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        class_name = class_dir.name
        img_paths  = list_images(class_dir)
        count      = 0
        corrupted  = 0
        wrong_shape = 0
        wrong_dtype = 0

        for img_path in img_paths:
            img = safe_imread(img_path)

            # Check 1: Đọc được
            if img is None:
                corrupted += 1
                errors.append(f"CORRUPT [{split_name}/{class_name}]: {img_path.name}")
                logger.error(f"CORRUPT: {img_path}")
                continue

            # Check 2: Đúng shape
            if img.shape != (expected_size, expected_size, 3):
                wrong_shape += 1
                msg = (f"WRONG_SHAPE [{split_name}/{class_name}]: "
                       f"{img_path.name} — {img.shape} ≠ ({expected_size},{expected_size},3)")
                errors.append(msg)
                if wrong_shape <= 3:
                    logger.error(msg)

            # Check 3: Dtype uint8
            if img.dtype != np.uint8:
                wrong_dtype += 1

            # Check 4: Không phải ảnh đồng nhất
            if img.std() < 3.0:
                warnings.append(f"UNIFORM [{split_name}/{class_name}]: {img_path.name} std={img.std():.1f}")

            count += 1

        class_counts[class_name] = count

        if corrupted > 0:
            errors.append(f"[{split_name}/{class_name}] {corrupted} ảnh bị hỏng")
        if wrong_shape > 3:
            errors.append(f"[{split_name}/{class_name}] {wrong_shape} ảnh sai shape")
        if wrong_dtype > 0:
            errors.append(f"[{split_name}/{class_name}] {wrong_dtype} ảnh sai dtype (cần uint8)")

    # Check 5: Không có ảnh augmented trong val/test
    if split_name in ("val", "test"):
        for class_dir in split_dir.iterdir():
            if not class_dir.is_dir():
                continue
            aug_images = [p for p in list_images(class_dir) if p.name.startswith("aug_")]
            if aug_images:
                errors.append(
                    f"AUG_IN_{split_name.upper()}: {len(aug_images)} ảnh aug "
                    f"trong [{split_name}/{class_dir.name}] — VI PHẠM NGHIÊM TRỌNG!"
                )

    # Check 6: Class balance trong train
    if split_name == "train" and len(class_counts) > 1:
        max_c = max(class_counts.values())
        min_c = min(class_counts.values())
        ratio = max_c / max(min_c, 1)
        if ratio > 5.0:
            errors.append(f"CRITICAL IMBALANCE [{split_name}]: {ratio:.1f}x ({class_counts})")
        elif ratio > 2.0:
            warnings.append(f"Imbalance [{split_name}]: {ratio:.1f}x — cân nhắc aug thêm")

    return warnings, errors


def run_integrity_check(dataset_name: str) -> dict:
    print_banner("KIỂM TRA TOÀN VẸN DỮ LIỆU", STEP)
    logger = setup_logging(f"step{STEP}_{dataset_name}", LOGS_DIR)

    cfg           = DATASETS[dataset_name]
    classes       = cfg["classes"]
    expected_size = cfg["target_size"]
    split_path    = SPLITS_DIR / dataset_name

    create_dirs(REPORTS_DIR)

    if not split_path.exists():
        print_fail(f"Split không tồn tại: {split_path} → Chạy step7 trước")
        sys.exit(1)

    print_stat("Dataset", dataset_name)
    print_stat("Expected size", f"{expected_size}×{expected_size}×3 uint8")
    print_stat("Expected classes", classes)

    all_warnings = []
    all_errors   = []
    split_counts = {}

    for split in ["train", "val", "test"]:
        split_dir = split_path / split
        print(f"\n  Kiểm tra [{split}]...")
        warns, errs = check_dataset_split(
            split_dir, classes, expected_size, split, logger
        )
        all_warnings.extend(warns)
        all_errors.extend(errs)

        if split_dir.exists():
            total = sum(len(list_images(c)) for c in split_dir.iterdir() if c.is_dir())
            split_counts[split] = total
            print_stat(f"  [{split}] tổng ảnh", total)
            if errs:
                for e in errs[:3]:
                    print_fail(f"    {e}")
                if len(errs) > 3:
                    print(f"    ... và {len(errs)-3} lỗi khác")
            else:
                print_ok(f"    [{split}] PASS")

    # ─── Check class order vs Android ─────────
    print()
    print("  Kiểm tra class order (phải khớp Android):")
    train_dir = split_path / "train"
    if train_dir.exists():
        actual_classes = sorted([d.name for d in train_dir.iterdir() if d.is_dir()])
        if actual_classes == classes:
            print_ok(f"    Class order KHỚP: {actual_classes}")
            for i, c in enumerate(actual_classes):
                print(f"      index {i} = '{c}'")
        else:
            err = (f"CLASS ORDER KHÔNG KHỚP! "
                   f"Actual={actual_classes} | Expected={classes}")
            all_errors.append(err)
            print_fail(f"    {err}")

    # ─── Final verdict ─────────────────────────
    print()
    if all_errors:
        status = "FAIL"
        print_fail(f"INTEGRITY CHECK FAIL — {len(all_errors)} lỗi nghiêm trọng")
        print("  Danh sách lỗi:")
        for e in all_errors:
            print(f"    ✗ {e}")
        print()
        print("  ⛔ KHÔNG được train khi có lỗi này.")
        print("  ⛔ Hãy sửa lỗi và chạy lại bước 9 trước khi train.")
    else:
        status = "PASS"
        print_ok("INTEGRITY CHECK PASS — Dữ liệu sẵn sàng để train!")

    if all_warnings:
        print(f"\n  Warnings ({len(all_warnings)}):")
        for w in all_warnings[:5]:
            print_warn(f"    {w}")

    report = build_report(
        step=STEP, name=STEP_NAME, dataset=dataset_name,
        input_count=sum(split_counts.values()),
        output_count=sum(split_counts.values()),
        status=status,
        warnings=all_warnings[:50],
        errors=all_errors[:50],
        metrics={
            "split_counts": split_counts,
            "expected_shape": [expected_size, expected_size, 3],
            "expected_classes": classes,
            "class_order_check": "PASS" if not all_errors else "FAIL",
        }
    )
    save_json(report, REPORTS_DIR / f"step{STEP}_{dataset_name}_{STEP_NAME}.json")

    if status == "FAIL":
        logger.error(f"Integrity check FAIL — {len(all_errors)} errors")
        sys.exit(1)

    return report


def main():
    parser = argparse.ArgumentParser(description="Bước 9: Integrity Check")
    parser.add_argument("--dataset", default="cnn_eye",
                        choices=list(DATASETS.keys()) + ["all"])
    args = parser.parse_args()

    if args.dataset == "all":
        all_pass = True
        for ds in DATASETS:
            try:
                run_integrity_check(ds)
            except SystemExit:
                all_pass = False
        if not all_pass:
            print("\n⛔ Một số dataset FAIL integrity check — xem log chi tiết")
            sys.exit(1)
    else:
        run_integrity_check(args.dataset)


if __name__ == "__main__":
    main()
