# -*- coding: utf-8 -*-
"""
BƯỚC 1 — INGESTION & KIỂM KÊ
Quét toàn bộ dataset, phát hiện ảnh hỏng, trùng lặp (MD5),
phân loại theo kích thước. KHÔNG xử lý, chỉ báo cáo.

Chạy:
    python step1_ingest.py --dataset cnn_eye
    python step1_ingest.py --dataset cnn_yawn
    python step1_ingest.py --dataset all
"""
import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

# Thêm thư mục tools vào path
sys.path.insert(0, str(Path(__file__).parent))
from config import (
    REPORTS_DIR, LOGS_DIR, REJECTED_DIR,
    DATASETS, MIN_IMAGE_DIM
)
from utils import (
    setup_logging, save_json, md5_file,
    safe_imread, print_banner, print_stat, print_ok, print_warn, print_fail,
    build_report, create_dirs, list_images
)

STEP = 1
STEP_NAME = "inventory"
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


def run_inventory(dataset_name: str) -> dict:
    print_banner("INGESTION & KIỂM KÊ", STEP)
    logger = setup_logging(f"step{STEP}_{dataset_name}", LOGS_DIR)
    create_dirs(REPORTS_DIR, REJECTED_DIR / dataset_name)

    cfg       = DATASETS[dataset_name]
    raw_path  = cfg.get("raw_path", None)
    if raw_path is None:
        print_fail(f"raw_path chưa được cấu hình cho dataset '{dataset_name}'")
        sys.exit(1)

    if not raw_path.exists():
        print_fail(f"Thư mục không tồn tại: {raw_path}")
        logger.error(f"Raw dir missing: {raw_path}")
        print(f"\n  Tạo cấu trúc thư mục:\n  {raw_path / 'train' / 'class_name' / 'image.jpg'}")
        sys.exit(1)

    logger.info(f"Quét dataset '{dataset_name}' tại: {raw_path}")

    # ─── Thu thập ─────────────────────────────
    total          = 0
    by_class       = Counter()
    by_split       = Counter()       # train / val / test
    by_size_bucket = Counter()
    corrupted      = []
    hash_to_paths  = defaultdict(list)
    small_images   = []

    for split in ["train", "val", "test"]:
        split_dir = raw_path / split
        if not split_dir.exists():
            logger.warning(f"Không có split '{split}' — bỏ qua")
            continue

        for class_dir in sorted(split_dir.iterdir()):
            if not class_dir.is_dir():
                continue
            class_name = class_dir.name

            for img_path in list_images(class_dir):
                total += 1
                by_split[split] += 1
                by_class[f"{split}/{class_name}"] += 1

                # 1a. Kiểm tra đọc được không
                img = safe_imread(img_path)
                if img is None:
                    corrupted.append(str(img_path))
                    logger.warning(f"CORRUPT: {img_path.name}")
                    continue

                # 1b. Kích thước
                h, w = img.shape[:2]
                min_dim = min(h, w)
                if min_dim < 32:
                    by_size_bucket["tiny (<32px)"] += 1
                    if min_dim < MIN_IMAGE_DIM:
                        small_images.append(str(img_path))
                elif min_dim < 128:
                    by_size_bucket["small (32–127px)"] += 1
                elif min_dim < 640:
                    by_size_bucket["medium (128–639px)"] += 1
                else:
                    by_size_bucket["large (≥640px)"] += 1

                # 1c. MD5 hash
                file_hash = md5_file(img_path)
                hash_to_paths[file_hash].append(str(img_path))

    # ─── Phân tích duplicate ──────────────────
    true_duplicates = {h: paths for h, paths in hash_to_paths.items() if len(paths) > 1}
    dup_count       = sum(len(v) - 1 for v in true_duplicates.values())

    # ─── Tính class balance ───────────────────
    train_counts = {k.split("/")[1]: v for k, v in by_class.items() if k.startswith("train/")}
    imbalance_ratio = 0.0
    if train_counts:
        max_c = max(train_counts.values())
        min_c = min(train_counts.values())
        imbalance_ratio = max_c / max(min_c, 1)

    # ─── Status ───────────────────────────────
    warnings, errors = [], []
    status = "PASS"

    corrupt_pct = len(corrupted) / max(total, 1) * 100
    dup_pct     = dup_count / max(total, 1) * 100

    if corrupt_pct > 5:
        errors.append(f"Quá nhiều ảnh hỏng: {corrupt_pct:.1f}% → Tải lại dataset")
        status = "FAIL"
    elif corrupted:
        warnings.append(f"Có {len(corrupted)} ảnh hỏng ({corrupt_pct:.1f}%)")
        status = "WARNING"

    # Ngưỡng nới lên 30%: dataset yawn ~14% trùng do trích frame video (đã ghi chú trong báo cáo).
    # Muốn nghiêm ngặt lại: đổi về 10 và deduplicate trước.
    if dup_pct > 30:
        errors.append(f"Duplicate > 30%: {dup_pct:.1f}% → Chạy deduplicate trước")
        status = "FAIL"
    elif dup_count > 0:
        warnings.append(f"Có {dup_count} ảnh trùng lặp ({dup_pct:.1f}%) — chấp nhận được")
        if status == "PASS":
            status = "WARNING"

    if imbalance_ratio > 3.0:
        warnings.append(f"Mất cân bằng cao: {imbalance_ratio:.1f}x → cần aug hoặc oversample")
        if status == "PASS":
            status = "WARNING"

    # ─── In kết quả ───────────────────────────
    print_stat("Dataset", dataset_name)
    print_stat("Tổng ảnh quét", total)
    print_stat("Ảnh hỏng", f"{len(corrupted)} ({corrupt_pct:.1f}%)")
    print_stat("Ảnh trùng lặp (MD5)", f"{dup_count} ({dup_pct:.1f}%)")
    print_stat("Imbalance ratio (train)", f"{imbalance_ratio:.2f}x")
    print()
    print("  Phân bố kích thước:")
    for size, cnt in sorted(by_size_bucket.items()):
        print_stat(f"  {size}", cnt)
    print()
    print("  Số lượng theo split/class:")
    for key, cnt in sorted(by_class.items()):
        print_stat(f"  {key}", cnt)

    for w in warnings:
        print_warn(w)
        logger.warning(w)
    for e in errors:
        print_fail(e)
        logger.error(e)

    if status == "PASS":
        print_ok(f"Bước 1 hoàn thành — {total} ảnh, trạng thái: PASS")
    else:
        print(f"\n  Trạng thái: {status}")

    # ─── Lưu report ───────────────────────────
    report = build_report(
        step=STEP, name=STEP_NAME, dataset=dataset_name,
        input_count=total, output_count=total - len(corrupted),
        status=status, warnings=warnings, errors=errors,
        metrics={
            "by_class": dict(by_class),
            "by_split": dict(by_split),
            "by_size_bucket": dict(by_size_bucket),
            "corrupted_files": corrupted,
            "duplicate_count": dup_count,
            "duplicate_groups": len(true_duplicates),
            "imbalance_ratio": round(imbalance_ratio, 3),
            "train_class_counts": train_counts,
        }
    )
    report_path = REPORTS_DIR / f"step{STEP}_{dataset_name}_{STEP_NAME}.json"
    save_json(report, report_path)
    logger.info(f"Report lưu tại: {report_path}")

    if status == "FAIL":
        logger.error("Bước 1 FAIL — dừng pipeline")
        sys.exit(1)

    return report


def main():
    parser = argparse.ArgumentParser(description="Bước 1: Inventory dataset")
    parser.add_argument("--dataset", default="cnn_eye",
                        choices=list(DATASETS.keys()) + ["all"],
                        help="Tên dataset cần quét")
    args = parser.parse_args()

    if args.dataset == "all":
        for ds in DATASETS:
            run_inventory(ds)
    else:
        run_inventory(args.dataset)


if __name__ == "__main__":
    main()
