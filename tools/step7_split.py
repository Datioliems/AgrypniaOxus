# -*- coding: utf-8 -*-
"""
BƯỚC 7 — PHÂN CHIA DỮ LIỆU (Train / Val / Test Split)
Stratified split đảm bảo tỷ lệ class đồng đều ở 3 tập.
Verify KHÔNG có ảnh xuất hiện ở cả 2 tập (data leakage).

LƯU Ý: Nếu bước 6 (augment) đã tạo sẵn split train/val/test theo cấu trúc
thư mục → bước này xác nhận lại và tạo manifest. Không tái split.

Chạy:
    python step7_split.py --dataset cnn_eye
"""
import argparse
import sys
import hashlib
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    PROCESSED_DIR, SPLITS_DIR, REPORTS_DIR, LOGS_DIR, DATASETS,
    SPLIT_TRAIN, SPLIT_VAL, SPLIT_TEST, SPLIT_SEED
)
from utils import (
    setup_logging, save_json, safe_imread, save_image, copy_image,
    print_banner, print_stat, print_ok, print_warn, print_fail,
    build_report, create_dirs, list_images, md5_file
)

STEP = 7
STEP_NAME = "split"


def compute_split_hashes(split_dir: Path) -> set:
    """Tính set MD5 của tất cả ảnh trong một split."""
    hashes = set()
    for img_path in list_images(split_dir):
        hashes.add(md5_file(img_path))
    return hashes


def verify_no_leakage(splits: dict[str, Path]) -> list[str]:
    """
    Kiểm tra không có ảnh nào xuất hiện ở 2 split khác nhau.
    Trả về list lỗi (rỗng = OK).
    """
    errors = []
    split_names = list(splits.keys())
    print("  Đang kiểm tra data leakage...", end=" ")

    hash_sets = {}
    for name, path in splits.items():
        if path.exists():
            hash_sets[name] = compute_split_hashes(path)
        else:
            hash_sets[name] = set()

    for i in range(len(split_names)):
        for j in range(i + 1, len(split_names)):
            n1, n2 = split_names[i], split_names[j]
            overlap = hash_sets[n1] & hash_sets[n2]
            if overlap:
                errors.append(
                    f"DATA LEAKAGE: {len(overlap)} ảnh trùng giữa {n1} và {n2}"
                )

    if not errors:
        print("OK — không có leakage!")
    else:
        print(f"FAIL!")
    return errors


def run_split(dataset_name: str) -> dict:
    print_banner("PHÂN CHIA DỮ LIỆU (Split)", STEP)
    logger = setup_logging(f"step{STEP}_{dataset_name}", LOGS_DIR)

    cfg      = DATASETS[dataset_name]
    target   = cfg["target_size"]
    classes  = cfg["classes"]

    # Nguồn ưu tiên: augmented → resized → quality_filtered
    for subdir in ["augmented", f"resized_{target}", "quality_filtered"]:
        src_path = PROCESSED_DIR / cfg["raw_subdir"] / subdir
        if src_path.exists():
            break
    else:
        print_fail("Không tìm thấy dữ liệu đã qua xử lý")
        sys.exit(1)

    out_path = SPLITS_DIR / dataset_name
    create_dirs(out_path, REPORTS_DIR)

    print_stat("Dataset", dataset_name)
    print_stat("Nguồn", src_path)
    print_stat("Đích", out_path)
    print_stat("Classes", classes)
    print_stat("Split ratio", f"train={SPLIT_TRAIN} / val={SPLIT_VAL} / test={SPLIT_TEST}")

    # ─── Kiểm tra cấu trúc thư mục ──────────
    splits_exist = all(
        (src_path / split).exists()
        for split in ["train", "val", "test"]
    )

    if splits_exist:
        # Dữ liệu đã có cấu trúc train/val/test (từ bước 6 hoặc Roboflow)
        # Chỉ copy + verify
        print("\n  Phát hiện cấu trúc train/val/test đã có → Copy + Verify")
        split_counts  = {}

        for split in ["train", "val", "test"]:
            for class_dir in sorted((src_path / split).iterdir()):
                if not class_dir.is_dir():
                    continue
                class_name = class_dir.name
                dst = out_path / split / class_name
                img_paths = list_images(class_dir)
                for img_path in img_paths:
                    copy_image(img_path, dst / img_path.name)
                split_counts[f"{split}/{class_name}"] = len(img_paths)
                print_stat(f"  {split}/{class_name}", len(img_paths))

    else:
        # Cần thực hiện split
        print("\n  Chưa có train/val/test → Thực hiện stratified split")
        try:
            from sklearn.model_selection import train_test_split
        except ImportError:
            print_fail("scikit-learn chưa cài. Chạy: pip install scikit-learn")
            sys.exit(1)

        import random
        random.seed(SPLIT_SEED)

        split_counts = {}

        for class_dir in sorted(src_path.iterdir()):
            if not class_dir.is_dir():
                continue
            class_name = class_dir.name
            img_paths  = list_images(class_dir)
            random.shuffle(img_paths)

            # Stratified split
            tv_paths, test_paths = train_test_split(
                img_paths, test_size=SPLIT_TEST, random_state=SPLIT_SEED
            )
            adj_val = SPLIT_VAL / (SPLIT_TRAIN + SPLIT_VAL)
            train_paths, val_paths = train_test_split(
                tv_paths, test_size=adj_val, random_state=SPLIT_SEED
            )

            for split, paths in [("train", train_paths),
                                   ("val",   val_paths),
                                   ("test",  test_paths)]:
                dst = out_path / split / class_name
                for p in paths:
                    copy_image(p, dst / p.name)
                split_counts[f"{split}/{class_name}"] = len(paths)
                print_stat(f"  {split}/{class_name}", len(paths))

    # ─── Verify không leakage ────────────────
    split_dirs = {
        "train": out_path / "train",
        "val":   out_path / "val",
        "test":  out_path / "test",
    }
    leakage_errors = verify_no_leakage(split_dirs)

    # ─── Verify class order nhất quán với Android ────
    class_order_ok   = True
    class_order_errors = []
    actual_classes   = sorted([
        d.name for d in (out_path / "train").iterdir()
        if d.is_dir()
    ])
    expected_classes = classes

    if actual_classes != expected_classes:
        class_order_errors.append(
            f"Class order KHÔNG khớp! Actual: {actual_classes} | Expected: {expected_classes}"
        )
        class_order_ok = False
    else:
        print_ok(f"Class order đúng: {actual_classes} → index {list(range(len(actual_classes)))}")

    # ─── Status ───────────────────────────────
    all_errors = leakage_errors + class_order_errors
    warnings   = []
    status     = "PASS"
    if all_errors:
        status = "FAIL"
    elif not class_order_ok:
        status = "WARNING"

    print()
    for e in all_errors:
        print_fail(e)
        logger.error(e)

    if status == "PASS":
        total = sum(split_counts.values())
        print_ok(f"Bước 7 hoàn thành — {total:,} ảnh phân chia thành công")

    # Lưu split manifest
    manifest = {
        "dataset": dataset_name,
        "split_counts": split_counts,
        "class_to_index": {c: i for i, c in enumerate(classes)},
        "ratios": {"train": SPLIT_TRAIN, "val": SPLIT_VAL, "test": SPLIT_TEST},
        "seed": SPLIT_SEED,
        "leakage_check": "PASS" if not leakage_errors else "FAIL",
        "class_order_check": "PASS" if class_order_ok else "FAIL",
    }
    save_json(manifest, out_path / "split_manifest.json")
    print(f"  Manifest: {out_path / 'split_manifest.json'}")

    report = build_report(
        step=STEP, name=STEP_NAME, dataset=dataset_name,
        input_count=sum(v for k, v in split_counts.items() if k.startswith("train")),
        output_count=sum(split_counts.values()),
        status=status, warnings=warnings, errors=all_errors,
        metrics=manifest
    )
    save_json(report, REPORTS_DIR / f"step{STEP}_{dataset_name}_{STEP_NAME}.json")

    if status == "FAIL":
        sys.exit(1)
    return report


def main():
    parser = argparse.ArgumentParser(description="Bước 7: Dataset Split")
    parser.add_argument("--dataset", default="cnn_eye",
                        choices=list(DATASETS.keys()) + ["all"])
    args = parser.parse_args()

    if args.dataset == "all":
        for ds in DATASETS:
            run_split(ds)
    else:
        run_split(args.dataset)


if __name__ == "__main__":
    main()
