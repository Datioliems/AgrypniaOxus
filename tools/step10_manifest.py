# -*- coding: utf-8 -*-
"""
BƯỚC 10 — TẠO DATASET MANIFEST
Tổng hợp toàn bộ thông tin pipeline vào một file manifest cuối cùng.
File này là "hợp đồng" giữa pipeline dữ liệu và quá trình training.

Chạy:
    python step10_manifest.py --dataset cnn_eye
    python step10_manifest.py --dataset all
"""
import argparse
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    PROJECT_ROOT, SPLITS_DIR, REPORTS_DIR, LOGS_DIR, OUTPUTS_DIR, DATASETS,
    CNN_INPUT_SIZE, TFLITE_INPUT_SHAPE, TFLITE_VALUE_RANGE,
    CNN_EYE_CONF_THRESHOLD, CNN_YAWN_CONF_THRESHOLD,
    EAR_THRESHOLD, MAR_THRESHOLD, SPLIT_SEED, AUG_SEED, STEP_KEYS
)
from utils import (
    setup_logging, save_json, load_json, md5_dir,
    print_banner, print_stat, print_ok, print_warn,
    build_report, create_dirs, list_images
)

STEP = 10
STEP_NAME = "manifest"


def load_step_report(step_num: int, dataset: str) -> dict:
    """Load report từ bước trước (trả về {} nếu không có)."""
    key  = STEP_KEYS.get(step_num, f"step{step_num}")
    path = REPORTS_DIR / f"step{step_num}_{dataset}_{key}.json"
    if path.exists():
        try:
            return load_json(path)
        except Exception:
            return {}
    return {}


def count_split_images(split_path: Path) -> dict:
    """Đếm số ảnh trong từng split/class."""
    counts = {}
    for split in ["train", "val", "test"]:
        split_dir = split_path / split
        if not split_dir.exists():
            continue
        for class_dir in sorted(split_dir.iterdir()):
            if class_dir.is_dir():
                n = len(list_images(class_dir))
                counts[f"{split}/{class_dir.name}"] = n
    return counts


def run_manifest(dataset_name: str) -> dict:
    print_banner("TẠO DATASET MANIFEST", STEP)
    logger = setup_logging(f"step{STEP}_{dataset_name}", LOGS_DIR)

    cfg        = DATASETS[dataset_name]
    classes    = cfg["classes"]
    target_sz  = cfg["target_size"]
    split_path = SPLITS_DIR / dataset_name

    create_dirs(REPORTS_DIR, OUTPUTS_DIR)

    print_stat("Dataset", dataset_name)

    # ─── Load tất cả step reports ─────────────
    step_reports = {}
    for i in range(1, 10):
        r = load_step_report(i, dataset_name)
        if r:
            step_reports[f"step{i}"] = {
                "status":        r.get("status", "UNKNOWN"),
                "input_count":   r.get("input_count", 0),
                "output_count":  r.get("output_count", 0),
                "rejected_count": r.get("rejected_count", 0),
            }
            status_icon = "✓" if r.get("status") == "PASS" else "⚠"
            print_stat(f"  Bước {i} ({STEP_KEYS.get(i, '?')})",
                       f"{status_icon} {r.get('status', 'N/A')}")

    # ─── Đếm ảnh cuối cùng ────────────────────
    final_counts = count_split_images(split_path)
    print()
    print("  Số ảnh cuối cùng:")
    for key, cnt in sorted(final_counts.items()):
        print_stat(f"    {key}", f"{cnt:,}")

    # ─── Hash dataset để detect thay đổi ──────
    print("\n  Tính dataset hash...", end=" ")
    dataset_hash = "N/A"
    if split_path.exists():
        try:
            dataset_hash = md5_dir(split_path)
            print(f"{dataset_hash[:12]}...")
        except Exception as e:
            print(f"error: {e}")

    # ─── Preprocessing config ─────────────────
    preproc_path = split_path / "preprocessing_config.json"
    preproc_cfg  = load_json(preproc_path) if preproc_path.exists() else {}

    # ─── Build manifest ───────────────────────
    manifest = {
        "schema_version": "1.0.0",
        "created_at":     datetime.now().isoformat(),
        "pipeline_seed":  {"augmentation": AUG_SEED, "split": SPLIT_SEED},

        # Dataset info
        "dataset": {
            "name":        dataset_name,
            "classes":     classes,
            "class_to_index": {c: i for i, c in enumerate(classes)},
            "roi_type":    cfg["roi_type"],
            "target_size": target_sz,
        },

        # Final data counts
        "final_counts": final_counts,
        "final_total":  sum(final_counts.values()),
        "dataset_hash": dataset_hash,

        # Step summary
        "pipeline_steps": step_reports,

        # Preprocessing spec
        "preprocessing": {
            "normalize":     "divide_by_255",
            "value_range":   list(TFLITE_VALUE_RANGE),
            "color_order":   "RGB",
            "note":          "Normalize NGOÀI model — KHÔNG trong Keras layer",
        },

        # Android / TFLite spec — PHẢI KHỚP Constants.kt
        "android_tflite_spec": {
            "input_shape": TFLITE_INPUT_SHAPE,
            "input_dtype": "float32",
            "output_shape": [1, len(classes)],
            "output_meaning": {str(i): c for i, c in enumerate(classes)},
        },
        "android_thresholds": {
            "cnn_eye_conf":  CNN_EYE_CONF_THRESHOLD,
            "cnn_yawn_conf": CNN_YAWN_CONF_THRESHOLD,
            "ear_threshold": EAR_THRESHOLD,
            "mar_threshold": MAR_THRESHOLD,
        },

        # Trạng thái các bước
        "all_steps_passed": all(
            r.get("status") == "PASS"
            for r in step_reports.values()
        ),

        # Paths
        "paths": {
            "split_dir":       str(split_path),
            "preprocessing_config": str(preproc_path),
            "reports_dir":     str(REPORTS_DIR),
        },
    }

    # ─── Lưu manifest ─────────────────────────
    manifest_path = split_path / "dataset_manifest.json"
    save_json(manifest, manifest_path)

    # Cũng copy lên outputs/ cho dễ tìm
    global_manifest_path = OUTPUTS_DIR / f"manifest_{dataset_name}.json"
    save_json(manifest, global_manifest_path)

    print()
    print_ok(f"Manifest lưu tại: {manifest_path}")
    print_ok(f"Copy lưu tại:     {global_manifest_path}")

    if manifest["all_steps_passed"]:
        print_ok("✅ TẤT CẢ BƯỚC ĐÃ PASS — Dataset sẵn sàng để train!")
    else:
        print_warn("⚠  Một số bước chưa chạy hoặc có WARNING")
        print("   → Chạy từng step và kiểm tra lại")

    # ─── In bảng tóm tắt cho training ─────────
    print()
    print("  ┌─────────────────────────────────────────────────────")
    print(f"  │  TRAINING CHECKLIST — {dataset_name.upper()}")
    print("  ├─────────────────────────────────────────────────────")
    for split in ["train", "val", "test"]:
        for cls in classes:
            key = f"{split}/{cls}"
            cnt = final_counts.get(key, 0)
            print(f"  │  {key:<30} {cnt:>6,} ảnh")
    print("  ├─────────────────────────────────────────────────────")
    print(f"  │  Input tensor : {TFLITE_INPUT_SHAPE} float32 [0,1]")
    print(f"  │  Class order  : {dict(enumerate(classes))}")
    print("  └─────────────────────────────────────────────────────")

    report = build_report(
        step=STEP, name=STEP_NAME, dataset=dataset_name,
        input_count=sum(final_counts.values()),
        output_count=sum(final_counts.values()),
        status="PASS",
        metrics={"manifest_path": str(manifest_path), "hash": dataset_hash}
    )
    save_json(report, REPORTS_DIR / f"step{STEP}_{dataset_name}_{STEP_NAME}.json")
    return manifest


def main():
    parser = argparse.ArgumentParser(description="Bước 10: Dataset Manifest")
    parser.add_argument("--dataset", default="cnn_eye",
                        choices=list(DATASETS.keys()) + ["all"])
    args = parser.parse_args()

    if args.dataset == "all":
        for ds in DATASETS:
            run_manifest(ds)
    else:
        run_manifest(args.dataset)


if __name__ == "__main__":
    main()
