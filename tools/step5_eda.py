# -*- coding: utf-8 -*-
"""
BƯỚC 5 — PHÂN TÍCH THỐNG KÊ (EDA)
Phân tích đặc điểm dữ liệu TRƯỚC khi augment để ra quyết định đúng.
Tạo các biểu đồ PNG và báo cáo JSON.

Chạy:
    python step5_eda.py --dataset cnn_eye
    python step5_eda.py --dataset all --no-plots   # Nếu không có display
"""
import argparse
import sys
from pathlib import Path
from collections import Counter

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    PROCESSED_DIR, REPORTS_DIR, LOGS_DIR, EDA_DIR, DATASETS
)
from utils import (
    setup_logging, save_json, safe_imread,
    print_banner, print_stat, print_ok, print_warn,
    build_report, create_dirs, list_images
)

STEP = 5
STEP_NAME = "eda"

# Tắt display nếu chạy headless
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# ─────────────────────────────────────────────
# TÍNH THỐNG KÊ
# ─────────────────────────────────────────────

def compute_pixel_stats(images: list[np.ndarray]) -> dict:
    """Mean, std, percentiles trên sample ảnh (tối đa 500)."""
    sample = images[:500]
    all_pixels = np.concatenate([img.flatten().astype(np.float32) for img in sample])
    return {
        "mean":  round(float(np.mean(all_pixels)), 3),
        "std":   round(float(np.std(all_pixels)), 3),
        "p5":    round(float(np.percentile(all_pixels, 5)), 1),
        "p25":   round(float(np.percentile(all_pixels, 25)), 1),
        "p50":   round(float(np.percentile(all_pixels, 50)), 1),
        "p75":   round(float(np.percentile(all_pixels, 75)), 1),
        "p95":   round(float(np.percentile(all_pixels, 95)), 1),
        "dark_pct":   round(float((all_pixels < 30).sum() / len(all_pixels) * 100), 2),
        "bright_pct": round(float((all_pixels > 225).sum() / len(all_pixels) * 100), 2),
    }


def compute_channel_stats(images: list[np.ndarray]) -> dict:
    """Mean và std từng kênh BGR trên tập train (chuẩn hóa /255)."""
    sample = images[:500]
    stats  = {}
    for ch_idx, ch_name in enumerate(["B", "G", "R"]):
        vals = np.concatenate([
            (img[:, :, ch_idx].flatten().astype(np.float32) / 255.0)
            for img in sample
        ])
        stats[ch_name] = {
            "mean": round(float(np.mean(vals)), 4),
            "std":  round(float(np.std(vals)), 4),
        }
    return stats


def compute_brightness_per_image(images: list[np.ndarray]) -> list[float]:
    """Tính độ sáng trung bình của mỗi ảnh."""
    return [float(np.mean(img)) for img in images]


# ─────────────────────────────────────────────
# VẼ BIỂU ĐỒ
# ─────────────────────────────────────────────

def plot_class_distribution(class_counts: dict, title: str, save_path: Path):
    fig, ax = plt.subplots(figsize=(10, 5))
    classes = list(class_counts.keys())
    counts  = list(class_counts.values())
    colors  = ["#e74c3c" if "closed" in c or "yawn" in c or "drowsy" in c
               else "#2ecc71" for c in classes]
    bars = ax.bar(classes, counts, color=colors, edgecolor="white", linewidth=0.8)
    ax.set_title(title, fontsize=14, pad=15)
    ax.set_xlabel("Class")
    ax.set_ylabel("Số lượng ảnh")
    for bar, cnt in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 50,
                f"{cnt:,}", ha="center", va="bottom", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_pixel_histogram(images: list[np.ndarray], title: str, save_path: Path):
    fig, ax = plt.subplots(figsize=(10, 4))
    sample = images[:200]
    all_pixels = np.concatenate([img.flatten() for img in sample])
    ax.hist(all_pixels, bins=128, range=(0, 256), color="#3498db", alpha=0.8, edgecolor="none")
    ax.axvline(x=20,  color="red",    linestyle="--", alpha=0.6, label="Brightness min=20")
    ax.axvline(x=235, color="orange", linestyle="--", alpha=0.6, label="Brightness max=235")
    ax.set_title(title, fontsize=13)
    ax.set_xlabel("Pixel Value (0–255)")
    ax.set_ylabel("Frequency")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_brightness_distribution(brightness_list: list[float],
                                  labels: list[str], title: str, save_path: Path):
    fig, ax = plt.subplots(figsize=(10, 4))
    unique_labels = list(set(labels))
    colors = ["#e74c3c", "#2ecc71", "#3498db", "#f39c12"]
    for i, lbl in enumerate(unique_labels):
        vals = [b for b, l in zip(brightness_list, labels) if l == lbl]
        ax.hist(vals, bins=50, alpha=0.7, label=lbl, color=colors[i % len(colors)])
    ax.axvline(x=20,  color="red",    linestyle="--", alpha=0.5)
    ax.axvline(x=235, color="orange", linestyle="--", alpha=0.5)
    ax.set_title(title, fontsize=13)
    ax.set_xlabel("Mean Brightness")
    ax.set_ylabel("Số ảnh")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_sample_grid(images_by_class: dict, title: str, save_path: Path,
                     cols: int = 5, rows_per_class: int = 2):
    """Lưới ảnh mẫu cho từng class."""
    classes   = list(images_by_class.keys())
    n_classes = len(classes)
    fig, axes = plt.subplots(n_classes * rows_per_class, cols,
                              figsize=(cols * 2, n_classes * rows_per_class * 2))
    if n_classes == 1:
        axes = [axes]

    for ci, cls in enumerate(classes):
        imgs = images_by_class[cls][:cols * rows_per_class]
        for ri in range(rows_per_class):
            for ci2 in range(cols):
                ax = axes[ci * rows_per_class + ri][ci2] if n_classes > 1 else axes[ri][ci2]
                idx = ri * cols + ci2
                if idx < len(imgs):
                    img = imgs[idx]
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    ax.imshow(img_rgb)
                ax.axis("off")
                if ri == 0 and ci2 == 0:
                    ax.set_title(cls, fontsize=8, color="blue")

    fig.suptitle(title, fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()


# ─────────────────────────────────────────────
# MAIN RUNNER
# ─────────────────────────────────────────────

def run_eda(dataset_name: str, make_plots: bool = True) -> dict:
    print_banner("PHÂN TÍCH THỐNG KÊ (EDA)", STEP)
    logger = setup_logging(f"step{STEP}_{dataset_name}", LOGS_DIR)

    cfg      = DATASETS[dataset_name]
    target   = cfg["target_size"]
    src_path = PROCESSED_DIR / cfg["raw_subdir"] / f"resized_{target}"
    eda_out  = EDA_DIR / dataset_name

    create_dirs(eda_out, REPORTS_DIR)

    if not src_path.exists():
        print_warn(f"resized_{target} không tồn tại, thử quality_filtered")
        src_path = PROCESSED_DIR / cfg["raw_subdir"] / "quality_filtered"

    if not src_path.exists():
        print_warn(f"Không tìm thấy nguồn ảnh EDA: {src_path}")
        return {}

    print_stat("Dataset", dataset_name)
    print_stat("Nguồn EDA", src_path)

    # ─── Load train images ────────────────────
    train_path = src_path / "train"
    if not train_path.exists():
        print_warn("Không có thư mục train/ — EDA không đầy đủ")
        return {}

    class_counts       = {}
    all_images         = []
    all_labels         = []
    images_by_class    = {}
    brightness_list    = []
    brightness_labels  = []

    for class_dir in sorted(train_path.iterdir()):
        if not class_dir.is_dir():
            continue
        class_name = class_dir.name
        img_paths  = list_images(class_dir)
        class_counts[class_name] = len(img_paths)
        images_by_class[class_name] = []

        print(f"  [{class_name}] loading {len(img_paths)} ảnh...")

        for img_path in img_paths[:1000]:  # Giới hạn 1000/class để nhanh
            img = safe_imread(img_path)
            if img is None:
                continue
            all_images.append(img)
            all_labels.append(class_name)
            brightness_list.append(float(np.mean(img)))
            brightness_labels.append(class_name)
            if len(images_by_class[class_name]) < 10:
                images_by_class[class_name].append(img)

    # ─── Tính thống kê ────────────────────────
    pixel_stats   = compute_pixel_stats(all_images)
    channel_stats = compute_channel_stats(all_images)

    # Imbalance
    if class_counts:
        max_c = max(class_counts.values())
        min_c = min(class_counts.values())
        imbalance_ratio = max_c / max(min_c, 1)
        if imbalance_ratio > 3.0:
            strategy = "CRITICAL: Thu thập thêm dữ liệu"
        elif imbalance_ratio > 1.5:
            strategy = "Dùng class_weight hoặc oversample"
        else:
            strategy = "Cân bằng tốt — không cần xử lý đặc biệt"
    else:
        imbalance_ratio = 0.0
        strategy = "N/A"

    # Gợi ý normalize
    r_mean = channel_stats.get("R", {}).get("mean", 0)
    g_mean = channel_stats.get("G", {}).get("mean", 0)
    b_mean = channel_stats.get("B", {}).get("mean", 0)

    print()
    print_stat("Tổng ảnh train load", len(all_images))
    print_stat("Pixel mean (sample)", round(pixel_stats["mean"], 2))
    print_stat("Pixel std (sample)",  round(pixel_stats["std"], 2))
    print_stat("Ảnh quá tối (<30px)", f"{pixel_stats['dark_pct']}%")
    print_stat("Ảnh quá sáng (>225px)", f"{pixel_stats['bright_pct']}%")
    print_stat("Imbalance ratio", f"{imbalance_ratio:.2f}x → {strategy}")
    print()
    print("  Channel stats (normalized /255):")
    for ch, s in channel_stats.items():
        print_stat(f"    {ch}", f"mean={s['mean']}, std={s['std']}")

    # ─── Vẽ biểu đồ ──────────────────────────
    if make_plots and all_images:
        print("\n  Đang vẽ biểu đồ...", end=" ")

        plot_class_distribution(
            class_counts,
            f"[{dataset_name}] Phân bố class (Train)",
            eda_out / "class_distribution.png"
        )
        plot_pixel_histogram(
            all_images,
            f"[{dataset_name}] Phân phối pixel values",
            eda_out / "pixel_histogram.png"
        )
        plot_brightness_distribution(
            brightness_list, brightness_labels,
            f"[{dataset_name}] Độ sáng theo class",
            eda_out / "brightness_distribution.png"
        )
        plot_sample_grid(
            images_by_class,
            f"[{dataset_name}] Ảnh mẫu",
            eda_out / "sample_grid.png"
        )
        print("xong!")
        print(f"  Biểu đồ lưu tại: {eda_out}")

    # ─── Lưu report ───────────────────────────
    metrics = {
        "class_counts_train": class_counts,
        "imbalance_ratio": round(imbalance_ratio, 3),
        "imbalance_strategy": strategy,
        "pixel_stats": pixel_stats,
        "channel_stats_normalized": channel_stats,
        "total_images_analyzed": len(all_images),
        "plots_dir": str(eda_out),
    }
    save_json(metrics, eda_out / "eda_report.json")
    print_ok(f"EDA report lưu tại: {eda_out / 'eda_report.json'}")

    report = build_report(
        step=STEP, name=STEP_NAME, dataset=dataset_name,
        input_count=len(all_images), output_count=len(all_images),
        status="PASS", metrics=metrics
    )
    save_json(report, REPORTS_DIR / f"step{STEP}_{dataset_name}_{STEP_NAME}.json")

    # Gợi ý augmentation
    print()
    print("  ► Gợi ý cho bước tiếp theo:")
    for cls, cnt in class_counts.items():
        target_aug = cfg["train_per_class"]
        if cnt < target_aug:
            need = target_aug - cnt
            print(f"    • {cls}: cần aug thêm {need:,} ảnh ({cnt} → {target_aug})")
        else:
            print(f"    • {cls}: đủ số lượng ({cnt} ≥ {target_aug})")

    return report


def main():
    parser = argparse.ArgumentParser(description="Bước 5: EDA")
    parser.add_argument("--dataset", default="cnn_eye",
                        choices=list(DATASETS.keys()) + ["all"])
    parser.add_argument("--no-plots", action="store_true",
                        help="Bỏ qua vẽ biểu đồ (headless mode)")
    args = parser.parse_args()

    make_plots = not args.no_plots

    if args.dataset == "all":
        for ds in DATASETS:
            run_eda(ds, make_plots)
    else:
        run_eda(args.dataset, make_plots)


if __name__ == "__main__":
    main()
