# -*- coding: utf-8 -*-
"""
BƯỚC 6 — DATA AUGMENTATION
Tăng cường dữ liệu CHỈ trên tập train đến target_per_class.
Mỗi ảnh aug có parent_id để traceback.
TUYỆT ĐỐI KHÔNG aug val và test.

Chạy:
    python step6_augment.py --dataset cnn_eye
    python step6_augment.py --dataset cnn_yawn --target 5000
"""
import argparse
import random
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    PROCESSED_DIR, REPORTS_DIR, LOGS_DIR, REJECTED_DIR, DATASETS, AUG_SEED
)
from utils import (
    setup_logging, save_json, safe_imread, save_image, copy_image,
    print_banner, print_stat, print_ok, print_warn, print_fail,
    build_report, create_dirs, list_images, md5_file
)

STEP = 6
STEP_NAME = "augmentation"


# ─────────────────────────────────────────────
# AUGMENTATION PIPELINES (Albumentations)
# ─────────────────────────────────────────────

def _make_eye_pipeline():
    """Pipeline cho CNN Eye (ảnh crop mắt 64×64)."""
    try:
        import albumentations as A
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=15, border_mode=cv2.BORDER_REFLECT, p=0.6),
            A.ShiftScaleRotate(shift_limit=0.10, scale_limit=0.15,
                               rotate_limit=0, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.25,
                                       contrast_limit=0.20, p=0.7),
            A.RandomGamma(gamma_limit=(70, 130), p=0.4),
            A.CLAHE(clip_limit=2.0, tile_grid_size=(4, 4), p=0.3),
            A.GaussianBlur(blur_limit=(1, 3), p=0.3),
            A.MotionBlur(blur_limit=5, p=0.2),
            A.ImageCompression(quality_lower=70, quality_upper=95, p=0.3),
            _make_gauss_noise(),
        ])
    except ImportError:
        return None


def _make_mouth_pipeline():
    """Pipeline cho CNN Yawn (ảnh crop miệng 64×64)."""
    try:
        import albumentations as A
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=12, border_mode=cv2.BORDER_REFLECT, p=0.5),
            A.ShiftScaleRotate(shift_limit=0.08, scale_limit=0.12,
                               rotate_limit=0, p=0.4),
            A.RandomBrightnessContrast(brightness_limit=0.30,
                                       contrast_limit=0.25, p=0.7),
            A.RandomGamma(gamma_limit=(65, 140), p=0.4),
            A.GaussianBlur(blur_limit=(1, 3), p=0.25),
            A.MotionBlur(blur_limit=5, p=0.15),
            A.ImageCompression(quality_lower=75, quality_upper=95, p=0.25),
            _make_gauss_noise(),
        ])
    except ImportError:
        return None


def _make_gauss_noise():
    """GaussNoise với API compat cho cả albumentations v1 và v2."""
    try:
        import albumentations as A
        # v2.x dùng std_range
        return A.GaussNoise(std_range=(0.01, 0.1), p=0.4)
    except Exception:
        try:
            import albumentations as A
            return A.GaussNoise(var_limit=(5.0, 25.0), p=0.4)
        except Exception:
            import albumentations as A
            return A.NoOp()


def _manual_augment(img: np.ndarray, seed: int) -> np.ndarray:
    """
    Fallback augmentation dùng OpenCV thuần (không cần albumentations).
    Dùng khi albumentations không cài.
    """
    rng = np.random.RandomState(seed)
    result = img.copy()

    # Flip
    if rng.rand() > 0.5:
        result = cv2.flip(result, 1)

    # Brightness
    delta = rng.uniform(-40, 40)
    result = np.clip(result.astype(np.int16) + int(delta), 0, 255).astype(np.uint8)

    # Rotate nhẹ
    angle = rng.uniform(-12, 12)
    h, w  = result.shape[:2]
    M     = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    result = cv2.warpAffine(result, M, (w, h), borderMode=cv2.BORDER_REFLECT)

    # Blur nhẹ 50% thời gian
    if rng.rand() > 0.5:
        k = rng.choice([1, 3])
        if k > 1:
            result = cv2.GaussianBlur(result, (k, k), 0)

    return result


# ─────────────────────────────────────────────
# AUGMENT CLASS ĐẾN TARGET
# ─────────────────────────────────────────────

def augment_class_to_target(
    src_dir: Path,
    dst_dir: Path,
    class_name: str,
    target_count: int,
    pipeline,
    logger,
    seed: int = AUG_SEED
) -> tuple[int, int]:
    """
    Augment hoặc Sample ảnh của một class đến target_count.
    - Nếu current > target: SAMPLE ngẫu nhiên xuống target (không aug)
    - Nếu current < target: AUG lên target
    Trả về (ảnh gốc copy, ảnh aug tạo mới).
    """
    random.seed(seed)
    np.random.seed(seed)

    img_paths = list_images(src_dir)
    current   = len(img_paths)
    dst_dir.mkdir(parents=True, exist_ok=True)

    # ── TRƯỜNG HỢP 1: Quá nhiều ảnh → SAMPLE xuống ──
    if current >= target_count:
        # Chọn ngẫu nhiên target_count ảnh để giữ lại
        selected = random.sample(img_paths, target_count)
        for img_path in selected:
            copy_image(img_path, dst_dir / img_path.name)
        logger.info(f"[{class_name}] SAMPLE {current} → {target_count} ảnh (không aug)")
        return target_count, 0

    need = target_count - current

    # Bước 1: Copy ảnh gốc
    original_copied = 0
    for img_path in img_paths:
        dst = dst_dir / img_path.name
        copy_image(img_path, dst)
        original_copied += 1

    if need == 0:
        logger.info(f"[{class_name}] Đủ {current} ≥ {target_count}, không cần aug")
        return original_copied, 0

    logger.info(f"[{class_name}] Cần aug {need} ảnh ({current} → {target_count})")

    # Bước 2: Augment
    aug_count = 0
    for i in range(need):
        # Chọn ngẫu nhiên ảnh gốc (có replacement)
        src_path   = random.choice(img_paths)
        parent_img = safe_imread(src_path)
        if parent_img is None:
            continue

        parent_hash = md5_file(src_path)[:8]

        # Augment
        if pipeline is not None:
            try:
                augmented = pipeline(image=parent_img)["image"]
            except Exception as e:
                logger.debug(f"Albumentations error: {e}, dùng manual aug")
                augmented = _manual_augment(parent_img, seed + i)
        else:
            augmented = _manual_augment(parent_img, seed + i)

        # Lưu với tên có parent_id
        aug_name = f"aug_{i:05d}_p{parent_hash}_{src_path.stem}.jpg"
        dst      = dst_dir / aug_name

        if save_image(augmented, dst):
            aug_count += 1
        else:
            logger.warning(f"Không lưu được aug ảnh #{i}")

    logger.info(f"[{class_name}] Done: {original_copied} orig + {aug_count} aug = {original_copied + aug_count}")
    return original_copied, aug_count


# ─────────────────────────────────────────────
# MAIN RUNNER
# ─────────────────────────────────────────────

def run_augmentation(dataset_name: str, target_override: int = None) -> dict:
    print_banner("DATA AUGMENTATION", STEP)
    logger = setup_logging(f"step{STEP}_{dataset_name}", LOGS_DIR)

    cfg      = DATASETS[dataset_name]
    target   = cfg["target_size"]
    roi_type = cfg["roi_type"]

    src_path = PROCESSED_DIR / cfg["raw_subdir"] / f"resized_{target}"
    out_path = PROCESSED_DIR / cfg["raw_subdir"] / "augmented"

    create_dirs(out_path, REPORTS_DIR)

    if not src_path.exists():
        print_fail(f"Nguồn không tồn tại: {src_path} → Chạy step4 trước")
        sys.exit(1)

    # Chọn pipeline
    try:
        import albumentations as A
        if roi_type == "eye" or cfg["already_cropped"]:
            pipeline = _make_eye_pipeline()
        elif roi_type == "mouth":
            pipeline = _make_mouth_pipeline()
        else:
            pipeline = None   # YOLO: aug ở trong training notebook
        print_ok(f"Albumentations pipeline: {roi_type}")
    except ImportError:
        print_warn("albumentations không tìm thấy → dùng manual augmentation (OpenCV)")
        pipeline = None

    print_stat("Dataset", dataset_name)
    print_stat("ROI type", roi_type)

    total_orig = 0
    total_aug  = 0
    by_class   = {}

    # QUAN TRỌNG: Chỉ aug TRAIN, copy nguyên val/test
    for split in ["train", "val", "test"]:
        split_src = src_path / split
        if not split_src.exists():
            continue

        for class_dir in sorted(split_src.iterdir()):
            if not class_dir.is_dir():
                continue
            class_name = class_dir.name
            dst_class  = out_path / split / class_name

            if split == "train":
                # Aug để đạt target
                target_count = target_override or cfg["train_per_class"]
                print(f"\n  [train/{class_name}] target={target_count:,}")
                orig, aug = augment_class_to_target(
                    class_dir, dst_class, class_name,
                    target_count, pipeline, logger
                )
                total_orig += orig
                total_aug  += aug
                by_class[f"train/{class_name}"] = {
                    "original": orig,
                    "augmented": aug,
                    "total": orig + aug,
                }
                print_stat("    Original", orig)
                print_stat("    Augmented", aug)
                print_stat("    Total", orig + aug)
            else:
                # val / test: chỉ copy, KHÔNG aug
                img_paths = list_images(class_dir)
                for img_path in img_paths:
                    copy_image(img_path, dst_class / img_path.name)
                by_class[f"{split}/{class_name}"] = {
                    "original": len(img_paths),
                    "augmented": 0,
                    "total": len(img_paths),
                }
                print(f"  [{split}/{class_name}] copy {len(img_paths)} ảnh (NO aug)")

    print()
    print_stat("Original train", total_orig)
    print_stat("Augmented train", total_aug)
    print_stat("Tổng train", total_orig + total_aug)

    status   = "PASS"
    warnings = []
    if total_aug == 0 and total_orig > 0:
        warnings.append("Không cần aug — dataset đã đủ số lượng target")

    print_ok(f"Bước 6 hoàn thành — {total_orig + total_aug:,} train ảnh")

    report = build_report(
        step=STEP, name=STEP_NAME, dataset=dataset_name,
        input_count=total_orig, output_count=total_orig + total_aug,
        status=status, warnings=warnings,
        metrics={
            "by_class": by_class,
            "total_original_train": total_orig,
            "total_augmented_train": total_aug,
            "aug_pipeline": roi_type,
            "output_dir": str(out_path),
            "note": "Val/Test KHÔNG được aug",
        }
    )
    save_json(report, REPORTS_DIR / f"step{STEP}_{dataset_name}_{STEP_NAME}.json")
    return report


def main():
    parser = argparse.ArgumentParser(description="Bước 6: Augmentation")
    parser.add_argument("--dataset", default="cnn_eye",
                        choices=list(DATASETS.keys()) + ["all"])
    parser.add_argument("--target", type=int, default=None,
                        help="Override target per class (mặc định theo config)")
    args = parser.parse_args()

    if args.dataset == "all":
        for ds in DATASETS:
            run_augmentation(ds, args.target)
    else:
        run_augmentation(args.dataset, args.target)


if __name__ == "__main__":
    main()
