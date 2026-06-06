"""
Chuẩn bị dataset Yawn — chia thành train/val/test cố định.

Input:
    rawdata/data/yawn/
        no yawn/   (2,591 ảnh)   ← tên có dấu cách
        yawn/      (2,528 ảnh)

Output:
    dataset_yawn/
        train/
            no_yawn/   (~2,044 ảnh, 80%)
            yawn/      (~2,022 ảnh, 80%)
        val/
            no_yawn/   (~256 ảnh, 10%)
            yawn/      (~253 ảnh, 10%)
        test/
            no_yawn/   (~256 ảnh, 10%)
            yawn/      (~253 ảnh, 10%)

Usage:
    cd D:\\2026.AI\\DrowsyDriverAndroid
    python tools\\prepare_yawn_dataset.py
"""

import random
import shutil
from pathlib import Path

# ── Cấu hình ──────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent

SOURCE_DIR   = PROJECT_ROOT / "rawdata" / "data" / "yawn"
OUTPUT_DIR   = PROJECT_ROOT / "dataset_yawn"

TRAIN_RATIO  = 0.80
VAL_RATIO    = 0.10
TEST_RATIO   = 0.10      # = 1 - TRAIN - VAL

SEED         = 42
MAX_PER_CLASS = None     # None = dùng hết. Đặt số (ví dụ 2000) để giới hạn

# Map tên thư mục gốc → tên chuẩn (bỏ dấu cách, lowercase)
CLASS_NAME_MAP = {
    "no yawn":  "no_yawn",   # "no yawn" có dấu cách → "no_yawn"
    "no_yawn":  "no_yawn",
    "yawn":     "yawn",
}

VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def find_source_classes(source_dir: Path) -> dict[str, Path]:
    """Tìm các thư mục class trong source, map sang tên chuẩn"""
    found = {}
    for d in sorted(source_dir.iterdir()):
        if not d.is_dir():
            continue
        canonical = CLASS_NAME_MAP.get(d.name.lower())
        if canonical:
            found[canonical] = d
            print(f"  ✅ Tìm thấy: '{d.name}' → class '{canonical}'")
        else:
            print(f"  ⚠️  Bỏ qua thư mục không xác định: '{d.name}'")
    return found


def get_image_files(class_dir: Path) -> list[Path]:
    """Lấy danh sách ảnh trong thư mục"""
    files = [
        f for f in class_dir.iterdir()
        if f.is_file() and f.suffix.lower() in VALID_EXTS
    ]
    return sorted(files)


def split_files(files: list[Path], train_r: float, val_r: float, seed: int):
    """Chia danh sách file thành train/val/test"""
    rng = random.Random(seed)
    files = files.copy()
    rng.shuffle(files)

    n       = len(files)
    n_train = int(n * train_r)
    n_val   = int(n * val_r)

    train = files[:n_train]
    val   = files[n_train:n_train + n_val]
    test  = files[n_train + n_val:]
    return train, val, test


def copy_files(files: list[Path], dest_dir: Path) -> int:
    """Copy danh sách file vào thư mục đích"""
    dest_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    for src in files:
        dst = dest_dir / src.name
        # Xử lý trùng tên
        if dst.exists():
            dst = dest_dir / f"{src.stem}_{copied}{src.suffix}"
        shutil.copy2(src, dst)
        copied += 1
    return copied


def main():
    print("=" * 55)
    print("📂 PREPARE YAWN DATASET")
    print("=" * 55)

    # ── Kiểm tra source ───────────────────────────────────────────
    if not SOURCE_DIR.exists():
        print(f"\n❌ Không tìm thấy: {SOURCE_DIR}")
        print(f"   Kiểm tra lại đường dẫn rawdata/data/yawn/")
        return

    print(f"\nSource : {SOURCE_DIR}")
    print(f"Output : {OUTPUT_DIR}")
    print(f"Split  : {TRAIN_RATIO*100:.0f}% / {VAL_RATIO*100:.0f}% / {TEST_RATIO*100:.0f}%")

    # ── Tìm classes ───────────────────────────────────────────────
    print(f"\n📁 Tìm class folders trong source:")
    classes = find_source_classes(SOURCE_DIR)

    if len(classes) < 2:
        print(f"\n❌ Cần ít nhất 2 class, chỉ tìm thấy: {list(classes.keys())}")
        return

    # ── Xoá output cũ ────────────────────────────────────────────
    if OUTPUT_DIR.exists():
        print(f"\n🗑️  Xoá output cũ: {OUTPUT_DIR}")
        shutil.rmtree(OUTPUT_DIR)

    # ── Xử lý từng class ─────────────────────────────────────────
    print(f"\n📊 Phân chia:")
    print(f"{'Class':>10} {'Tổng':>6} {'Train':>6} {'Val':>5} {'Test':>5}")
    print(f"{'-'*40}")

    total_stats = {"train": 0, "val": 0, "test": 0}

    for class_name, class_dir in sorted(classes.items()):
        files = get_image_files(class_dir)

        # Giới hạn số lượng nếu cần
        if MAX_PER_CLASS and len(files) > MAX_PER_CLASS:
            rng = random.Random(SEED)
            files = rng.sample(files, MAX_PER_CLASS)

        train_f, val_f, test_f = split_files(files, TRAIN_RATIO, VAL_RATIO, SEED)

        # Copy vào output
        copy_files(train_f, OUTPUT_DIR / "train" / class_name)
        copy_files(val_f,   OUTPUT_DIR / "val"   / class_name)
        copy_files(test_f,  OUTPUT_DIR / "test"  / class_name)

        print(f"{class_name:>10} {len(files):>6} {len(train_f):>6} {len(val_f):>5} {len(test_f):>5}")
        total_stats["train"] += len(train_f)
        total_stats["val"]   += len(val_f)
        total_stats["test"]  += len(test_f)

    total = sum(total_stats.values())
    print(f"{'-'*40}")
    print(f"{'TOTAL':>10} {total:>6} {total_stats['train']:>6} {total_stats['val']:>5} {total_stats['test']:>5}")

    # ── Kiểm tra cân bằng class ───────────────────────────────────
    print(f"\n⚖️  Kiểm tra cân bằng class (train set):")
    class_train_counts = {}
    for class_name in classes:
        n = len(list((OUTPUT_DIR / "train" / class_name).glob("*")))
        class_train_counts[class_name] = n
        print(f"   {class_name}: {n} ảnh")

    counts = list(class_train_counts.values())
    ratio  = max(counts) / min(counts) if min(counts) > 0 else 999
    if ratio > 1.2:
        print(f"   ⚠️  Mất cân bằng {ratio:.2f}x — train_eye_classifier sẽ tự xử lý class_weight")
    else:
        print(f"   ✅ Cân bằng tốt (ratio={ratio:.2f}x)")

    # ── Kết quả ───────────────────────────────────────────────────
    print(f"""
✅ Dataset yawn đã sẵn sàng!

   {OUTPUT_DIR}/
   ├── train/  ({total_stats['train']} ảnh)
   │   ├── no_yawn/
   │   └── yawn/
   ├── val/    ({total_stats['val']} ảnh)
   │   ├── no_yawn/
   │   └── yawn/
   └── test/   ({total_stats['test']} ảnh)
       ├── no_yawn/
       └── yawn/

📋 Bước tiếp theo:
   Mở tune_experiments.py → Cell 1
   Đổi MODEL_TARGET = "yawn"
   Đổi DATASET = "dataset_yawn"
   Run All Cells
""")


if __name__ == "__main__":
    main()
