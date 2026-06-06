"""
Bước 1: Chuẩn bị dataset sạch từ MRL Eye Dataset.

Nguồn dữ liệu ĐÃ CÓ NHÃN TRỰC TIẾP (mắt mở / mắt nhắm):
  rawdata/closed_eye/  ← ảnh mắt nhắm
  rawdata/open_eye/    ← ảnh mắt mở

Output:
  dataset_mrl/
    train/eyes_closed/  (70%)
    train/eyes_open/
    val/eyes_closed/    (15%)
    val/eyes_open/
    test/eyes_closed/   (15%)
    test/eyes_open/

Tại sao dùng dataset này thay vì dataset/ cũ:
  - dataset/ cũ map "sleepy" → eyes_closed, "awake" → eyes_open
  - Người buồn ngủ không nhất thiết đang nhắm mắt → label noise cao
  - rawdata/closed_eye có nhãn trực tiếp từng frame → sạch hơn nhiều

Usage:
  python tools/prepare_mrl_dataset.py
  python tools/prepare_mrl_dataset.py --rawdata rawdata --out dataset_mrl
"""

import argparse
import random
import shutil
from pathlib import Path


def count_images(folder: Path) -> int:
    if not folder.exists():
        return 0
    return sum(
        1 for f in folder.rglob("*")
        if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".pgm"}
    )


def collect_images(folder: Path) -> list:
    if not folder.exists():
        return []
    return sorted(
        f for f in folder.rglob("*")
        if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".pgm"}
    )


def split_and_copy(
    src_files: list,
    out_root: Path,
    class_name: str,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42,
    copy: bool = True,
) -> dict:
    """
    Chia files thành train/val/test và copy (hoặc symlink) vào thư mục đích.
    Trả về dict: {'train': N, 'val': N, 'test': N}
    """
    random.seed(seed)
    files = list(src_files)
    random.shuffle(files)

    n = len(files)
    n_train = int(n * train_ratio)
    n_val   = int(n * val_ratio)

    splits = {
        "train": files[:n_train],
        "val":   files[n_train : n_train + n_val],
        "test":  files[n_train + n_val :],
    }

    counts = {}
    for split_name, split_files in splits.items():
        dest_dir = out_root / split_name / class_name
        dest_dir.mkdir(parents=True, exist_ok=True)
        for src in split_files:
            dest = dest_dir / src.name
            if not dest.exists():
                if copy:
                    shutil.copy2(src, dest)
                else:
                    dest.symlink_to(src.resolve())
        counts[split_name] = len(split_files)

    return counts


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Chuẩn bị dataset sạch từ rawdata/closed_eye + open_eye"
    )
    parser.add_argument(
        "--rawdata", default="rawdata",
        help="Thư mục rawdata chứa closed_eye/ và open_eye/"
    )
    parser.add_argument(
        "--out", default="dataset_mrl",
        help="Thư mục output dataset"
    )
    parser.add_argument(
        "--train-ratio", type=float, default=0.70,
        help="Tỉ lệ train (default 0.70)"
    )
    parser.add_argument(
        "--val-ratio", type=float, default=0.15,
        help="Tỉ lệ val (default 0.15), test = 1 - train - val"
    )
    parser.add_argument(
        "--max-per-class", type=int, default=0,
        help="Giới hạn số ảnh mỗi class (0 = không giới hạn)"
    )
    parser.add_argument(
        "--seed", type=int, default=42
    )
    args = parser.parse_args()

    rawdata_root = Path(args.rawdata)
    out_root     = Path(args.out)

    # ── Tìm thư mục nguồn ─────────────────────────────────────────────
    # Hỗ trợ nhiều tên folder khác nhau có thể có trong dự án
    CLOSED_CANDIDATES = [
        rawdata_root / "closed_eye",
        rawdata_root / "Close-Eyes",
        Path("mrleyedataset") / "Close-Eyes",
    ]
    OPEN_CANDIDATES = [
        rawdata_root / "open_eye",
        rawdata_root / "Open-Eyes",
        Path("mrleyedataset") / "Open-Eyes",
    ]

    def find_folder(candidates):
        for c in candidates:
            if c.exists() and count_images(c) > 0:
                return c
        return None

    closed_dir = find_folder(CLOSED_CANDIDATES)
    open_dir   = find_folder(OPEN_CANDIDATES)

    print("=" * 55)
    print("📁 PREPARE MRL EYE DATASET")
    print("=" * 55)

    if closed_dir is None:
        print("❌ Không tìm thấy thư mục ảnh mắt nhắm!")
        print("   Tìm ở:", [str(c) for c in CLOSED_CANDIDATES])
        return
    if open_dir is None:
        print("❌ Không tìm thấy thư mục ảnh mắt mở!")
        print("   Tìm ở:", [str(c) for c in OPEN_CANDIDATES])
        return

    print(f"  closed_eye  : {closed_dir}  ({count_images(closed_dir):,} ảnh)")
    print(f"  open_eye    : {open_dir}  ({count_images(open_dir):,} ảnh)")

    # ── Thu thập files ─────────────────────────────────────────────────
    closed_files = collect_images(closed_dir)
    open_files   = collect_images(open_dir)

    if args.max_per_class > 0:
        random.seed(args.seed)
        closed_files = random.sample(closed_files,
                                     min(args.max_per_class, len(closed_files)))
        open_files   = random.sample(open_files,
                                     min(args.max_per_class, len(open_files)))
        print(f"\n  Giới hạn: {args.max_per_class:,} ảnh/class")

    print(f"\nSẽ dùng: {len(closed_files):,} closed + {len(open_files):,} open")

    # ── Cân bằng 2 class ──────────────────────────────────────────────
    min_count = min(len(closed_files), len(open_files))
    if len(closed_files) != len(open_files):
        print(f"⚖️  Cân bằng về {min_count:,} ảnh/class")
        random.seed(args.seed)
        closed_files = random.sample(closed_files, min_count)
        open_files   = random.sample(open_files, min_count)

    # ── Xoá output cũ nếu có ──────────────────────────────────────────
    if out_root.exists():
        print(f"\n🗑️  Xoá output cũ: {out_root}")
        shutil.rmtree(out_root)

    print(f"\n📦 Copying vào {out_root} ...")
    print(f"   Tỉ lệ: train={args.train_ratio} / "
          f"val={args.val_ratio} / "
          f"test={1-args.train_ratio-args.val_ratio:.2f}")

    counts_closed = split_and_copy(
        closed_files, out_root, "eyes_closed",
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
        copy=True,
    )
    counts_open = split_and_copy(
        open_files, out_root, "eyes_open",
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed + 1,
        copy=True,
    )

    # ── Báo cáo ───────────────────────────────────────────────────────
    print("\n✅ Hoàn thành! Tổng kết:")
    print(f"\n{'Split':<8} {'eyes_closed':>12} {'eyes_open':>12} {'Total':>10}")
    print("-" * 44)
    for split in ["train", "val", "test"]:
        nc = counts_closed[split]
        no = counts_open[split]
        print(f"{split:<8} {nc:>12,} {no:>12,} {nc+no:>10,}")
    total = (counts_closed["train"] + counts_closed["val"] + counts_closed["test"] +
             counts_open["train"]   + counts_open["val"]   + counts_open["test"])
    print(f"\n  Tổng cộng: {total:,} ảnh")
    print(f"  Output   : {out_root.absolute()}")

    # ── Kiểm tra thực tế ──────────────────────────────────────────────
    print("\n🔍 Kiểm tra thực tế:")
    for split in ["train", "val", "test"]:
        for cls in ["eyes_closed", "eyes_open"]:
            d = out_root / split / cls
            n = count_images(d)
            print(f"  {split}/{cls}: {n:,} ảnh {'✅' if n > 0 else '❌'}")

    print(f"""
─────────────────────────────────────────────
Bước tiếp theo:
  python tools\\train_eye_classifier.py \\
    --data {out_root} \\
    --out app\\src\\main\\assets\\drowsiness_model.tflite \\
    --epochs 25

  python tools\\evaluate_eye_classifier.py \\
    --data {out_root} \\
    --tflite app\\src\\main\\assets\\drowsiness_model.tflite
─────────────────────────────────────────────
""")


if __name__ == "__main__":
    main()
