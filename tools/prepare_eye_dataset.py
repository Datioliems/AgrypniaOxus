"""
Prepare the eye-state dataset expected by the training scripts.

Default source:
  rawdata/data/
    train/awake   -> dataset/train/eyes_open
    train/sleepy  -> dataset/train/eyes_closed
    val/awake     -> dataset/val/eyes_open
    val/sleepy    -> dataset/val/eyes_closed
    test/awake    -> dataset/test/eyes_open
    test/sleepy   -> dataset/test/eyes_closed

Usage:
  python tools/prepare_eye_dataset.py --source rawdata/data --out dataset
  python tools/prepare_eye_dataset.py --source rawdata/data --out dataset --mode copy
  python tools/prepare_eye_dataset.py --source rawdata/data --out dataset --clean
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
LABEL_MAP = {
    "awake": "eyes_open",
    "sleepy": "eyes_closed",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="rawdata/data", help="Source root with train/val/test")
    parser.add_argument("--out", default="dataset", help="Output dataset root")
    parser.add_argument("--mode", choices=["link", "copy"], default="link")
    parser.add_argument("--clean", action="store_true", help="Remove output directory before preparing")
    args = parser.parse_args()

    source_root = Path(args.source)
    out_root = Path(args.out)

    if not source_root.exists():
        raise SystemExit(f"Source not found: {source_root}")

    if args.clean and out_root.exists():
        shutil.rmtree(out_root)

    total = 0
    for split in ["train", "val", "test"]:
        for source_label, target_label in LABEL_MAP.items():
            src_dir = source_root / split / source_label
            dst_dir = out_root / split / target_label
            if not src_dir.exists():
                raise SystemExit(f"Missing source folder: {src_dir}")

            dst_dir.mkdir(parents=True, exist_ok=True)
            count = prepare_folder(src_dir, dst_dir, args.mode)
            total += count
            print(f"{src_dir} -> {dst_dir}: {count} files")

    print(f"Prepared {total} image files in {out_root}")
    print("Next:")
    print(f"  python tools/summarize_dataset.py --data {out_root} --out outputs/dataset_summary")


def prepare_folder(src_dir: Path, dst_dir: Path, mode: str) -> int:
    count = 0
    for src in sorted(src_dir.iterdir()):
        if not src.is_file() or src.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        dst = dst_dir / src.name
        if dst.exists():
            count += 1
            continue
        if mode == "link":
            try:
                os.link(src, dst)
            except OSError:
                shutil.copy2(src, dst)
        else:
            shutil.copy2(src, dst)
        count += 1
    return count


if __name__ == "__main__":
    main()
