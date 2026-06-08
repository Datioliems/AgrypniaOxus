"""
data_pipeline/preprocess.py
===========================
Toan bo code XU LY DU LIEU cho model CNN phan loai mat (eyes_closed / eyes_open).
Doc tham so tu best_params.json (cung folder).

Gom 2 phan:
  A. CHUAN BI DATASET  : lam sach anh -> chia train/val/test -> sap xep thu muc
  B. PIPELINE TRAIN    : load + resize + normalize(/255) + augmentation (tf.data)

Cach dung
---------
# 1) Chuan bi dataset tu anh goc (Open-Eyes / Close-Eyes) -> dataset/
python data_pipeline/preprocess.py --prepare --src-root . --out dataset

# 2) Trong code train (vd Colab), import de lay tf.data datasets:
from data_pipeline.preprocess import load_params, make_tf_datasets
params = load_params()
train_ds, val_ds, test_ds, class_names = make_tf_datasets("dataset", params)
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
PARAMS_PATH = HERE / "best_params.json"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ---------------------------------------------------------------------------
# Tham so
# ---------------------------------------------------------------------------
def load_params(path: str | Path = PARAMS_PATH) -> dict:
    """Doc bo tham so tot nhat tu best_params.json."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ===========================================================================
# A. CHUAN BI DATASET (lam sach -> chia -> sap xep)
# ===========================================================================
def list_valid_images(folder: Path) -> list[Path]:
    """Loc anh hop le, bo file hong / 0 byte."""
    from PIL import Image

    folder = Path(folder)
    good, bad = [], 0
    for p in folder.iterdir():
        if not p.is_file() or p.suffix.lower() not in IMAGE_EXTS:
            continue
        try:
            if p.stat().st_size == 0:
                bad += 1
                continue
            with Image.open(p) as im:
                im.verify()
            good.append(p)
        except Exception:
            bad += 1
    print(f"  {folder}: {len(good)} anh OK, {bad} bo qua")
    return good


def split_list(items: list[Path], ratios: dict, seed: int) -> dict:
    """Tron ngau nhien + chia train/val/test."""
    items = list(items)
    random.Random(seed).shuffle(items)
    n = len(items)
    n_tr = int(n * ratios["train"])
    n_va = int(n * ratios["val"])
    return {
        "train": items[:n_tr],
        "val": items[n_tr:n_tr + n_va],
        "test": items[n_tr + n_va:],
    }


def prepare_dataset(src_root: str | Path, out: str | Path, params: dict) -> None:
    """Tao dataset/train|val|test/<class> tu anh goc co nhan theo thu muc."""
    src_root = Path(src_root)
    out = Path(out)
    sources = params["sources"]            # {label: relative_path}
    ratios = params["split"]
    seed = ratios.get("seed", 42)

    print("== A1. Quet & lam sach ==")
    clean = {label: list_valid_images(src_root / rel) for label, rel in sources.items()}

    print("\n== A2. Chia train/val/test ==")
    splits = {label: split_list(files, ratios, seed) for label, files in clean.items()}

    print("\n== A3. Sap xep vao", out, "==")
    if out.exists():
        shutil.rmtree(out)
    total = 0
    for label, parts in splits.items():
        for split_name, files in parts.items():
            dst = out / split_name / label
            dst.mkdir(parents=True, exist_ok=True)
            for src in files:
                shutil.copy2(src, dst / src.name)
            total += len(files)
            print(f"  {split_name}/{label}: {len(files)}")
    print(f"Tong: {total} anh -> {out}")

    print("\n== A4. Thong ke ==")
    for split_name in ["train", "val", "test"]:
        for label in sources:
            n = len(list((out / split_name / label).glob("*")))
            print(f"  {split_name}/{label}: {n}")


# ===========================================================================
# B. PIPELINE TRAIN (resize + normalize + augmentation)
# ===========================================================================
def make_tf_datasets(data_root: str | Path, params: dict):
    """Tra ve (train_ds, val_ds, test_ds, class_names) da normalize + augment."""
    import tensorflow as tf

    data_root = Path(data_root)
    pp = params["preprocess"]
    img = pp["image_size"]
    batch = params["train"]["batch_size"]
    AUTOTUNE = tf.data.AUTOTUNE

    def load(split, shuffle):
        return tf.keras.utils.image_dataset_from_directory(
            data_root / split, image_size=(img, img), batch_size=batch,
            label_mode="categorical", shuffle=shuffle)

    train_ds = load("train", True)
    val_ds = load("val", False)
    test_ds = load("test", False)
    class_names = train_ds.class_names
    print("Classes:", class_names)

    # Chuan hoa /255 trong pipeline (KHONG dung Rescaling trong model)
    norm = lambda x, y: (tf.cast(x, tf.float32) / 255.0, y)
    train_ds = train_ds.map(norm, num_parallel_calls=AUTOTUNE)
    val_ds = val_ds.map(norm, num_parallel_calls=AUTOTUNE)
    test_ds = test_ds.map(norm, num_parallel_calls=AUTOTUNE)

    # Augmentation nhe, chi cho train
    a = pp["augmentation"]
    aug = tf.keras.Sequential([
        tf.keras.layers.RandomFlip(a["random_flip"]),
        tf.keras.layers.RandomRotation(a["random_rotation"]),
        tf.keras.layers.RandomZoom(a["random_zoom"]),
        tf.keras.layers.RandomContrast(a["random_contrast"]),
    ])
    train_ds = train_ds.map(lambda x, y: (aug(x, training=True), y),
                            num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)
    val_ds = val_ds.prefetch(AUTOTUNE)
    test_ds = test_ds.prefetch(AUTOTUNE)
    return train_ds, val_ds, test_ds, class_names


# ===========================================================================
# CLI
# ===========================================================================
def main() -> None:
    ap = argparse.ArgumentParser(description="Xu ly du lieu CNN phan loai mat")
    ap.add_argument("--prepare", action="store_true", help="Chuan bi dataset tu anh goc")
    ap.add_argument("--src-root", default=".", help="Thu muc goc chua mrleyedataset/...")
    ap.add_argument("--out", default="dataset", help="Thu muc dataset dau ra")
    ap.add_argument("--params", default=str(PARAMS_PATH))
    args = ap.parse_args()

    params = load_params(args.params)
    if args.prepare:
        prepare_dataset(args.src_root, args.out, params)
    else:
        print("Dung --prepare de chuan bi dataset, hoac import make_tf_datasets() trong code train.")
        print("Tham so hien tai:", json.dumps(params["preprocess"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
