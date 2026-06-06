# %% [markdown]
"""
# 🚗 DROWSY DRIVER DETECTION — MULTI-MODEL NOTEBOOK
## Google Colab / VS Code Jupyter

### 3 Models:
| Model | Data | Deploy |
|-------|------|--------|
| A: CNN Eye      | MRL 84K ảnh (mrleyedataset)     | ✅ Android TFLite |
| B: CNN Yawn     | Yawn 5K ảnh (rawdata/data/yawn) | ✅ Android TFLite |
| C: YOLOv8       | Roboflow drowsiness driver      | 📊 Colab demo     |

### Pipeline tổng thể:
```
Dataset Survey → Download Roboflow → Train CNN Eye → Train CNN Yawn
    → Train YOLOv8 → So sánh kết quả → Export TFLite → Android
```
"""

# %% ──────────────────────────────────────────────────────────────────
# CELL 0 ▸ DETECT ENVIRONMENT
# ─────────────────────────────────────────────────────────────────────
import sys
IN_COLAB = "google.colab" in sys.modules
print(f"Môi trường: {'Google Colab' if IN_COLAB else 'VS Code / Local'}")

# %% ──────────────────────────────────────────────────────────────────
# CELL 1 ▸ CÀI ĐẶT THƯ VIỆN
# ─────────────────────────────────────────────────────────────────────
if IN_COLAB:
    import subprocess
    subprocess.run(["pip", "install", "-q",
        "tensorflow>=2.15", "mediapipe>=0.10",
        "opencv-python-headless", "matplotlib", "seaborn",
        "scikit-learn", "pillow", "gradio", "pandas", "numpy",
        "roboflow", "ultralytics",   # YOLOv8
    ], check=True)
    print("✅ Đã cài đặt (Colab)")
else:
    # Chạy lệnh này trong Terminal VS Code:
    # pip install tensorflow mediapipe opencv-python matplotlib seaborn
    #     scikit-learn pillow gradio pandas numpy roboflow ultralytics
    print("✅ Đang chạy local — đảm bảo đã cài packages")

# %% ──────────────────────────────────────────────────────────────────
# CELL 2 ▸ IMPORT & THIẾT LẬP ĐƯỜNG DẪN
# ─────────────────────────────────────────────────────────────────────
import os, json, math, shutil, warnings, random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path
from PIL import Image

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")
plt.style.use("seaborn-v0_8-darkgrid")

# ── Cấu hình đường dẫn ────────────────────────────────────────────────
if IN_COLAB:
    from google.colab import drive
    drive.mount("/content/drive", force_remount=False)
    PROJECT_ROOT = Path("/content/drive/MyDrive/DrowsyDriverAndroid")
else:
    PROJECT_ROOT = Path(".")   # chạy từ D:\2026.AI\DrowsyDriverAndroid

RAWDATA     = PROJECT_ROOT / "rawdata"
MRLEYE      = PROJECT_ROOT / "mrleyedataset"
OUTPUTS     = PROJECT_ROOT / "outputs"
DATASET_EYE  = PROJECT_ROOT / "dataset_mrl"    # Model A — CNN Eye
DATASET_YAWN = PROJECT_ROOT / "dataset_yawn"   # Model B — CNN Yawn
ROBOFLOW_DIR = PROJECT_ROOT / "roboflow_data"  # Model C — YOLOv8

for d in [OUTPUTS/"training_eye", OUTPUTS/"training_yawn",
          OUTPUTS/"training_yolo", OUTPUTS/"evaluation"]:
    d.mkdir(parents=True, exist_ok=True)

print(f"PROJECT_ROOT: {PROJECT_ROOT.absolute()}")

# %% ──────────────────────────────────────────────────────────────────
# CELL 3 ▸ KIỂM KÊ TOÀN BỘ DỮ LIỆU
# ─────────────────────────────────────────────────────────────────────
def count_images(folder: Path) -> int:
    if not folder.exists():
        return 0
    return sum(1 for f in folder.rglob("*")
               if f.suffix.lower() in {".jpg",".jpeg",".png",".bmp",".pgm"})

datasets = {
    "rawdata/closed_eye":          RAWDATA / "closed_eye",
    "rawdata/open_eye":            RAWDATA / "open_eye",
    "rawdata/data/eyes/train":     RAWDATA / "data" / "eyes" / "train",
    "rawdata/data/eyes/val":       RAWDATA / "data" / "eyes" / "val",
    "rawdata/data/eyes/test":      RAWDATA / "data" / "eyes" / "test",
    "rawdata/data/yawn/yawn":      RAWDATA / "data" / "yawn" / "yawn",
    "rawdata/data/yawn/no yawn":   RAWDATA / "data" / "yawn" / "no yawn",
    "rawdata/data/train/awake":    RAWDATA / "data" / "train" / "awake",
    "rawdata/data/train/sleepy":   RAWDATA / "data" / "train" / "sleepy",
    "mrleyedataset/Close-Eyes":    MRLEYE / "Close-Eyes",
    "mrleyedataset/Open-Eyes":     MRLEYE / "Open-Eyes",
}

print("=" * 60)
print("📦  KIỂM KÊ DỮ LIỆU")
print("=" * 60)
rows = []
for name, path in datasets.items():
    n = count_images(path)
    exists = "✅" if path.exists() else "❌"
    print(f"  {exists} {name:<38} {n:>8,} ảnh")
    rows.append({"dataset": name, "count": n, "exists": path.exists()})

df_inv = pd.DataFrame(rows)
total_unique = (
    df_inv[df_inv.dataset.str.contains("mrleye|yawn")]["count"].sum()
)
print(f"\n  Tổng (MRL eye + yawn): ~{total_unique:,} ảnh")
print(f"\n  Roboflow datasets: cần download ở Cell 5")

# %% ──────────────────────────────────────────────────────────────────
# CELL 4 ▸ VISUALIZE KIỂM KÊ DATASET
# ─────────────────────────────────────────────────────────────────────
df_plot = df_inv[df_inv["exists"]].copy()
df_plot["category"] = df_plot["dataset"].apply(lambda x:
    "MRL Eye" if "mrleye" in x else
    "Yawn"    if "yawn"   in x else
    "Awake/Sleepy" if ("awake" in x or "sleepy" in x) else
    "Eyes Split"
)

fig, ax = plt.subplots(figsize=(12, 5))
colors = {"MRL Eye": "#4ECDC4", "Yawn": "#FF6B6B",
          "Awake/Sleepy": "#95E1D3", "Eyes Split": "#F8B500"}
bar_colors = [colors[c] for c in df_plot["category"]]
bars = ax.barh(df_plot["dataset"], df_plot["count"],
               color=bar_colors, edgecolor="white")
ax.set_xlabel("Số lượng ảnh")
ax.set_title("Kiểm kê dataset", fontweight="bold", fontsize=13)
for bar, val in zip(bars, df_plot["count"]):
    if val > 0:
        ax.text(bar.get_width() + 200, bar.get_y() + bar.get_height()/2,
                f"{val:,}", va="center", fontsize=9)
patches = [mpatches.Patch(color=v, label=k) for k, v in colors.items()]
ax.legend(handles=patches)
plt.tight_layout()
plt.savefig(OUTPUTS / "eda_dataset_inventory.png", bbox_inches="tight", dpi=150)
plt.show()
print("✅ Saved → outputs/eda_dataset_inventory.png")

# %% ──────────────────────────────────────────────────────────────────
# CELL 5 ▸ DOWNLOAD ROBOFLOW DATASET (YOLOv8)
# ─────────────────────────────────────────────────────────────────────
"""
Download Roboflow "drowsiness driver" dataset cho YOLOv8 training.
API key và project info đã được cung cấp.
"""
ROBOFLOW_DIR.mkdir(parents=True, exist_ok=True)

try:
    from roboflow import Roboflow

    rf = Roboflow(api_key="qI3lEKlNpIZpNENdk3MH")
    project = rf.workspace("nguyen-tuan-dat").project("drowsiness-driver")

    print("📥 Downloading YOLOv8 format dataset...")
    dataset_yolo = project.version(1).download(
        "yolov8",
        location=str(ROBOFLOW_DIR / "yolov8"),
        overwrite=False
    )
    print(f"✅ YOLOv8 dataset saved: {dataset_yolo.location}")

    # Đọc data.yaml để xem classes
    yaml_path = Path(dataset_yolo.location) / "data.yaml"
    if yaml_path.exists():
        import yaml
        with open(yaml_path) as f:
            data_cfg = yaml.safe_load(f)
        print(f"\nYOLOv8 Classes: {data_cfg.get('names', 'N/A')}")
        print(f"nc (num classes): {data_cfg.get('nc', 'N/A')}")

except ImportError:
    print("⚠️  roboflow chưa được cài. Chạy: pip install roboflow")
except Exception as e:
    print(f"⚠️  Không tải được Roboflow: {e}")
    print("   Tiếp tục với CNN Eye + Yawn vẫn được")

# %% ──────────────────────────────────────────────────────────────────
# CELL 6 ▸ CHUẨN BỊ DATASET A — CNN EYE
# ─────────────────────────────────────────────────────────────────────
"""
Dùng mrleyedataset (84,898 ảnh) — nhãn TRỰC TIẾP mắt mở/nhắm.
Chia 70% train / 15% val / 15% test
"""
import tensorflow as tf
print(f"TensorFlow: {tf.__version__}")

def prepare_split_dataset(src_map: dict, out_root: Path,
                           train=0.70, val=0.15, seed=42,
                           max_per_class=0):
    """
    src_map: {class_name: source_folder}
    Chia train/val/test và copy ảnh.
    """
    if out_root.exists():
        shutil.rmtree(out_root)
    random.seed(seed)
    stats = {}
    for cls_name, src_dir in src_map.items():
        files = sorted(f for f in Path(src_dir).rglob("*")
                       if f.suffix.lower() in {".jpg",".jpeg",".png",".bmp",".pgm"})
        if max_per_class > 0:
            files = random.sample(files, min(max_per_class, len(files)))
        random.shuffle(files)
        n = len(files)
        n_tr = int(n * train)
        n_vl = int(n * val)
        splits = {"train": files[:n_tr],
                  "val":   files[n_tr:n_tr+n_vl],
                  "test":  files[n_tr+n_vl:]}
        stats[cls_name] = {}
        for split_name, split_files in splits.items():
            dest = out_root / split_name / cls_name
            dest.mkdir(parents=True, exist_ok=True)
            for f in split_files:
                shutil.copy2(f, dest / f.name)
            stats[cls_name][split_name] = len(split_files)
        print(f"  {cls_name}: "
              f"train={stats[cls_name]['train']:,}  "
              f"val={stats[cls_name]['val']:,}  "
              f"test={stats[cls_name]['test']:,}")
    return stats

print("=" * 55)
print("📦  CHUẨN BỊ DATASET A — CNN EYE")
print("=" * 55)

# Dùng mrleyedataset — nhiều ảnh nhất, nhãn trực tiếp
eye_src = {
    "eyes_closed": MRLEYE / "Close-Eyes",
    "eyes_open":   MRLEYE / "Open-Eyes",
}
# Nếu không có mrleyedataset, fallback về rawdata
if not (MRLEYE / "Close-Eyes").exists():
    eye_src = {
        "eyes_closed": RAWDATA / "closed_eye",
        "eyes_open":   RAWDATA / "open_eye",
    }
    print("  (dùng rawdata/closed_eye + open_eye thay vì mrleyedataset)")

# Giới hạn 30K/class để cân bằng và train nhanh (tăng lên 0 nếu muốn full)
MAX_EYE = 30_000
stats_eye = prepare_split_dataset(eye_src, DATASET_EYE,
                                   max_per_class=MAX_EYE, seed=42)

total_eye = sum(v for cls in stats_eye.values() for v in cls.values())
print(f"\nTổng: {total_eye:,} ảnh → {DATASET_EYE}")

# %% ──────────────────────────────────────────────────────────────────
# CELL 7 ▸ CHUẨN BỊ DATASET B — CNN YAWN
# ─────────────────────────────────────────────────────────────────────
"""
Dùng rawdata/data/yawn (5,119 ảnh) — nhãn trực tiếp yawn/no yawn.
Dataset nhỏ → train nhanh, cần augmentation mạnh hơn.
"""
print("=" * 55)
print("📦  CHUẨN BỊ DATASET B — CNN YAWN")
print("=" * 55)

yawn_src_dir = RAWDATA / "data" / "yawn"
yawn_src = {}

# Tự động detect tên subfolder (có thể là "yawn"/"no yawn" hoặc tên khác)
if yawn_src_dir.exists():
    subfolders = [d for d in yawn_src_dir.iterdir() if d.is_dir()]
    print(f"  Subfolders tìm thấy: {[d.name for d in subfolders]}")
    for d in subfolders:
        name_lower = d.name.lower()
        if "no" in name_lower or "normal" in name_lower:
            yawn_src["no_yawn"] = d
        elif "yawn" in name_lower:
            yawn_src["yawn"] = d

    if len(yawn_src) == 2:
        stats_yawn = prepare_split_dataset(yawn_src, DATASET_YAWN, seed=42)
        total_yawn = sum(v for cls in stats_yawn.values() for v in cls.values())
        print(f"\nTổng: {total_yawn:,} ảnh → {DATASET_YAWN}")
        print("⚠️  Dataset nhỏ (5K) → cần augmentation mạnh khi train")
    else:
        print(f"⚠️  Không xác định được class trong {yawn_src_dir}")
        print(f"   Folders: {[d.name for d in subfolders]}")
else:
    print(f"❌ Không tìm thấy yawn dataset tại {yawn_src_dir}")

# %% ──────────────────────────────────────────────────────────────────
# CELL 8 ▸ VISUALIZE ẢNH MẪU — TẤT CẢ DATASET
# ─────────────────────────────────────────────────────────────────────
def show_sample_grid(dataset_root: Path, title: str, n_cols=6):
    """Hiển thị ảnh mẫu từ từng class trong dataset"""
    class_dirs = sorted([d for d in (dataset_root/"train").iterdir() if d.is_dir()])
    if not class_dirs:
        print(f"⚠️  Không tìm thấy classes trong {dataset_root}/train")
        return
    n_rows = len(class_dirs)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*2.5, n_rows*2.5))
    if n_rows == 1:
        axes = [axes]
    fig.suptitle(title, fontsize=13, fontweight="bold")
    for row, cls_dir in enumerate(class_dirs):
        imgs = sorted(cls_dir.glob("*.jpg"))[:n_cols] or \
               sorted(cls_dir.glob("*.png"))[:n_cols]
        for col, img_path in enumerate(imgs[:n_cols]):
            ax = axes[row][col] if n_rows > 1 else axes[col]
            try:
                img = Image.open(img_path).resize((64,64)).convert("RGB")
                ax.imshow(img)
            except Exception:
                ax.set_facecolor("gray")
            ax.axis("off")
            if col == 0:
                ax.set_title(cls_dir.name, fontsize=9, fontweight="bold", pad=3)
    plt.tight_layout()
    plt.show()

if DATASET_EYE.exists():
    show_sample_grid(DATASET_EYE, "Dataset A: CNN Eye (mrleyedataset)")
if DATASET_YAWN.exists():
    show_sample_grid(DATASET_YAWN, "Dataset B: CNN Yawn")

# %% ──────────────────────────────────────────────────────────────────
# CELL 9 ▸ HELPER: BUILD CNN MODEL
# ─────────────────────────────────────────────────────────────────────
def normalize_batch(images, labels):
    """Normalize [0,255] → [0.0, 1.0] — NGOÀI model để TFLite không bị kép"""
    return tf.cast(images, tf.float32) / 255.0, labels


def build_cnn(image_size: int, num_classes: int, name: str = "DrowsyCNN"):
    """
    CNN 3-block + BatchNorm:
      Conv32 → BN → MaxPool
      Conv64 → BN → MaxPool
      Conv128 → BN → GAP
      Dropout(0.30) → Dense(num_classes, softmax)
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(image_size, image_size, 3)),
        tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(128, 3, padding="same", activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dropout(0.30),
        tf.keras.layers.Dense(num_classes, activation="softmax"),
    ], name=name)
    return model


def train_cnn(dataset_root: Path,
              model_name: str,
              out_dir: Path,
              tflite_path: Path,
              epochs: int = 25,
              image_size: int = 64,
              batch_size: int = 32,
              lr: float = 1e-3,
              augment: bool = True):
    """
    Pipeline đầy đủ: load data → train → export TFLite.
    Trả về (model, history, class_names).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    img_size = (image_size, image_size)

    # ── Load datasets ─────────────────────────────────────────────────
    train_ds = tf.keras.utils.image_dataset_from_directory(
        dataset_root / "train", image_size=img_size,
        batch_size=batch_size, label_mode="categorical", seed=42)
    val_ds   = tf.keras.utils.image_dataset_from_directory(
        dataset_root / "val",   image_size=img_size,
        batch_size=batch_size, label_mode="categorical", shuffle=False)

    class_names = train_ds.class_names
    num_classes = len(class_names)
    print(f"  Classes: {class_names}")

    # Lưu class_names để Android biết thứ tự
    (out_dir / "class_names.json").write_text(
        json.dumps(class_names, indent=2), encoding="utf-8")

    # ── Optional augmentation layer ──────────────────────────────────
    if augment:
        aug = tf.keras.Sequential([
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.08),
            tf.keras.layers.RandomZoom(0.10),
            tf.keras.layers.RandomContrast(0.20),
        ])
        train_ds = (train_ds
                    .map(lambda x, y: (aug(x, training=True), y),
                         num_parallel_calls=tf.data.AUTOTUNE)
                    .map(normalize_batch, num_parallel_calls=tf.data.AUTOTUNE)
                    .prefetch(tf.data.AUTOTUNE))
    else:
        train_ds = (train_ds
                    .map(normalize_batch, num_parallel_calls=tf.data.AUTOTUNE)
                    .prefetch(tf.data.AUTOTUNE))

    val_ds = (val_ds
              .map(normalize_batch, num_parallel_calls=tf.data.AUTOTUNE)
              .prefetch(tf.data.AUTOTUNE))

    # ── Model ─────────────────────────────────────────────────────────
    model = build_cnn(image_size, num_classes, name=model_name)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr),
        loss="categorical_crossentropy",
        metrics=["accuracy",
                 tf.keras.metrics.Precision(name="precision"),
                 tf.keras.metrics.Recall(name="recall")])
    model.summary()

    # ── Callbacks ─────────────────────────────────────────────────────
    cbs = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=5,
            restore_best_weights=True, verbose=1, min_delta=0.001),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=2,
            min_lr=1e-7, verbose=1),
        tf.keras.callbacks.ModelCheckpoint(
            str(out_dir / "best_model.keras"),
            monitor="val_accuracy", save_best_only=True, verbose=1),
    ]

    # ── Train ─────────────────────────────────────────────────────────
    history = model.fit(train_ds, validation_data=val_ds,
                        epochs=epochs, callbacks=cbs, verbose=2)
    epochs_ran = len(history.history["accuracy"])
    best_acc   = max(history.history["val_accuracy"])
    print(f"\n✅ {model_name}: {epochs_ran} epochs, best val_acc={best_acc:.4f}")

    # ── Save ──────────────────────────────────────────────────────────
    keras_path = out_dir / f"{model_name}.keras"
    model.save(str(keras_path))
    (out_dir / "training_history.json").write_text(
        json.dumps({k:[float(v) for v in vals]
                    for k,vals in history.history.items()}, indent=2),
        encoding="utf-8")

    # ── Export TFLite ─────────────────────────────────────────────────
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    tflite_path.parent.mkdir(parents=True, exist_ok=True)
    tflite_path.write_bytes(tflite_model)
    kb = tflite_path.stat().st_size / 1024
    print(f"✅ TFLite: {tflite_path} ({kb:.1f} KB)")

    return model, history, class_names


print("✅ Helper functions sẵn sàng: normalize_batch, build_cnn, train_cnn")

# %% ──────────────────────────────────────────────────────────────────
# CELL 10 ▸ TRAIN MODEL A — CNN EYE
# ─────────────────────────────────────────────────────────────────────
print("=" * 55)
print("🏋️  TRAIN MODEL A — CNN EYE")
print("=" * 55)

TFLITE_EYE = PROJECT_ROOT / "app/src/main/assets/drowsiness_model.tflite"

model_eye, hist_eye, classes_eye = train_cnn(
    dataset_root = DATASET_EYE,
    model_name   = "cnn_eye",
    out_dir      = OUTPUTS / "training_eye",
    tflite_path  = TFLITE_EYE,
    epochs       = 25,
    image_size   = 64,
    augment      = True,
)

print(f"\nClass order: {classes_eye}")
print("⚠️  Android labels array phải đúng thứ tự này!")

# %% ──────────────────────────────────────────────────────────────────
# CELL 11 ▸ TRAIN MODEL B — CNN YAWN
# ─────────────────────────────────────────────────────────────────────
print("=" * 55)
print("🏋️  TRAIN MODEL B — CNN YAWN")
print("=" * 55)

TFLITE_YAWN = PROJECT_ROOT / "app/src/main/assets/yawn_model.tflite"

if DATASET_YAWN.exists() and any((DATASET_YAWN/"train").iterdir()):
    model_yawn, hist_yawn, classes_yawn = train_cnn(
        dataset_root = DATASET_YAWN,
        model_name   = "cnn_yawn",
        out_dir      = OUTPUTS / "training_yawn",
        tflite_path  = TFLITE_YAWN,
        epochs       = 30,         # nhiều epoch hơn vì dataset nhỏ
        image_size   = 64,
        augment      = True,       # augmentation mạnh vì ít data
    )
    print(f"\nYawn class order: {classes_yawn}")
else:
    print("⚠️  Dataset yawn chưa sẵn sàng, bỏ qua Model B")
    model_yawn = None

# %% ──────────────────────────────────────────────────────────────────
# CELL 12 ▸ PLOT TRAINING CURVES — MODEL A + B
# ─────────────────────────────────────────────────────────────────────
def plot_training_curves(history, title: str, save_path: Path):
    hist = history.history
    epochs_ran = range(1, len(hist["accuracy"]) + 1)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(title, fontsize=15, fontweight="bold")
    plot_cfg = [
        ("accuracy",  "val_accuracy",  "Accuracy",  "#2196F3"),
        ("loss",      "val_loss",      "Loss",       "#F44336"),
        ("precision", "val_precision", "Precision",  "#4CAF50"),
        ("recall",    "val_recall",    "Recall",     "#FF9800"),
    ]
    for ax, (tr, vl, lbl, clr) in zip(axes.flat, plot_cfg):
        ax.plot(epochs_ran, hist[tr],  "o-", label="Train", color=clr, lw=2, ms=4)
        if vl in hist:
            ax.plot(epochs_ran, hist[vl], "s--", label="Val", color=clr, alpha=0.6, lw=2, ms=4)
        ax.set_title(lbl, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.legend()
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.show()
    print(f"✅ Saved: {save_path}")

plot_training_curves(hist_eye, "Training Curves — Model A (CNN Eye)",
                     OUTPUTS / "training_eye" / "training_curves.png")
if model_yawn:
    plot_training_curves(hist_yawn, "Training Curves — Model B (CNN Yawn)",
                         OUTPUTS / "training_yawn" / "training_curves.png")

# %% ──────────────────────────────────────────────────────────────────
# CELL 13 ▸ EVALUATE MODEL A — CNN EYE
# ─────────────────────────────────────────────────────────────────────
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             precision_recall_fscore_support,
                             classification_report)

def evaluate_model(model, dataset_root: Path, class_names: list,
                   title: str, save_dir: Path, image_size: int = 64):
    """Evaluate trên test set, vẽ confusion matrix, lưu metrics"""
    save_dir.mkdir(parents=True, exist_ok=True)
    test_ds = tf.keras.utils.image_dataset_from_directory(
        dataset_root / "test", image_size=(image_size, image_size),
        batch_size=32, label_mode="categorical", shuffle=False)
    test_ds = test_ds.map(normalize_batch).prefetch(tf.data.AUTOTUNE)

    y_pred_probs = model.predict(test_ds, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = np.concatenate([np.argmax(y.numpy(), axis=1) for _, y in test_ds])

    acc  = accuracy_score(y_true, y_pred)
    prec, rec, f1, sup = precision_recall_fscore_support(
        y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    print(f"\n{'='*50}")
    print(f"  {title}")
    print(f"  Accuracy: {acc:.4f} ({acc*100:.2f}%)")
    print(f"{'='*50}")
    print(classification_report(y_true, y_pred, target_names=class_names))

    # Confusion matrix
    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names,
                linewidths=2, linecolor="white",
                annot_kws={"size": 16, "weight": "bold"}, ax=ax)
    ax.set_title(f"Confusion Matrix — {title}", fontsize=13, fontweight="bold", pad=15)
    ax.set_ylabel("True Label")
    ax.set_xlabel("Predicted Label")
    ax.text(0.98, 0.02, f"Acc: {acc*100:.2f}%",
            transform=ax.transAxes, fontsize=12, color="darkgreen",
            ha="right", va="bottom",
            bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.8))
    plt.tight_layout()
    plt.savefig(save_dir / "confusion_matrix.png", bbox_inches="tight", dpi=150)
    plt.show()

    # Lưu metrics
    metrics = {
        "accuracy": float(acc),
        "per_class": [
            {"class": class_names[i], "precision": float(prec[i]),
             "recall": float(rec[i]), "f1": float(f1[i]),
             "support": int(sup[i])}
            for i in range(len(class_names))
        ],
        "confusion_matrix": cm.tolist()
    }
    (save_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8")

    return metrics


print("Đánh giá Model A (CNN Eye)...")
metrics_eye = evaluate_model(
    model_eye, DATASET_EYE, classes_eye,
    "CNN Eye", OUTPUTS / "evaluation" / "cnn_eye")

# %% ──────────────────────────────────────────────────────────────────
# CELL 14 ▸ EVALUATE MODEL B — CNN YAWN
# ─────────────────────────────────────────────────────────────────────
if model_yawn:
    print("Đánh giá Model B (CNN Yawn)...")
    metrics_yawn = evaluate_model(
        model_yawn, DATASET_YAWN, classes_yawn,
        "CNN Yawn", OUTPUTS / "evaluation" / "cnn_yawn")
else:
    print("⚠️  Model Yawn chưa được train, bỏ qua đánh giá")

# %% ──────────────────────────────────────────────────────────────────
# CELL 15 ▸ TRAIN MODEL C — YOLOV8
# ─────────────────────────────────────────────────────────────────────
"""
Train YOLOv8 với Roboflow drowsiness dataset.
⚠️  Cần GPU (Colab T4 ~20-40 phút / VS Code với CUDA)
Nếu không có GPU, bỏ qua cell này và dùng Model A+B.
"""
print("=" * 55)
print("🎯  TRAIN MODEL C — YOLOV8")
print("=" * 55)

yolo_data_yaml = ROBOFLOW_DIR / "yolov8" / "data.yaml"

if not yolo_data_yaml.exists():
    print("⚠️  Chưa có Roboflow dataset (chạy Cell 5 trước)")
    print("   Bỏ qua YOLOv8 training, tiếp tục với CNN models")
else:
    try:
        from ultralytics import YOLO

        # Dùng YOLOv8 nano (nhỏ nhất, phù hợp dataset nhỏ)
        yolo_model = YOLO("yolov8n.pt")

        results = yolo_model.train(
            data   = str(yolo_data_yaml),
            epochs = 50,
            imgsz  = 640,
            batch  = 16,
            name   = "drowsy_yolov8n",
            project= str(OUTPUTS / "training_yolo"),
            device = 0 if tf.config.list_physical_devices("GPU") else "cpu",
            patience = 10,
            save    = True,
            verbose = True,
        )

        print(f"✅ YOLOv8 training xong!")
        print(f"   Results: {results.save_dir}")

        # Export sang TFLite (tùy chọn — model nặng hơn CNN)
        # yolo_model.export(format="tflite")

    except ImportError:
        print("⚠️  ultralytics chưa được cài. Chạy: pip install ultralytics")
    except Exception as e:
        print(f"⚠️  YOLOv8 training failed: {e}")

# %% ──────────────────────────────────────────────────────────────────
# CELL 16 ▸ SO SÁNH KẾT QUẢ 3 MODEL
# ─────────────────────────────────────────────────────────────────────
print("=" * 60)
print("📊  SO SÁNH KẾT QUẢ")
print("=" * 60)

comparison = []

# Model A — CNN Eye
acc_eye = metrics_eye["accuracy"]
comparison.append({
    "Model": "CNN Eye (Model A)",
    "Task": "eyes_closed vs eyes_open",
    "Accuracy": f"{acc_eye*100:.2f}%",
    "F1 closed": f"{metrics_eye['per_class'][0]['f1']:.4f}",
    "F1 open":   f"{metrics_eye['per_class'][1]['f1']:.4f}",
    "TFLite KB": f"{TFLITE_EYE.stat().st_size/1024:.1f}" if TFLITE_EYE.exists() else "N/A",
    "Deploy":    "✅ Android",
})

# Model B — CNN Yawn
if model_yawn:
    acc_yawn = metrics_yawn["accuracy"]
    yawn_idx = next((i for i,c in enumerate(classes_yawn) if "yawn" in c and "no" not in c), 1)
    no_yawn_idx = 1 - yawn_idx
    comparison.append({
        "Model": "CNN Yawn (Model B)",
        "Task": "yawn vs no_yawn",
        "Accuracy": f"{acc_yawn*100:.2f}%",
        "F1 closed": f"{metrics_yawn['per_class'][yawn_idx]['f1']:.4f}",
        "F1 open":   f"{metrics_yawn['per_class'][no_yawn_idx]['f1']:.4f}",
        "TFLite KB": f"{TFLITE_YAWN.stat().st_size/1024:.1f}" if TFLITE_YAWN.exists() else "N/A",
        "Deploy":    "✅ Android",
    })

# Model C — YOLOv8
yolo_results_dir = OUTPUTS / "training_yolo"
yolo_results_csv = next(yolo_results_dir.rglob("results.csv"), None) if yolo_results_dir.exists() else None
if yolo_results_csv:
    df_yolo = pd.read_csv(yolo_results_csv)
    best_map50 = df_yolo["metrics/mAP50(B)"].max() if "metrics/mAP50(B)" in df_yolo.columns else 0
    comparison.append({
        "Model": "YOLOv8n (Model C)",
        "Task": "face drowsiness detect",
        "Accuracy": f"mAP50={best_map50:.3f}",
        "F1 closed": "N/A",
        "F1 open": "N/A",
        "TFLite KB": "~3500 (quá lớn cho Android)",
        "Deploy":    "📊 Colab demo only",
    })

df_compare = pd.DataFrame(comparison)
print(df_compare.to_string(index=False))

# Visualize accuracy comparison
fig, ax = plt.subplots(figsize=(10, 5))
model_names = [r["Model"] for r in comparison if "%" in r["Accuracy"]]
acc_values  = [float(r["Accuracy"].replace("%","")) for r in comparison if "%" in r["Accuracy"]]
colors_bar  = ["#4ECDC4", "#FF6B6B"][:len(model_names)]
bars = ax.bar(model_names, acc_values, color=colors_bar, width=0.4,
              edgecolor="white", linewidth=2)
ax.set_ylim(0, 105)
ax.set_ylabel("Accuracy (%)")
ax.set_title("So sánh Accuracy — CNN Eye vs CNN Yawn", fontweight="bold")
for bar, val in zip(bars, acc_values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f"{val:.2f}%", ha="center", fontweight="bold", fontsize=12)
plt.tight_layout()
plt.savefig(OUTPUTS / "model_comparison.png", bbox_inches="tight", dpi=150)
plt.show()

# %% ──────────────────────────────────────────────────────────────────
# CELL 17 ▸ VERIFY TFLITE — KIỂM TRA TRƯỚC KHI ĐƯA LÊN ANDROID
# ─────────────────────────────────────────────────────────────────────
def verify_tflite(tflite_path: Path, dataset_root: Path,
                  class_names: list, n_per_class: int = 100):
    """
    Verify TFLite model bằng cách chạy trên n_per_class ảnh test.
    In kết quả accuracy + confusion matrix nhỏ.
    """
    if not tflite_path.exists():
        print(f"⚠️  TFLite không tồn tại: {tflite_path}")
        return

    interp = tf.lite.Interpreter(model_path=str(tflite_path))
    interp.allocate_tensors()
    inp_det = interp.get_input_details()[0]
    out_det = interp.get_output_details()[0]

    print(f"\nVerify: {tflite_path.name}")
    print(f"  Input : {inp_det['shape']}  dtype={inp_det['dtype']}")
    print(f"  Output: {out_det['shape']}  dtype={out_det['dtype']}")
    print(f"  Classes: {class_names}")

    correct = 0
    total   = 0
    preds   = []
    trues   = []

    for cls_idx, cls in enumerate(class_names):
        cls_dir = dataset_root / "test" / cls
        if not cls_dir.exists():
            continue
        imgs = sorted(cls_dir.glob("*.jpg"))[:n_per_class] or \
               sorted(cls_dir.glob("*.png"))[:n_per_class]
        for img_path in imgs:
            img = Image.open(img_path).resize((64, 64)).convert("RGB")
            x   = np.array(img, dtype=np.float32)[None] / 255.0
            interp.set_tensor(inp_det["index"], x)
            interp.invoke()
            probs = interp.get_tensor(out_det["index"])[0]
            pred  = int(np.argmax(probs))
            preds.append(pred)
            trues.append(cls_idx)
            if pred == cls_idx:
                correct += 1
            total += 1

    if total == 0:
        print("  ⚠️  Không có ảnh để verify")
        return

    acc_tflite = correct / total
    print(f"\n  TFLite accuracy ({total} samples): {acc_tflite:.4f} ({acc_tflite*100:.2f}%)")
    print(f"  Confusion matrix:")
    print(confusion_matrix(trues, preds))

    if acc_tflite < 0.85:
        print("  ⚠️  Accuracy thấp — kiểm tra lại dataset hoặc training")
    elif acc_tflite < 0.90:
        print("  ⚠️  Accuracy chưa đạt yêu cầu (> 0.90)")
    else:
        print("  ✅ TFLite OK — sẵn sàng đưa lên Android")


verify_tflite(TFLITE_EYE, DATASET_EYE, classes_eye)
if model_yawn and TFLITE_YAWN.exists():
    verify_tflite(TFLITE_YAWN, DATASET_YAWN, classes_yawn)

# %% ──────────────────────────────────────────────────────────────────
# CELL 18 ▸ COPY TFLITE VÀO ANDROID PROJECT
# ─────────────────────────────────────────────────────────────────────
android_assets = PROJECT_ROOT / "app" / "src" / "main" / "assets"

print("=" * 55)
print("📱  COPY TFLite → Android Assets")
print("=" * 55)

if android_assets.exists():
    # Eye model
    if TFLITE_EYE.exists():
        print(f"  ✅ drowsiness_model.tflite: {TFLITE_EYE.stat().st_size/1024:.1f} KB")

    # Yawn model
    if model_yawn and TFLITE_YAWN.exists():
        print(f"  ✅ yawn_model.tflite: {TFLITE_YAWN.stat().st_size/1024:.1f} KB")
        # File đã nằm đúng chỗ từ train_cnn()

    print(f"\n  Assets folder: {android_assets}")
    print("  Files:")
    for f in sorted(android_assets.iterdir()):
        print(f"    {f.name}: {f.stat().st_size/1024:.1f} KB")
else:
    print(f"  ⚠️  Android assets không tìm thấy: {android_assets}")
    print("  (Normal nếu đang chạy Colab — download outputs.zip thay)")

# Colab: download toàn bộ outputs
if IN_COLAB:
    shutil.make_archive("/content/drowsy_outputs_multi", "zip", str(OUTPUTS))
    from google.colab import files
    files.download("/content/drowsy_outputs_multi.zip")
    print("\n✅ Downloaded drowsy_outputs_multi.zip")

# %% ──────────────────────────────────────────────────────────────────
# CELL 19 ▸ GRADIO DEMO — TẤT CẢ MODEL
# ─────────────────────────────────────────────────────────────────────
import mediapipe as mp
import cv2

mp_face_mesh = mp.solutions.face_mesh
TFLITE_EYE_PATH  = str(TFLITE_EYE)
TFLITE_YAWN_PATH = str(TFLITE_YAWN) if TFLITE_YAWN.exists() else None

# Load interpreters
interp_eye  = tf.lite.Interpreter(TFLITE_EYE_PATH)
interp_eye.allocate_tensors()
inp_eye  = interp_eye.get_input_details()[0]
out_eye  = interp_eye.get_output_details()[0]

interp_yawn = None
if TFLITE_YAWN_PATH and Path(TFLITE_YAWN_PATH).exists():
    interp_yawn = tf.lite.Interpreter(TFLITE_YAWN_PATH)
    interp_yawn.allocate_tensors()
    inp_yawn = interp_yawn.get_input_details()[0]
    out_yawn = interp_yawn.get_output_details()[0]

LEFT_EYE  = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH_IDX = [61, 291, 39, 181, 0, 17, 269, 405]

def tflite_predict(interp, inp_det, out_det, img_pil, label_names):
    x = np.array(img_pil.resize((64,64)).convert("RGB"), dtype=np.float32)[None] / 255.0
    interp.set_tensor(inp_det["index"], x)
    interp.invoke()
    probs = interp.get_tensor(out_det["index"])[0]
    idx   = int(np.argmax(probs))
    return label_names[idx], float(probs[idx]), probs

def full_analysis(image: np.ndarray):
    if image is None:
        return image, "## ❌ Chưa có ảnh"

    pil_img  = Image.fromarray(image)
    h, w     = image.shape[:2]
    img_rgb  = image.copy()

    result_text  = []
    annotated    = image.copy()

    # ── MediaPipe ─────────────────────────────────────────────────────
    with mp_face_mesh.FaceMesh(
        static_image_mode=True, max_num_faces=1,
        min_detection_confidence=0.5
    ) as fm:
        mp_result = fm.process(img_rgb)

    ear_val = mar_val = None
    eye_state_ear = "N/A"
    mouth_state_mar = "N/A"

    if mp_result.multi_face_landmarks:
        lm = mp_result.multi_face_landmarks[0].landmark

        def dist(p1, p2):
            return math.sqrt((p1.x-p2.x)**2 + (p1.y-p2.y)**2)
        def ear(idx_list):
            pts = [lm[i] for i in idx_list]
            A = dist(pts[1], pts[5]); B = dist(pts[2], pts[4]); C = dist(pts[0], pts[3])
            return (A+B)/(2*C) if C > 1e-9 else 0
        def mar(idx_list):
            pts = [lm[i] for i in idx_list]
            A = dist(pts[2], pts[6]); B = dist(pts[3], pts[7]); C = dist(pts[0], pts[1])
            return (A+B)/(2*C) if C > 1e-9 else 0

        ear_val = (ear(LEFT_EYE) + ear(RIGHT_EYE)) / 2
        mar_val = mar(MOUTH_IDX)
        eye_state_ear   = "CLOSED" if ear_val < 0.24 else "OPEN"
        mouth_state_mar = "YAWNING" if mar_val > 0.58 else "NORMAL"

        # Vẽ landmarks
        mp.solutions.drawing_utils.draw_landmarks(
            annotated, mp_result.multi_face_landmarks[0],
            mp_face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(
                color=(0,200,0), thickness=1))

        result_text.append(f"### 📐 MediaPipe")
        result_text.append(f"- EAR: `{ear_val:.4f}` → **{eye_state_ear}**")
        result_text.append(f"- MAR: `{mar_val:.4f}` → **{mouth_state_mar}**")

        # ── CNN Eye ───────────────────────────────────────────────────
        eye_crop = pil_img.crop((w//5, h//6, 4*w//5, h//2))
        lbl_e, conf_e, probs_e = tflite_predict(
            interp_eye, inp_eye, out_eye, eye_crop, classes_eye)
        result_text.append(f"\n### 🧠 CNN Eye")
        result_text.append(f"- Dự đoán: **{lbl_e}** ({conf_e*100:.1f}%)")
        for i, cn in enumerate(classes_eye):
            result_text.append(f"  - `{cn}`: {probs_e[i]*100:.1f}%")

        # ── CNN Yawn ──────────────────────────────────────────────────
        if interp_yawn:
            mouth_crop = pil_img.crop((w//4, h//2, 3*w//4, h))
            lbl_y, conf_y, probs_y = tflite_predict(
                interp_yawn, inp_yawn, out_yawn, mouth_crop, classes_yawn)
            result_text.append(f"\n### 😮 CNN Yawn")
            result_text.append(f"- Dự đoán: **{lbl_y}** ({conf_y*100:.1f}%)")

        # ── Final state ───────────────────────────────────────────────
        if mar_val > 0.58 or (interp_yawn and "yawn" in lbl_y.lower() and conf_y > 0.55):
            final = "😴 YAWNING"; icon = "🟠"
        elif (lbl_e == "eyes_closed" and conf_e > 0.55) or eye_state_ear == "CLOSED":
            final = "👁️ EYES CLOSED"; icon = "🔴"
        else:
            final = "✅ AWAKE"; icon = "🟢"

        cv2.putText(annotated, final, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (255,80,80) if "CLOSED" in final or "YAWN" in final
                    else (80,200,80), 2)
        result_text.insert(0, f"## {icon} Kết quả: **{final}**\n")
    else:
        result_text.append("## ❌ Không phát hiện khuôn mặt")

    return annotated, "\n".join(result_text)


import gradio as gr

demo = gr.Interface(
    fn=full_analysis,
    inputs=gr.Image(label="📸 Upload ảnh / webcam", sources=["upload","webcam"]),
    outputs=[
        gr.Image(label="🎭 MediaPipe Landmarks"),
        gr.Markdown(label="📊 Kết quả"),
    ],
    title="🚗 Drowsy Driver Detection — Multi-Model Demo",
    description=(
        "**3 Models:** MediaPipe EAR/MAR + CNN Eye TFLite + CNN Yawn TFLite\n"
        "Upload ảnh khuôn mặt để phân tích trạng thái buồn ngủ"
    ),
    flagging_mode="never",
    theme=gr.themes.Soft(),
)

if IN_COLAB:
    demo.launch(share=True, quiet=True)
    print("✅ Public URL đã tạo ở trên")
else:
    demo.launch(server_port=7860, quiet=True)
    print("✅ Demo: http://localhost:7860")

# %% ──────────────────────────────────────────────────────────────────
# CELL 20 ▸ SUMMARY REPORT ĐẦY ĐỦ
# ─────────────────────────────────────────────────────────────────────
print("=" * 65)
print("🏆  TỔNG KẾT DỰ ÁN — MULTI-MODEL APPROACH")
print("=" * 65)

print(f"""
┌─────────────────────────────────────────────────────────────┐
│  MODEL A: CNN Eye Classifier                                 │
│  Data   : mrleyedataset ({MAX_EYE*2:,} ảnh / class)        │
│  Classes: eyes_closed (0) / eyes_open (1)                   │
│  Acc    : {metrics_eye['accuracy']*100:.2f}%                │
│  TFLite : drowsiness_model.tflite                           │
│  Android: TfliteDrowsinessClassifier.kt                     │
├─────────────────────────────────────────────────────────────┤""")

if model_yawn:
    print(f"""│  MODEL B: CNN Yawn Classifier                                │
│  Data   : rawdata/data/yawn (5,119 ảnh)                     │
│  Classes: no_yawn (0) / yawn (1)                            │
│  Acc    : {metrics_yawn['accuracy']*100:.2f}%               │
│  TFLite : yawn_model.tflite                                 │
│  Android: YawnClassifier.kt (cần thêm)                     │
├─────────────────────────────────────────────────────────────┤""")

print(f"""│  MODEL C: YOLOv8n Drowsiness Detector                        │
│  Data   : Roboflow "drowsiness driver"                      │
│  Deploy : Colab demo / báo cáo (không Android)              │
│  So sánh: end-to-end vs MediaPipe+CNN                       │
└─────────────────────────────────────────────────────────────┘

Android Pipeline sau khi tích hợp cả 3:
  Camera → MediaPipe landmarks
    ├── EAR/MAR (fallback)
    ├── CNN Eye TFLite   (eyes_closed/eyes_open)
    └── CNN Yawn TFLite  (yawn/no_yawn)
                    ↓
             FUSION LOGIC
    → AWAKE / EYES_CLOSED / YAWNING / DROWSY
""")
