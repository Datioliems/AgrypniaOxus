# %% [markdown]
"""
# 🚗 DROWSY DRIVER DETECTION — FULL PIPELINE
### Google Colab / VS Code Jupyter Notebook
---
**Pipeline:**
`Cài đặt` → `Upload Dataset` → `EDA` → `Tiền xử lý` → `Train CNN` → `Đánh giá` → `Xuất TFLite` → `Demo Gradio`

**Kết quả kỳ vọng:** Accuracy ~97% trên tập test (đã đạt 97.21% trong thực nghiệm)
"""

# %% ───────────────────────────────────────────────────────────────
# CELL 1 ▸ CÀI ĐẶT THƯ VIỆN
# ──────────────────────────────────────────────────────────────────
import sys
IN_COLAB = "google.colab" in sys.modules

if IN_COLAB:
    # Chạy trong Google Colab
    import subprocess
    subprocess.run([
        "pip", "install", "-q",
        "tensorflow>=2.15",
        "mediapipe>=0.10",
        "opencv-python-headless",
        "matplotlib", "seaborn",
        "scikit-learn", "pillow",
        "gradio", "pandas", "numpy"
    ], check=True)
    print("✅ Đã cài xong (Colab)")
else:
    # VS Code / Local — chạy lệnh này trong Terminal trước:
    # pip install tensorflow mediapipe opencv-python matplotlib seaborn
    #             scikit-learn pillow gradio pandas numpy
    print("✅ Đang chạy local (VS Code) — đảm bảo đã cài đủ packages")

# %% ───────────────────────────────────────────────────────────────
# CELL 2 ▸ IMPORT & CẤU HÌNH CHUNG
# ──────────────────────────────────────────────────────────────────
import os, json, math, warnings, shutil
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

# ── Hyperparameters ────────────────────────────────────────────────
IMG_SIZE    = (64, 64)      # kích thước ảnh đầu vào CNN
BATCH_SIZE  = 32
EPOCHS      = 15
LR          = 1e-3
SEED        = 42
CLASSES     = ["eyes_closed", "eyes_open"]  # thứ tự này là class index 0, 1
EAR_THRESH  = 0.24          # EAR < ngưỡng → nhắm mắt
MAR_THRESH  = 0.58          # MAR > ngưỡng → ngáp

print(f"Python  : {sys.version.split()[0]}")
print(f"Platform: {'Google Colab' if IN_COLAB else 'Local / VS Code'}")

# %% ───────────────────────────────────────────────────────────────
# CELL 3 ▸ MOUNT GOOGLE DRIVE  (CHỈ CẦN KHI DÙNG COLAB)
# ──────────────────────────────────────────────────────────────────
"""
NẾU DÙNG GOOGLE COLAB:
  1. Chạy cell này để mount Drive
  2. Upload thư mục 'dataset' lên Drive trước
     (kéo thả thư mục dataset/ vào Google Drive → MyDrive/)
  3. Sửa DRIVE_DATASET bên dưới đúng đường dẫn

NẾU DÙNG VS CODE LOCAL:
  - Bỏ qua cell này, tiếp tục cell 4
"""

if IN_COLAB:
    from google.colab import drive
    drive.mount("/content/drive", force_remount=False)
    print("✅ Google Drive đã mount tại /content/drive")

# %% ───────────────────────────────────────────────────────────────
# CELL 4 ▸ THIẾT LẬP ĐƯỜNG DẪN
# ──────────────────────────────────────────────────────────────────
if IN_COLAB:
    # ── Thay đường dẫn này cho đúng vị trí trên Drive của bạn ──
    DATASET_ROOT = Path("/content/drive/MyDrive/DrowsyDriverAndroid/dataset")
    # Hoặc nếu đã upload thẳng vào Colab runtime (không qua Drive):
    # DATASET_ROOT = Path("/content/dataset")
    OUTPUT_ROOT  = Path("/content/outputs")
else:
    # VS Code — chạy từ thư mục gốc dự án D:\2026.AI\DrowsyDriverAndroid
    PROJECT_ROOT = Path(__file__).parent if "__file__" in dir() else Path(".")
    DATASET_ROOT = PROJECT_ROOT / "dataset"
    OUTPUT_ROOT  = PROJECT_ROOT / "outputs"

# Tạo thư mục output
for d in ["training", "evaluation", "eda"]:
    (OUTPUT_ROOT / d).mkdir(parents=True, exist_ok=True)

TRAIN_DIR = DATASET_ROOT / "train"
VAL_DIR   = DATASET_ROOT / "val"
TEST_DIR  = DATASET_ROOT / "test"

# ── Kiểm tra dataset ───────────────────────────────────────────────
print("📁 Kiểm tra dataset:")
all_ok = True
for split, d in [("train", TRAIN_DIR), ("val", VAL_DIR), ("test", TEST_DIR)]:
    if d.exists():
        counts = {}
        for cls in CLASSES:
            cls_dir = d / cls
            if cls_dir.exists():
                n = sum(1 for f in cls_dir.iterdir()
                        if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"})
                counts[cls] = n
            else:
                counts[cls] = 0
        total = sum(counts.values())
        print(f"  ✅ {split:5s} → {total:7,} ảnh  |  "
              f"eyes_closed={counts.get('eyes_closed',0):,}  "
              f"eyes_open={counts.get('eyes_open',0):,}")
    else:
        print(f"  ❌ {split:5s} → KHÔNG TÌM THẤY: {d}")
        all_ok = False

if not all_ok:
    print("\n⚠️  Kiểm tra lại đường dẫn DATASET_ROOT ở cell trên!")
    print(f"   Hiện tại: {DATASET_ROOT.absolute()}")
else:
    print(f"\n✅ Dataset OK — Output sẽ lưu tại: {OUTPUT_ROOT.absolute()}")

# %% ───────────────────────────────────────────────────────────────
# CELL 5 ▸ UPLOAD DATASET LÊN COLAB (nếu chưa có)
# ──────────────────────────────────────────────────────────────────
"""
NẾU MUỐN UPLOAD NHANH DATASET .ZIP LÊN COLAB RUNTIME:
Bỏ comment 3 dòng dưới và chạy.
"""
# if IN_COLAB:
#     from google.colab import files
#     uploaded = files.upload()   # chọn file dataset.zip từ máy tính
#     !unzip -q dataset.zip -d /content/dataset

# %% ───────────────────────────────────────────────────────────────
# CELL 6 ▸ EDA — THỐNG KÊ & PHÂN BỐ DATASET
# ──────────────────────────────────────────────────────────────────
print("=" * 55)
print("📊  EXPLORATORY DATA ANALYSIS")
print("=" * 55)

stats = {}
for split in ["train", "val", "test"]:
    split_dir = DATASET_ROOT / split
    stats[split] = {}
    for cls in CLASSES:
        cls_dir = split_dir / cls
        if cls_dir.exists():
            imgs = [f for f in cls_dir.iterdir()
                    if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}]
            stats[split][cls] = len(imgs)
        else:
            stats[split][cls] = 0

df_stats = pd.DataFrame(stats).T
df_stats["total"]   = df_stats.sum(axis=1)
df_stats["balance"] = (df_stats[CLASSES[0]] / df_stats["total"] * 100).round(1)
grand_total = int(df_stats["total"].sum())

print(df_stats.to_string())
print(f"\n  Tổng cộng: {grand_total:,} ảnh")

# ── Visualize phân bố ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("Phân bố dữ liệu theo class", fontsize=15, fontweight="bold", y=1.02)

COLORS = {"eyes_closed": "#FF6B6B", "eyes_open": "#4ECDC4"}

for ax, split in zip(axes, ["train", "val", "test"]):
    values  = [stats[split][cls] for cls in CLASSES]
    bar_col = [COLORS[cls] for cls in CLASSES]
    bars = ax.bar(CLASSES, values, color=bar_col, edgecolor="white", linewidth=2, width=0.6)
    ax.set_title(f"{split.upper()} SET\n({sum(values):,} ảnh)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Số lượng ảnh")
    ax.set_ylim(0, max(values) * 1.25)
    ax.tick_params(axis="x", rotation=10)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 100,
                f"{val:,}", ha="center", va="bottom", fontweight="bold", fontsize=11)

plt.tight_layout()
plt.savefig(OUTPUT_ROOT / "eda" / "class_distribution.png", bbox_inches="tight", dpi=150)
plt.show()
print("✅ Saved → eda/class_distribution.png")

# %% ───────────────────────────────────────────────────────────────
# CELL 7 ▸ EDA — HIỂN THỊ ẢNH MẪU
# ──────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 8, figsize=(20, 6))
fig.suptitle("Ảnh mẫu từ tập train — 8 ảnh mỗi class", fontsize=14, fontweight="bold")

for row, cls in enumerate(CLASSES):
    cls_dir = TRAIN_DIR / cls
    all_imgs = sorted([f for f in cls_dir.iterdir()
                       if f.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    # Chọn ảnh đều từ đầu, giữa, cuối
    indices = np.linspace(0, min(len(all_imgs)-1, 1000), 8, dtype=int)
    sample_imgs = [all_imgs[i] for i in indices]

    for col, img_path in enumerate(sample_imgs):
        ax = axes[row, col]
        img = Image.open(img_path).resize((64, 64)).convert("RGB")
        ax.imshow(img)
        ax.axis("off")
        if col == 0:
            ax.set_title(cls, fontsize=10, fontweight="bold",
                         color=COLORS[cls], pad=3)

plt.tight_layout()
plt.savefig(OUTPUT_ROOT / "eda" / "sample_grid.png", bbox_inches="tight", dpi=150)
plt.show()
print("✅ Saved → eda/sample_grid.png")

# %% ───────────────────────────────────────────────────────────────
# CELL 8 ▸ EDA — PHÂN TÍCH KÍCH THƯỚC ẢNH
# ──────────────────────────────────────────────────────────────────
print("Đang đọc kích thước ảnh (sampling 200 ảnh mỗi class)...")
widths, heights, cls_labels = [], [], []

for cls in CLASSES:
    cls_dir = TRAIN_DIR / cls
    all_imgs = sorted(cls_dir.glob("*.jpg"))[:200]
    for p in all_imgs:
        try:
            w, h = Image.open(p).size
            widths.append(w)
            heights.append(h)
            cls_labels.append(cls)
        except Exception:
            pass

df_dims = pd.DataFrame({"width": widths, "height": heights, "class": cls_labels})

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, dim in zip(axes, ["width", "height"]):
    for cls in CLASSES:
        data = df_dims[df_dims["class"] == cls][dim]
        ax.hist(data, bins=30, alpha=0.6, label=cls, color=COLORS[cls])
    ax.set_title(f"Phân bố {dim}", fontsize=12, fontweight="bold")
    ax.set_xlabel(f"{dim} (px)")
    ax.set_ylabel("Số lượng")
    ax.legend()

plt.suptitle("Phân bố kích thước ảnh (sample 200/class)", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(OUTPUT_ROOT / "eda" / "image_size_distribution.png", bbox_inches="tight", dpi=150)
plt.show()

print(f"\nKích thước trung bình: {df_dims['width'].mean():.0f}×{df_dims['height'].mean():.0f} px")
print(f"Kích thước min: {df_dims['width'].min()}×{df_dims['height'].min()} px")
print(f"Kích thước max: {df_dims['width'].max()}×{df_dims['height'].max()} px")

# %% ───────────────────────────────────────────────────────────────
# CELL 9 ▸ CHUẨN BỊ DỮ LIỆU — ImageDataGenerator
# ──────────────────────────────────────────────────────────────────
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

print(f"TensorFlow version: {tf.__version__}")
print(f"GPU available: {len(tf.config.list_physical_devices('GPU')) > 0}")
print("=" * 55)
print("⚙️  CHUẨN BỊ DATA PIPELINE")
print("=" * 55)

# ── Train: có augmentation ────────────────────────────────────────
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=12,
    width_shift_range=0.10,
    height_shift_range=0.10,
    horizontal_flip=True,
    zoom_range=0.12,
    brightness_range=[0.75, 1.25],
    shear_range=5,
)

# ── Val / Test: chỉ normalize ─────────────────────────────────────
eval_datagen = ImageDataGenerator(rescale=1.0 / 255)

train_data = train_datagen.flow_from_directory(
    TRAIN_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode="categorical", classes=CLASSES, seed=SEED, shuffle=True,
)
val_data = eval_datagen.flow_from_directory(
    VAL_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode="categorical", classes=CLASSES, seed=SEED, shuffle=False,
)
test_data = eval_datagen.flow_from_directory(
    TEST_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode="categorical", classes=CLASSES, seed=SEED, shuffle=False,
)

print(f"\nClass mapping: {train_data.class_indices}")
print(f"Steps/epoch  : {len(train_data)}")

# Lưu class names
class_names = {str(v): k for k, v in train_data.class_indices.items()}
with open(OUTPUT_ROOT / "training" / "class_names.json", "w") as f:
    json.dump(class_names, f, indent=2)

# ── Hiển thị ảnh sau augmentation ────────────────────────────────
batch_imgs, batch_labels = next(train_data)
fig, axes = plt.subplots(2, 8, figsize=(20, 6))
fig.suptitle("Ảnh SAU augmentation (batch đầu tiên)", fontsize=13, fontweight="bold")

for idx, (img, lbl) in enumerate(zip(batch_imgs[:16], batch_labels[:16])):
    ax = axes[idx // 8, idx % 8]
    ax.imshow(img)
    cls_name = CLASSES[int(np.argmax(lbl))]
    ax.set_title(cls_name, fontsize=8, color=COLORS[cls_name])
    ax.axis("off")

plt.tight_layout()
plt.savefig(OUTPUT_ROOT / "eda" / "augmented_samples.png", bbox_inches="tight", dpi=150)
plt.show()
print("✅ Saved → eda/augmented_samples.png")

# %% ───────────────────────────────────────────────────────────────
# CELL 10 ▸ XÂY DỰNG MÔ HÌNH CNN
# ──────────────────────────────────────────────────────────────────
from tensorflow.keras import layers, models, callbacks, regularizers

print("=" * 55)
print("🧠  XÂY DỰNG MÔ HÌNH CNN")
print("=" * 55)

def build_drowsy_cnn(input_shape=(64, 64, 3), num_classes=2):
    """
    Kiến trúc CNN 3-block:
      Block 1: Conv(24) + BN + MaxPool
      Block 2: Conv(48) + BN + MaxPool
      Block 3: Conv(64) + BN + GlobalAvgPool
      Head   : Dropout(0.25) + Dense(2, softmax)

    Tổng ~38K parameters — nhỏ gọn, phù hợp TFLite trên Android
    """
    inp = layers.Input(shape=input_shape, name="input_eye")

    # Block 1
    x = layers.Conv2D(24, (3, 3), padding="same", activation="relu",
                      kernel_regularizer=regularizers.l2(1e-4), name="conv1")(inp)
    x = layers.BatchNormalization(name="bn1")(x)
    x = layers.MaxPooling2D(2, 2, name="pool1")(x)

    # Block 2
    x = layers.Conv2D(48, (3, 3), padding="same", activation="relu",
                      kernel_regularizer=regularizers.l2(1e-4), name="conv2")(x)
    x = layers.BatchNormalization(name="bn2")(x)
    x = layers.MaxPooling2D(2, 2, name="pool2")(x)

    # Block 3
    x = layers.Conv2D(64, (3, 3), padding="same", activation="relu",
                      kernel_regularizer=regularizers.l2(1e-4), name="conv3")(x)
    x = layers.BatchNormalization(name="bn3")(x)
    x = layers.GlobalAveragePooling2D(name="gap")(x)

    # Head
    x = layers.Dropout(0.25, name="dropout")(x)
    out = layers.Dense(num_classes, activation="softmax", name="output")(x)

    return models.Model(inputs=inp, outputs=out, name="DrowsyCNN")


model = build_drowsy_cnn()
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LR),
    loss="categorical_crossentropy",
    metrics=[
        "accuracy",
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall"),
    ],
)
model.summary()

n_params = model.count_params()
print(f"\n✅ Model: {n_params:,} parameters ({n_params*4/1024:.1f} KB weights)")

# Lưu model summary
with open(OUTPUT_ROOT / "training" / "model_summary.txt", "w") as f:
    model.summary(print_fn=lambda line: f.write(line + "\n"))

# Visualize feature maps (batch đầu)
print("\nKiểm tra forward pass...")
dummy = tf.zeros((1, 64, 64, 3))
out   = model(dummy, training=False)
print(f"Input shape : {dummy.shape}")
print(f"Output shape: {out.shape}  (probs sum = {float(tf.reduce_sum(out)):.4f})")

# %% ───────────────────────────────────────────────────────────────
# CELL 11 ▸ HUẤN LUYỆN MÔ HÌNH
# ──────────────────────────────────────────────────────────────────
print("=" * 55)
print("🏋️  TRAINING")
print("=" * 55)

cbs = [
    callbacks.EarlyStopping(
        monitor="val_accuracy", patience=4,
        restore_best_weights=True, verbose=1,
        min_delta=0.001,
    ),
    callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5,
        patience=2, min_lr=1e-7, verbose=1,
    ),
    callbacks.ModelCheckpoint(
        filepath=str(OUTPUT_ROOT / "training" / "best_model.keras"),
        monitor="val_accuracy", save_best_only=True,
        verbose=1,
    ),
]

history = model.fit(
    train_data,
    epochs=EPOCHS,
    validation_data=val_data,
    callbacks=cbs,
    verbose=1,
)

# Lưu model & history
model.save(str(OUTPUT_ROOT / "training" / "drowsiness_model.keras"))
hist_dict = {k: [float(v) for v in vals] for k, vals in history.history.items()}
with open(OUTPUT_ROOT / "training" / "training_history.json", "w") as f:
    json.dump(hist_dict, f, indent=2)

best_val_acc = max(history.history["val_accuracy"])
print(f"\n🎯 Training xong! Best val_accuracy = {best_val_acc:.4f} ({best_val_acc*100:.2f}%)")
print(f"   Epochs chạy: {len(history.history['accuracy'])}/{EPOCHS}")

# %% ───────────────────────────────────────────────────────────────
# CELL 12 ▸ VISUALIZE TRAINING CURVES
# ──────────────────────────────────────────────────────────────────
hist = history.history
epochs_ran = range(1, len(hist["accuracy"]) + 1)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Training Curves — DrowsyCNN", fontsize=16, fontweight="bold")

plot_cfg = [
    ("accuracy",  "val_accuracy",  "Accuracy",  "#2196F3"),
    ("loss",      "val_loss",      "Loss",       "#F44336"),
    ("precision", "val_precision", "Precision",  "#4CAF50"),
    ("recall",    "val_recall",    "Recall",     "#FF9800"),
]

for ax, (tr_key, vl_key, title, clr) in zip(axes.flat, plot_cfg):
    ax.plot(epochs_ran, hist[tr_key], "o-",
            label="Train", color=clr, linewidth=2, markersize=5)
    if vl_key in hist:
        ax.plot(epochs_ran, hist[vl_key], "s--",
                label="Val", color=clr, alpha=0.55, linewidth=2, markersize=5)
    # Đánh dấu best epoch
    if vl_key in hist:
        best_ep = int(np.argmax(hist[vl_key]) if "loss" not in vl_key
                      else np.argmin(hist[vl_key]))
        ax.axvline(x=best_ep + 1, color="gray", linestyle=":", alpha=0.6, label=f"Best ep={best_ep+1}")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(title)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_ROOT / "training" / "training_curves.png", bbox_inches="tight", dpi=150)
plt.show()
print("✅ Saved → training/training_curves.png")

# %% ───────────────────────────────────────────────────────────────
# CELL 13 ▸ ĐÁNH GIÁ TRÊN TẬP TEST
# ──────────────────────────────────────────────────────────────────
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    classification_report, confusion_matrix,
)

print("=" * 55)
print("📊  ĐÁNH GIÁ TRÊN TẬP TEST")
print("=" * 55)

# Predict toàn bộ test set
test_data.reset()
y_pred_probs = model.predict(test_data, verbose=1)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = test_data.classes

# Metrics
acc  = accuracy_score(y_true, y_pred)
prec, rec, f1, sup = precision_recall_fscore_support(y_true, y_pred, zero_division=0)
cm   = confusion_matrix(y_true, y_pred)

print(f"\n{'='*45}")
print(f"  Accuracy tổng: {acc:.4f}  ({acc*100:.2f}%)")
print(f"{'='*45}")
print(f"\n{'Class':<16} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
print("-" * 56)
for i, cls in enumerate(CLASSES):
    print(f"  {cls:<14} {prec[i]:>10.4f} {rec[i]:>10.4f} {f1[i]:>10.4f} {int(sup[i]):>10,}")

print(f"\n{classification_report(y_true, y_pred, target_names=CLASSES)}")

# Lưu metrics
metrics_out = {
    "accuracy": float(acc),
    "per_class": [
        {"class": CLASSES[i], "precision": float(prec[i]),
         "recall": float(rec[i]), "f1": float(f1[i]), "support": int(sup[i])}
        for i in range(len(CLASSES))
    ],
    "confusion_matrix": cm.tolist(),
}
with open(OUTPUT_ROOT / "evaluation" / "metrics.json", "w") as f:
    json.dump(metrics_out, f, indent=2)
print("✅ Saved → evaluation/metrics.json")

# %% ───────────────────────────────────────────────────────────────
# CELL 14 ▸ CONFUSION MATRIX (đẹp)
# ──────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 7))

sns.heatmap(
    cm, annot=True, fmt="d", cmap="Blues",
    xticklabels=CLASSES, yticklabels=CLASSES,
    linewidths=2, linecolor="white",
    annot_kws={"size": 18, "weight": "bold"},
    ax=ax,
)
ax.set_title("Confusion Matrix — Test Set", fontsize=14, fontweight="bold", pad=15)
ax.set_ylabel("True Label",      fontsize=12)
ax.set_xlabel("Predicted Label", fontsize=12)
ax.tick_params(axis="both", labelsize=11)

# Thêm % accuracy
ax.text(0.98, 0.02, f"Accuracy: {acc*100:.2f}%",
        transform=ax.transAxes, fontsize=13, color="darkgreen",
        ha="right", va="bottom",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="lightgreen", alpha=0.75))

plt.tight_layout()
plt.savefig(OUTPUT_ROOT / "evaluation" / "confusion_matrix.png", bbox_inches="tight", dpi=150)
plt.show()
print("✅ Saved → evaluation/confusion_matrix.png")

# %% ───────────────────────────────────────────────────────────────
# CELL 15 ▸ XUẤT TFLITE MODEL
# ──────────────────────────────────────────────────────────────────
print("=" * 55)
print("📱  XUẤT TFLITE MODEL")
print("=" * 55)

# Load best model checkpoint
best_model_path = OUTPUT_ROOT / "training" / "best_model.keras"
if best_model_path.exists():
    model_to_convert = tf.keras.models.load_model(str(best_model_path))
    print("Dùng best checkpoint model")
else:
    model_to_convert = model
    print("Dùng model cuối cùng")

# ── Convert với Dynamic Range Quantization ────────────────────────
converter = tf.lite.TFLiteConverter.from_keras_model(model_to_convert)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

tflite_model  = converter.convert()
tflite_path   = OUTPUT_ROOT / "training" / "drowsiness_model.tflite"
with open(tflite_path, "wb") as f:
    f.write(tflite_model)

keras_kb  = best_model_path.stat().st_size / 1024 if best_model_path.exists() else 0
tflite_kb = tflite_path.stat().st_size / 1024
print(f"\nKeras  model : {keras_kb:.1f} KB")
print(f"TFLite model : {tflite_kb:.1f} KB  (giảm {(1-tflite_kb/keras_kb)*100:.0f}%)")
print(f"✅ Saved → training/drowsiness_model.tflite")

# %% ───────────────────────────────────────────────────────────────
# CELL 16 ▸ KIỂM TRA TFLITE MODEL
# ──────────────────────────────────────────────────────────────────
print("Kiểm tra TFLite model...")

interp = tf.lite.Interpreter(model_path=str(tflite_path))
interp.allocate_tensors()
inp_det = interp.get_input_details()
out_det = interp.get_output_details()
print(f"Input shape : {inp_det[0]['shape']}  dtype={inp_det[0]['dtype']}")
print(f"Output shape: {out_det[0]['shape']}  dtype={out_det[0]['dtype']}")

# Test 50 ảnh mỗi class
correct = total_t = 0
for cls_idx, cls in enumerate(CLASSES):
    cls_dir  = TEST_DIR / cls
    imgs     = sorted(cls_dir.glob("*.jpg"))[:50]
    for img_path in imgs:
        img = Image.open(img_path).resize((64, 64)).convert("RGB")
        x   = np.array(img, dtype=np.float32)[None] / 255.0
        interp.set_tensor(inp_det[0]["index"], x)
        interp.invoke()
        pred = int(np.argmax(interp.get_tensor(out_det[0]["index"])[0]))
        if pred == cls_idx:
            correct += 1
        total_t += 1

tflite_acc = correct / total_t
print(f"\nTFLite accuracy (100 samples): {tflite_acc:.4f} ({tflite_acc*100:.2f}%)")
diff = abs(acc - tflite_acc)
print(f"Độ chênh lệch vs Keras       : {diff:.4f}  {'✅ OK' if diff < 0.02 else '⚠️ Kiểm tra lại'}")

# %% ───────────────────────────────────────────────────────────────
# CELL 17 ▸ MEDIAPIPE — EAR / MAR CALCULATION
# ──────────────────────────────────────────────────────────────────
"""
EAR (Eye Aspect Ratio) và MAR (Mouth Aspect Ratio)
được tính từ 478 facial landmarks của MediaPipe.

EAR = (|A| + |B|) / (2 * |C|)
  A = khoảng cách dọc trên
  B = khoảng cách dọc dưới
  C = khoảng cách ngang
  → EAR < 0.24: mắt nhắm

MAR = (|A| + |B|) / (2 * |C|)
  → MAR > 0.58: đang ngáp
"""
import mediapipe as mp
import cv2

# Landmark indices (MediaPipe 478-point)
LEFT_EYE_IDX  = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]
MOUTH_IDX     = [61, 291, 39, 181, 0, 17, 269, 405]

mp_face_mesh = mp.solutions.face_mesh
mp_draw      = mp.solutions.drawing_utils
mp_styles    = mp.solutions.drawing_styles


def _dist(p1, p2) -> float:
    return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)


def compute_ear(landmarks, indices) -> float:
    """Eye Aspect Ratio"""
    pts = [landmarks[i] for i in indices]
    A = _dist(pts[1], pts[5])
    B = _dist(pts[2], pts[4])
    C = _dist(pts[0], pts[3])
    return (A + B) / (2.0 * C) if C > 1e-9 else 0.0


def compute_mar(landmarks, indices) -> float:
    """Mouth Aspect Ratio"""
    pts = [landmarks[i] for i in indices]
    A = _dist(pts[2], pts[6])
    B = _dist(pts[3], pts[7])
    C = _dist(pts[0], pts[1])
    return (A + B) / (2.0 * C) if C > 1e-9 else 0.0


def analyze_face(image_rgb: np.ndarray) -> dict | None:
    """
    Nhận ảnh RGB numpy → trả về dict: ear, mar, eye_state, mouth_state
    """
    with mp_face_mesh.FaceMesh(
        static_image_mode=True, max_num_faces=1,
        min_detection_confidence=0.5, refine_landmarks=True
    ) as fm:
        results = fm.process(image_rgb)

    if not results.multi_face_landmarks:
        return None

    lm = results.multi_face_landmarks[0].landmark
    ear_l = compute_ear(lm, LEFT_EYE_IDX)
    ear_r = compute_ear(lm, RIGHT_EYE_IDX)
    ear   = (ear_l + ear_r) / 2.0
    mar   = compute_mar(lm, MOUTH_IDX)

    return {
        "landmarks"  : results.multi_face_landmarks[0],
        "ear"        : ear,
        "mar"        : mar,
        "eye_state"  : "CLOSED"  if ear < EAR_THRESH else "OPEN",
        "mouth_state": "YAWNING" if mar > MAR_THRESH else "NORMAL",
    }


def draw_analysis(image_rgb: np.ndarray, result: dict,
                  cnn_pred: dict | None = None) -> np.ndarray:
    """Vẽ landmarks + EAR/MAR lên ảnh"""
    img = image_rgb.copy()
    mp_draw.draw_landmarks(
        img, result["landmarks"],
        mp_face_mesh.FACEMESH_CONTOURS,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp_draw.DrawingSpec(
            color=(0, 200, 0), thickness=1),
    )
    ear, mar = result["ear"], result["mar"]
    eye_col = (255, 80, 80)  if result["eye_state"] == "CLOSED"  else (80, 200, 80)
    mth_col = (255, 165, 0) if result["mouth_state"] == "YAWNING" else (80, 200, 80)
    cv2.putText(img, f"EAR: {ear:.3f}  [{result['eye_state']}]",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, eye_col, 2)
    cv2.putText(img, f"MAR: {mar:.3f}  [{result['mouth_state']}]",
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, mth_col, 2)
    if cnn_pred:
        cv2.putText(img,
                    f"CNN: {cnn_pred['label']} ({cnn_pred['conf']:.2f})",
                    (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 180, 255), 2)
    return img


def classify_eye_tflite(pil_img: Image.Image) -> dict:
    """Phân loại ảnh mắt bằng TFLite model đã export"""
    x = np.array(pil_img.resize((64, 64)).convert("RGB"),
                 dtype=np.float32)[None] / 255.0
    interp.set_tensor(inp_det[0]["index"], x)
    interp.invoke()
    probs = interp.get_tensor(out_det[0]["index"])[0]
    idx   = int(np.argmax(probs))
    return {
        "label" : class_names[str(idx)],
        "conf"  : float(probs[idx]),
        "probs" : {class_names[str(i)]: float(p) for i, p in enumerate(probs)},
    }


print("✅ MediaPipe helpers sẵn sàng (analyze_face, draw_analysis, classify_eye_tflite)")

# %% ───────────────────────────────────────────────────────────────
# CELL 18 ▸ TEST MEDIAPIPE + CNN TRÊN ẢNH MẪU
# ──────────────────────────────────────────────────────────────────
print("Test MediaPipe + CNN trên ảnh mẫu từ dataset...")

fig, axes = plt.subplots(2, 4, figsize=(20, 10))
fig.suptitle("Kết quả MediaPipe + CNN phân tích ảnh mắt", fontsize=14, fontweight="bold")

for row, cls in enumerate(CLASSES):
    imgs = sorted((TEST_DIR / cls).glob("*.jpg"))
    indices_test = np.linspace(0, min(len(imgs)-1, 200), 4, dtype=int)
    for col_idx, img_idx in enumerate(indices_test):
        img_pil = Image.open(imgs[img_idx]).convert("RGB")
        img_np  = np.array(img_pil)

        result   = analyze_face(img_np)
        cnn_pred = classify_eye_tflite(img_pil)
        ax       = axes[row, col_idx]

        if result:
            annotated = draw_analysis(img_np, result, cnn_pred)
            ax.imshow(annotated)
            ear, mar = result["ear"], result["mar"]
            title_color = "red" if result["eye_state"] == "CLOSED" else "green"
            ax.set_title(
                f"True: {cls}\n"
                f"EAR={ear:.3f} MAR={mar:.3f}\n"
                f"CNN: {cnn_pred['label']} ({cnn_pred['conf']:.2f})",
                fontsize=8, color=title_color
            )
        else:
            ax.imshow(img_np)
            ax.set_title(f"True: {cls}\n⚠️ No face detected", fontsize=8, color="gray")

        ax.axis("off")

plt.tight_layout()
plt.savefig(OUTPUT_ROOT / "evaluation" / "mediapipe_cnn_demo.png", bbox_inches="tight", dpi=120)
plt.show()
print("✅ Saved → evaluation/mediapipe_cnn_demo.png")

# %% ───────────────────────────────────────────────────────────────
# CELL 19 ▸ BẢNG TỔNG HỢP KẾT QUẢ
# ──────────────────────────────────────────────────────────────────
print("=" * 60)
print("🏆  KẾT QUẢ TỔNG HỢP DỰ ÁN")
print("=" * 60)

with open(OUTPUT_ROOT / "training" / "training_history.json") as f:
    hist_saved = json.load(f)
with open(OUTPUT_ROOT / "evaluation" / "metrics.json") as f:
    metrics_saved = json.load(f)

print(f"""
📁 DATASET
   Tổng:    {grand_total:,} ảnh
   Train:   {int(df_stats.loc['train', 'total']):,}  |  Val: {int(df_stats.loc['val','total']):,}  |  Test: {int(df_stats.loc['test','total']):,}
   Balance: ~{float(df_stats.loc['train','balance']):.1f}% eyes_closed

🧠 MÔ HÌNH CNN
   Kiến trúc: Conv2D(24) → Conv2D(48) → Conv2D(64) → GAP → Dense(2)
   Tham số:   {n_params:,}
   Input:     64×64×3   Output: 2 classes

📈 TRAINING
   Epochs chạy:        {len(hist_saved['accuracy'])}/{EPOCHS}
   Train acc cuối:     {hist_saved['accuracy'][-1]:.4f}
   Best val acc:       {max(hist_saved['val_accuracy']):.4f}

🎯 ĐÁNH GIÁ TEST SET
   Accuracy:  {metrics_saved['accuracy']:.4f}  ({metrics_saved['accuracy']*100:.2f}%)
""")

for cls_data in metrics_saved["per_class"]:
    print(f"   {cls_data['class']:<15} "
          f"Precision={cls_data['precision']:.4f}  "
          f"Recall={cls_data['recall']:.4f}  "
          f"F1={cls_data['f1']:.4f}  "
          f"Support={cls_data['support']:,}")

print(f"""
📱 TFLITE EXPORT
   Kích thước:         {tflite_kb:.1f} KB
   TFLite accuracy:    {tflite_acc:.4f}  ({tflite_acc*100:.2f}%)
   Path:               {tflite_path}

📂 OUTPUT FILES
""")
for p in sorted(OUTPUT_ROOT.rglob("*")):
    if p.is_file():
        size = p.stat().st_size / 1024
        print(f"   {str(p.relative_to(OUTPUT_ROOT)):<45} {size:>7.1f} KB")

# %% ───────────────────────────────────────────────────────────────
# CELL 20 ▸ INTERACTIVE DEMO — GRADIO
# ──────────────────────────────────────────────────────────────────
"""
Demo tương tác bằng Gradio:
- Upload ảnh khuôn mặt
- Phân tích MediaPipe (EAR / MAR)
- Phân loại CNN/TFLite
- Trả kết quả: AWAKE / EYES_CLOSED / YAWNING / DROWSY
"""
import gradio as gr

def predict_drowsiness(image: np.ndarray):
    """Hàm xử lý chính của Gradio"""
    if image is None:
        return None, "## ❌ Chưa có ảnh — vui lòng upload ảnh khuôn mặt"

    # ── MediaPipe analysis ────────────────────────────────────────
    mp_result = analyze_face(image)
    if mp_result is None:
        return image, "## ❌ Không phát hiện khuôn mặt\nHãy đảm bảo ảnh chụp rõ khuôn mặt, đủ ánh sáng"

    ear, mar = mp_result["ear"], mp_result["mar"]

    # ── CNN prediction (crop vùng mắt gần đúng) ──────────────────
    pil_img = Image.fromarray(image)
    h, w = image.shape[:2]
    # Crop 1/4 trên-giữa của ảnh làm vùng mắt
    eye_crop  = pil_img.crop((w//5, h//6, 4*w//5, h//2))
    cnn_pred  = classify_eye_tflite(eye_crop)

    # ── Fusion logic ──────────────────────────────────────────────
    if mar > MAR_THRESH:
        final_state = "😴 YAWNING"
        state_icon  = "🟠"
    elif ear < EAR_THRESH:
        final_state = "👁️ EYES CLOSED (EAR)"
        state_icon  = "🔴"
    elif cnn_pred["label"] == "eyes_closed" and cnn_pred["conf"] >= 0.70:
        final_state = "👁️ EYES CLOSED (CNN)"
        state_icon  = "🔴"
    else:
        final_state = "✅ AWAKE"
        state_icon  = "🟢"

    # ── Vẽ lên ảnh ───────────────────────────────────────────────
    annotated = draw_analysis(image, mp_result, cnn_pred)

    # ── Báo cáo Markdown ─────────────────────────────────────────
    ear_warn  = " ⚠️" if ear < EAR_THRESH  else " ✅"
    mar_warn  = " ⚠️" if mar > MAR_THRESH  else " ✅"
    cnn_warn  = " ⚠️" if cnn_pred["label"] == "eyes_closed" else " ✅"

    report = f"""## {state_icon} Kết quả: **{final_state}**

---
### 📐 MediaPipe Metrics
| Chỉ số | Giá trị | Ngưỡng | Trạng thái |
|--------|---------|--------|------------|
| EAR (Eye Aspect Ratio) | `{ear:.4f}` | < {EAR_THRESH} | {ear_warn} {mp_result['eye_state']} |
| MAR (Mouth Aspect Ratio) | `{mar:.4f}` | > {MAR_THRESH} | {mar_warn} {mp_result['mouth_state']} |

---
### 🧠 CNN/TFLite Prediction
| | |
|---|---|
| Nhãn dự đoán | **{cnn_pred['label']}** {cnn_warn} |
| Độ tin cậy | **{cnn_pred['conf']*100:.1f}%** |
| `eyes_closed` | {cnn_pred['probs'].get('eyes_closed', 0)*100:.1f}% |
| `eyes_open`   | {cnn_pred['probs'].get('eyes_open', 0)*100:.1f}% |

---
### 💡 Giải thích
- **EAR < {EAR_THRESH}** → mắt nhắm (buồn ngủ có thể)
- **MAR > {MAR_THRESH}** → miệng há to (đang ngáp)
- **CNN confidence ≥ 70%** → dùng CNN, ngược lại dùng EAR/MAR
"""
    return annotated, report


# ── Giao diện Gradio ─────────────────────────────────────────────
demo = gr.Interface(
    fn=predict_drowsiness,
    inputs=gr.Image(
        label="📸 Upload ảnh khuôn mặt (hoặc chụp webcam)",
        sources=["upload", "webcam"],
    ),
    outputs=[
        gr.Image(label="🎭 Kết quả MediaPipe Landmarks"),
        gr.Markdown(label="📊 Báo cáo phân tích"),
    ],
    title="🚗 Drowsy Driver Detection — IS54A Demo",
    description=(
        "**Upload ảnh khuôn mặt** để hệ thống phân tích:\n"
        "- MediaPipe tính EAR (Eye Aspect Ratio) và MAR (Mouth Aspect Ratio)\n"
        "- CNN/TFLite phân loại trạng thái mắt (open/closed)\n"
        "- Hệ thống kết hợp (Fusion) đưa ra kết luận: AWAKE / EYES_CLOSED / YAWNING"
    ),
    flagging_mode="never",
    theme=gr.themes.Soft(primary_hue="blue"),
)

print("\n🚀 Khởi động Gradio demo...")
if IN_COLAB:
    demo.launch(share=True, quiet=True)   # Colab → tạo public URL
    print("✅ Public URL đã được tạo ở trên (dạng https://xxx.gradio.live)")
else:
    demo.launch(server_port=7860, quiet=True)
    print("✅ Demo đang chạy tại: http://localhost:7860")

# %% ───────────────────────────────────────────────────────────────
# CELL 21 ▸ COPY TFLITE VÀO ANDROID PROJECT  (chỉ cần khi local)
# ──────────────────────────────────────────────────────────────────
"""
Sau khi train xong, copy file .tflite vào dự án Android:
"""
if not IN_COLAB:
    android_assets = Path("app/src/main/assets")
    if android_assets.exists():
        dest = android_assets / "drowsiness_model.tflite"
        shutil.copy2(tflite_path, dest)
        print(f"✅ Đã copy TFLite → {dest} ({dest.stat().st_size/1024:.1f} KB)")
    else:
        print(f"ℹ️  Không tìm thấy Android assets folder ({android_assets})")
        print(f"   Copy thủ công: {tflite_path} → app/src/main/assets/drowsiness_model.tflite")
else:
    # Colab → download
    print("📥 Download file TFLite về máy (nếu cần dùng cho Android):")
    print(f"   Path trên Colab: {tflite_path}")
    try:
        from google.colab import files
        files.download(str(tflite_path))
    except Exception:
        print("   Gọi files.download() thủ công trong cell riêng nếu cần")

# %% ───────────────────────────────────────────────────────────────
# CELL 22 ▸ DOWNLOAD ALL OUTPUTS  (Colab)
# ──────────────────────────────────────────────────────────────────
if IN_COLAB:
    zip_path = "/content/drowsy_outputs"
    shutil.make_archive(zip_path, "zip", str(OUTPUT_ROOT))
    from google.colab import files
    files.download(zip_path + ".zip")
    print("✅ Tất cả outputs đã được download (drowsy_outputs.zip)")
else:
    print("✅ Tất cả outputs đã lưu tại:")
    for p in sorted(OUTPUT_ROOT.rglob("*")):
        if p.is_file():
            print(f"   {p.relative_to(OUTPUT_ROOT)}")
