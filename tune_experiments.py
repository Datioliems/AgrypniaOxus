# %% [markdown]
# # 🔬 Tinh chỉnh mô hình — Drowsy Driver Detection
#
# **Cách dùng trong VS Code:**
# 1. Cài extension: `ms-toolsai.jupyter` (Jupyter) + `ms-python.python`
# 2. Mở file này trong VS Code
# 3. Bấm **"Run Cell"** (▶) trên từng cell `# %%`
# 4. Kết quả + biểu đồ hiện ngay bên phải
#
# **Quy trình tinh chỉnh:**
# Cell 1 → Cấu hình  →  Cell 2 → Load data  →  Cell 3 → Build model
# → Cell 4 → Train  →  Cell 5 → Đánh giá  →  Cell 6 → So sánh thực nghiệm

# %% ════════════════════════════════════════════════════════════════
# CELL 1 — ⚙️  CẤU HÌNH  (THAY ĐỔI Ở ĐÂY)
# ═══════════════════════════════════════════════════════════════════
# fmt: off
# ┌─────────────────────────────────────────────────────────────────┐
# │  Thay đổi bất kỳ giá trị nào bên dưới rồi Run Cell → xem kết  │
# │  quả thay đổi ngay, không cần chạy lại toàn bộ script          │
# └─────────────────────────────────────────────────────────────────┘

# ── 1. Chọn mô hình cần tinh chỉnh ──────────────────────────────────
MODEL_TARGET = "eye"          # "eye" | "yawn"

# ── 2. Dataset ────────────────────────────────────────────────────
DATASET = {
    "eye":  "dataset_mrl",    # đã chia sẵn train/val/test bởi prepare_mrl_dataset.py
    "yawn": "dataset_yawn",   # đã chia sẵn train/val/test bởi prepare_yawn_dataset.py
}[MODEL_TARGET]

# ── 3. Kiến trúc CNN ────────────────────────────────────────────────
IMAGE_SIZE   = 64           # ← thử: 48 (nhanh hơn) | 96 (chính xác hơn)
FILTERS      = [32, 64, 128] # ← thử: [64,128,256] cho model lớn hơn
DROPOUT      = 0.30         # ← thử: 0.20 (ít dropout) | 0.45 (nhiều dropout)
DENSE_UNITS  = 0            # ← 0 = không có Dense ẩn. Thử: 64 | 128
USE_BN       = True         # ← BatchNormalization sau mỗi Conv block

# ── 4. Training hyperparameters ─────────────────────────────────────
LEARNING_RATE  = 1e-3       # ← thử: 1e-4 (chậm, ổn định) | 3e-3 (nhanh)
BATCH_SIZE     = 32         # ← thử: 16 | 64
EPOCHS         = 30         # ← tối đa, EarlyStopping sẽ dừng sớm
OPTIMIZER_NAME = "adam"     # ← "adam" | "sgd" | "rmsprop"
LR_DECAY       = True       # ← ReduceLROnPlateau
EARLY_STOP_PATIENCE = 5     # ← số epoch không cải thiện → dừng

# ── 5. Data Augmentation ─────────────────────────────────────────────
AUG_ROTATION   = 10         # ← độ xoay tối đa. 0 = tắt, thử: 5 | 15
AUG_ZOOM       = 0.10       # ← zoom in/out. 0 = tắt, thử: 0.05 | 0.15
AUG_BRIGHTNESS = 0.15       # ← thay đổi độ sáng. 0 = tắt, thử: 0.10 | 0.20
AUG_FLIP_H     = True       # ← lật ngang. Hợp lý cho mắt (đối xứng)
AUG_FLIP_V     = False      # ← lật dọc. KHÔNG dùng cho khuôn mặt

# ── 6. Android threshold (tuỳ chỉnh sau khi model xong) ─────────────
CNN_CONF_THRESHOLD  = 0.55  # ← thử: 0.50 (nhạy hơn) | 0.65 (ít false alarm)
YAWN_CONF_THRESHOLD = 0.60  # ← thử: 0.55 | 0.70
DROWSY_MS_THRESHOLD = 1200  # ← ms mắt nhắm liên tục → alert. Thử: 800 | 1500
# fmt: on

# In config ra để xác nhận
print("=" * 55)
print(f"🎯  Model target : {MODEL_TARGET.upper()}")
print(f"   Dataset      : {DATASET}")
print(f"   Image size   : {IMAGE_SIZE}×{IMAGE_SIZE}")
print(f"   Filters      : {FILTERS}")
print(f"   Dropout      : {DROPOUT}")
print(f"   Dense units  : {DENSE_UNITS if DENSE_UNITS else 'None'}")
print(f"   BatchNorm    : {USE_BN}")
print(f"   LR           : {LEARNING_RATE}")
print(f"   Batch        : {BATCH_SIZE}")
print(f"   Epochs       : {EPOCHS} (max)")
print(f"   Optimizer    : {OPTIMIZER_NAME}")
print(f"   Aug: rot={AUG_ROTATION}° zoom={AUG_ZOOM} bright={AUG_BRIGHTNESS}")
print("=" * 55)

# %% ════════════════════════════════════════════════════════════════
# CELL 2 — 📦  LOAD DATASET
# ═══════════════════════════════════════════════════════════════════
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

print(f"TensorFlow: {tf.__version__}")
print(f"GPU: {tf.config.list_physical_devices('GPU')}")

PROJECT_ROOT = Path(".")
data_path    = PROJECT_ROOT / DATASET
img_size     = (IMAGE_SIZE, IMAGE_SIZE)


def load_datasets(data_path: Path):
    """
    Load train / val / test từ thư mục đã chia sẵn.

    Yêu cầu cấu trúc:
        dataset_mrl/   hoặc   dataset_yawn/
            train/
                class_a/
                class_b/
            val/
                class_a/
                class_b/
            test/          ← chỉ dùng ở Cell 5 để đánh giá cuối
                class_a/
                class_b/

    Nếu chưa có thư mục train/ → nhắc chạy prepare script trước.
    """
    if not (data_path / "train").exists():
        raise FileNotFoundError(
            f"\n❌ Không tìm thấy {data_path / 'train'}\n"
            f"   Chạy trước:\n"
            f"   • eye:  python tools/prepare_mrl_dataset.py\n"
            f"   • yawn: python tools/prepare_yawn_dataset.py"
        )

    def make_ds(split: str, shuffle: bool):
        return tf.keras.utils.image_dataset_from_directory(
            data_path / split,
            image_size=img_size,
            batch_size=BATCH_SIZE,
            label_mode="categorical",
            seed=42,
            shuffle=shuffle,
        )

    train_ds = make_ds("train", shuffle=True)
    val_ds   = make_ds("val",   shuffle=False)

    # test set — tải nếu có, None nếu chưa có
    test_ds = make_ds("test", shuffle=False) if (data_path / "test").exists() else None

    class_names = train_ds.class_names
    print(f"\n✅ Classes ({len(class_names)}): {class_names}")

    # Đếm nhanh bằng cách đọc thư mục (không unbatch — nhanh hơn nhiều)
    def count_files(split_dir):
        return sum(
            len(list(d.glob("*")))
            for d in (data_path / split_dir).iterdir()
            if d.is_dir()
        )

    n_train = count_files("train")
    n_val   = count_files("val")
    n_test  = count_files("test") if test_ds else 0

    print(f"   Train : {n_train:,} ảnh")
    print(f"   Val   : {n_val:,} ảnh")
    if test_ds:
        print(f"   Test  : {n_test:,} ảnh  ← chỉ dùng ở Cell 5")
    else:
        print(f"   Test  : (chưa có)")

    return train_ds, val_ds, test_ds, class_names, n_train, n_val


train_ds_raw, val_ds_raw, test_ds_raw, CLASS_NAMES, n_train, n_val = load_datasets(data_path)
NUM_CLASSES = len(CLASS_NAMES)
print(f"\n   Ratio: {n_train/(n_train+n_val)*100:.0f}% train / {n_val/(n_train+n_val)*100:.0f}% val")

# Xem phân phối class
class_counts = {}
for _, labels in train_ds_raw.unbatch():
    idx = int(tf.argmax(labels))
    class_counts[CLASS_NAMES[idx]] = class_counts.get(CLASS_NAMES[idx], 0) + 1

print(f"\n📊 Phân phối Train:")
for cls, cnt in class_counts.items():
    bar = "█" * (cnt // 1000)
    pct = cnt / n_train * 100
    print(f"   {cls:15s}: {cnt:6,}  ({pct:.1f}%)  {bar}")

# Cân bằng class không?
counts = list(class_counts.values())
imbalance = max(counts) / min(counts) if min(counts) > 0 else 999
if imbalance > 1.5:
    print(f"\n⚠️  Class imbalance ratio: {imbalance:.2f}x — nên dùng class_weight!")
else:
    print(f"\n✅ Class balanced (ratio: {imbalance:.2f}x)")

# %% ════════════════════════════════════════════════════════════════
# CELL 3 — 🏗️  XEM MẪU + BUILD MODEL
# ═══════════════════════════════════════════════════════════════════

# ── Xem 12 ảnh mẫu ──────────────────────────────────────────────
fig, axes = plt.subplots(2, 6, figsize=(14, 5))
for images, labels in train_ds_raw.take(1):
    for i, ax in enumerate(axes.flat):
        if i < len(images):
            ax.imshow(images[i].numpy().astype("uint8"))
            ax.set_title(CLASS_NAMES[int(tf.argmax(labels[i]))], fontsize=8)
        ax.axis("off")
plt.suptitle(f"Mẫu dataset — {MODEL_TARGET.upper()}", fontsize=12)
plt.tight_layout()
plt.show()

# ── Build model từ config ────────────────────────────────────────

def build_model_from_config():
    """Xây dựng CNN dựa hoàn toàn vào config ở Cell 1"""
    layers = [tf.keras.layers.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3))]

    # Augmentation layers (chỉ active khi training=True)
    # KHÔNG đặt Rescaling ở đây — normalize /255 thực hiện NGOÀI model
    # → đảm bảo Android (TfliteDrowsinessClassifier.kt) nhất quán
    if AUG_ROTATION > 0:
        layers.append(tf.keras.layers.RandomRotation(AUG_ROTATION / 360))
    if AUG_ZOOM > 0:
        layers.append(tf.keras.layers.RandomZoom(AUG_ZOOM))
    if AUG_BRIGHTNESS > 0:
        layers.append(tf.keras.layers.RandomBrightness(AUG_BRIGHTNESS))
    if AUG_FLIP_H:
        layers.append(tf.keras.layers.RandomFlip("horizontal"))
    if AUG_FLIP_V:
        layers.append(tf.keras.layers.RandomFlip("vertical"))

    # Conv blocks
    for n_filters in FILTERS:
        layers.append(tf.keras.layers.Conv2D(n_filters, 3, padding="same", activation="relu"))
        if USE_BN:
            layers.append(tf.keras.layers.BatchNormalization())
        if n_filters == FILTERS[-1]:
            # Cuối cùng dùng GlobalAveragePooling thay MaxPool
            layers.append(tf.keras.layers.GlobalAveragePooling2D())
        else:
            layers.append(tf.keras.layers.MaxPooling2D())

    # Dense ẩn (optional)
    if DENSE_UNITS > 0:
        layers.append(tf.keras.layers.Dense(DENSE_UNITS, activation="relu"))
        if USE_BN:
            layers.append(tf.keras.layers.BatchNormalization())

    # Dropout + Output
    layers.append(tf.keras.layers.Dropout(DROPOUT))
    layers.append(tf.keras.layers.Dense(NUM_CLASSES, activation="softmax"))

    model = tf.keras.Sequential(layers, name=f"DrowsyCNN_{MODEL_TARGET}")
    return model


# Chọn optimizer
def get_optimizer():
    opts = {
        "adam":    tf.keras.optimizers.Adam(LEARNING_RATE),
        "sgd":     tf.keras.optimizers.SGD(LEARNING_RATE, momentum=0.9, nesterov=True),
        "rmsprop": tf.keras.optimizers.RMSprop(LEARNING_RATE),
    }
    return opts.get(OPTIMIZER_NAME, opts["adam"])


model = build_model_from_config()
model.compile(
    optimizer=get_optimizer(),
    loss="categorical_crossentropy",
    metrics=["accuracy",
             tf.keras.metrics.Precision(name="precision"),
             tf.keras.metrics.Recall(name="recall")],
)

# In kiến trúc
model.summary()

# Tính params
total_params = model.count_params()
print(f"\n📊 Total parameters: {total_params:,}")
print(f"   Model complexity: {'Nhẹ ✅' if total_params < 500_000 else 'Vừa ⚠️' if total_params < 2_000_000 else 'Nặng ❌ (Android chậm)'}")

# Test forward pass
dummy = tf.random.uniform((1, IMAGE_SIZE, IMAGE_SIZE, 3))
out   = model(dummy, training=False)
print(f"   Output shape: {out.shape}  (classes: {NUM_CLASSES})")
print(f"   Sum of probs: {float(tf.reduce_sum(out)):.4f}  (phải = 1.0)")

# %% ════════════════════════════════════════════════════════════════
# CELL 4 — 🏋️  TRAIN
# ═══════════════════════════════════════════════════════════════════

# Normalize /255 NGOÀI model — nhất quán với Android và train_eye_classifier.py
# Augmentation nằm trong model nên vẫn chạy TRƯỚC normalize (đúng thứ tự)
def normalize(images, labels):
    return tf.cast(images, tf.float32) / 255.0, labels

train_ds = train_ds_raw.map(normalize, num_parallel_calls=tf.data.AUTOTUNE)\
                        .cache().shuffle(2000).prefetch(tf.data.AUTOTUNE)
val_ds   = val_ds_raw.map(normalize,   num_parallel_calls=tf.data.AUTOTUNE)\
                      .cache().prefetch(tf.data.AUTOTUNE)

# Class weights nếu imbalanced
class_weight = None
if imbalance > 1.5:
    total = sum(counts)
    class_weight = {
        i: total / (NUM_CLASSES * cnt)
        for i, (cls, cnt) in enumerate(class_counts.items())
    }
    print(f"⚖️  Class weights: {class_weight}")

# Callbacks
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy", patience=EARLY_STOP_PATIENCE,
        restore_best_weights=True, verbose=1, min_delta=0.001,
    ),
]
if LR_DECAY:
    callbacks.append(tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=3,
        min_lr=1e-7, verbose=1,
    ))

# Train
print(f"\n🏋️  Training {MODEL_TARGET.upper()} model...")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks,
    class_weight=class_weight,
    verbose=1,
)

# Kết quả ngay
best_val_acc = max(history.history["val_accuracy"])
epochs_ran   = len(history.history["accuracy"])
print(f"\n{'='*50}")
print(f"✅ Done! Epochs: {epochs_ran}/{EPOCHS}")
print(f"   Best val_accuracy: {best_val_acc:.4f} ({best_val_acc*100:.2f}%)")
print(f"{'='*50}")

# %% ════════════════════════════════════════════════════════════════
# CELL 5 — 📊  ĐỒ THỊ & ĐÁNH GIÁ
# ═══════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Accuracy
axes[0].plot(history.history["accuracy"],     label="Train", linewidth=2)
axes[0].plot(history.history["val_accuracy"], label="Val",   linewidth=2)
axes[0].set_title("Accuracy", fontsize=12)
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Accuracy")
axes[0].legend()
axes[0].grid(True, alpha=0.3)
axes[0].set_ylim(0, 1.05)

# Loss
axes[1].plot(history.history["loss"],     label="Train", linewidth=2)
axes[1].plot(history.history["val_loss"], label="Val",   linewidth=2)
axes[1].set_title("Loss", fontsize=12)
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Loss")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# Learning rate (nếu có)
if "lr" in history.history:
    axes[2].plot(history.history["lr"], color="orange", linewidth=2)
    axes[2].set_title("Learning Rate", fontsize=12)
    axes[2].set_xlabel("Epoch")
    axes[2].set_yscale("log")
    axes[2].grid(True, alpha=0.3)
else:
    axes[2].text(0.5, 0.5, "No LR schedule", ha="center", va="center",
                 transform=axes[2].transAxes)
    axes[2].set_title("Learning Rate", fontsize=12)

plt.suptitle(
    f"{MODEL_TARGET.upper()} CNN — img={IMAGE_SIZE} filters={FILTERS} "
    f"dropout={DROPOUT} lr={LEARNING_RATE}",
    fontsize=10,
)
plt.tight_layout()
plt.savefig(f"outputs/tune_{MODEL_TARGET}_curves.png", dpi=100)
plt.show()

# ── Hàm đánh giá chung cho bất kỳ split nào ────────────────────
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

def evaluate_split(ds_raw, split_name: str, color="Blues"):
    """Chạy model trên 1 split, vẽ confusion matrix, in classification report"""
    if ds_raw is None:
        print(f"⚠️  Không có {split_name} set — bỏ qua")
        return None, None

    y_true, y_pred = [], []
    # Dùng ds đã normalize (không dùng raw)
    ds_norm = ds_raw.map(normalize, num_parallel_calls=tf.data.AUTOTUNE)
    for images, labels in ds_norm:
        preds = model(images, training=False)
        y_pred.extend(tf.argmax(preds, axis=1).numpy())
        y_true.extend(tf.argmax(labels, axis=1).numpy())

    acc = sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true)
    cm  = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap=color,
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix — {split_name} set  (acc={acc*100:.2f}%)")
    plt.tight_layout()
    plt.savefig(f"outputs/tune_{MODEL_TARGET}_cm_{split_name.lower()}.png", dpi=100)
    plt.show()

    print(f"\n📋 Classification Report ({split_name} set):")
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4))
    return acc, cm

# Đánh giá Val set (theo dõi trong lúc train)
val_acc_final, _ = evaluate_split(val_ds_raw, "Val", color="Blues")

# Đánh giá Test set (kết quả thật — chỉ nhìn 1 lần sau cùng)
print("\n" + "═"*50)
print("📌 TEST SET — kết quả THẬT (không dùng để tune)")
print("═"*50)
test_acc_final, _ = evaluate_split(test_ds_raw, "Test", color="Greens")

if test_acc_final:
    print(f"\n{'='*50}")
    print(f"  Val  accuracy : {val_acc_final*100:.2f}%")
    print(f"  Test accuracy : {test_acc_final*100:.2f}%")
    gap = abs(val_acc_final - test_acc_final) * 100
    if gap > 3:
        print(f"  ⚠️  Chênh lệch {gap:.1f}% — có thể overfit val set")
    else:
        print(f"  ✅ Chênh lệch {gap:.1f}% — model tổng quát tốt")
    print(f"{'='*50}")

# %% ════════════════════════════════════════════════════════════════
# CELL 6 — 💾  EXPORT TFLITE (chạy khi hài lòng với kết quả)
# ═══════════════════════════════════════════════════════════════════

# ✅ Normalize /255 thực hiện NGOÀI model (nhất quán với Android)
# → Android (TfliteDrowsinessClassifier.kt + YawnClassifier.kt) đưa pixel /255 vào TFLite
# → Python eval cũng dùng /255 trước khi gọi TFLite
# → Không có double normalization

OUTPUT_ASSETS = {
    "eye":  "app/src/main/assets/drowsiness_model.tflite",
    "yawn": "app/src/main/assets/yawn_model.tflite",
}
out_path = Path(OUTPUT_ASSETS[MODEL_TARGET])
out_path.parent.mkdir(parents=True, exist_ok=True)

# Export TFLite với INT8 quantization
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Đại diện dataset cho quantization — phải normalize /255 trước (giống lúc train)
def representative_data():
    for images, _ in val_ds_raw.take(50):
        yield [tf.cast(images, tf.float32) / 255.0]

converter.representative_dataset   = representative_data
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type      = tf.float32   # Android input vẫn float
converter.inference_output_type     = tf.float32   # output vẫn float

tflite_bytes = converter.convert()
out_path.write_bytes(tflite_bytes)

print(f"✅ TFLite saved: {out_path}")
print(f"   Size: {len(tflite_bytes)/1024:.1f} KB")
print(f"   Best val_accuracy: {best_val_acc*100:.2f}%")

# Verify
interp = tf.lite.Interpreter(model_path=str(out_path))
interp.allocate_tensors()
inp_d = interp.get_input_details()[0]
out_d = interp.get_output_details()[0]
print(f"\n🔍 TFLite verify:")
print(f"   Input : {inp_d['shape']}  dtype={inp_d['dtype']}")
print(f"   Output: {out_d['shape']}  dtype={out_d['dtype']}")
print(f"   Classes: {CLASS_NAMES}")

# ⚠️ Ghi chú quan trọng về normalize
print(f"""
╔══════════════════════════════════════════════════════════╗
║  ✅ NORMALIZE NHẤT QUÁN                                  ║
║                                                          ║
║  Python (file này):                                      ║
║    normalize /255 NGOÀI model qua .map()                 ║
║                                                          ║
║  Android (Kotlin):                                       ║
║    buffer.putFloat(channel / 255f)   ← ĐÚNG             ║
║                                                          ║
║  → Cả hai đưa giá trị [0.0, 1.0] vào TFLite             ║
║  → Không có double normalization                         ║
╚══════════════════════════════════════════════════════════╝
""")

# %% ════════════════════════════════════════════════════════════════
# CELL 7 — 🧪  SO SÁNH NHIỀU THỬ NGHIỆM (chạy nhiều lần Cell 1→5)
# ═══════════════════════════════════════════════════════════════════
import json
from pathlib import Path

# Lưu kết quả thực nghiệm hiện tại
EXPERIMENTS_LOG = Path("outputs/experiments_log.json")
EXPERIMENTS_LOG.parent.mkdir(parents=True, exist_ok=True)

# Load lịch sử
if EXPERIMENTS_LOG.exists():
    all_experiments = json.loads(EXPERIMENTS_LOG.read_text())
else:
    all_experiments = []

# Thêm thực nghiệm hiện tại
current_exp = {
    "model":      MODEL_TARGET,
    "image_size": IMAGE_SIZE,
    "filters":    FILTERS,
    "dropout":    DROPOUT,
    "dense_units":DENSE_UNITS,
    "lr":         LEARNING_RATE,
    "batch":      BATCH_SIZE,
    "optimizer":  OPTIMIZER_NAME,
    "aug_rot":    AUG_ROTATION,
    "aug_zoom":   AUG_ZOOM,
    "aug_bright": AUG_BRIGHTNESS,
    "epochs_ran": epochs_ran,
    "best_val_acc": round(best_val_acc, 6),
    "total_params": int(model.count_params()),
}
all_experiments.append(current_exp)
EXPERIMENTS_LOG.write_text(json.dumps(all_experiments, indent=2))

# In bảng so sánh
print(f"\n{'='*80}")
print(f"{'#':>3} {'img':>4} {'filters':>18} {'drop':>5} {'lr':>7} {'val_acc':>9} {'params':>10}")
print(f"{'-'*80}")
for i, e in enumerate(all_experiments):
    if e.get("model") != MODEL_TARGET:
        continue
    marker = " ← hiện tại" if i == len(all_experiments) - 1 else ""
    print(
        f"{i+1:>3} "
        f"{e['image_size']:>4} "
        f"{str(e['filters']):>18} "
        f"{e['dropout']:>5.2f} "
        f"{e['lr']:>7.0e} "
        f"{e['best_val_acc']*100:>8.2f}% "
        f"{e['total_params']:>10,}"
        f"{marker}"
    )

# Model tốt nhất
best = max(
    (e for e in all_experiments if e.get("model") == MODEL_TARGET),
    key=lambda e: e["best_val_acc"],
    default=None,
)
if best:
    print(f"\n🏆 Best so far: val_acc={best['best_val_acc']*100:.2f}%")
    print(f"   Config: img={best['image_size']} filters={best['filters']} "
          f"drop={best['dropout']} lr={best['lr']}")

print(f"\n💡 Để thử nghiệm mới: quay lại Cell 1, thay đổi config → Run All Cells")

# %% ════════════════════════════════════════════════════════════════
# CELL 8 — 📱  CẬP NHẬT THRESHOLD ANDROID (sau khi model đã tốt)
# ═══════════════════════════════════════════════════════════════════
# Sau khi quyết định model cuối cùng, cập nhật ngưỡng trong Android

ANDROID_CNN_THRESHOLD  = CNN_CONF_THRESHOLD
ANDROID_YAWN_THRESHOLD = YAWN_CONF_THRESHOLD
ANDROID_DROWSY_MS      = DROWSY_MS_THRESHOLD

print("📋 Cập nhật file Android với ngưỡng mới:")
print()
print("═" * 55)
print("▶  TfliteDrowsinessClassifier.kt  ←  CNN Eye threshold")
print(f"   const val CNN_CONF_THRESHOLD = {ANDROID_CNN_THRESHOLD}f")
print()
print("▶  YawnClassifier.kt  ←  CNN Yawn threshold")
print(f"   const val YAWN_CONF_THRESHOLD = {ANDROID_YAWN_THRESHOLD}f")
print()
print("▶  MainActivity.kt  ←  Thời gian mắt nhắm → DROWSY")
print(f"   cnnClosedMs >= {ANDROID_DROWSY_MS}_L -> DriverState.DROWSY")
print("═" * 55)
print()
print("💡 Ngưỡng thấp hơn → phát hiện sớm hơn, nhiều false alarm hơn")
print("💡 Ngưỡng cao hơn  → ít false alarm, có thể bỏ sót")
print()
print("Gợi ý theo use case:")
print("  Ban đêm / mệt mỏi  → CNN_CONF=0.50, DROWSY_MS=800")
print("  Ban ngày bình thường → CNN_CONF=0.55, DROWSY_MS=1200  (mặc định)")
print("  Tránh false alarm  → CNN_CONF=0.65, DROWSY_MS=1500")
