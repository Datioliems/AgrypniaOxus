# %% [markdown]
# # Train CNN Yawn Classifier
# Phan loai mieng: no_yawn (0) / yawn (1)
# Mo trong VS Code → bam **Run Cell** tung o theo thu tu
#
# Thu tu chay: Cell 1 → Cell 2 → Cell 3 → Cell 4 → Cell 5 → Cell 6 → Cell 7

# %% Cell 1 — Chuan bi dataset yawn (tu dong split 80/10/10)
import os, random, shutil
from pathlib import Path

os.environ["PYTHONUTF8"] = "1"

PROJECT_ROOT = Path(__file__).resolve().parent
SOURCE_DIR   = PROJECT_ROOT / "rawdata" / "data" / "yawn"
DATA_DIR     = PROJECT_ROOT / "dataset_yawn"
OUT_DIR      = PROJECT_ROOT / "outputs" / "cnn_yawn"
TFLITE_OUT   = PROJECT_ROOT / "app" / "src" / "main" / "assets" / "yawn_model.tflite"

OUT_DIR.mkdir(parents=True, exist_ok=True)
TFLITE_OUT.parent.mkdir(parents=True, exist_ok=True)

TRAIN_RATIO = 0.80
VAL_RATIO   = 0.10
SEED        = 42

CLASS_NAME_MAP = {"no yawn": "no_yawn", "no_yawn": "no_yawn", "yawn": "yawn"}
VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

if DATA_DIR.exists() and any(DATA_DIR.rglob("*.jpg")):
    print("dataset_yawn da co:")
    for sp in ["train", "val", "test"]:
        for cls in ["no_yawn", "yawn"]:
            d = DATA_DIR / sp / cls
            if d.exists():
                n = sum(1 for _ in d.rglob("*.*"))
                print(f"  {sp}/{cls}: {n} anh")
else:
    print("Dang split dataset yawn (80/10/10)...")
    random.seed(SEED)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    for src_folder in sorted(SOURCE_DIR.iterdir()):
        if not src_folder.is_dir(): continue
        cls_name = CLASS_NAME_MAP.get(src_folder.name.lower())
        if not cls_name: continue

        imgs = sorted(f for f in src_folder.rglob("*") if f.suffix.lower() in VALID_EXTS)
        random.shuffle(imgs)

        n = len(imgs)
        n_train = int(n * TRAIN_RATIO)
        n_val   = int(n * VAL_RATIO)
        splits  = {"train": imgs[:n_train],
                   "val":   imgs[n_train:n_train+n_val],
                   "test":  imgs[n_train+n_val:]}

        for split, files in splits.items():
            dst = DATA_DIR / split / cls_name
            dst.mkdir(parents=True, exist_ok=True)
            for f in files:
                shutil.copy2(f, dst / f.name)
            print(f"  {split}/{cls_name}: {len(files)} anh")

    print("Split xong!")

# %% Cell 2 — Cau hinh
IMAGE_SIZE    = 64
BATCH_SIZE    = 32
EPOCHS        = 20
LEARNING_RATE = 1e-3
DROPOUT       = 0.30
PATIENCE      = 5

print("=== CONFIG ===")
print(f"  Image size   : {IMAGE_SIZE}x{IMAGE_SIZE}")
print(f"  Batch size   : {BATCH_SIZE}")
print(f"  Epochs       : {EPOCHS}")
print(f"  Learning rate: {LEARNING_RATE}")
print(f"  Output       : {TFLITE_OUT}")

import tensorflow as tf
print(f"\nTensorFlow: {tf.__version__}")
print(f"GPU: {tf.config.list_physical_devices('GPU') or 'CPU only'}")

# %% Cell 3 — Load dataset
import tensorflow as tf, json

IMG = (IMAGE_SIZE, IMAGE_SIZE)

train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR / "train",
    image_size=IMG, batch_size=BATCH_SIZE,
    label_mode="categorical", seed=42, shuffle=True,
)
val_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR / "val",
    image_size=IMG, batch_size=BATCH_SIZE,
    label_mode="categorical", shuffle=False,
)
test_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR / "test",
    image_size=IMG, batch_size=BATCH_SIZE,
    label_mode="categorical", shuffle=False,
)

CLASS_NAMES = train_ds.class_names
print("=== CLASS ORDER ===")
for i, name in enumerate(CLASS_NAMES):
    print(f"  {i} = '{name}'")
# no_yawn=0, yawn=1 (alphabetical order)

(OUT_DIR / "class_names.json").write_text(
    json.dumps(CLASS_NAMES, indent=2), encoding="utf-8"
)

def normalize(x, y):
    return tf.cast(x, tf.float32) / 255.0, y

train_ds = train_ds.map(normalize, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
val_ds   = val_ds.map(normalize,   num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
test_ds  = test_ds.map(normalize,  num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)

print(f"\nTrain: {len(train_ds)} batches")
print(f"Val  : {len(val_ds)} batches")
print(f"Test : {len(test_ds)} batches")

# %% Cell 4 — Xay dung model CNN
import tensorflow as tf

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3)),

    # Block 1
    tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(),

    # Block 2
    tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(),

    # Block 3
    tf.keras.layers.Conv2D(128, 3, padding="same", activation="relu"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.GlobalAveragePooling2D(),

    tf.keras.layers.Dropout(DROPOUT),
    tf.keras.layers.Dense(len(CLASS_NAMES), activation="softmax"),
], name="DrowsyCNN_Yawn")

model.compile(
    optimizer=tf.keras.optimizers.Adam(LEARNING_RATE),
    loss="categorical_crossentropy",
    metrics=["accuracy",
             tf.keras.metrics.Precision(name="precision"),
             tf.keras.metrics.Recall(name="recall")],
)
model.summary()

# %% Cell 5 — TRAIN  (~10-15 phut)
import tensorflow as tf

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy", patience=PATIENCE,
        restore_best_weights=True, verbose=1, min_delta=0.001,
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=2,
        min_lr=1e-7, verbose=1,
    ),
    tf.keras.callbacks.ModelCheckpoint(
        filepath=str(OUT_DIR / "best_model.keras"),
        monitor="val_accuracy", save_best_only=True, verbose=1,
    ),
]

print(f"Bat dau train {EPOCHS} epochs...")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=2,
)

best_acc = max(history.history["val_accuracy"])
print(f"\nXong! Best val_accuracy: {best_acc*100:.2f}%")

import json
(OUT_DIR / "training_history.json").write_text(
    json.dumps({k: [float(v) for v in vals]
                for k, vals in history.history.items()}, indent=2),
    encoding="utf-8"
)

# %% Cell 6 — Danh gia tren Test set
import tensorflow as tf

best_path = OUT_DIR / "best_model.keras"
eval_model = tf.keras.models.load_model(str(best_path)) if best_path.exists() else model

metrics_dict = eval_model.evaluate(test_ds, verbose=1, return_dict=True)

print("\n=== KET QUA TEST SET ===")
print(f"  Accuracy  : {metrics_dict.get('accuracy',0)*100:.2f}%")
print(f"  Precision : {metrics_dict.get('precision',0)*100:.2f}%")
print(f"  Recall    : {metrics_dict.get('recall',0)*100:.2f}%")

val_acc  = max(history.history["val_accuracy"])
test_acc = metrics_dict.get("accuracy", 0)
gap = abs(val_acc - test_acc)
print(f"\n  Val acc   : {val_acc*100:.2f}%")
print(f"  Test acc  : {test_acc*100:.2f}%")
print(f"  Gap       : {gap*100:.2f}% {'[OK]' if gap < 0.05 else '[Kiem tra overfitting]'}")

pred_probs = eval_model.predict(test_ds, verbose=0)
y_pred = tf.argmax(pred_probs, axis=1).numpy()
y_true = tf.concat([tf.argmax(y, axis=1) for _, y in test_ds], axis=0).numpy()
cm = tf.math.confusion_matrix(y_true, y_pred, num_classes=len(CLASS_NAMES)).numpy()
per_class_metrics = {}
print("\n=== PER-CLASS METRICS ===")
for i, name in enumerate(CLASS_NAMES):
    tp = int(cm[i, i])
    fp = int(cm[:, i].sum() - tp)
    fn = int(cm[i, :].sum() - tp)
    precision_i = tp / max(tp + fp, 1)
    recall_i = tp / max(tp + fn, 1)
    f1_i = 2 * precision_i * recall_i / max(precision_i + recall_i, 1e-12)
    per_class_metrics[name] = {
        "precision": round(float(precision_i), 6),
        "recall": round(float(recall_i), 6),
        "f1": round(float(f1_i), 6),
        "support": int(cm[i, :].sum()),
    }
    print(f"  {name:<8} precision={precision_i*100:.2f}% recall={recall_i*100:.2f}% f1={f1_i*100:.2f}%")

# %% Cell 7 — Export TFLite + kiem tra
import tensorflow as tf, json

best_path = OUT_DIR / "best_model.keras"
export_model = tf.keras.models.load_model(str(best_path)) if best_path.exists() else model

converter = tf.lite.TFLiteConverter.from_keras_model(export_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_bytes = converter.convert()

TFLITE_OUT.write_bytes(tflite_bytes)

tflite_kb = TFLITE_OUT.stat().st_size / 1024
print(f"TFLite saved: {TFLITE_OUT}")
print(f"  Size: {tflite_kb:.1f} KB")

# Verify
interp = tf.lite.Interpreter(model_path=str(TFLITE_OUT))
interp.allocate_tensors()
inp_d = interp.get_input_details()[0]
out_d = interp.get_output_details()[0]
print(f"Input : shape={inp_d['shape']}  dtype={inp_d['dtype'].__name__}")
print(f"Output: shape={out_d['shape']}  dtype={out_d['dtype'].__name__}")
print(f"Classes: {CLASS_NAMES}")

summary = {
    "model": "DrowsyCNN_Yawn",
    "classes": CLASS_NAMES,
    "image_size": IMAGE_SIZE,
    "best_val_accuracy": round(max(history.history["val_accuracy"]), 6),
    "test_accuracy": round(float(metrics_dict.get("accuracy", 0)), 6),
    "test_precision": round(float(metrics_dict.get("precision", 0)), 6),
    "test_recall": round(float(metrics_dict.get("recall", 0)), 6),
    "per_class_metrics": per_class_metrics,
    "confusion_matrix": cm.tolist(),
    "tflite_path": str(TFLITE_OUT),
    "tflite_kb": round(tflite_kb, 1),
}
(OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

print("\n" + "="*50)
print("  HOAN THANH CNN YAWN TRAINING!")
print(f"  Best val acc : {max(history.history['val_accuracy'])*100:.2f}%")
print(f"  TFLite       : {TFLITE_OUT}")
print(f"  Class order  : no_yawn=0, yawn=1")
print("="*50)
print("\nBuoc tiep theo: Mo train_clean_local.py (YOLO)")
