# %% [markdown]
# # Train CNN Eye Classifier
# Phan loai mat: eyes_closed (0) / eyes_open (1)
# Mo trong VS Code → bam **Run Cell** tung o theo thu tu
#
# Thu tu chay: Cell 1 → Cell 2 → Cell 3 → Cell 4 → Cell 5 → Cell 6 → Cell 7

# %% Cell 1 — Kiem tra dataset & GPU
import os, json
from pathlib import Path

os.environ["PYTHONUTF8"] = "1"

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR     = PROJECT_ROOT / "dataset_mrl"
OUT_DIR      = PROJECT_ROOT / "outputs" / "cnn_eye"
TFLITE_OUT   = PROJECT_ROOT / "app" / "src" / "main" / "assets" / "drowsiness_model.tflite"

OUT_DIR.mkdir(parents=True, exist_ok=True)
TFLITE_OUT.parent.mkdir(parents=True, exist_ok=True)

# Dem anh
print("=== DATASET ===")
total = 0
for split in ["train", "val", "test"]:
    for cls in ["eyes_closed", "eyes_open"]:
        d = DATA_DIR / split / cls
        if d.exists():
            n = sum(1 for _ in d.rglob("*.*"))
            total += n
            print(f"  {split}/{cls}: {n:,} anh")
print(f"  Tong: {total:,} anh")

# Kiem tra TensorFlow
import tensorflow as tf
print(f"\nTensorFlow: {tf.__version__}")
gpus = tf.config.list_physical_devices("GPU")
print(f"GPU: {gpus if gpus else 'Khong co GPU (dung CPU)'}")

# %% Cell 2 — Cau hinh (chinh tai day neu muon)
IMAGE_SIZE   = 64      # kich thuoc anh dau vao (64x64)
BATCH_SIZE   = 32      # so anh moi batch
EPOCHS       = 25      # so epoch toi da (EarlyStopping se dung som)
LEARNING_RATE= 1e-3    # Adam learning rate
DROPOUT      = 0.30    # Dropout rate
PATIENCE     = 5       # EarlyStopping patience

print("=== CONFIG ===")
print(f"  Image size  : {IMAGE_SIZE}x{IMAGE_SIZE}")
print(f"  Batch size  : {BATCH_SIZE}")
print(f"  Epochs      : {EPOCHS} (max, EarlyStopping co the dung som hon)")
print(f"  Learning rate: {LEARNING_RATE}")
print(f"  Dropout     : {DROPOUT}")
print(f"  Output TFLite: {TFLITE_OUT}")

# %% Cell 3 — Load dataset
import tensorflow as tf

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
print("=== CLASS ORDER (phai khop Android) ===")
for i, name in enumerate(CLASS_NAMES):
    print(f"  {i} = '{name}'")

# Luu class names
(OUT_DIR / "class_names.json").write_text(
    json.dumps(CLASS_NAMES, indent=2), encoding="utf-8"
)

# Normalize /255 NGOAI model (quan trong!)
def normalize(x, y):
    return tf.cast(x, tf.float32) / 255.0, y

train_ds = train_ds.map(normalize, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
val_ds   = val_ds.map(normalize,   num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
test_ds  = test_ds.map(normalize,  num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)

print(f"\nTrain batches: {len(train_ds)}")
print(f"Val   batches: {len(val_ds)}")
print(f"Test  batches: {len(test_ds)}")

# %% Cell 4 — Xay dung model CNN
import tensorflow as tf

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3)),

    # Block 1: 64x64 → 32x32
    tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(),

    # Block 2: 32x32 → 16x16
    tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(),

    # Block 3: 16x16 → GAP
    tf.keras.layers.Conv2D(128, 3, padding="same", activation="relu"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.GlobalAveragePooling2D(),

    tf.keras.layers.Dropout(DROPOUT),
    tf.keras.layers.Dense(len(CLASS_NAMES), activation="softmax"),
], name="DrowsyCNN_Eye")

model.compile(
    optimizer=tf.keras.optimizers.Adam(LEARNING_RATE),
    loss="categorical_crossentropy",
    metrics=["accuracy",
             tf.keras.metrics.Precision(name="precision"),
             tf.keras.metrics.Recall(name="recall")],
)
model.summary()

# %% Cell 5 — TRAIN  (~15-20 phut)
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

# Luu history
import json
(OUT_DIR / "training_history.json").write_text(
    json.dumps({k: [float(v) for v in vals]
                for k, vals in history.history.items()}, indent=2),
    encoding="utf-8"
)

# %% Cell 6 — Danh gia tren Test set
import tensorflow as tf, numpy as np

# Load best model
best_model_path = OUT_DIR / "best_model.keras"
if best_model_path.exists():
    eval_model = tf.keras.models.load_model(str(best_model_path))
    print("Dung best checkpoint")
else:
    eval_model = model

metrics_dict = eval_model.evaluate(test_ds, verbose=1, return_dict=True)

print("\n=== KET QUA TEST SET ===")
print(f"  Loss      : {metrics_dict.get('loss', 0):.4f}")
print(f"  Accuracy  : {metrics_dict.get('accuracy', 0)*100:.2f}%")
print(f"  Precision : {metrics_dict.get('precision', 0)*100:.2f}%")
print(f"  Recall    : {metrics_dict.get('recall', 0)*100:.2f}%")

val_acc = max(history.history["val_accuracy"])
test_acc = metrics_dict.get("accuracy", 0)
gap = abs(val_acc - test_acc)
print(f"\n  Val acc   : {val_acc*100:.2f}%")
print(f"  Test acc  : {test_acc*100:.2f}%")
print(f"  Gap       : {gap*100:.2f}% {'[OK]' if gap < 0.05 else '[Kiem tra overfitting]'}")

pred_probs = eval_model.predict(test_ds, verbose=0)
y_pred = np.argmax(pred_probs, axis=1)
y_true = np.concatenate([np.argmax(y.numpy(), axis=1) for _, y in test_ds], axis=0)
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
    print(f"  {name:<12} precision={precision_i*100:.2f}% recall={recall_i*100:.2f}% f1={f1_i*100:.2f}%")

# %% Cell 7 — Export TFLite + kiem tra
import tensorflow as tf, json

best_model_path = OUT_DIR / "best_model.keras"
export_model = tf.keras.models.load_model(str(best_model_path)) if best_model_path.exists() else model

# Convert → TFLite INT8 quantization
converter = tf.lite.TFLiteConverter.from_keras_model(export_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_bytes = converter.convert()

TFLITE_OUT.parent.mkdir(parents=True, exist_ok=True)
TFLITE_OUT.write_bytes(tflite_bytes)

# Kich thuoc
keras_kb  = best_model_path.stat().st_size / 1024 if best_model_path.exists() else 0
tflite_kb = TFLITE_OUT.stat().st_size / 1024
print(f"TFLite saved: {TFLITE_OUT}")
print(f"  Keras  : {keras_kb:.1f} KB")
print(f"  TFLite : {tflite_kb:.1f} KB")

# Verify TFLite
interp = tf.lite.Interpreter(model_path=str(TFLITE_OUT))
interp.allocate_tensors()
inp_d = interp.get_input_details()[0]
out_d = interp.get_output_details()[0]
print(f"\nTFLite Input : shape={inp_d['shape']}  dtype={inp_d['dtype'].__name__}")
print(f"TFLite Output: shape={out_d['shape']}  dtype={out_d['dtype'].__name__}")
print(f"Classes      : {CLASS_NAMES}")

# Luu summary
summary = {
    "model": "DrowsyCNN_Eye",
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
print("  HOAN THANH CNN EYE TRAINING!")
print(f"  Best val acc : {max(history.history['val_accuracy'])*100:.2f}%")
print(f"  TFLite       : {TFLITE_OUT}")
print(f"  Android class: eyes_closed=0, eyes_open=1")
print("="*50)
print("\nBuoc tiep theo: Mo train_cnn_yawn.py")
