# %% [markdown]
# # CNN Eye + Yawn voi CBAM Attention
# **Phuong phap 2: CBAM (Convolutional Block Attention Module)**
#
# CBAM them 2 lop Attention vao sau moi Conv block:
#   1. Channel Attention — "Feature map nao quan trong?"
#      Avg-pool + Max-pool -> Shared MLP -> Sigmoid -> scale tung channel
#
#   2. Spatial Attention — "Vi tri nao trong anh quan trong?"
#      Concat(avg_spatial, max_spatial) -> Conv7x7 -> Sigmoid -> scale tung pixel
#      => Model hoc focus vao vung MAT / MIENG, bo qua nen!
#
# Kien truc so sanh:
#   Baseline : Conv(32)+BN+MaxPool -> Conv(64)+BN+MaxPool -> Conv(128)+BN+GAP -> Drop+Dense
#   CBAM     : Conv(32)+BN+[CBAM]+MaxPool -> Conv(64)+BN+[CBAM]+MaxPool -> Conv(128)+BN+[CBAM]+GAP -> Drop+Dense
#
# Chay trong VS Code (Run Cell tung o) hoac Colab
# Thu tu: Cell 1 -> 2 -> 3 -> 4 -> 5 -> 6 -> 7 -> (8 cho Yawn)

# %% Cell 1 — Kiem tra dataset & GPU
import os, json
from pathlib import Path

os.environ["PYTHONUTF8"] = "1"

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR_EYE  = PROJECT_ROOT / "dataset_mrl"
DATA_DIR_YAWN = PROJECT_ROOT / "dataset_yawn"
OUT_DIR_EYE   = PROJECT_ROOT / "outputs" / "cnn_eye_cbam"
OUT_DIR_YAWN  = PROJECT_ROOT / "outputs" / "cnn_yawn_cbam"
TFLITE_EYE    = PROJECT_ROOT / "app" / "src" / "main" / "assets" / "drowsiness_cbam.tflite"
TFLITE_YAWN   = PROJECT_ROOT / "app" / "src" / "main" / "assets" / "yawn_cbam.tflite"

for d in [OUT_DIR_EYE, OUT_DIR_YAWN]:
    d.mkdir(parents=True, exist_ok=True)
TFLITE_EYE.parent.mkdir(parents=True, exist_ok=True)

print("=== DATASET EYE ===")
for split in ["train", "val", "test"]:
    for cls in ["eyes_closed", "eyes_open"]:
        d = DATA_DIR_EYE / split / cls
        if d.exists():
            n = sum(1 for _ in d.rglob("*.*"))
            print(f"  {split}/{cls}: {n:,}")

import tensorflow as tf
print(f"\nTensorFlow: {tf.__version__}")
gpus = tf.config.list_physical_devices("GPU")
print(f"GPU: {gpus if gpus else 'CPU only'}")

# %% Cell 2 — Config
IMAGE_SIZE    = 64       # Phai khop Android INPUT_SIZE = 64
BATCH_SIZE    = 32
EPOCHS        = 25
LEARNING_RATE = 1e-3
DROPOUT       = 0.30
PATIENCE      = 5
CBAM_RATIO    = 8        # Channel reduction ratio: channels / CBAM_RATIO

print("=== CONFIG ===")
print(f"  IMAGE_SIZE    : {IMAGE_SIZE}x{IMAGE_SIZE}  (Android = 64)")
print(f"  CBAM_RATIO    : 1/{CBAM_RATIO}  (chia channel cho {CBAM_RATIO} trong MLP)")
print(f"  EPOCHS        : {EPOCHS}")
print(f"  DROPOUT       : {DROPOUT}")

# %% Cell 3 — Load dataset Eye
import tensorflow as tf

IMG = (IMAGE_SIZE, IMAGE_SIZE)

train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR_EYE / "train", image_size=IMG, batch_size=BATCH_SIZE,
    label_mode="categorical", seed=42, shuffle=True,
)
val_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR_EYE / "val", image_size=IMG, batch_size=BATCH_SIZE,
    label_mode="categorical", shuffle=False,
)
test_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR_EYE / "test", image_size=IMG, batch_size=BATCH_SIZE,
    label_mode="categorical", shuffle=False,
)

CLASS_NAMES = train_ds.class_names
print(f"CLASS ORDER (phai la ['eyes_closed','eyes_open']):")
for i, n in enumerate(CLASS_NAMES):
    print(f"  {i} = '{n}'")

def normalize(x, y):
    return tf.cast(x, tf.float32) / 255.0, y

train_ds = train_ds.map(normalize, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
val_ds   = val_ds.map(normalize,   num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
test_ds  = test_ds.map(normalize,  num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)

(OUT_DIR_EYE / "class_names.json").write_text(json.dumps(CLASS_NAMES, indent=2), encoding="utf-8")

# %% Cell 4 — Dinh nghia CBAM + xay dung model
import tensorflow as tf

# ──────────────────────────────────────────────────────────────
# Channel Attention Module
# Hoc CHANNEL nao quan trong (32, 64, hoac 128 feature map nao?)
# ──────────────────────────────────────────────────────────────
def channel_attention(x, ratio=CBAM_RATIO):
    channels = x.shape[-1]

    # Shared 2-layer MLP (dung chung cho avg va max path)
    shared_l1 = tf.keras.layers.Dense(channels // ratio, activation="relu",
                                        use_bias=True, kernel_initializer="he_normal")
    shared_l2 = tf.keras.layers.Dense(channels,
                                        use_bias=True, kernel_initializer="he_normal")

    # Path 1: Global Average Pool -> MLP
    avg = tf.keras.layers.GlobalAveragePooling2D()(x)           # [B, C]
    avg = shared_l2(shared_l1(avg))                              # [B, C]

    # Path 2: Global Max Pool -> MLP
    mx  = tf.keras.layers.GlobalMaxPooling2D()(x)               # [B, C]
    mx  = shared_l2(shared_l1(mx))                              # [B, C]

    # Cong 2 path, qua Sigmoid
    scale = tf.keras.layers.Activation("sigmoid")(
        tf.keras.layers.Add()([avg, mx])
    )                                                            # [B, C]
    scale = tf.keras.layers.Reshape((1, 1, channels))(scale)    # [B, 1, 1, C]

    # Nhan voi feature map goc
    return tf.keras.layers.Multiply()([x, scale])               # [B, H, W, C]


# ──────────────────────────────────────────────────────────────
# Spatial Attention Module
# Hoc VI TRI nao trong anh quan trong (focus vao mat / mieng)
# ──────────────────────────────────────────────────────────────
def spatial_attention(x, kernel_size=7):
    # Tinh average va max tren chieu channel
    avg_spatial = tf.keras.layers.Lambda(
        lambda t: tf.reduce_mean(t, axis=-1, keepdims=True))(x)  # [B, H, W, 1]
    max_spatial = tf.keras.layers.Lambda(
        lambda t: tf.reduce_max(t,  axis=-1, keepdims=True))(x)  # [B, H, W, 1]

    concat = tf.keras.layers.Concatenate(axis=-1)(
        [avg_spatial, max_spatial]
    )                                                             # [B, H, W, 2]

    # 7x7 conv de capture vung anh lon (mat, mieng)
    scale = tf.keras.layers.Conv2D(
        1, kernel_size, padding="same",
        activation="sigmoid", use_bias=False,
        kernel_initializer="glorot_uniform"
    )(concat)                                                     # [B, H, W, 1]

    return tf.keras.layers.Multiply()([x, scale])                # [B, H, W, C]


def cbam_block(x, ratio=CBAM_RATIO):
    """CBAM = Channel Attention -> Spatial Attention (thu tu trong paper goc)"""
    x = channel_attention(x, ratio)
    x = spatial_attention(x)
    return x


# ──────────────────────────────────────────────────────────────
# Model: Phai dung Functional API (khong dung Sequential)
# vi CBAM co non-sequential flow (GlobalPool -> Reshape -> Multiply)
# ──────────────────────────────────────────────────────────────
def build_cbam_model(image_size, num_classes, dropout=DROPOUT, name="DrowsyCNN_CBAM_Eye"):
    inputs = tf.keras.Input(shape=(image_size, image_size, 3))

    # Block 1: 64x64 -> 32x32
    x = tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu")(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = cbam_block(x)                         # <- Channel + Spatial Attention
    x = tf.keras.layers.MaxPooling2D()(x)

    # Block 2: 32x32 -> 16x16
    x = tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = cbam_block(x)                         # <- Channel + Spatial Attention
    x = tf.keras.layers.MaxPooling2D()(x)

    # Block 3: 16x16 -> GAP
    x = tf.keras.layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = cbam_block(x)                         # <- Focus vao pixel mat/mieng!
    x = tf.keras.layers.GlobalAveragePooling2D()(x)

    x = tf.keras.layers.Dropout(dropout)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    return tf.keras.Model(inputs, outputs, name=name)


model_eye = build_cbam_model(IMAGE_SIZE, len(CLASS_NAMES), name="DrowsyCNN_CBAM_Eye")
model_eye.compile(
    optimizer=tf.keras.optimizers.Adam(LEARNING_RATE),
    loss="categorical_crossentropy",
    metrics=["accuracy",
             tf.keras.metrics.Precision(name="precision"),
             tf.keras.metrics.Recall(name="recall")],
)
model_eye.summary()

n_params = model_eye.count_params()
print(f"\nCBAM CNN Eye: {n_params:,} params")
print(f"Baseline CNN Eye: ~59,234 params  (khong co CBAM)")
print(f"Overhead CBAM   : +{n_params - 59234:,} params (+{(n_params/59234-1)*100:.1f}%)")

# %% Cell 5 — TRAIN CNN Eye CBAM  (~20-25 phut)
import tensorflow as tf

callbacks_eye = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy", patience=PATIENCE,
        restore_best_weights=True, verbose=1, min_delta=0.001,
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=2,
        min_lr=1e-7, verbose=1,
    ),
    tf.keras.callbacks.ModelCheckpoint(
        filepath=str(OUT_DIR_EYE / "best_model.keras"),
        monitor="val_accuracy", save_best_only=True, verbose=1,
    ),
]

print(f"Bat dau train CNN Eye CBAM ({EPOCHS} epochs max)...")
history_eye = model_eye.fit(
    train_ds, validation_data=val_ds,
    epochs=EPOCHS, callbacks=callbacks_eye, verbose=2,
)
best_val_eye = max(history_eye.history["val_accuracy"])
print(f"\nBest val_accuracy (CBAM Eye): {best_val_eye*100:.2f}%")

import json
(OUT_DIR_EYE / "training_history.json").write_text(
    json.dumps({k: [float(v) for v in vals] for k, vals in history_eye.history.items()}, indent=2),
    encoding="utf-8",
)

# %% Cell 6 — Danh gia Eye + So sanh baseline
import tensorflow as tf

eval_model = model_eye

metrics = eval_model.evaluate(test_ds, verbose=1, return_dict=True)

print("\n=== CBAM CNN Eye — Test Set ===")
print(f"  Accuracy  : {metrics.get('accuracy',0)*100:.2f}%")
print(f"  Precision : {metrics.get('precision',0)*100:.2f}%")
print(f"  Recall    : {metrics.get('recall',0)*100:.2f}%")

pred_probs = eval_model.predict(test_ds, verbose=0)
y_pred = tf.argmax(pred_probs, axis=1).numpy()
y_true = tf.concat([tf.argmax(y, axis=1) for _, y in test_ds], axis=0).numpy()
cm_eye = tf.math.confusion_matrix(y_true, y_pred, num_classes=len(CLASS_NAMES)).numpy()
per_class_eye = {}
print("\n=== CBAM Eye PER-CLASS METRICS ===")
for i, name in enumerate(CLASS_NAMES):
    tp = int(cm_eye[i, i])
    fp = int(cm_eye[:, i].sum() - tp)
    fn = int(cm_eye[i, :].sum() - tp)
    precision_i = tp / max(tp + fp, 1)
    recall_i = tp / max(tp + fn, 1)
    f1_i = 2 * precision_i * recall_i / max(precision_i + recall_i, 1e-12)
    per_class_eye[name] = {
        "precision": round(float(precision_i), 6),
        "recall": round(float(recall_i), 6),
        "f1": round(float(f1_i), 6),
        "support": int(cm_eye[i, :].sum()),
    }
    print(f"  {name:<12} precision={precision_i*100:.2f}% recall={recall_i*100:.2f}% f1={f1_i*100:.2f}%")

# So sanh voi baseline
baseline_path = PROJECT_ROOT / "outputs" / "cnn_eye" / "summary.json"
if baseline_path.exists():
    with open(baseline_path) as f:
        bl = json.load(f)
    print()
    print(f"  Baseline  : {bl['best_val_accuracy']*100:.2f}%  ({bl.get('tflite_kb',0):.1f} KB)")
    print(f"  CBAM      : {best_val_eye*100:.2f}%")
    print(f"  Delta     : {(best_val_eye - bl['best_val_accuracy'])*100:+.2f}%")

# %% Cell 7 — Export TFLite Eye CBAM
import tensorflow as tf, json

export_model = model_eye

converter = tf.lite.TFLiteConverter.from_keras_model(export_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_bytes = converter.convert()
TFLITE_EYE.write_bytes(tflite_bytes)

# Verify
interp = tf.lite.Interpreter(model_path=str(TFLITE_EYE))
interp.allocate_tensors()
inp_d = interp.get_input_details()[0]
out_d = interp.get_output_details()[0]
print(f"TFLite Input : shape={inp_d['shape']}  dtype={inp_d['dtype'].__name__}")
print(f"TFLite Output: shape={out_d['shape']}  dtype={out_d['dtype'].__name__}")
print(f"Classes      : {CLASS_NAMES}")

ok_shape = inp_d["shape"].tolist() == [1, 64, 64, 3]
ok_dtype = inp_d["dtype"].__name__ == "float32"
print(f"Android compat: {'[OK]' if ok_shape and ok_dtype else '[FAIL — kiem tra lai]'}")

summary_eye = {
    "model": "DrowsyCNN_CBAM_Eye",
    "classes": CLASS_NAMES,
    "image_size": IMAGE_SIZE,
    "cbam_ratio": CBAM_RATIO,
    "total_params": int(model_eye.count_params()),
    "best_val_accuracy": round(best_val_eye, 6),
    "test_accuracy": round(float(metrics.get("accuracy", 0)), 6),
    "test_precision": round(float(metrics.get("precision", 0)), 6),
    "test_recall": round(float(metrics.get("recall", 0)), 6),
    "per_class_metrics": per_class_eye,
    "confusion_matrix": cm_eye.tolist(),
    "tflite_path": str(TFLITE_EYE),
    "tflite_kb": round(TFLITE_EYE.stat().st_size / 1024, 1),
}
(OUT_DIR_EYE / "summary.json").write_text(json.dumps(summary_eye, indent=2), encoding="utf-8")
print(f"\nSaved: {TFLITE_EYE.name}  ({summary_eye['tflite_kb']:.1f} KB)")

# %% Cell 8 — Train CNN Yawn CBAM (cung kien truc, khac data)
# Chay cell nay sau khi Eye da xong
# dataset_yawn/ can ton tai (Cell 1 cua train_cnn_yawn.ipynb se tao)

import tensorflow as tf

if not DATA_DIR_YAWN.exists() or not any(DATA_DIR_YAWN.rglob("*.jpg")):
    print("dataset_yawn/ chua co!")
    print("Hay chay Cell 1 cua train_cnn_yawn.ipynb truoc, roi quay lai day.")
else:
    # Load Yawn dataset
    train_yawn = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR_YAWN / "train", image_size=IMG, batch_size=BATCH_SIZE,
        label_mode="categorical", seed=42, shuffle=True,
    )
    val_yawn = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR_YAWN / "val", image_size=IMG, batch_size=BATCH_SIZE,
        label_mode="categorical", shuffle=False,
    )
    test_yawn = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR_YAWN / "test", image_size=IMG, batch_size=BATCH_SIZE,
        label_mode="categorical", shuffle=False,
    )

    YAWN_CLASSES = train_yawn.class_names
    print(f"Yawn classes (phai la ['no_yawn','yawn']): {YAWN_CLASSES}")

    train_yawn = train_yawn.map(normalize, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
    val_yawn   = val_yawn.map(normalize,   num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
    test_yawn  = test_yawn.map(normalize,  num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)

    # Xay dung model Yawn CBAM (cung ham build_cbam_model)
    model_yawn = build_cbam_model(IMAGE_SIZE, len(YAWN_CLASSES),
                                   dropout=0.30, name="DrowsyCNN_CBAM_Yawn")
    model_yawn.compile(
        optimizer=tf.keras.optimizers.Adam(LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy",
                 tf.keras.metrics.Precision(name="precision"),
                 tf.keras.metrics.Recall(name="recall")],
    )

    callbacks_yawn = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=PATIENCE,
            restore_best_weights=True, verbose=1, min_delta=0.001,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=2, min_lr=1e-7, verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(OUT_DIR_YAWN / "best_model.keras"),
            monitor="val_accuracy", save_best_only=True, verbose=1,
        ),
    ]

    print("\nBat dau train CNN Yawn CBAM...")
    history_yawn = model_yawn.fit(
        train_yawn, validation_data=val_yawn,
        epochs=20, callbacks=callbacks_yawn, verbose=2,
    )
    best_val_yawn = max(history_yawn.history["val_accuracy"])
    print(f"Best val_accuracy (CBAM Yawn): {best_val_yawn*100:.2f}%")

    yawn_metrics = model_yawn.evaluate(test_yawn, verbose=1, return_dict=True)
    yawn_probs = model_yawn.predict(test_yawn, verbose=0)
    yawn_pred = tf.argmax(yawn_probs, axis=1).numpy()
    yawn_true = tf.concat([tf.argmax(y, axis=1) for _, y in test_yawn], axis=0).numpy()
    cm_yawn = tf.math.confusion_matrix(yawn_true, yawn_pred, num_classes=len(YAWN_CLASSES)).numpy()
    per_class_yawn = {}
    print("\n=== CBAM Yawn PER-CLASS METRICS ===")
    for i, name in enumerate(YAWN_CLASSES):
        tp = int(cm_yawn[i, i])
        fp = int(cm_yawn[:, i].sum() - tp)
        fn = int(cm_yawn[i, :].sum() - tp)
        precision_i = tp / max(tp + fp, 1)
        recall_i = tp / max(tp + fn, 1)
        f1_i = 2 * precision_i * recall_i / max(precision_i + recall_i, 1e-12)
        per_class_yawn[name] = {
            "precision": round(float(precision_i), 6),
            "recall": round(float(recall_i), 6),
            "f1": round(float(f1_i), 6),
            "support": int(cm_yawn[i, :].sum()),
        }
        print(f"  {name:<8} precision={precision_i*100:.2f}% recall={recall_i*100:.2f}% f1={f1_i*100:.2f}%")

    # Export Yawn TFLite
    exp_yawn = model_yawn

    converter_y = tf.lite.TFLiteConverter.from_keras_model(exp_yawn)
    converter_y.optimizations = [tf.lite.Optimize.DEFAULT]
    TFLITE_YAWN.write_bytes(converter_y.convert())

    interp_y = tf.lite.Interpreter(model_path=str(TFLITE_YAWN))
    interp_y.allocate_tensors()
    inp_y = interp_y.get_input_details()[0]
    out_y = interp_y.get_output_details()[0]
    print(f"\nYawn TFLite Input : shape={inp_y['shape']}  dtype={inp_y['dtype'].__name__}")
    print(f"Yawn TFLite Output: shape={out_y['shape']}  dtype={out_y['dtype'].__name__}")
    print(f"Classes           : {YAWN_CLASSES}")

    summary_yawn = {
        "model": "DrowsyCNN_CBAM_Yawn",
        "classes": YAWN_CLASSES,
        "image_size": IMAGE_SIZE,
        "cbam_ratio": CBAM_RATIO,
        "best_val_accuracy": round(best_val_yawn, 6),
        "test_accuracy": round(float(yawn_metrics.get("accuracy", 0)), 6),
        "test_precision": round(float(yawn_metrics.get("precision", 0)), 6),
        "test_recall": round(float(yawn_metrics.get("recall", 0)), 6),
        "per_class_metrics": per_class_yawn,
        "confusion_matrix": cm_yawn.tolist(),
        "tflite_kb": round(TFLITE_YAWN.stat().st_size / 1024, 1),
    }
    (OUT_DIR_YAWN / "summary.json").write_text(json.dumps(summary_yawn, indent=2), encoding="utf-8")

    print("\n" + "="*50)
    print("  HOAN THANH CBAM CNN TRAINING!")
    print(f"  Eye  CBAM val acc: {best_val_eye*100:.2f}%")
    print(f"  Yawn CBAM val acc: {best_val_yawn*100:.2f}%")
    print("="*50)
    print("\nDe dung CBAM model tren Android:")
    print("  1. Doi MODEL_FILE trong TfliteDrowsinessClassifier.kt")
    print('     tu "drowsiness_model.tflite" -> "drowsiness_cbam.tflite"')
    print("  2. Lam tuong tu cho YawnClassifier.kt -> yawn_cbam.tflite")
    print("  3. Rebuild APK")
