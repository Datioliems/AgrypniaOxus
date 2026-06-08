# %% [markdown]
# # Temporal Transformer — Phat hien ngu gat qua chuoi thoi gian
# **Phuong phap 5: Multi-Head Self-Attention tren chuoi EAR**
#
# Y tuong:
#   Nguoi lay xe khong "dot ngot" ngu gat — mat tu tu nham lai qua nhieu giay.
#   Model hien tai (CNN Eye) chi nhin 1 frame -> mat tran the cham.
#   Temporal Transformer nhin chuoi 20 frame lien tiep -> phat hien xu huong.
#
# Input: chuoi EAR (Eye Aspect Ratio) tren N=20 frame
#   frame t-19: EAR = 0.38  (tinh tao)
#   frame t-15: EAR = 0.33  (bat dau giam)
#   frame t-10: EAR = 0.25  (tiep tuc giam)
#   frame t-5:  EAR = 0.18  (gan nham)
#   frame t:    EAR = 0.12  (NHAM MAT)
#            ^ Transformer detect xu huong nay rat tot!
#
# Kien truc:
#   Input [batch, 20, 1]
#     Linear(1 -> d_model=16)
#     + Positional Embedding
#     Transformer Encoder x2 (4 heads, FFN=64)
#     GlobalAvgPool1D
#     Dense(16) -> Dense(2, softmax)
#   Output: [batch, 2] = P(awake), P(drowsy)
#
# Android integration:
#   - Du tri EAR values 20 frame gan nhat trong MainActivity.kt
#   - Moi frame: them EAR moi vao buffer, xoa EAR cu nhat
#   - Goi TFLite tren buffer nay -> probability drowsy
#
# Chay trong VS Code (Run Cell) hoac Colab

# %% Cell 1 — Setup va kiem tra
import os, json
from pathlib import Path

os.environ["PYTHONUTF8"] = "1"

PROJECT_ROOT = Path(__file__).resolve().parent
OUT_DIR      = PROJECT_ROOT / "outputs" / "temporal_transformer"
TFLITE_OUT   = PROJECT_ROOT / "app" / "src" / "main" / "assets" / "temporal_model.tflite"

OUT_DIR.mkdir(parents=True, exist_ok=True)
TFLITE_OUT.parent.mkdir(parents=True, exist_ok=True)

import tensorflow as tf
print(f"TensorFlow: {tf.__version__}")
gpus = tf.config.list_physical_devices("GPU")
print(f"GPU: {gpus if gpus else 'CPU only'}")
print(f"Output TFLite: {TFLITE_OUT}")

# %% Cell 2 — Config
SEQ_LEN       = 20      # So frame lien tiep dua vao model
D_MODEL       = 16      # Chieu feature sau khi embed EAR
N_HEADS       = 4       # So head trong Multi-Head Attention
N_LAYERS      = 2       # So Transformer Encoder layers
FFN_DIM       = 64      # Chieu FFN (Feed-Forward Network)
DROPOUT_RATE  = 0.10
BATCH_SIZE    = 256     # Lon vi dataset synthetic
EPOCHS        = 50
LEARNING_RATE = 1e-3
PATIENCE      = 8
N_TRAIN       = 20000   # So sequence synthetic train
N_VAL         = 4000
N_TEST        = 4000

print("=== CONFIG ===")
print(f"  Sequence length : {SEQ_LEN} frames")
print(f"  d_model         : {D_MODEL}  (EAR 1 -> {D_MODEL} dims)")
print(f"  Attention heads : {N_HEADS}")
print(f"  Encoder layers  : {N_LAYERS}")
print(f"  FFN dim         : {FFN_DIM}")
print(f"  Train sequences : {N_TRAIN:,}")

# %% Cell 3 — Tao du lieu Synthetic EAR Sequences
# Khong can video dataset — tao synthetic sequences hoc duoc
# cac pattern EAR thuong gap khi tinh tao / ngu gat / chop mat

import numpy as np

np.random.seed(42)

def generate_sequences(n_samples: int, seq_len: int = SEQ_LEN) -> tuple:
    """
    Tao synthetic EAR time series.

    Class 0 — Awake (tinh tao):
      - EAR dao dong quanh 0.30-0.38
      - Chop mat nhanh (< 300ms): EAR giam dot ngot roi phuc hoi nhanh

    Class 1 — Drowsy (buon ngu):
      - EAR bat dau binh thuong, giam dan
      - Cuoi sequence: EAR thap va on dinh (mat nam nham)

    Class 2 — Microsleep (ngu gat nga):
      - EAR binh thuong -> dot ngot NHAM HOAN TOAN -> mo mat lai
      (Gop vao class 1 = "drowsy")

    EAR binh thuong:
      - Mat mo rong : ~0.35-0.42
      - Chop mat    : < 0.20 trong 1-2 frame
      - Buon ngu    : < 0.24 lien tuc trong nhieu frame
    """
    X_list, y_list = [], []

    for _ in range(n_samples):
        cls = np.random.randint(0, 2)

        if cls == 0:
            # AWAKE: EAR cao, on dinh, co chop mat ngau nhien
            base_ear = np.random.uniform(0.30, 0.40)
            ear = np.random.normal(base_ear, 0.025, seq_len)

            # Them chop mat ngau nhien (30% khả năng)
            if np.random.random() < 0.30:
                blink_start = np.random.randint(2, seq_len - 3)
                blink_dur   = np.random.randint(1, 3)     # 1-2 frame
                ear[blink_start : blink_start + blink_dur] = np.random.uniform(0.08, 0.18)
                ear[blink_start + blink_dur] = base_ear * np.random.uniform(0.7, 0.9)

        else:
            # DROWSY: EAR giam dan
            pattern = np.random.choice(["gradual", "sudden", "microsleep"])

            if pattern == "gradual":
                # Giam dan tu ~0.33 xuong ~0.14
                start = np.random.normal(0.33, 0.03)
                end   = np.random.normal(0.14, 0.03)
                trend = np.linspace(start, end, seq_len)
                ear   = trend + np.random.normal(0, 0.02, seq_len)

            elif pattern == "sudden":
                # Binh thuong mot luc, sau do NHAM dot ngot va giu nham
                switch = np.random.randint(seq_len // 3, seq_len * 2 // 3)
                ear    = np.zeros(seq_len)
                ear[:switch]  = np.random.normal(0.33, 0.03, switch)
                ear[switch:]  = np.random.normal(0.12, 0.02, seq_len - switch)

            else:
                # Microsleep: nham hoan toan trong 5-8 frame
                ear = np.random.normal(0.33, 0.03, seq_len)
                ms_start = np.random.randint(seq_len // 3, seq_len * 2 // 3)
                ms_dur   = np.random.randint(5, min(10, seq_len - ms_start))
                ear[ms_start : ms_start + ms_dur] = np.random.uniform(0.06, 0.14)

        ear = np.clip(ear, 0.04, 0.55).astype(np.float32)
        X_list.append(ear.reshape(seq_len, 1))
        y_list.append(cls)

    return np.stack(X_list), np.array(y_list, dtype=np.int32)


# Tao dataset
X_train, y_train = generate_sequences(N_TRAIN)
X_val,   y_val   = generate_sequences(N_VAL)
X_test,  y_test  = generate_sequences(N_TEST)

print(f"Train: {X_train.shape}  labels: awake={np.sum(y_train==0)}, drowsy={np.sum(y_train==1)}")
print(f"Val  : {X_val.shape}")
print(f"Test : {X_test.shape}")

# %% Cell 4 — Visualize cac loai EAR sequence
import matplotlib.pyplot as plt
import numpy as np

fig, axes = plt.subplots(2, 3, figsize=(14, 6))
LABELS = ["AWAKE", "DROWSY"]
COLORS = ["#2196F3", "#F44336"]

patterns = [
    ("awake_blink",   X_train[y_train==0][0].flatten()),
    ("awake_stable",  X_train[y_train==0][1].flatten()),
    ("awake_variant", X_train[y_train==0][2].flatten()),
    ("drowsy_grad",   X_train[y_train==1][0].flatten()),
    ("drowsy_sudden", X_train[y_train==1][1].flatten()),
    ("drowsy_micro",  X_train[y_train==1][2].flatten()),
]

for ax, (name, seq) in zip(axes.flat, patterns):
    label_idx = 0 if "awake" in name else 1
    color = COLORS[label_idx]
    ax.plot(range(SEQ_LEN), seq, color=color, linewidth=2)
    ax.axhline(0.24, color="orange", linestyle="--", linewidth=1.0,
               label="EAR threshold=0.24")
    ax.set_ylim(0, 0.55)
    ax.set_xlabel("Frame")
    ax.set_ylabel("EAR")
    ax.set_title(f"{LABELS[label_idx]} — {name.split('_')[1]}")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

plt.suptitle("EAR Sequences: AWAKE vs DROWSY patterns", fontsize=12)
plt.tight_layout()
plt.savefig(str(OUT_DIR / "ear_sequence_patterns.png"), dpi=100, bbox_inches="tight")
plt.show()
print("Saved: ear_sequence_patterns.png")

# %% Cell 5 — Xay dung Temporal Transformer
import tensorflow as tf

@tf.keras.utils.register_keras_serializable(package="DrowsyDriver")
class AddPositionalEmbedding(tf.keras.layers.Layer):
    """Add learnable positional embeddings while preserving dynamic batch size."""

    def __init__(self, seq_len: int, d_model: int, **kwargs):
        super().__init__(**kwargs)
        self.seq_len = seq_len
        self.d_model = d_model
        self.pos_embed = tf.keras.layers.Embedding(
            input_dim=seq_len, output_dim=d_model, name="pos_embed"
        )

    def call(self, x):
        positions = tf.range(start=0, limit=tf.shape(x)[1], delta=1)
        positions = self.pos_embed(positions)
        return x + tf.expand_dims(positions, axis=0)

    def get_config(self):
        config = super().get_config()
        config.update({"seq_len": self.seq_len, "d_model": self.d_model})
        return config

def build_temporal_transformer(
    seq_len=SEQ_LEN,
    d_model=D_MODEL,
    n_heads=N_HEADS,
    n_layers=N_LAYERS,
    ffn_dim=FFN_DIM,
    dropout=DROPOUT_RATE,
    num_classes=2,
):
    """
    Temporal Transformer Encoder cho chuoi EAR.

    Architecture:
      Input: [batch, seq_len=20, 1]  (20 EAR values)
        |
      Linear(1 -> d_model=16)        -- Embed EAR scalar vao khong gian D
        |
      + Positional Embedding          -- Hoc vi tri tung frame trong chuoi
        |
      [Transformer Encoder] x n_layers
        |  Multi-Head Self-Attention  -- Frame nao "quan trong" de quyet dinh?
        |  Add & LayerNorm
        |  Feed-Forward (d_model -> ffn_dim -> d_model)
        |  Add & LayerNorm
        |
      GlobalAveragePooling1D           -- Tong hop thong tin tu toan bo chuoi
        |
      Dense(16, relu) -> Dropout
        |
      Dense(2, softmax)
      Output: [batch, 2] = [P(awake), P(drowsy)]
    """
    inputs = tf.keras.Input(shape=(seq_len, 1), name="ear_sequence")

    # 1) Linear embedding: EAR scalar (1 dim) -> d_model (16 dims)
    x = tf.keras.layers.Dense(d_model, name="ear_embed")(inputs)

    # 2) Learnable positional embedding
    # Dung Embedding layer (hoc tu data) thay vi sin/cos cung
    x = AddPositionalEmbedding(seq_len, d_model, name="add_pos_embed")(x)
    x = tf.keras.layers.Dropout(dropout)(x)

    # 3) Transformer Encoder layers
    for i in range(n_layers):
        # --- Multi-Head Self-Attention ---
        attn_out = tf.keras.layers.MultiHeadAttention(
            num_heads=n_heads,
            key_dim=d_model // n_heads,
            dropout=dropout,
            name=f"mhsa_{i}",
        )(x, x)  # query=x, key=x, value=x (self-attention)

        # Residual + LayerNorm
        x = tf.keras.layers.LayerNormalization(name=f"norm1_{i}")(
            tf.keras.layers.Add()([x, attn_out])
        )

        # --- Feed-Forward Network ---
        ffn = tf.keras.layers.Dense(ffn_dim, activation="relu", name=f"ffn1_{i}")(x)
        ffn = tf.keras.layers.Dropout(dropout)(ffn)
        ffn = tf.keras.layers.Dense(d_model, name=f"ffn2_{i}")(ffn)
        ffn = tf.keras.layers.Dropout(dropout)(ffn)

        # Residual + LayerNorm
        x = tf.keras.layers.LayerNormalization(name=f"norm2_{i}")(
            tf.keras.layers.Add()([x, ffn])
        )

    # 4) Aggregate toan bo chuoi: [batch, seq_len, d_model] -> [batch, d_model]
    x = tf.keras.layers.GlobalAveragePooling1D(name="time_pool")(x)

    # 5) Classifier
    x = tf.keras.layers.Dense(16, activation="relu", name="cls_hidden")(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="cls_out")(x)

    return tf.keras.Model(inputs, outputs, name="TemporalTransformer_EAR")


model = build_temporal_transformer()
model.compile(
    optimizer=tf.keras.optimizers.Adam(LEARNING_RATE),
    loss="sparse_categorical_crossentropy",  # y la integer (0 hoac 1)
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
)
model.summary()

n_params = model.count_params()
print(f"\nTotal params : {n_params:,}  (nhe va nhanh cho realtime!)")

# %% Cell 6 — TRAIN  (~5-10 phut)
import tensorflow as tf

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy", patience=PATIENCE,
        restore_best_weights=True, verbose=1, min_delta=0.001,
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=3,
        min_lr=1e-7, verbose=1,
    ),
    tf.keras.callbacks.ModelCheckpoint(
        filepath=str(OUT_DIR / "best_model.keras"),
        monitor="val_accuracy", save_best_only=True, verbose=1,
    ),
]

print(f"Training Temporal Transformer ({EPOCHS} epochs max)...")
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=2,
)

best_val_acc = max(history.history["val_accuracy"])
print(f"\nBest val_accuracy: {best_val_acc*100:.2f}%")

(OUT_DIR / "training_history.json").write_text(
    json.dumps({k: [float(v) for v in vs] for k, vs in history.history.items()}, indent=2),
    encoding="utf-8",
)

# %% Cell 7 — Danh gia tren Test set
import tensorflow as tf
import numpy as np

eval_model = model

test_results = eval_model.evaluate(X_test, y_test, verbose=1)
test_loss = float(test_results[0])
test_acc = float(test_results[1]) if len(test_results) > 1 else 0.0

print("\n=== Temporal Transformer — Test Set ===")
print(f"  Loss      : {test_loss:.4f}")
print(f"  Accuracy  : {test_acc*100:.2f}%")

# Confusion matrix
from sklearn.metrics import confusion_matrix
preds = eval_model.predict(X_test, verbose=0)
pred_classes = np.argmax(preds, axis=1)
cm = confusion_matrix(y_test, pred_classes)
precision_drowsy = cm[1][1] / max(cm[0][1] + cm[1][1], 1)
recall_drowsy = cm[1][1] / max(cm[1][0] + cm[1][1], 1)
print(f"  Precision(drowsy): {precision_drowsy*100:.2f}%")
print(f"  Recall(drowsy)   : {recall_drowsy*100:.2f}%")
print(f"\nConfusion Matrix:")
print(f"  {'':>12} Pred_Awake  Pred_Drowsy")
print(f"  True_Awake   {cm[0][0]:>8}     {cm[0][1]:>8}")
print(f"  True_Drowsy  {cm[1][0]:>8}     {cm[1][1]:>8}")
print(f"\nFalse Negative Rate (miss drowsy): {cm[1][0]/(cm[1][0]+cm[1][1])*100:.1f}%  <- QUAN TRONG!")
print(f"False Positive Rate (false alarm): {cm[0][1]/(cm[0][0]+cm[0][1])*100:.1f}%")

# Loss/Accuracy curves
import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(history.history["accuracy"],     label="Train")
axes[0].plot(history.history["val_accuracy"], label="Val")
axes[0].set_title("Accuracy"); axes[0].legend(); axes[0].grid(True, alpha=0.3)

axes[1].plot(history.history["loss"],     label="Train")
axes[1].plot(history.history["val_loss"], label="Val")
axes[1].set_title("Loss"); axes[1].legend(); axes[1].grid(True, alpha=0.3)

plt.suptitle("Temporal Transformer Training Curves", fontsize=12)
plt.tight_layout()
plt.savefig(str(OUT_DIR / "training_curves.png"), dpi=100, bbox_inches="tight")
plt.show()

# %% Cell 8 — Export TFLite + Android Integration
import tensorflow as tf, json

export_model = model

converter = tf.lite.TFLiteConverter.from_keras_model(export_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Temporal Transformer dung MultiHeadAttention -> can enable select_tf_ops
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,
    tf.lite.OpsSet.SELECT_TF_OPS,         # Can cho Transformer ops
]
converter._experimental_lower_tensor_list_ops = False  # Tranh loi voi Transformer

tflite_bytes = converter.convert()
TFLITE_OUT.write_bytes(tflite_bytes)

# Verify
interp = tf.lite.Interpreter(model_path=str(TFLITE_OUT))
interp.allocate_tensors()
inp_d = interp.get_input_details()[0]
out_d = interp.get_output_details()[0]
print(f"TFLite Input : shape={inp_d['shape']}  dtype={inp_d['dtype'].__name__}")
print(f"TFLite Output: shape={out_d['shape']}  dtype={out_d['dtype'].__name__}")

# Test: chay inference tren 1 sequence
test_seq = X_test[:1]  # shape [1, 20, 1]
interp.set_tensor(inp_d["index"], test_seq)
interp.invoke()
out = interp.get_output_details()[0]
result = interp.get_tensor(out["index"])
print(f"\nTest inference:")
print(f"  Input EAR: {test_seq[0,:,0]}")
print(f"  P(awake) = {result[0][0]:.4f}")
print(f"  P(drowsy)= {result[0][1]:.4f}")
print(f"  -> {'DROWSY' if result[0][1] >= 0.50 else 'AWAKE'}")

# Luu summary
summary = {
    "model": "TemporalTransformer_EAR",
    "input_spec": {
        "shape":  [1, SEQ_LEN, 1],
        "dtype":  "float32",
        "desc":   f"Last {SEQ_LEN} EAR values (Eye Aspect Ratio)",
        "range":  [0.0, 0.55],
    },
    "output_spec": {
        "shape":  [1, 2],
        "labels": ["awake", "drowsy"],
    },
    "config": {
        "seq_len": SEQ_LEN, "d_model": D_MODEL,
        "n_heads": N_HEADS, "n_layers": N_LAYERS,
    },
    "total_params": int(model.count_params()),
    "best_val_accuracy": round(best_val_acc, 6),
    "test_accuracy": round(test_acc, 6),
    "test_precision_drowsy": round(float(precision_drowsy), 6),
    "test_recall_drowsy": round(float(recall_drowsy), 6),
    "confusion_matrix": cm.tolist(),
    "tflite_kb": round(TFLITE_OUT.stat().st_size / 1024, 1),
}
(OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

print("\n" + "="*55)
print("  HOAN THANH TEMPORAL TRANSFORMER!")
print(f"  Val accuracy  : {best_val_acc*100:.2f}%")
print(f"  Total params  : {model.count_params():,}")
print(f"  TFLite size   : {TFLITE_OUT.stat().st_size/1024:.1f} KB")
print(f"  TFLite output : {TFLITE_OUT}")
print("="*55)
print()
print("Android integration (MainActivity.kt):")
print("  1. Them earBuffer: FloatArray(20) { 0.35f }  // rolling buffer")
print("  2. Moi frame: earBuffer = earBuffer.drop(1) + currentEAR")
print("  3. Goi temporalModel.predict(earBuffer)")
print("  4. isTemporalDrowsy = drowsyProb >= 0.65f")
print()
print("NOTE: Dung voi EAR tu DrowsinessAnalyzer.kt (da co landmarks)")
