"""
Train a small CNN for eye-state classification and export TensorFlow Lite.

THAY ĐỔI SO VỚI PHIÊN BẢN CŨ:
  - Thêm BatchNormalization vào mỗi Conv block → training ổn định hơn
  - Tăng default epochs từ 12 lên 25 (EarlyStopping sẽ dừng sớm nếu cần)
  - EarlyStopping monitor val_accuracy thay vì val_loss
  - Lưu class_names.json tường minh và in ra để kiểm tra
  - Thêm ReduceLROnPlateau để tránh stuck ở local minimum
  - Augmentation nằm trong pipeline data (không trong model) → TFLite không
    bị dính normalize kép

QUAN TRỌNG - Class order:
  image_dataset_from_directory sắp xếp alphabetically:
    "eyes_closed" (index 0) < "eyes_open" (index 1)
  Android TfliteDrowsinessClassifier.kt phải dùng cùng thứ tự này!

Expected dataset structure:
  dataset_mrl/
    train/
      eyes_closed/
      eyes_open/
    val/
      eyes_closed/
      eyes_open/

Usage:
  python tools/train_eye_classifier.py \\
    --data dataset_mrl \\
    --out app/src/main/assets/drowsiness_model.tflite \\
    --epochs 25
"""

import argparse
import json
from pathlib import Path


def normalize_batch(images, labels):
    """
    Normalize ảnh về [0, 1].
    QUAN TRỌNG: Hàm này normalize NGOÀI model, không phải trong model.
    → TFLite model sẽ nhận input ĐÃ normalize.
    → Cả Python eval và Android đều phải normalize trước khi gọi TFLite.
    """
    import tensorflow as tf
    return tf.cast(images, tf.float32) / 255.0, labels


def build_model(image_size: int, num_classes: int):
    """
    CNN 3-block với BatchNormalization.

    Kiến trúc:
      Input (64×64×3)
      Conv2D(32) + BN + ReLU + MaxPool → (32×32×32)
      Conv2D(64) + BN + ReLU + MaxPool → (16×16×64)
      Conv2D(128) + BN + ReLU + GAP   → (128,)
      Dropout(0.30)
      Dense(num_classes, softmax)

    Tại sao BatchNorm:
      - Ổn định gradient → có thể dùng LR cao hơn
      - Giảm nhu cầu Dropout quá lớn
      - Training nhanh hội tụ hơn
    """
    import tensorflow as tf

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(image_size, image_size, 3)),

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

            # Head
            tf.keras.layers.Dropout(0.30),
            tf.keras.layers.Dense(num_classes, activation="softmax"),
        ],
        name="DrowsyCNN_v2",
    )
    return model


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train CNN eye-state classifier và export TFLite"
    )
    parser.add_argument("--data",       required=True,
                        help="Dataset root với train/ và val/")
    parser.add_argument("--out",        default="app/src/main/assets/drowsiness_model.tflite",
                        help="Đường dẫn output TFLite")
    parser.add_argument("--keras-out",  default="outputs/training/drowsiness_model.keras")
    parser.add_argument("--artifacts-dir", default="outputs/training")
    parser.add_argument("--epochs",     type=int, default=25)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--early-stop-patience", type=int, default=5)
    args = parser.parse_args()

    import tensorflow as tf
    print(f"TensorFlow: {tf.__version__}")
    print(f"GPU: {tf.config.list_physical_devices('GPU')}")

    data_root = Path(args.data)
    img_size  = (args.image_size, args.image_size)

    # ── Load dataset ──────────────────────────────────────────────────
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_root / "train",
        image_size=img_size,
        batch_size=args.batch_size,
        label_mode="categorical",
        seed=42,
        shuffle=True,
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_root / "val",
        image_size=img_size,
        batch_size=args.batch_size,
        label_mode="categorical",
        shuffle=False,
    )

    class_names = train_ds.class_names
    num_classes = len(class_names)

    # ── QUAN TRỌNG: Kiểm tra và lưu class order ──────────────────────
    print("\n" + "=" * 50)
    print("CLASS ORDER (phải khớp với Android labels array!):")
    for i, name in enumerate(class_names):
        print(f"  index {i} → '{name}'")
    print("=" * 50)

    # Validate expected order
    if class_names != sorted(class_names):
        print("⚠️  Class names không theo thứ tự alphabet!")
    expected = ["eyes_closed", "eyes_open"]
    if class_names != expected:
        print(f"⚠️  Class names = {class_names}, expected = {expected}")
        print("   Kiểm tra lại tên thư mục trong dataset!")
    else:
        print("✅ Class order khớp với Android: eyes_closed=0, eyes_open=1")

    # Lưu artifacts
    artifacts_dir = Path(args.artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    (artifacts_dir / "class_names.json").write_text(
        json.dumps(class_names, indent=2), encoding="utf-8"
    )

    # ── Count samples ─────────────────────────────────────────────────
    n_train = sum(
        len(list((data_root / "train" / c).glob("*")))
        for c in class_names
        if (data_root / "train" / c).is_dir()
    )
    n_val = sum(
        len(list((data_root / "val" / c).glob("*")))
        for c in class_names
        if (data_root / "val" / c).is_dir()
    )
    print(f"\nTrain samples: {n_train:,}  |  Val samples: {n_val:,}")

    # ── Normalize (NGOÀI model) ───────────────────────────────────────
    # KHÔNG đặt normalize bên trong model → TFLite nhận input [0,1]
    # → Python eval và Android đều chia /255 trước khi gọi TFLite
    train_ds = train_ds.map(normalize_batch, num_parallel_calls=tf.data.AUTOTUNE)
    val_ds   = val_ds.map(normalize_batch,   num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    val_ds   = val_ds.prefetch(tf.data.AUTOTUNE)

    # ── Build model ───────────────────────────────────────────────────
    model = build_model(args.image_size, num_classes)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
        loss="categorical_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )

    summary_lines = []
    model.summary(print_fn=lambda line: summary_lines.append(line))
    (artifacts_dir / "model_summary.txt").write_text(
        "\n".join(summary_lines), encoding="utf-8"
    )
    model.summary()

    # ── Callbacks ─────────────────────────────────────────────────────
    callbacks = []

    if args.early_stop_patience > 0:
        callbacks.append(
            tf.keras.callbacks.EarlyStopping(
                monitor="val_accuracy",
                patience=args.early_stop_patience,
                restore_best_weights=True,
                verbose=1,
                min_delta=0.001,
            )
        )

    # Giảm LR khi val_loss không cải thiện
    callbacks.append(
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-7,
            verbose=1,
        )
    )

    # Lưu model tốt nhất theo val_accuracy
    best_keras_path = artifacts_dir / "best_model.keras"
    callbacks.append(
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(best_keras_path),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        )
    )

    # ── Train ─────────────────────────────────────────────────────────
    print(f"\n🏋️  Training {args.epochs} epochs (EarlyStopping patience={args.early_stop_patience})...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=2,
    )

    epochs_ran = len(history.history["accuracy"])
    best_val_acc = max(history.history["val_accuracy"])
    print(f"\n✅ Training xong! Epochs: {epochs_ran}/{args.epochs}  "
          f"Best val_accuracy: {best_val_acc:.4f} ({best_val_acc*100:.2f}%)")

    # ── Lưu Keras model ───────────────────────────────────────────────
    (artifacts_dir / "training_history.json").write_text(
        json.dumps(
            {k: [float(v) for v in vals] for k, vals in history.history.items()},
            indent=2,
        ),
        encoding="utf-8",
    )

    keras_out = Path(args.keras_out)
    keras_out.parent.mkdir(parents=True, exist_ok=True)

    # Dùng best checkpoint nếu có
    if best_keras_path.exists():
        import shutil
        shutil.copy2(best_keras_path, keras_out)
        model = tf.keras.models.load_model(str(best_keras_path))
        print(f"Dùng best checkpoint: {best_keras_path}")
    else:
        model.save(keras_out)

    print(f"Saved Keras model: {keras_out}")

    # ── Export TFLite ─────────────────────────────────────────────────
    print("\n📱 Exporting TFLite...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(tflite_model)

    keras_kb  = keras_out.stat().st_size / 1024
    tflite_kb = out_path.stat().st_size / 1024
    print(f"Saved TFLite: {out_path}")
    print(f"  Keras size : {keras_kb:.1f} KB")
    print(f"  TFLite size: {tflite_kb:.1f} KB (giảm {(1-tflite_kb/keras_kb)*100:.0f}%)")

    # ── Verify TFLite output ──────────────────────────────────────────
    print("\n🔍 Verify TFLite model...")
    interp = tf.lite.Interpreter(model_path=str(out_path))
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]
    print(f"  Input  : shape={inp['shape']}  dtype={inp['dtype']}")
    print(f"  Output : shape={out['shape']}  dtype={out['dtype']}")
    print(f"  Classes: {class_names}")

    if out["shape"][1] != num_classes:
        print(f"⚠️  Output shape {out['shape']} không khớp num_classes={num_classes}")
    else:
        print("✅ TFLite output shape OK")

    print(f"""
╔══════════════════════════════════════════════╗
║  QUAN TRỌNG — Android setup                  ║
╠══════════════════════════════════════════════╣
║  Class order được lưu: {class_names}
║  Android labels array phải là:               ║
║    arrayOf("eyes_closed", "eyes_open")        ║
║  (index 0 = eyes_closed, index 1 = eyes_open)║
╠══════════════════════════════════════════════╣
║  Bước tiếp theo:                             ║
║  1. Evaluate: python tools/evaluate_eye_...  ║
║  2. Copy TFLite vào Android assets           ║
║  3. Build APK và test                        ║
╚══════════════════════════════════════════════╝
""")


if __name__ == "__main__":
    main()
