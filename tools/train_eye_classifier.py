"""
Train a small CNN for eye-state classification and export TensorFlow Lite.

Expected dataset structure:

dataset/
  train/
    eyes_open/
    eyes_closed/
  val/
    eyes_open/
    eyes_closed/

Usage:
  python tools/train_eye_classifier.py --data dataset --out app/src/main/assets/drowsiness_model.tflite
"""

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Dataset root with train/ and val/")
    parser.add_argument("--out", default="drowsiness_model.tflite")
    parser.add_argument("--keras-out", default="outputs/training/drowsiness_model.keras")
    parser.add_argument("--artifacts-dir", default="outputs/training")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--early-stop-patience", type=int, default=3)
    args = parser.parse_args()

    import tensorflow as tf

    data_root = Path(args.data)
    img_size = (args.image_size, args.image_size)

    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_root / "train",
        image_size=img_size,
        batch_size=32,
        label_mode="categorical",
        seed=42,
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_root / "val",
        image_size=img_size,
        batch_size=32,
        label_mode="categorical",
        shuffle=False,
    )

    class_names = train_ds.class_names
    print("Classes:", class_names)
    artifacts_dir = Path(args.artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    (artifacts_dir / "class_names.json").write_text(
        json.dumps(class_names, indent=2),
        encoding="utf-8",
    )

    train_ds = train_ds.map(normalize_batch, num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.map(normalize_batch, num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(args.image_size, args.image_size, 3)),
            tf.keras.layers.Conv2D(24, 3, activation="relu"),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(48, 3, activation="relu"),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(64, 3, activation="relu"),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dropout(0.25),
            tf.keras.layers.Dense(len(class_names), activation="softmax"),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )
    summary_lines = []
    model.summary(print_fn=summary_lines.append)
    (artifacts_dir / "model_summary.txt").write_text("\n".join(summary_lines), encoding="utf-8")

    callbacks = []
    if args.early_stop_patience > 0:
        callbacks.append(
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=args.early_stop_patience,
                restore_best_weights=True,
            )
        )
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=2,
    )
    (artifacts_dir / "training_history.json").write_text(
        json.dumps({key: [float(value) for value in values] for key, values in history.history.items()}, indent=2),
        encoding="utf-8",
    )

    keras_out = Path(args.keras_out)
    keras_out.parent.mkdir(parents=True, exist_ok=True)
    model.save(keras_out)
    print(f"Saved {keras_out}")

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(tflite_model)
    print(f"Saved {out_path}")
    print("Class order:", class_names)


def normalize_batch(images, labels):
    import tensorflow as tf

    return tf.cast(images, tf.float32) / 255.0, labels


if __name__ == "__main__":
    main()
