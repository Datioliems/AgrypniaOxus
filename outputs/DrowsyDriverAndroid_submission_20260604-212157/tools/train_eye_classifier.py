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
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Dataset root with train/ and val/")
    parser.add_argument("--out", default="drowsiness_model.tflite")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--image-size", type=int, default=64)
    args = parser.parse_args()

    import tensorflow as tf

    data_root = Path(args.data)
    img_size = (args.image_size, args.image_size)

    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_root / "train",
        image_size=img_size,
        batch_size=32,
        label_mode="categorical",
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_root / "val",
        image_size=img_size,
        batch_size=32,
        label_mode="categorical",
    )

    class_names = train_ds.class_names
    print("Classes:", class_names)

    normalization = tf.keras.layers.Rescaling(1.0 / 255.0)
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(args.image_size, args.image_size, 3)),
            normalization,
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
    model.fit(train_ds, validation_data=val_ds, epochs=args.epochs)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(tflite_model)
    print(f"Saved {out_path}")
    print("Class order:", class_names)


if __name__ == "__main__":
    main()
