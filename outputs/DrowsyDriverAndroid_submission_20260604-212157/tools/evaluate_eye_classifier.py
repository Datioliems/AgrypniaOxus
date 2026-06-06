"""
Evaluate an eye-state classifier and print report-ready metrics.

Expected dataset structure:

dataset/
  test/
    eyes_closed/
    eyes_open/

Usage with a Keras model:
  python tools/evaluate_eye_classifier.py --data dataset --keras model.keras

Usage with a TensorFlow Lite model:
  python tools/evaluate_eye_classifier.py --data dataset --tflite app/src/main/assets/drowsiness_model.tflite
"""

import argparse
import csv
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Dataset root with test/")
    parser.add_argument("--keras", help="Path to .keras/.h5 model")
    parser.add_argument("--tflite", help="Path to .tflite model")
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--out-dir", default="outputs/evaluation")
    args = parser.parse_args()

    if not args.keras and not args.tflite:
        raise SystemExit("Provide either --keras or --tflite")

    import numpy as np
    import tensorflow as tf

    test_root = Path(args.data) / "test"
    class_names = sorted([p.name for p in test_root.iterdir() if p.is_dir()])
    if not class_names:
        raise SystemExit(f"No class folders found in {test_root}")

    image_paths = []
    y_true = []
    for class_index, class_name in enumerate(class_names):
        for path in sorted((test_root / class_name).glob("*")):
            if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                image_paths.append(path)
                y_true.append(class_index)

    if not image_paths:
        raise SystemExit(f"No images found in {test_root}")

    x = np.stack([load_image(tf, path, args.image_size) for path in image_paths], axis=0)

    if args.keras:
        model = tf.keras.models.load_model(args.keras)
        probs = model.predict(x, verbose=0)
    else:
        probs = predict_tflite(tf, np, args.tflite, x, len(class_names))

    y_pred = probs.argmax(axis=1)
    metrics = compute_metrics(class_names, y_true, y_pred)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    write_confusion_csv(out_dir / "confusion_matrix.csv", class_names, metrics["confusion_matrix"])
    write_predictions_csv(out_dir / "predictions.csv", image_paths, class_names, y_true, y_pred, probs)

    print("Classes:", class_names)
    print("Accuracy:", f"{metrics['accuracy']:.4f}")
    print("\nPer-class metrics")
    for row in metrics["per_class"]:
        print(
            f"- {row['class']}: precision={row['precision']:.4f}, "
            f"recall={row['recall']:.4f}, f1={row['f1']:.4f}, support={row['support']}"
        )
    print("\nConfusion matrix")
    print("actual/pred," + ",".join(class_names))
    for class_name, row in zip(class_names, metrics["confusion_matrix"]):
        print(class_name + "," + ",".join(str(v) for v in row))
    print(f"\nSaved reports to {out_dir}")


def load_image(tf, path: Path, image_size: int):
    image = tf.io.read_file(str(path))
    image = tf.image.decode_image(image, channels=3, expand_animations=False)
    image = tf.image.resize(image, (image_size, image_size))
    image = tf.cast(image, tf.float32) / 255.0
    return image.numpy()


def predict_tflite(tf, np, model_path: str, x, class_count: int):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    output = []
    for sample in x:
        batch = np.expand_dims(sample, axis=0).astype(input_details["dtype"])
        interpreter.set_tensor(input_details["index"], batch)
        interpreter.invoke()
        probs = interpreter.get_tensor(output_details["index"])[0]
        if len(probs) != class_count:
            raise SystemExit(f"Model output has {len(probs)} classes, dataset has {class_count}")
        output.append(probs)
    return np.array(output)


def compute_metrics(class_names, y_true, y_pred):
    class_count = len(class_names)
    confusion = [[0 for _ in range(class_count)] for _ in range(class_count)]
    for actual, predicted in zip(y_true, y_pred):
        confusion[actual][predicted] += 1

    total = len(y_true)
    correct = sum(confusion[i][i] for i in range(class_count))
    per_class = []
    for i, class_name in enumerate(class_names):
        tp = confusion[i][i]
        fp = sum(confusion[row][i] for row in range(class_count)) - tp
        fn = sum(confusion[i][col] for col in range(class_count)) - tp
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        support = sum(confusion[i])
        per_class.append(
            {
                "class": class_name,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": support,
            }
        )

    return {
        "accuracy": correct / total if total else 0.0,
        "per_class": per_class,
        "confusion_matrix": confusion,
    }


def write_confusion_csv(path: Path, class_names, confusion):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["actual/predicted", *class_names])
        for class_name, row in zip(class_names, confusion):
            writer.writerow([class_name, *row])


def write_predictions_csv(path: Path, image_paths, class_names, y_true, y_pred, probs):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "actual", "predicted", "confidence"])
        for image_path, actual, predicted, probability in zip(image_paths, y_true, y_pred, probs):
            writer.writerow([image_path, class_names[actual], class_names[predicted], float(probability[predicted])])


if __name__ == "__main__":
    main()
