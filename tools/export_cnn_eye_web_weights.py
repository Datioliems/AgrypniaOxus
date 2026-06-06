"""
Export the trained Keras eye CNN to a small JSON file for the iPhone PWA.

The browser PWA cannot use Android's .tflite file directly without an extra
runtime. This script exports the same trained CNN weights so app.js can run a
lightweight forward pass on cropped eye ROIs.

Usage:
  python tools/export_cnn_eye_web_weights.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def flatten(values):
    return [float(x) for x in values.reshape(-1)]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--keras",
        default="outputs/training/drowsiness_model.keras",
        help="Input Keras model trained by tools/train_eye_classifier.py",
    )
    parser.add_argument(
        "--classes",
        default="outputs/training/class_names.json",
        help="Class name JSON, expected: ['eyes_closed', 'eyes_open']",
    )
    parser.add_argument(
        "--out",
        default="web/iphone-pwa/models/cnn_eye_weights.json",
        help="Output JSON used by the iPhone PWA",
    )
    args = parser.parse_args()

    import tensorflow as tf

    model_path = Path(args.keras)
    classes_path = Path(args.classes)
    out_path = Path(args.out)

    if not model_path.exists():
        raise FileNotFoundError(f"Missing Keras model: {model_path}")
    if not classes_path.exists():
        raise FileNotFoundError(f"Missing class names: {classes_path}")

    model = tf.keras.models.load_model(str(model_path))
    class_names = json.loads(classes_path.read_text(encoding="utf-8"))

    layers = []
    for layer in model.layers:
        weights = layer.get_weights()
        if layer.__class__.__name__ == "Conv2D":
            kernel, bias = weights
            layers.append(
                {
                    "type": "conv2d",
                    "name": layer.name,
                    "activation": layer.activation.__name__,
                    "padding": layer.padding,
                    "strides": list(layer.strides),
                    "kernelShape": list(kernel.shape),
                    "kernel": flatten(kernel),
                    "bias": flatten(bias),
                }
            )
        elif layer.__class__.__name__ == "MaxPooling2D":
            layers.append(
                {
                    "type": "maxpool2d",
                    "name": layer.name,
                    "poolSize": list(layer.pool_size),
                    "strides": list(layer.strides),
                    "padding": layer.padding,
                }
            )
        elif layer.__class__.__name__ == "GlobalAveragePooling2D":
            layers.append({"type": "gap2d", "name": layer.name})
        elif layer.__class__.__name__ == "Dropout":
            continue
        elif layer.__class__.__name__ == "Dense":
            kernel, bias = weights
            layers.append(
                {
                    "type": "dense",
                    "name": layer.name,
                    "activation": layer.activation.__name__,
                    "kernelShape": list(kernel.shape),
                    "kernel": flatten(kernel),
                    "bias": flatten(bias),
                }
            )

    payload = {
        "format": "drowsy-guard-cnn-eye-v1",
        "source": str(model_path),
        "inputSize": 64,
        "inputChannels": 3,
        "normalization": "rgb_0_1",
        "classes": class_names,
        "layers": layers,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    print(f"Saved {out_path} ({out_path.stat().st_size / 1024:.1f} KB)")
    print(f"Classes: {class_names}")
    print(f"Layers : {[layer['type'] for layer in layers]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
