"""
Create report-ready plots from training and evaluation outputs.

Outputs:
  outputs/training/training_curves.png
  outputs/evaluation/confusion_matrix.png

Usage:
  python tools/plot_model_results.py
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--history", default="outputs/training/training_history.json")
    parser.add_argument("--confusion", default="outputs/evaluation/confusion_matrix.csv")
    args = parser.parse_args()

    plot_training_curves(Path(args.history))
    plot_confusion_matrix(Path(args.confusion))
    print("Saved outputs/training/training_curves.png")
    print("Saved outputs/evaluation/confusion_matrix.png")


def plot_training_curves(history_path: Path) -> None:
    import matplotlib.pyplot as plt

    history = json.loads(history_path.read_text(encoding="utf-8"))
    epochs = list(range(1, len(history.get("accuracy", [])) + 1))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epochs, history.get("accuracy", []), marker="o", label="train accuracy")
    axes[0].plot(epochs, history.get("val_accuracy", []), marker="o", label="val accuracy")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].plot(epochs, history.get("loss", []), marker="o", label="train loss")
    axes[1].plot(epochs, history.get("val_loss", []), marker="o", label="val loss")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].grid(alpha=0.25)
    axes[1].legend()

    fig.suptitle("CNN eye-state training history")
    fig.tight_layout()
    out = Path("outputs/training/training_curves.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    plt.close(fig)


def plot_confusion_matrix(confusion_path: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    with confusion_path.open(newline="", encoding="utf-8") as f:
        reader = list(csv.reader(f))
    labels = reader[0][1:]
    matrix = np.array([[int(value) for value in row[1:]] for row in reader[1:]])

    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_title("Confusion matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticks(range(len(labels)), labels, rotation=20, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    max_value = matrix.max() if matrix.size else 0
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            color = "white" if matrix[row, col] > max_value / 2 else "black"
            ax.text(col, row, str(matrix[row, col]), ha="center", va="center", color=color)

    fig.tight_layout()
    out = Path("outputs/evaluation/confusion_matrix.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
