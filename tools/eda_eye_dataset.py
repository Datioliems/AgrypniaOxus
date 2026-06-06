"""
Run report-ready EDA for the prepared eye-state dataset.

Outputs:
  outputs/eda/eda_summary.json
  outputs/eda/eda_summary.csv
  outputs/eda/class_distribution.png
  outputs/eda/sample_grid.png

Usage:
  python tools/eda_eye_dataset.py --data dataset --out outputs/eda
  python tools/eda_eye_dataset.py --data dataset --out outputs/eda --max-open-per-class 2000
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageOps


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_SPLITS = ["train", "val", "test"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="dataset", help="Prepared dataset root")
    parser.add_argument("--out", default="outputs/eda", help="Output directory")
    parser.add_argument("--sample-per-class", type=int, default=8)
    parser.add_argument(
        "--max-open-per-class",
        type=int,
        default=2000,
        help="Maximum images to open per split/class for quick corruption and dimension checks",
    )
    args = parser.parse_args()

    data_root = Path(args.data)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows, summary, sample_paths = inspect_dataset(data_root, args.sample_per_class, args.max_open_per_class)
    write_outputs(out_dir, rows, summary)
    make_class_distribution_plot(out_dir / "class_distribution.png", summary)
    make_sample_grid(out_dir / "sample_grid.png", sample_paths)

    print("EDA summary")
    print("split,class,count,checked,valid_checked,corrupt_checked,extensions,top_dimensions")
    for row in rows:
        print(
            f"{row['split']},{row['class']},{row['count']},{row['checked']},"
            f"{row['valid_checked']},{row['corrupt_checked']},{row['extensions']},{row['top_dimensions']}"
        )
    print(f"Saved EDA reports to {out_dir}")


def inspect_dataset(data_root: Path, sample_per_class: int, max_open_per_class: int):
    if not data_root.exists():
        raise SystemExit(f"Dataset root not found: {data_root}")

    rows = []
    summary = {
        "data_root": str(data_root),
        "splits": {},
        "totals": defaultdict(int),
        "notes": [
            "Label mapping used in this project: awake -> eyes_open, sleepy -> eyes_closed.",
            "MRL eye images are suitable for eye-state classification but are not full cabin/dashcam frames.",
        ],
    }
    sample_paths: dict[str, list[Path]] = defaultdict(list)

    for split in DEFAULT_SPLITS:
        split_dir = data_root / split
        if not split_dir.exists():
            continue
        summary["splits"][split] = {}

        for class_dir in sorted([p for p in split_dir.iterdir() if p.is_dir()]):
            files = [
                path
                for path in sorted(class_dir.iterdir())
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            ]
            ext_counter = Counter()
            dim_counter = Counter()
            corrupt = []
            valid_checked = 0
            checked = 0

            for path in files:
                ext_counter[path.suffix.lower()] += 1

            for path in files[:max_open_per_class]:
                checked += 1
                try:
                    with Image.open(path) as image:
                        dim_counter[f"{image.width}x{image.height}"] += 1
                    valid_checked += 1
                    if len(sample_paths[class_dir.name]) < sample_per_class:
                        sample_paths[class_dir.name].append(path)
                except Exception as exc:  # pragma: no cover - report-only details
                    corrupt.append({"path": str(path), "error": str(exc)})

            row = {
                "split": split,
                "class": class_dir.name,
                "count": len(files),
                "checked": checked,
                "valid_checked": valid_checked,
                "corrupt_checked": len(corrupt),
                "extensions": "; ".join(f"{k}:{v}" for k, v in sorted(ext_counter.items())),
                "top_dimensions": "; ".join(f"{k}:{v}" for k, v in dim_counter.most_common(5)),
            }
            rows.append(row)
            summary["splits"][split][class_dir.name] = {
                **row,
                "corrupt_files": corrupt[:20],
            }
            summary["totals"][class_dir.name] += len(files)

    summary["totals"] = dict(summary["totals"])
    return rows, summary, sample_paths


def write_outputs(out_dir: Path, rows: list[dict[str, object]], summary: dict[str, object]) -> None:
    (out_dir / "eda_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    with (out_dir / "eda_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "split",
                "class",
                "count",
                "checked",
                "valid_checked",
                "corrupt_checked",
                "extensions",
                "top_dimensions",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def make_class_distribution_plot(path: Path, summary: dict[str, object]) -> None:
    import matplotlib.pyplot as plt

    splits = [split for split in DEFAULT_SPLITS if split in summary["splits"]]
    classes = sorted(summary["totals"].keys())
    x = range(len(splits))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for class_index, class_name in enumerate(classes):
        values = [
            summary["splits"][split].get(class_name, {}).get("count", 0)
            for split in splits
        ]
        offset = (class_index - (len(classes) - 1) / 2) * width
        ax.bar([i + offset for i in x], values, width=width, label=class_name)

    ax.set_title("Eye-state dataset distribution")
    ax.set_xlabel("Split")
    ax.set_ylabel("Image count")
    ax.set_xticks(list(x), splits)
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def make_sample_grid(path: Path, sample_paths: dict[str, list[Path]]) -> None:
    import matplotlib.pyplot as plt

    classes = sorted(sample_paths.keys())
    if not classes:
        return

    max_cols = max(len(paths) for paths in sample_paths.values())
    fig, axes = plt.subplots(len(classes), max_cols, figsize=(max_cols * 1.45, len(classes) * 1.7))
    if len(classes) == 1:
        axes = [axes]

    for row_index, class_name in enumerate(classes):
        for col_index in range(max_cols):
            ax = axes[row_index][col_index] if max_cols > 1 else axes[row_index]
            ax.axis("off")
            if col_index >= len(sample_paths[class_name]):
                continue
            with Image.open(sample_paths[class_name][col_index]) as image:
                preview = ImageOps.grayscale(image).resize((96, 96))
            ax.imshow(preview, cmap="gray")
            if col_index == 0:
                ax.set_ylabel(class_name, fontsize=10)

    fig.suptitle("Sample eye images by class", fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
