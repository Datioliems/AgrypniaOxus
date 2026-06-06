"""
Create dataset statistics for the report.

Expected structure:

dataset/
  train/
    eyes_closed/
    eyes_open/
  val/
    eyes_closed/
    eyes_open/
  test/
    eyes_closed/
    eyes_open/

Usage:
  python tools/summarize_dataset.py --data dataset --out outputs/dataset_summary
"""

import argparse
import csv
import json
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Dataset root")
    parser.add_argument("--out", default="outputs/dataset_summary")
    args = parser.parse_args()

    data_root = Path(args.data)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    splits = [p.name for p in data_root.iterdir() if p.is_dir()]
    preferred = ["train", "val", "test"]
    splits = [s for s in preferred if s in splits] + sorted([s for s in splits if s not in preferred])

    classes = sorted(
        {
            class_dir.name
            for split in splits
            for class_dir in (data_root / split).iterdir()
            if class_dir.is_dir()
        }
    )

    summary = {}
    manifest_rows = []
    for split in splits:
        summary[split] = {}
        for class_name in classes:
            class_dir = data_root / split / class_name
            files = []
            if class_dir.exists():
                files = [
                    path
                    for path in sorted(class_dir.iterdir())
                    if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
                ]
            summary[split][class_name] = len(files)
            for path in files:
                manifest_rows.append(
                    {
                        "split": split,
                        "label": class_name,
                        "path": str(path),
                    }
                )

    totals = {
        class_name: sum(summary[split].get(class_name, 0) for split in splits)
        for class_name in classes
    }
    summary["_totals"] = totals

    (out_dir / "dataset_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    with (out_dir / "dataset_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["split", *classes, "total"])
        for split in splits:
            row = [summary[split].get(class_name, 0) for class_name in classes]
            writer.writerow([split, *row, sum(row)])
        writer.writerow(["total", *[totals[class_name] for class_name in classes], sum(totals.values())])

    with (out_dir / "dataset_manifest.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["split", "label", "path"])
        writer.writeheader()
        writer.writerows(manifest_rows)

    print("Dataset summary")
    print("split," + ",".join(classes) + ",total")
    for split in splits:
        row = [summary[split].get(class_name, 0) for class_name in classes]
        print(",".join([split, *[str(value) for value in row], str(sum(row))]))
    print(",".join(["total", *[str(totals[class_name]) for class_name in classes], str(sum(totals.values()))]))
    print(f"Saved reports to {out_dir}")


if __name__ == "__main__":
    main()
