import argparse
from pathlib import Path


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(description="Check a YOLO object-detection dataset.")
    parser.add_argument(
        "--root",
        default="dataset_yolo",
        help="Dataset root folder that contains data.yaml.",
    )
    return parser.parse_args()


def count_files(path, suffixes):
    if not path.exists():
        return 0
    return sum(1 for file in path.rglob("*") if file.is_file() and file.suffix.lower() in suffixes)


def find_split_dirs(root, split):
    candidates = [
        (root / split / "images", root / split / "labels"),
        (root / "images" / split, root / "labels" / split),
    ]
    if split == "val":
        candidates.extend(
            [
                (root / "valid" / "images", root / "valid" / "labels"),
                (root / "images" / "valid", root / "labels" / "valid"),
            ]
        )
    for images_dir, labels_dir in candidates:
        if images_dir.exists() or labels_dir.exists():
            return images_dir, labels_dir
    return candidates[0]


def validate_label_file(label_file):
    problems = []
    lines = label_file.read_text(encoding="utf-8", errors="ignore").splitlines()
    for index, line in enumerate(lines, start=1):
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            problems.append(f"{label_file}:{index} expected 5 values, got {len(parts)}")
            continue
        try:
            class_id = int(float(parts[0]))
            values = [float(value) for value in parts[1:]]
        except ValueError:
            problems.append(f"{label_file}:{index} contains non-numeric values")
            continue
        if class_id < 0:
            problems.append(f"{label_file}:{index} class id is negative")
        if any(value < 0 or value > 1 for value in values):
            problems.append(f"{label_file}:{index} box values must be normalized from 0 to 1")
    return problems


def main():
    args = parse_args()
    root = Path(args.root).resolve()
    yaml_path = root / "data.yaml"

    print(f"Dataset root: {root}")
    if not root.exists():
        print("FAIL: dataset root does not exist.")
        return 1
    if not yaml_path.exists():
        print("FAIL: data.yaml was not found.")
        return 1
    print(f"Found: {yaml_path}")

    total_images = 0
    total_labels = 0
    all_problems = []
    for split in ["train", "val", "test"]:
        images_dir, labels_dir = find_split_dirs(root, split)
        image_count = count_files(images_dir, IMAGE_SUFFIXES)
        label_count = count_files(labels_dir, {".txt"})
        total_images += image_count
        total_labels += label_count
        print(f"{split:5} images={image_count:5} labels={label_count:5}")

        if image_count > 0 and label_count == 0:
            all_problems.append(f"{split}: images exist but labels are missing")
        if label_count > image_count * 2 and image_count > 0:
            all_problems.append(f"{split}: label count looks unusually high")

        if labels_dir.exists():
            for label_file in labels_dir.rglob("*.txt"):
                all_problems.extend(validate_label_file(label_file))
                if len(all_problems) > 20:
                    break

    if total_images == 0:
        print("FAIL: no images were found.")
        return 1
    if total_labels == 0:
        print("FAIL: no YOLO label files were found.")
        return 1

    if all_problems:
        print("WARN: possible problems:")
        for problem in all_problems[:20]:
            print(f"  - {problem}")
        if len(all_problems) > 20:
            print(f"  - ... {len(all_problems) - 20} more")
        return 2

    print("PASS: YOLO dataset structure looks usable.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
