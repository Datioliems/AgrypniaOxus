import argparse
import os
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download a Roboflow object-detection dataset for YOLO training."
    )
    parser.add_argument(
        "--workspace",
        required=True,
        help="Roboflow workspace name, for example: universidad-carlos-3-madrid",
    )
    parser.add_argument(
        "--project",
        required=True,
        help="Roboflow project name/slug, for example: driver-yawn",
    )
    parser.add_argument(
        "--version",
        required=True,
        type=int,
        help="Roboflow dataset version number.",
    )
    parser.add_argument(
        "--out",
        default="dataset_yolo",
        help="Output folder for the downloaded dataset.",
    )
    parser.add_argument(
        "--format",
        default="yolov8",
        help="Export format. Use yolov8 unless Roboflow snippet says otherwise.",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="Roboflow API key. Prefer ROBOFLOW_API_KEY environment variable.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    api_key = args.api_key or os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        print(
            "Missing Roboflow API key.\n"
            "Option 1: set it for this terminal:\n"
            "  $env:ROBOFLOW_API_KEY='YOUR_KEY'\n"
            "Option 2: pass --api-key YOUR_KEY when running this script.",
            file=sys.stderr,
        )
        return 2

    try:
        from roboflow import Roboflow
    except ImportError:
        print(
            "The 'roboflow' package is not installed.\n"
            "Install it with:\n"
            "  python -m pip install roboflow",
            file=sys.stderr,
        )
        return 2

    out_dir = Path(args.out).resolve()
    out_dir.parent.mkdir(parents=True, exist_ok=True)

    print("Connecting to Roboflow...")
    rf = Roboflow(api_key=api_key)
    project = rf.workspace(args.workspace).project(args.project)
    version = project.version(args.version)

    print(f"Downloading {args.workspace}/{args.project}/{args.version} as {args.format}...")
    dataset = version.download(model_format=args.format, location=str(out_dir))

    location = Path(getattr(dataset, "location", out_dir)).resolve()
    print(f"Downloaded dataset path: {location}")
    print(f"Expected YAML file: {location / 'data.yaml'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
