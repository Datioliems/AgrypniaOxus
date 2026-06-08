"""Finalize a saved CBAM PyTorch checkpoint into metrics + TFLite."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from train_cbam_fast_gpu import (
    TASKS,
    FastCbamCNN,
    evaluate,
    export_tflite_from_torch,
    load_friend_params,
    make_loaders,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=sorted(TASKS))
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--cbam-ratio", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.30)
    parser.add_argument("--note", default="finalized from interrupted run checkpoint")
    args = parser.parse_args()

    task = TASKS[args.task]
    checkpoint = Path(args.checkpoint) if args.checkpoint else task.out_dir / "best_cbam_torch_state.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)

    params = load_friend_params()
    classes, _, val_loader, test_loader = make_loaders(task, args.batch_size, args.workers, params)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FastCbamCNN(num_classes=len(classes), ratio=args.cbam_ratio, dropout=args.dropout)
    state = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(state)
    model = model.to(device)

    started = time.perf_counter()
    val = evaluate(model, val_loader, device, classes)
    test = evaluate(model, test_loader, device, classes)
    elapsed = time.perf_counter() - started

    export_tflite_from_torch(task, model.cpu(), classes, args.cbam_ratio, args.dropout)

    summary = {
        "model": f"DrowsyCNN_CBAM_{args.task}",
        "classes": classes,
        "image_size": 64,
        "cbam_ratio": args.cbam_ratio,
        "dropout": args.dropout,
        "epochs_completed_known_minimum": 1,
        "epochs_completed_exact": None,
        "checkpoint": str(checkpoint),
        "checkpoint_last_write_time": checkpoint.stat().st_mtime,
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "val_accuracy": round(float(val["accuracy"]), 6),
        "val_loss": round(float(val["loss"]), 6),
        "test_accuracy": round(float(test["accuracy"]), 6),
        "test_loss": round(float(test["loss"]), 6),
        "confusion_matrix": test["confusion_matrix"],
        "per_class": test["per_class"],
        "preprocess_source": "data_pipeline/best_params.json" if params else "script defaults",
        "augmentation": params.get("preprocess", {}).get("augmentation", {}),
        "tflite_path": str(task.tflite_out),
        "tflite_kb": round(task.tflite_out.stat().st_size / 1024, 1),
        "elapsed_seconds": round(elapsed, 1),
        "note": args.note,
    }
    task.out_dir.mkdir(parents=True, exist_ok=True)
    out = task.out_dir / "summary_partial.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Saved partial summary: {out}")


if __name__ == "__main__":
    main()
