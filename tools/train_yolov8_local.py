"""
Train YOLOv8 trên máy LOCAL với RTX 4050 (6GB VRAM).

Tối ưu cho RTX 4050:
  - FP16 half precision → giảm VRAM 50%
  - batch=16, imgsz=640 (vừa 6GB VRAM)
  - cache='ram' → tải ảnh vào RAM 1 lần, training nhanh hơn
  - workers=4  → không quá tải CPU khi GPU đang train
  - cos_lr=True → cosine learning rate schedule

Usage:
  cd D:\\2026.AI\\DrowsyDriverAndroid
  python tools\\train_yolov8_local.py

  # Tuỳ chọn model size:
  python tools\\train_yolov8_local.py --model yolov8n  # nano  ~3.2M params
  python tools\\train_yolov8_local.py --model yolov8s  # small ~11M  params
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path


def check_gpu():
    """Kiểm tra GPU có sẵn không"""
    try:
        import torch
        if torch.cuda.is_available():
            gpu = torch.cuda.get_device_properties(0)
            vram_gb = gpu.total_memory / 1024**3
            print(f"✅ GPU: {gpu.name}")
            print(f"   VRAM: {vram_gb:.1f} GB")
            if vram_gb < 4:
                print("⚠️  VRAM < 4GB — giảm batch xuống 8")
                return 0, 8
            elif vram_gb < 6:
                print("   VRAM 4-6GB — batch=12 an toàn")
                return 0, 12
            else:
                print("   VRAM 6GB (RTX 4050) — batch=16 OK")
                return 0, 16
        else:
            print("⚠️  Không có GPU CUDA — dùng CPU (rất chậm)")
            return "cpu", 4
    except ImportError:
        print("❌ PyTorch chưa được cài")
        sys.exit(1)


def install_if_needed():
    """Cài ultralytics nếu chưa có"""
    try:
        import ultralytics
        print(f"✅ Ultralytics {ultralytics.__version__} đã cài")
    except ImportError:
        print("Cài ultralytics...")
        subprocess.run([sys.executable, "-m", "pip", "install",
                        "ultralytics", "-q"], check=True)
        print("✅ Đã cài ultralytics")

    # Cài roboflow nếu chưa có
    try:
        import roboflow
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install",
                        "roboflow", "-q"], check=True)


def download_roboflow_yolov8(out_dir: Path):
    """Download Roboflow dataset format YOLOv8"""
    yaml_path = out_dir / "yolov8" / "data.yaml"
    if yaml_path.exists():
        print(f"✅ Dataset YOLOv8 đã có: {yaml_path}")
        return yaml_path

    print("📥 Download Roboflow YOLOv8 dataset...")
    from roboflow import Roboflow
    rf = Roboflow(api_key="qI3lEKlNpIZpNENdk3MH")
    project = rf.workspace("nguyen-tuan-dat").project("drowsiness driver")
    dataset  = project.version(1).download(
        "yolov8",
        location=str(out_dir / "yolov8"),
        overwrite=False,
    )
    yaml_path = Path(dataset.location) / "data.yaml"
    print(f"✅ Downloaded: {yaml_path}")
    return yaml_path


def read_dataset_info(yaml_path: Path) -> dict:
    """Đọc data.yaml để biết classes"""
    try:
        import yaml
        with open(yaml_path) as f:
            cfg = yaml.safe_load(f)
        return cfg
    except ImportError:
        # Đọc thủ công nếu pyyaml chưa có
        info = {}
        with open(yaml_path) as f:
            for line in f:
                if line.startswith("nc:"):
                    info["nc"] = int(line.split(":")[1].strip())
                elif line.startswith("names:"):
                    names_str = line.split(":")[1].strip()
                    info["names"] = [n.strip() for n in names_str.strip("[]").split(",")]
        return info


def main():
    parser = argparse.ArgumentParser(
        description="Train YOLOv8 cho drowsy driver detection (RTX 4050 optimized)"
    )
    parser.add_argument("--model",   default="yolov8n",
                        choices=["yolov8n","yolov8s","yolov8m"],
                        help="Model size: n=nano, s=small, m=medium")
    parser.add_argument("--epochs",  type=int, default=50)
    parser.add_argument("--data-dir",default="roboflow_data",
                        help="Thư mục chứa/sẽ chứa Roboflow dataset")
    parser.add_argument("--out-dir", default="outputs/training_yolo/yolov8",
                        help="Thư mục lưu kết quả")
    parser.add_argument("--no-download", action="store_true",
                        help="Bỏ qua download (nếu đã có dataset)")
    args = parser.parse_args()

    # ── 0. Setup ──────────────────────────────────────────────────────
    install_if_needed()
    device, batch = check_gpu()

    PROJECT_ROOT = Path(__file__).parent.parent
    data_dir     = PROJECT_ROOT / args.data_dir
    out_dir      = PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*55}")
    print(f"🎯  TRAIN YOLOV8 — LOCAL RTX 4050")
    print(f"{'='*55}")
    print(f"  Model   : {args.model}")
    print(f"  Epochs  : {args.epochs}")
    print(f"  Device  : {device}")
    print(f"  Batch   : {batch}")
    print(f"  VRAM    : FP16 half precision")

    # ── 1. Download dataset ───────────────────────────────────────────
    if not args.no_download:
        yaml_path = download_roboflow_yolov8(data_dir)
    else:
        yaml_path = data_dir / "yolov8" / "data.yaml"
        if not yaml_path.exists():
            print(f"❌ data.yaml không tìm thấy: {yaml_path}")
            sys.exit(1)

    # Đọc thông tin dataset
    cfg = read_dataset_info(yaml_path)
    print(f"\n📦 Dataset info:")
    print(f"  Classes ({cfg.get('nc','?')}): {cfg.get('names','?')}")

    # ── 2. Train YOLOv8 ──────────────────────────────────────────────
    from ultralytics import YOLO
    import torch

    model = YOLO(f"{args.model}.pt")  # tải pretrained weights từ Ultralytics
    print(f"\n🏋️  Bắt đầu training {args.model}...")
    print(f"   Dự kiến: ~{'30-50' if 'n' in args.model else '60-90'} phút")

    results = model.train(
        data    = str(yaml_path),
        epochs  = args.epochs,
        imgsz   = 640,
        batch   = batch,
        device  = device,
        half    = True if device != "cpu" else False,   # FP16 tiết kiệm VRAM
        cache   = "ram",          # cache ảnh vào RAM → nhanh hơn
        workers = 4,              # 4 worker threads
        cos_lr  = True,           # cosine LR schedule
        patience= 15,             # early stopping
        save    = True,
        exist_ok= True,
        name    = f"drowsy_{args.model}",
        project = str(out_dir),
        verbose = True,
        # Augmentation nhẹ (ảnh khuôn mặt không cần augment quá mạnh)
        hsv_h   = 0.015,
        hsv_s   = 0.4,
        hsv_v   = 0.4,
        degrees = 5.0,            # xoay nhẹ
        flipud  = 0.0,            # không lật dọc (khuôn mặt)
        fliplr  = 0.5,
        mosaic  = 0.5,
        mixup   = 0.0,
    )

    # ── 3. Kết quả ───────────────────────────────────────────────────
    save_dir = Path(results.save_dir)
    print(f"\n✅ Training xong!")
    print(f"   Results: {save_dir}")

    # Đọc metrics cuối cùng
    results_csv = save_dir / "results.csv"
    if results_csv.exists():
        import pandas as pd
        df = pd.read_csv(results_csv)
        df.columns = [c.strip() for c in df.columns]
        best_row   = df.loc[df.filter(like="mAP50").iloc[:,0].idxmax()]

        best_epoch = int(best_row.get("epoch", df.index[-1]))
        map50      = float(best_row.filter(like="mAP50").iloc[0]) if len(best_row.filter(like="mAP50")) > 0 else 0

        print(f"\n{'='*45}")
        print(f"  Best epoch : {best_epoch}")
        print(f"  Best mAP50 : {map50:.4f} ({map50*100:.2f}%)")
        print(f"{'='*45}")

        # Lưu summary để so sánh sau
        summary = {
            "model":      args.model,
            "format":     "yolov8",
            "device":     "local_rtx4050",
            "epochs_ran": len(df),
            "best_epoch": best_epoch,
            "best_map50": map50,
            "save_dir":   str(save_dir),
            "classes":    cfg.get("names", []),
        }
        summary_path = out_dir / f"summary_{args.model}.json"
        summary_path.write_text(json.dumps(summary, indent=2))
        print(f"  Summary: {summary_path}")

    # ── 4. Export TFLite (tùy chọn) ──────────────────────────────────
    print(f"\n📱 Export TFLite (optional — ~400MB, khó dùng Android real-time):")
    print(f"   model.export(format='tflite') # bỏ comment nếu cần")
    # best_model = YOLO(str(save_dir / 'weights' / 'best.pt'))
    # best_model.export(format='tflite', imgsz=320)  # dùng 320 cho nhẹ hơn

    print(f"\n📋 Bước tiếp theo:")
    print(f"   1. Chờ Colab xong YOLO26")
    print(f"   2. Chạy: python tools\\compare_yolo_results.py")
    print(f"   3. Xem outputs/yolo_comparison.png")


if __name__ == "__main__":
    main()
