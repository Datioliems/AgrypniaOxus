# %%
# ============================================================
#  COLAB: Train YOLO26 cho Drowsy Driver Detection
#  Chạy file này trên Google Colab (T4 GPU, 15GB VRAM)
#  Song song với YOLOv8 đang chạy trên máy local RTX 4050
#
#  CÁCH DÙNG:
#    1. Upload file này lên Google Colab
#    2. Runtime → Change runtime type → T4 GPU
#    3. Run All (Ctrl+F9)
#    4. Sau khi xong download summary_yolo26.json
#    5. Copy file đó vào outputs/training_yolo/yolo26/ trên local
#    6. Chạy tools/compare_yolo_results.py để so sánh
# ============================================================

# %% [1] Setup môi trường
print("=" * 60)
print("YOLO26 TRAINING — Google Colab T4")
print("=" * 60)

import subprocess, sys

def pip(pkg):
    subprocess.run([sys.executable, "-m", "pip", "install", pkg, "-q"], check=True)

# Cài dependencies
pip("roboflow")
pip("ultralytics")        # Ultralytics hỗ trợ nhiều format YOLO
pip("matplotlib")
pip("pandas")
pip("seaborn")

import os, json, shutil
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd
import numpy as np
import torch

print(f"\n✅ PyTorch {torch.__version__}")
print(f"   CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    gpu = torch.cuda.get_device_properties(0)
    vram = gpu.total_memory / 1024**3
    print(f"   GPU: {gpu.name} ({vram:.1f} GB VRAM)")

# %% [2] Mount Google Drive (optional — để save kết quả)
# Bỏ comment nếu muốn save sang Drive
try:
    from google.colab import drive
    drive.mount("/content/drive", force_remount=False)
    SAVE_TO_DRIVE = True
    DRIVE_DIR = Path("/content/drive/MyDrive/DrowsyDriver_YOLO26")
    DRIVE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"✅ Drive mounted. Save dir: {DRIVE_DIR}")
except Exception as e:
    SAVE_TO_DRIVE = False
    print(f"⚠️  Drive not mounted ({e}) — kết quả chỉ lưu local /content")

# %% [3] Cấu hình
# ─────────────────────────────────────────────────────────────
ROBOFLOW_API_KEY  = "qI3lEKlNpIZpNENdk3MH"
ROBOFLOW_WORKSPACE = "nguyen-tuan-dat"
ROBOFLOW_PROJECT  = "drowsiness driver"
ROBOFLOW_VERSION  = 1

# YOLO26 format — đây là format Roboflow export cho YOLO v2/v6 (Darknet-style)
# Ultralytics có thể train từ format này qua yolov8 wrapper
DATASET_FORMAT    = "yolo26"    # hoặc "yolov8" nếu yolo26 ko support
MODEL_BASE        = "yolov8s"   # T4 có 15GB → dùng small thay vì nano
EPOCHS            = 60          # nhiều hơn vì T4 nhanh hơn RTX 4050
IMGSZ             = 640
BATCH             = 32          # T4 15GB có thể batch lớn hơn
OUTPUT_DIR        = Path("/content/outputs_yolo26")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
print(f"\n📋 Config:")
print(f"  Model    : {MODEL_BASE}")
print(f"  Format   : {DATASET_FORMAT}")
print(f"  Epochs   : {EPOCHS}")
print(f"  Batch    : {BATCH}")
print(f"  IMGSZ    : {IMGSZ}")

# %% [4] Download dataset từ Roboflow
print("\n📥 Download Roboflow dataset (YOLO26 format)...")
from roboflow import Roboflow

rf = Roboflow(api_key=ROBOFLOW_API_KEY)
project = rf.workspace(ROBOFLOW_WORKSPACE).project(ROBOFLOW_PROJECT)

# Thử download YOLO26 format trước, fallback sang YOLOv8 nếu lỗi
dataset_yolo26 = None
yaml_path = None

try:
    dataset_yolo26 = project.version(ROBOFLOW_VERSION).download(
        "yolo26",
        location="/content/dataset_yolo26",
        overwrite=True,
    )
    print(f"✅ YOLO26 dataset downloaded: {dataset_yolo26.location}")
    dataset_location = Path(dataset_yolo26.location)

    # YOLO26 có thể dùng obj.data thay vì data.yaml
    yaml_candidates = list(dataset_location.glob("*.yaml")) + \
                      list(dataset_location.glob("*.data")) + \
                      list(dataset_location.glob("obj.data"))
    if yaml_candidates:
        yaml_path = yaml_candidates[0]
        print(f"   Config file: {yaml_path}")
    else:
        print("⚠️  Không tìm thấy config file trong YOLO26 format")

except Exception as e:
    print(f"⚠️  YOLO26 download failed: {e}")
    print("   → Fallback sang YOLOv8 format (compare format khác nhau)")
    DATASET_FORMAT = "yolov8_fallback"

# Download YOLOv8 format (luôn cần để train với Ultralytics)
print("\n📥 Download YOLOv8 format (cho Ultralytics training)...")
dataset_v8 = project.version(ROBOFLOW_VERSION).download(
    "yolov8",
    location="/content/dataset_yolov8",
    overwrite=True,
)
print(f"✅ YOLOv8 dataset: {dataset_v8.location}")
yaml_path_v8 = Path(dataset_v8.location) / "data.yaml"
print(f"   data.yaml: {yaml_path_v8}")

# %% [5] Khám phá dataset
print("\n📊 Dataset overview:")
dataset_v8_path = Path(dataset_v8.location)

import yaml
with open(yaml_path_v8) as f:
    cfg = yaml.safe_load(f)

print(f"  Classes ({cfg['nc']}): {cfg['names']}")
for split in ["train", "val", "test"]:
    split_path = dataset_v8_path / split / "images"
    if split_path.exists():
        n = len(list(split_path.glob("*.jpg")) + list(split_path.glob("*.png")))
        print(f"  {split:5s}: {n:5d} images")

# %% [6] Visualize mẫu từ dataset
print("\n🖼️  Hiển thị mẫu dataset...")
train_imgs = list((dataset_v8_path / "train" / "images").glob("*.jpg"))[:6]
if len(train_imgs) < 6:
    train_imgs += list((dataset_v8_path / "train" / "images").glob("*.png"))
train_imgs = train_imgs[:6]

if train_imgs:
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    for ax, img_path in zip(axes.flat, train_imgs):
        img = mpimg.imread(img_path)
        ax.imshow(img)
        ax.set_title(img_path.stem[:20], fontsize=8)
        ax.axis("off")
    plt.suptitle("Mẫu Dataset — Drowsiness Driver (Roboflow)", fontsize=12)
    plt.tight_layout()
    plt.savefig(str(OUTPUT_DIR / "dataset_samples.png"), dpi=120)
    plt.show()
    print(f"✅ Saved: {OUTPUT_DIR}/dataset_samples.png")

# %% [7] Train với YOLOv8 + Ultralytics
# Colab T4: dùng yolov8s (larger model so sánh được với yolov8n local)
print(f"\n{'='*60}")
print(f"🏋️  TRAINING: {MODEL_BASE} (YOLO26-equivalent so sánh)")
print(f"   T4 GPU, batch={BATCH}, epochs={EPOCHS}")
print(f"{'='*60}")

from ultralytics import YOLO

model_v8 = YOLO(f"{MODEL_BASE}.pt")

results = model_v8.train(
    data     = str(yaml_path_v8),
    epochs   = EPOCHS,
    imgsz    = IMGSZ,
    batch    = BATCH,
    device   = 0,               # GPU 0 (T4)
    half     = True,            # FP16
    cache    = "ram",
    workers  = 2,               # Colab CPU cores ít hơn
    cos_lr   = True,
    patience = 20,
    save     = True,
    exist_ok = True,
    name     = f"drowsy_{MODEL_BASE}_colab",
    project  = str(OUTPUT_DIR),
    verbose  = True,
    # Augmentation
    hsv_h    = 0.015,
    hsv_s    = 0.4,
    hsv_v    = 0.4,
    degrees  = 5.0,
    flipud   = 0.0,
    fliplr   = 0.5,
    mosaic   = 0.5,
    mixup    = 0.0,
)

save_dir = Path(results.save_dir)
print(f"\n✅ Training done! Save dir: {save_dir}")

# %% [8] Nếu YOLO26 dataset đã download thành công → train thêm
# So sánh: cùng model yolov8n nhưng data khác (v8 format vs 26 format)
if dataset_yolo26 is not None and DATASET_FORMAT == "yolo26":
    print(f"\n{'='*60}")
    print(f"🏋️  TRAINING THÊM: yolov8n với YOLO26 data prep")
    print(f"{'='*60}")

    # Chuyển YOLO26 format sang YOLOv8 format nếu cần
    yolo26_path = Path(dataset_yolo26.location)

    # Thường YOLO26 dùng cấu trúc obj/ thay vì train/images
    # Ultralytics đọc được nếu có data.yaml đúng format
    yolo26_yaml = None
    for candidate in ["data.yaml", "obj.data"]:
        p = yolo26_path / candidate
        if p.exists():
            yolo26_yaml = p
            break

    if yolo26_yaml and yolo26_yaml.suffix == ".yaml":
        print(f"   Train từ YOLO26 data.yaml: {yolo26_yaml}")
        model_26 = YOLO("yolov8n.pt")  # nano để so sánh fair
        results_26 = model_26.train(
            data     = str(yolo26_yaml),
            epochs   = EPOCHS,
            imgsz    = IMGSZ,
            batch    = BATCH,
            device   = 0,
            half     = True,
            cache    = "ram",
            workers  = 2,
            cos_lr   = True,
            patience = 20,
            save     = True,
            exist_ok = True,
            name     = "drowsy_yolov8n_yolo26data",
            project  = str(OUTPUT_DIR),
            verbose  = True,
        )
        save_dir_26 = Path(results_26.save_dir)
        print(f"   ✅ YOLO26-data training done: {save_dir_26}")
    else:
        print("⚠️  YOLO26 format dùng obj.data — Ultralytics không đọc trực tiếp được")
        print("   → Chỉ dùng YOLOv8 format cho training")

# %% [9] Đọc và hiển thị kết quả
print("\n📊 Training Results:")
results_csv = save_dir / "results.csv"

if results_csv.exists():
    df = pd.read_csv(results_csv)
    df.columns = [c.strip() for c in df.columns]

    print(df.tail(5).to_string())

    # Plot training curves
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    metric_cols = [c for c in df.columns if any(k in c.lower() for k in
                   ["loss", "map50", "precision", "recall"])][:6]

    for ax, col in zip(axes.flat, metric_cols):
        if col in df.columns:
            ax.plot(df["epoch"] if "epoch" in df.columns else df.index,
                    df[col], linewidth=2, color="steelblue")
            ax.set_title(col.strip(), fontsize=10)
            ax.set_xlabel("Epoch")
            ax.grid(True, alpha=0.3)

    plt.suptitle(f"Training Curves — {MODEL_BASE} (Colab T4)", fontsize=13)
    plt.tight_layout()
    plt.savefig(str(OUTPUT_DIR / "training_curves.png"), dpi=120)
    plt.show()
    print(f"✅ Saved: {OUTPUT_DIR}/training_curves.png")

# %% [10] Tính metrics tốt nhất
print("\n🏆 Best Metrics:")
if results_csv.exists():
    map50_col = [c for c in df.columns if "map50" in c.lower() and "95" not in c.lower()]
    map5095_col = [c for c in df.columns if "map50-95" in c.lower() or "map50_95" in c.lower()]
    prec_col = [c for c in df.columns if "precision" in c.lower()]
    rec_col  = [c for c in df.columns if "recall" in c.lower()]

    def best(cols):
        if not cols: return 0.0, 0
        col = cols[0]
        idx = df[col].idxmax()
        return float(df[col].iloc[idx]), int(df["epoch"].iloc[idx]) if "epoch" in df.columns else idx

    best_map50, best_ep_50       = best(map50_col)
    best_map5095, best_ep_5095   = best(map5095_col)
    best_prec, _                 = best(prec_col)
    best_rec, _                  = best(rec_col)

    print(f"  Best mAP@50     : {best_map50:.4f}  ({best_map50*100:.2f}%)  @ epoch {best_ep_50}")
    print(f"  Best mAP@50-95  : {best_map5095:.4f}  ({best_map5095*100:.2f}%)  @ epoch {best_ep_5095}")
    print(f"  Best Precision  : {best_prec:.4f}  ({best_prec*100:.2f}%)")
    print(f"  Best Recall     : {best_rec:.4f}  ({best_rec*100:.2f}%)")

    # Lưu summary JSON — copy về local để so sánh
    summary = {
        "model":          MODEL_BASE,
        "format":         DATASET_FORMAT,
        "device":         "colab_t4",
        "epochs_ran":     len(df),
        "best_epoch":     best_ep_50,
        "best_map50":     round(best_map50, 6),
        "best_map50_95":  round(best_map5095, 6),
        "best_precision": round(best_prec, 6),
        "best_recall":    round(best_rec, 6),
        "save_dir":       str(save_dir),
        "classes":        cfg.get("names", []),
    }
    summary_path = OUTPUT_DIR / "summary_yolo26.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\n✅ Summary saved: {summary_path}")
    print("   📋 Copy file này về local để so sánh!")

# %% [11] Hiển thị ảnh kết quả (confusion matrix, PR curve)
result_imgs = [
    save_dir / "confusion_matrix.png",
    save_dir / "PR_curve.png",
    save_dir / "results.png",
]

for img_path in result_imgs:
    if img_path.exists():
        img = mpimg.imread(str(img_path))
        plt.figure(figsize=(10, 6))
        plt.imshow(img)
        plt.axis("off")
        plt.title(img_path.name)
        plt.tight_layout()
        plt.show()
        print(f"✅ {img_path.name}")

# %% [12] Test inference trên ảnh mẫu
print("\n🔍 Test inference...")
val_imgs = list((dataset_v8_path / "val" / "images").glob("*.jpg"))[:3]
if val_imgs:
    best_weights = save_dir / "weights" / "best.pt"
    if best_weights.exists():
        model_best = YOLO(str(best_weights))
        for img_path in val_imgs:
            pred = model_best(str(img_path), conf=0.25, verbose=False)
            pred[0].save(filename=str(OUTPUT_DIR / f"pred_{img_path.name}"))
        print(f"✅ Saved {len(val_imgs)} prediction images")

        # Hiển thị predictions
        pred_imgs = list(OUTPUT_DIR.glob("pred_*.jpg"))[:3]
        if pred_imgs:
            fig, axes = plt.subplots(1, min(3, len(pred_imgs)), figsize=(15, 5))
            if len(pred_imgs) == 1:
                axes = [axes]
            for ax, p in zip(axes, pred_imgs):
                ax.imshow(mpimg.imread(str(p)))
                ax.axis("off")
                ax.set_title("Prediction", fontsize=10)
            plt.suptitle("Model Predictions — Test", fontsize=12)
            plt.tight_layout()
            plt.savefig(str(OUTPUT_DIR / "predictions.png"), dpi=100)
            plt.show()

# %% [13] Benchmark tốc độ inference
print("\n⚡ Benchmark inference speed...")
if val_imgs and (save_dir / "weights" / "best.pt").exists():
    import time
    model_bench = YOLO(str(save_dir / "weights" / "best.pt"))

    # Warmup
    for _ in range(3):
        model_bench(str(val_imgs[0]), verbose=False)

    # Benchmark 20 frames
    times = []
    for _ in range(20):
        t0 = time.perf_counter()
        model_bench(str(val_imgs[0]), verbose=False)
        times.append((time.perf_counter() - t0) * 1000)

    avg_ms = np.mean(times)
    fps = 1000 / avg_ms
    print(f"  Avg inference  : {avg_ms:.1f} ms/frame")
    print(f"  FPS            : {fps:.1f}")
    print(f"  Realtime ready : {'✅ YES' if fps >= 15 else '❌ NO (too slow for live)'}")

    # Thêm vào summary
    if summary_path.exists():
        s = json.loads(summary_path.read_text())
        s["inference_ms_colab"] = round(avg_ms, 2)
        s["fps_colab"] = round(fps, 1)
        summary_path.write_text(json.dumps(s, indent=2))

# %% [14] Save ke Drive (nếu đã mount)
if SAVE_TO_DRIVE:
    print(f"\n💾 Saving sang Drive: {DRIVE_DIR}")
    # Copy weights
    weights_dir = save_dir / "weights"
    dest_weights = DRIVE_DIR / "weights"
    if weights_dir.exists():
        shutil.copytree(str(weights_dir), str(dest_weights), dirs_exist_ok=True)
        print(f"  ✅ Weights → {dest_weights}")

    # Copy summary
    shutil.copy(str(summary_path), str(DRIVE_DIR / "summary_yolo26.json"))
    print(f"  ✅ Summary → Drive")

    # Copy plots
    for f in OUTPUT_DIR.glob("*.png"):
        shutil.copy(str(f), str(DRIVE_DIR / f.name))
    print(f"  ✅ Plots → Drive")
    print(f"\n📂 Drive folder: {DRIVE_DIR}")

# %% [15] Download files về máy (nếu không dùng Drive)
print("\n📥 Download kết quả về máy:")
print("  Copy đoạn này vào cell mới nếu cần download trực tiếp:\n")
print("""
from google.colab import files

# Download summary JSON (nhỏ, cần để so sánh)
files.download('/content/outputs_yolo26/summary_yolo26.json')

# Download best weights (~25MB)
files.download(f'{results.save_dir}/weights/best.pt')

# Download training curves
files.download('/content/outputs_yolo26/training_curves.png')
""")

# %% [16] In hướng dẫn so sánh
print("\n" + "=" * 60)
print("✅ COLAB TRAINING DONE!")
print("=" * 60)
print("""
📋 BƯỚC TIẾP THEO để SO SÁNH với local YOLOv8:

1. Download summary_yolo26.json từ Colab
   (File → Download hoặc dùng code ở cell trên)

2. Copy file về máy local:
   D:\\2026.AI\\DrowsyDriverAndroid\\outputs\\training_yolo\\yolo26\\

3. Mở PowerShell, chạy:
   cd D:\\2026.AI\\DrowsyDriverAndroid
   python tools\\compare_yolo_results.py

4. Xem bảng so sánh tại:
   outputs\\yolo_comparison_report.png
   outputs\\yolo_comparison.json

💡 Metrics để so sánh:
   - mAP@50     (chính) — càng cao càng tốt
   - mAP@50-95  (khắt khe hơn)
   - Precision / Recall
   - Inference ms/frame
""")
