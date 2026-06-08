# %% [markdown]
# # YOLO11s — Drowsy Driver Detection
# **Phuong phap 1: YOLO11s thay the YOLO26m**
#
# YOLO11s co C2PSA (Cross-Stage Partial + Parallel Spatial Attention) built-in:
#   - Nho hon: 9.4M params vs 20.4M (YOLO26m)
#   - Chinh xac hon COCO: mAP50=47.0% vs ~44% (YOLOv8s goc)
#   - Cung toc do, dung memory it hon
#
# Thay doi duy nhat so voi colab_drowsy_yolo26.ipynb:
#   model=yolo26m.pt  ->  model=yolo11s.pt
#
# Chay tren Google Colab T4 GPU
# Thu tu: Runtime -> Change runtime type -> T4 GPU -> Run All

# %% Cell 1 — Kiem tra GPU
!nvidia-smi

# %% Cell 2 — Cai packages
# %pip install -q "ultralytics>=8.4.0" supervision roboflow
import ultralytics
ultralytics.checks()

# %% Cell 3 — Setup HOME (bat buoc truoc Cell 4+)
import os
HOME = os.getcwd()
print("HOME:", HOME)

# %% Cell 4 — Download dataset Datio_drowsines
import zipfile, yaml
from pathlib import Path
from roboflow import Roboflow

rf      = Roboflow(api_key="qI3lEKlNpIZpNENdk3MH")
project = rf.workspace("nguyen-tuan-dat").project("datio_drowsines")
dataset = project.version(1).download("yolov8")

loc = Path(dataset.location)
yaml_hits = list(loc.rglob("data.yaml"))
if not yaml_hits:
    raise FileNotFoundError("Khong tim thay data.yaml")
DATA_YAML = str(yaml_hits[0])

# Fix duong dan tuyet doi trong data.yaml
with open(DATA_YAML) as f:
    cfg = yaml.safe_load(f)
base_dir = Path(DATA_YAML).parent
changed = False
for key in ["train", "val", "valid", "test"]:
    if key in cfg and not Path(cfg[key]).is_absolute():
        cfg[key] = str(base_dir / cfg[key])
        changed = True
if changed:
    with open(DATA_YAML, "w") as f:
        yaml.dump(cfg, f)

print("Classes:", cfg.get("names", []))
print("nc     :", cfg.get("nc", 0))
for split in ["train", "valid", "val", "test"]:
    p = Path(dataset.location) / split / "images"
    if p.exists():
        n = len(list(p.glob("*.jpg")) + list(p.glob("*.png")))
        print(f"  {split}: {n} images")

# %% Cell 5 — YOLO11 Architecture Overview
# YOLO11 thay the C2f bang C2PSA (Co Parallel Spatial Attention)
#
# YOLO11s Backbone:
#   Conv → C3k2 → C3k2 → [SPPF+PSA] → C3k2
#                                 ^
#                         C2PSA block: chia feature map thanh 2 nhanh
#                           Nhanh 1: giu nguyen
#                           Nhanh 2: Multi-Head Self-Attention (4 heads)
#                         -> Concat -> Project
#
# Ket qua: hieu qua hon voi khuon mat (spatial attention focus vao mat, mieng)

print("YOLO11 vs YOLOv8:")
print(f"  {'Model':<12} {'Params':>8} {'mAP50-COCO':>12} {'Latency T4':>12}")
print(f"  {'-'*46}")
for row in [
    ("yolo11n",  "2.6M",  "39.5%", "6.5ms"),
    ("yolo11s",  "9.4M",  "47.0%", "7.0ms"),
    ("yolo11m",  "20.1M", "51.5%", "9.5ms"),
    ("yolov8s",  "11.2M", "44.9%", "7.0ms"),
]:
    print(f"  {row[0]:<12} {row[1]:>8} {row[2]:>12} {row[3]:>12}")
print()
print("YOLO11s: nho hon + chinh xac hon YOLOv8s, cung toc do!")

# %% Cell 6 — TRAIN YOLO11s  (~40-60 phut tren T4 voi 50 epochs)
# imgsz=640 de dat accuracy tot nhat
# Voi 5h plan co the giam xuong imgsz=320 de nhanh ~4x

!yolo task=detect \
      mode=train \
      model=yolo11s.pt \
      data={DATA_YAML} \
      epochs=50 \
      imgsz=640 \
      batch=16 \
      patience=15 \
      plots=True \
      name=drowsy_yolo11s \
      project={HOME}/runs/detect \
      cos_lr=True \
      lr0=0.01 \
      lrf=0.01 \
      momentum=0.937 \
      weight_decay=0.0005 \
      warmup_epochs=3 \
      degrees=5.0 \
      fliplr=0.5 \
      hsv_v=0.4 \
      mosaic=0.5 \
      close_mosaic=10

# %% Cell 7 — Validate tren val set
best = f"{HOME}/runs/detect/drowsy_yolo11s/weights/best.pt"
!yolo task=detect mode=val \
      model={best} \
      data={DATA_YAML} \
      verbose=True

# %% Cell 8 — So sanh YOLO11s vs YOLO26m
import pandas as pd, json, os

df_11s = pd.read_csv(f"{HOME}/runs/detect/drowsy_yolo11s/results.csv")
df_11s.columns = [c.strip() for c in df_11s.columns]
map50_col = [c for c in df_11s.columns if "map50" in c.lower() and "95" not in c.lower()][0]
best_map50_11s = float(df_11s[map50_col].max())
best_epoch_11s = int(df_11s.loc[df_11s[map50_col].idxmax(), "epoch"])

print("=" * 50)
print("  Ket qua so sanh")
print("=" * 50)

# YOLO26 da chay truoc
yolo26_json = f"{HOME}/summary_yolo26.json"
if os.path.exists(yolo26_json):
    with open(yolo26_json) as f:
        r26 = json.load(f)
    print(f"  YOLO26m : {r26['best_map50']*100:.2f}%  (epoch {r26['best_epoch']}, {r26['epochs_ran']} epochs)")
print(f"  YOLO11s : {best_map50_11s*100:.2f}%  (epoch {best_epoch_11s})")

if os.path.exists(yolo26_json):
    delta = (best_map50_11s - r26["best_map50"]) * 100
    print(f"  Delta   : {delta:+.2f}%")
print()

# Luu summary
summary_11s = {
    "model":      "yolo11s",
    "device":     "colab_t4",
    "epochs_ran": len(df_11s),
    "best_epoch": best_epoch_11s,
    "best_map50": round(best_map50_11s, 6),
    "classes":    cfg.get("names", []),
}
with open(f"{HOME}/summary_yolo11s.json", "w") as f:
    json.dump(summary_11s, f, indent=2)
print("Saved: summary_yolo11s.json")

# %% Cell 9 — Xem anh ket qua
from IPython.display import Image as IPyImage, display

display(IPyImage(filename=f"{HOME}/runs/detect/drowsy_yolo11s/results.png", width=900))
try:
    display(IPyImage(filename=f"{HOME}/runs/detect/drowsy_yolo11s/confusion_matrix.png", width=600))
except:
    pass

# %% Cell 10 — Predict + hien thi
!yolo task=detect mode=predict \
      model={best} \
      source={dataset.location}/test/images \
      conf=0.3 save=True verbose=False

import glob, matplotlib.pyplot as plt, matplotlib.image as mpimg
pred_imgs = sorted(glob.glob(f"{HOME}/runs/detect/predict*/*.jpg"))[-4:]
if pred_imgs:
    fig, axes = plt.subplots(1, len(pred_imgs), figsize=(16, 5))
    if len(pred_imgs) == 1:
        axes = [axes]
    for ax, p in zip(axes, pred_imgs):
        ax.imshow(mpimg.imread(p))
        ax.axis("off")
    plt.suptitle("YOLO11s — Drowsy Detection Predictions", fontsize=12)
    plt.tight_layout()
    plt.show()

# %% Cell 11 — Download ve may
from google.colab import files
files.download(f"{HOME}/summary_yolo11s.json")
files.download(f"{HOME}/runs/detect/drowsy_yolo11s/weights/best.pt")
print("Da download: summary_yolo11s.json + best.pt")
