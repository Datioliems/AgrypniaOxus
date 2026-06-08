# %% [markdown]
# # RT-DETR — Real-Time Detection Transformer
# **Phuong phap 3: Transformer Decoder thay the YOLO Head + NMS**
#
# RT-DETR (Baidu, 2023) thay doi kien truc co ban:
#
#   YOLO:     Backbone -> Neck -> Grid anchors -> NMS (hau xu ly)
#   RT-DETR:  Backbone -> Hybrid Encoder -> Transformer Decoder -> Truc tiep ra boxes
#
# Loi ich chinh:
#   1. Khong can NMS (Non-Max Suppression) — Transformer tu xu ly duplicate boxes
#   2. Global attention: moi query "nhin" toan bo feature map
#   3. mAP50 cao hon YOLO26m ~3-5% tren COCO
#
# Luu y:
#   - imgsz PHAI la 640 (RT-DETR khong ho tro 320)
#   - batch=8 (model nang hon, can VRAM hon)
#   - lr0=1e-4 (thap hon YOLO vì da pretrained Transformer)
#
# Chay tren Google Colab T4 GPU

# %% Cell 1 — Kiem tra GPU
!nvidia-smi

# %% Cell 2 — Cai packages
# %pip install -q "ultralytics>=8.4.0" supervision roboflow
import ultralytics
ultralytics.checks()

# %% Cell 3 — Setup HOME
import os
HOME = os.getcwd()
print("HOME:", HOME)

# %% Cell 4 — Download dataset
import zipfile, yaml
from pathlib import Path
from roboflow import Roboflow

rf      = Roboflow(api_key="qI3lEKlNpIZpNENdk3MH")
project = rf.workspace("nguyen-tuan-dat").project("datio_drowsines")
dataset = project.version(1).download("yolov8")

loc = Path(dataset.location)
yaml_hits = list(loc.rglob("data.yaml"))
DATA_YAML = str(yaml_hits[0])

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
for split in ["train", "valid", "val", "test"]:
    p = Path(dataset.location) / split / "images"
    if p.exists():
        n = len(list(p.glob("*.jpg")) + list(p.glob("*.png")))
        print(f"  {split}: {n} images")

# %% Cell 5 — RT-DETR Architecture Overview
# RT-DETR Architecture (chi tiet):
#
#   Input (640x640)
#     |
#   ResNet-50 / ResNet-101 Backbone
#     |  (trich xuat features o 3 scale: S, M, L)
#     |
#   Hybrid Encoder
#     |  Intra-scale: Attention tren tung scale
#     |  Cross-scale: Fuse features giua cac scale
#     |
#   Transformer Decoder (300 queries)
#     |  Query: "Toi dang tim object o dau?"
#     |  Cross-attention: Query "nhin" vao Encoder features
#     |  Self-attention: Cac query noi chuyện voi nhau (tranh duplicate)
#     |
#   Outputs: (class_prob, box_coords) cho moi query
#   -> Chon top-K confident queries lam ket qua cuoi (khong can NMS!)
#
# Ultralytics models:
#   rtdetr-l : ResNet-50  backbone, 32M params, mAP50=53.0% COCO
#   rtdetr-x : ResNet-101 backbone, 67M params, mAP50=54.8% COCO

print("RT-DETR models:")
print(f"  {'Model':<12} {'Backbone':<15} {'Params':>8} {'mAP50':>8} {'ms T4':>8}")
print("  " + "-"*53)
for name, bb, params, map50, ms in [
    ("rtdetr-l", "ResNet-50",  "32M", "53.0%", "9.3ms"),
    ("rtdetr-x", "ResNet-101", "67M", "54.8%", "13.7ms"),
]:
    print(f"  {name:<12} {bb:<15} {params:>8} {map50:>8} {ms:>8}")
print()
print("Chon rtdetr-l cho T4 (rtdetr-x co the OOM tren 15GB VRAM)")

# %% Cell 6 — TRAIN RT-DETR-L  (~60-90 phut tren T4)
from ultralytics import RTDETR

model = RTDETR("rtdetr-l.pt")  # Auto-download pretrained weights (~140MB)

results = model.train(
    data=DATA_YAML,
    epochs=50,
    imgsz=640,        # RT-DETR yeu cau 640, khong giam duoc
    batch=8,          # Nho hon YOLO vi Transformer nang hon
    patience=15,
    plots=True,
    name="drowsy_rtdetr_l",
    project=f"{HOME}/runs/detect",
    cos_lr=True,
    lr0=1e-4,         # LR nho hon YOLO (Transformer da pretrained)
    lrf=0.01,
    warmup_epochs=2,
    weight_decay=1e-4,
    # Augmentation nhe hon cho Transformer
    degrees=5.0,
    fliplr=0.5,
    hsv_v=0.4,
    mosaic=0.5,
    close_mosaic=10,
)

# %% Cell 7 — Validate
import os
best = f"{HOME}/runs/detect/drowsy_rtdetr_l/weights/best.pt"
print(f"Best weights: {best}")
print(f"Exists: {os.path.exists(best)}")

model_val = RTDETR(best)
val_metrics = model_val.val(data=DATA_YAML, verbose=True)

# %% Cell 8 — So sanh RT-DETR vs YOLO26m
import pandas as pd, json

df = pd.read_csv(f"{HOME}/runs/detect/drowsy_rtdetr_l/results.csv")
df.columns = [c.strip() for c in df.columns]
map50_col = [c for c in df.columns if "map50" in c.lower() and "95" not in c.lower()][0]
best_map50_rtdetr = float(df[map50_col].max())
best_epoch_rtdetr = int(df.loc[df[map50_col].idxmax(), "epoch"])

print("=" * 55)
print("  So sanh ket qua")
print("=" * 55)
for json_file, label in [
    (f"{HOME}/summary_yolo26.json",  "YOLO26m"),
    (f"{HOME}/summary_yolo11s.json", "YOLO11s"),
]:
    if os.path.exists(json_file):
        with open(json_file) as f:
            r = json.load(f)
        print(f"  {label:<12}: {r['best_map50']*100:.2f}%  ({r['epochs_ran']} epochs)")

print(f"  {'RT-DETR-L':<12}: {best_map50_rtdetr*100:.2f}%  (epoch {best_epoch_rtdetr})")

summary = {
    "model":      "rtdetr-l",
    "device":     "colab_t4",
    "epochs_ran": len(df),
    "best_epoch": best_epoch_rtdetr,
    "best_map50": round(best_map50_rtdetr, 6),
    "classes":    cfg.get("names", []),
    "note":       "Transformer decoder, no NMS, imgsz=640",
}
with open(f"{HOME}/summary_rtdetr.json", "w") as f:
    json.dump(summary, f, indent=2)
print("\nSaved: summary_rtdetr.json")

# %% Cell 9 — Xem ket qua
from IPython.display import Image as IPyImage, display
import matplotlib.pyplot as plt

display(IPyImage(filename=f"{HOME}/runs/detect/drowsy_rtdetr_l/results.png", width=900))
try:
    display(IPyImage(filename=f"{HOME}/runs/detect/drowsy_rtdetr_l/confusion_matrix.png", width=600))
except:
    pass

# %% Cell 10 — Predict
model_pred = RTDETR(best)
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
    plt.suptitle("RT-DETR-L — Drowsy Detection (No NMS)", fontsize=12)
    plt.tight_layout()
    plt.show()

# %% Cell 11 — Download ve may
from google.colab import files
files.download(f"{HOME}/summary_rtdetr.json")
files.download(f"{HOME}/runs/detect/drowsy_rtdetr_l/weights/best.pt")
print("Da download: summary_rtdetr.json + best.pt")
