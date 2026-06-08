# %% [markdown]
# # YOLO26 — Drowsy Driver Detection (Colab)
# **Truoc khi chay:** `Runtime` → `Change runtime type` → **T4 GPU**
#
# **Lan dau:** Chay Cell 1 → Cell 11 theo thu tu
# **Session moi (data con cache):** Bat dau tu Cell 3 → Cell 7 (TRAIN)
#
# > Fix: format download la `yolov8` (khong phai `yolo26`) — yolo26 la ten *model weights*, khong phai format data.

# %% Cell 1 — Kiem tra GPU
get_ipython().system('nvidia-smi')

# %% Cell 2 — Cai thu vien
get_ipython().run_line_magic('pip', 'install -q "ultralytics>=8.4.0" roboflow supervision')
get_ipython().system('yolo settings sync=False')
import ultralytics
ultralytics.checks()

# %% Cell 3 — Setup HOME
import os
HOME = os.getcwd()
print("HOME:", HOME)   # phai la /content

# %% Cell 4 — Download dataset tu Roboflow
from roboflow import Roboflow
from pathlib import Path

rf      = Roboflow(api_key="qI3lEKlNpIZpNENdk3MH")
project = rf.workspace("nguyen-tuan-dat").project("datio_drowsines")
dataset = project.version(1).download("yolov8")   # ← "yolov8" khong phai "yolo26"

# Tim data.yaml linh hoat (tranh loi path cung)
loc       = Path(dataset.location)
yaml_hits = list(loc.rglob("data.yaml"))
if not yaml_hits:
    all_files = "\n".join(str(p) for p in sorted(loc.rglob("*"))[:40])
    raise FileNotFoundError(f"Khong tim thay data.yaml trong {loc}\nFiles:\n{all_files}")
DATA_YAML = str(yaml_hits[0])
print("Dataset  :", dataset.location)
print("data.yaml:", DATA_YAML)

# %% Cell 5 — Thong tin dataset
import yaml

with open(DATA_YAML) as f:
    cfg = yaml.safe_load(f)

print("Classes     :", cfg.get("names", []))
print("Num classes :", cfg.get("nc", 0))

base = Path(DATA_YAML).parent
for split in ["train", "valid", "val", "test"]:
    p = base / split / "images"
    if p.exists():
        n = len(list(p.glob("*.jpg")) + list(p.glob("*.png")))
        print(f"  {split}: {n} images")

# %% Cell 6 — Xem mau anh
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path

base = Path(DATA_YAML).parent
train_dir = next(
    (base / s / "images" for s in ["train", "valid"] if (base / s / "images").exists()),
    None
)

if train_dir:
    imgs = (list(train_dir.glob("*.jpg")) + list(train_dir.glob("*.png")))[:6]
    if imgs:
        fig, axes = plt.subplots(2, 3, figsize=(12, 7))
        for ax, p in zip(axes.flat, imgs):
            ax.imshow(mpimg.imread(str(p)))
            ax.axis("off")
        for ax in axes.flat[len(imgs):]:
            ax.axis("off")
        plt.suptitle(f"Mau anh dataset — {len(imgs)} anh", fontsize=13)
        plt.tight_layout()
        plt.show()
else:
    print("Khong tim thay thu muc anh train/valid")

# %% Cell 7 — TRAIN YOLO26  (~1h tren T4 voi 15 epoch)
# > Doi epochs=15 de vua voi ke hoach 5h; tang len 50 neu co nhieu thoi gian
get_ipython().system(
    f'yolo task=detect mode=train '
    f'model=yolo26m.pt '
    f'data={DATA_YAML} '
    f'epochs=15 '
    f'imgsz=640 '
    f'batch=16 '
    f'patience=10 '
    f'plots=True '
    f'name=drowsy_yolo26 '
    f'project={HOME}/runs/detect'
)

# %% Cell 8 — Ket qua training
from IPython.display import Image as IPyImage
import os

results_dir = f"{HOME}/runs/detect/drowsy_yolo26"
print("Files trong results dir:")
get_ipython().system(f'ls {results_dir}')
print()
IPyImage(filename=f"{results_dir}/results.png", width=900)

# %% Cell 9 — Confusion matrix + F1 / PR curve
from IPython.display import Image as IPyImage, display
import os

results_dir = f"{HOME}/runs/detect/drowsy_yolo26"
for fname in ["confusion_matrix.png", "F1_curve.png", "PR_curve.png"]:
    fp = f"{results_dir}/{fname}"
    if os.path.exists(fp):
        print(f"--- {fname} ---")
        display(IPyImage(filename=fp, width=620))

# %% Cell 10 — Validate tren tap val/test
best = f"{HOME}/runs/detect/drowsy_yolo26/weights/best.pt"
get_ipython().system(f'yolo task=detect mode=val model={best} data={DATA_YAML} verbose=True')

# %% Cell 11 — Predict + hien thi mau
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path

best     = f"{HOME}/runs/detect/drowsy_yolo26/weights/best.pt"
base     = Path(DATA_YAML).parent
test_src = next(
    (str(base / s / "images") for s in ["test", "valid"] if (base / s / "images").exists()),
    str(base)
)

get_ipython().system(
    f'yolo task=detect mode=predict model={best} '
    f'source={test_src} conf=0.3 save=True verbose=False '
    f'project={HOME}/runs/detect name=drowsy_predict'
)

pred_imgs = sorted(glob.glob(f"{HOME}/runs/detect/drowsy_predict*/*.jpg"))[:4]
if pred_imgs:
    fig, axes = plt.subplots(1, len(pred_imgs), figsize=(16, 5))
    if len(pred_imgs) == 1:
        axes = [axes]
    for ax, p in zip(axes, pred_imgs):
        ax.imshow(mpimg.imread(p))
        ax.axis("off")
    plt.suptitle("Predict results (conf=0.3)", fontsize=12)
    plt.tight_layout()
    plt.show()
else:
    print("Khong tim thay anh predict")

# %% Cell 12 — Luu summary JSON + tai ve may
import pandas as pd
import json
from google.colab import files

results_csv = f"{HOME}/runs/detect/drowsy_yolo26/results.csv"
df = pd.read_csv(results_csv)
df.columns = [c.strip() for c in df.columns]

map50_cols = [c for c in df.columns if "map50" in c.lower() and "95" not in c.lower()]
if map50_cols:
    best_idx   = df[map50_cols[0]].idxmax()
    best_map50 = float(df[map50_cols[0]].iloc[best_idx])
    best_epoch = int(df["epoch"].iloc[best_idx]) if "epoch" in df.columns else int(best_idx)
else:
    best_map50, best_epoch = 0.0, 0

print(f"Best mAP@50  : {best_map50*100:.2f}%")
print(f"Best epoch   : {best_epoch} / {len(df)}")

summary = {
    "model":      "yolo26m",
    "device":     "colab_t4",
    "dataset":    "datio_drowsines_v1",
    "classes":    cfg.get("names", []),
    "epochs_ran": len(df),
    "best_epoch": best_epoch,
    "best_map50": round(best_map50, 6),
    "data_yaml":  DATA_YAML,
}
summary_path = f"{HOME}/summary_yolo26.json"
with open(summary_path, "w") as f:
    json.dump(summary, f, indent=2)

print(f"\nTai ve may:")
files.download(summary_path)
files.download(f"{HOME}/runs/detect/drowsy_yolo26/weights/best.pt")
print("Xong! Luu vao: outputs/training_yolo/yolo26/")
