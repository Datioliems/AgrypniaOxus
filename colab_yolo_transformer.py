# %% [markdown]
# # YOLO + Transformer Experiments — Drowsy Driver (Colab)
# **Truoc khi chay:** `Runtime` → `Change runtime type` → **T4 GPU**
#
# ## 4 experiments trong file nay:
# | ID | Model | Ky thuat Transformer |
# |----|-------|---------------------|
# | A  | YOLO11s | C2PSA — Parallel Spatial Attention (built-in) |
# | B  | RT-DETR-l | Full Transformer Decoder (khong can NMS) |
# | C  | YOLO-World | CLIP Text Encoder (zero-shot + fine-tune) |
# | D  | YOLOv8s + SE | Squeeze-and-Excitation Channel Attention (tuy chinh) |
#
# **Cach chay:** Cell 1→5 bat buoc, sau do chay rieng tung experiment (A/B/C/D)

# %% Cell 1 — GPU + Cai thu vien
get_ipython().system('nvidia-smi')
get_ipython().run_line_magic('pip', 'install -q "ultralytics>=8.4.0" roboflow supervision')
get_ipython().system('yolo settings sync=False')
import ultralytics
ultralytics.checks()

# %% Cell 2 — Setup
import os
HOME = os.getcwd()
print("HOME:", HOME)

# %% Cell 3 — Download dataset (dung chung cho tat ca experiments)
from roboflow import Roboflow
from pathlib import Path
import yaml

rf      = Roboflow(api_key="qI3lEKlNpIZpNENdk3MH")
project = rf.workspace("nguyen-tuan-dat").project("datio_drowsines")
dataset = project.version(1).download("yolov8")

loc       = Path(dataset.location)
yaml_hits = list(loc.rglob("data.yaml"))
if not yaml_hits:
    raise FileNotFoundError(f"Khong tim thay data.yaml trong {loc}")
DATA_YAML = str(yaml_hits[0])

with open(DATA_YAML) as f:
    cfg = yaml.safe_load(f)

NUM_CLASSES = cfg.get("nc", 2)
CLASS_NAMES = cfg.get("names", [])

print(f"data.yaml  : {DATA_YAML}")
print(f"Classes    : {CLASS_NAMES}")
print(f"nc         : {NUM_CLASSES}")

base = Path(DATA_YAML).parent
for split in ["train", "valid", "val", "test"]:
    p = base / split / "images"
    if p.exists():
        n = len(list(p.glob("*.jpg")) + list(p.glob("*.png")))
        print(f"  {split}: {n} images")

# %% Cell 4 — Cau hinh chung
# Chinh o day truoc khi chay experiment
EPOCHS  = 30   # so epoch cho moi experiment (giam xuong 15 neu it thoi gian)
IMGSZ   = 640  # kich thuoc anh
BATCH   = -1   # -1 = tu dong theo VRAM
PATIENCE = 15  # early stopping

RESULTS = {}   # bien global luu ket qua tat ca experiments

print(f"Epochs  : {EPOCHS}")
print(f"imgsz   : {IMGSZ}")
print(f"batch   : {BATCH} (auto)")
print(f"patience: {PATIENCE}")
print()
print("San sang! Chay tung experiment o duoi theo thu tu.")

# %% [markdown]
# ---
# ## Experiment A — YOLO11s + C2PSA Attention
#
# YOLO11 them block **C2PSA (Cross Stage Partial + Parallel Spatial Attention)**
# vao backbone. So voi YOLOv8s:
# - Params it hon: 9.4M vs 11.2M (nhe hon 16%)
# - mAP50 cao hon: ~47% vs ~44.9% (tren COCO)
# - Toc do tuong duong
#
# **Day la upgrade de nhat** — chi doi ten model, khong can thay doi gi khac.

# %% Cell A1 — Train YOLO11s
from ultralytics import YOLO
print("=" * 55)
print("  [A] YOLO11s + C2PSA Spatial Attention")
print("=" * 55)

model_a = YOLO("yolo11s.pt")

results_a = model_a.train(
    data     = DATA_YAML,
    epochs   = EPOCHS,
    imgsz    = IMGSZ,
    batch    = BATCH,
    patience = PATIENCE,
    plots    = True,
    cos_lr   = True,
    name     = "exp_A_yolo11s",
    project  = f"{HOME}/runs/experiments",
    verbose  = True,
)

map50_a = float(results_a.results_dict.get("metrics/mAP50(B)", 0))
RESULTS["A_yolo11s"] = {
    "model":     "yolo11s",
    "attention": "C2PSA (Cross Stage Partial + Parallel Spatial Attention)",
    "params_M":  9.4,
    "best_map50": map50_a,
    "weights":   f"{HOME}/runs/experiments/exp_A_yolo11s/weights/best.pt",
}
print(f"\n[A] Done! Best mAP@50: {map50_a*100:.2f}%")

# %% Cell A2 — Xem ket qua A
from IPython.display import Image as IPyImage, display
import os

d = f"{HOME}/runs/experiments/exp_A_yolo11s"
for f in ["results.png", "confusion_matrix.png", "F1_curve.png"]:
    fp = f"{d}/{f}"
    if os.path.exists(fp):
        display(IPyImage(filename=fp, width=700))

# %% [markdown]
# ---
# ## Experiment B — RT-DETR-l (Real-Time Detection Transformer)
#
# RT-DETR thay toan bo **decoder** cua YOLO bang Transformer.
# - **Khong can NMS** — transformer tự chon box tot nhat qua cross-attention
# - Global attention nhin toan bo anh → tot hon trong dieu kien anh sang xau
# - Kem hon trong truong hop object nho (do attention expensive)
#
# ```
# Encoder (CNN features)
#     → Transformer Decoder (cross-attention giua query va features)
#         → Box predictions truc tiep (khong qua anchor)
# ```

# %% Cell B1 — Train RT-DETR-l
from ultralytics import RTDETR
print("=" * 55)
print("  [B] RT-DETR-l — Full Transformer Decoder")
print("=" * 55)

model_b = RTDETR("rtdetr-l.pt")

# RT-DETR yeu cau batch >= 4 va khong ho tro batch=-1
batch_b = 8   # T4 co 15GB, batch=8 an toan
results_b = model_b.train(
    data     = DATA_YAML,
    epochs   = EPOCHS,
    imgsz    = IMGSZ,
    batch    = batch_b,
    patience = PATIENCE,
    plots    = True,
    name     = "exp_B_rtdetr",
    project  = f"{HOME}/runs/experiments",
    verbose  = True,
)

map50_b = float(results_b.results_dict.get("metrics/mAP50(B)", 0))
RESULTS["B_rtdetr"] = {
    "model":     "rtdetr-l",
    "attention": "Multi-Head Self-Attention + Cross-Attention Decoder (no NMS)",
    "params_M":  32.0,
    "best_map50": map50_b,
    "weights":   f"{HOME}/runs/experiments/exp_B_rtdetr/weights/best.pt",
}
print(f"\n[B] Done! Best mAP@50: {map50_b*100:.2f}%")

# %% Cell B2 — Xem ket qua B
from IPython.display import Image as IPyImage, display
import os

d = f"{HOME}/runs/experiments/exp_B_rtdetr"
for f in ["results.png", "confusion_matrix.png"]:
    fp = f"{d}/{f}"
    if os.path.exists(fp):
        display(IPyImage(filename=fp, width=700))

# %% [markdown]
# ---
# ## Experiment C — YOLO-World (Open Vocabulary + CLIP)
#
# YOLO-World ket hop YOLO backbone voi **CLIP text encoder**:
# - Co the detect bang mo ta van ban, khong can train lai
# - Zero-shot: thu thach nhung cho thay kha nang tong quat hoa
# - Fine-tune: accuracy tiep tuc tang khi train them tren dataset chuyen biet
#
# ```
# Text prompt ("drowsy face", "closed eyes")
#         ↓ CLIP Text Encoder
# Text features → Cross-Attention voi image features
#         ↓
# Detection output
# ```

# %% Cell C1 — YOLO-World Zero-Shot (khong train)
from ultralytics import YOLOWorld
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path

print("=" * 55)
print("  [C.1] YOLO-World Zero-Shot (CLIP)")
print("=" * 55)

model_c_zs = YOLOWorld("yolov8s-world.pt")

# Mo ta theo ngu nghia drowsy detection (khong can class label)
TEXT_CLASSES = [
    "drowsy face",
    "alert face",
    "closed eyes",
    "open eyes",
    "yawning mouth",
]
model_c_zs.set_classes(TEXT_CLASSES)
print("Text classes:", TEXT_CLASSES)

# Predict tren 8 anh mau
base     = Path(DATA_YAML).parent
test_dir = next(
    (base / s / "images" for s in ["test", "valid"] if (base / s / "images").exists()),
    None
)

if test_dir:
    test_imgs = list(test_dir.glob("*.jpg"))[:8] + list(test_dir.glob("*.png"))[:8]
    test_imgs = test_imgs[:8]
    results_zs = model_c_zs.predict(
        [str(p) for p in test_imgs], conf=0.15, verbose=False
    )
    total_det = sum(len(r.boxes) for r in results_zs if r.boxes is not None)
    print(f"\nZero-shot: {total_det} detections tren {len(test_imgs)} anh")
    print("(Khong co mAP — zero-shot khong train, chi de xem kha nang phat hien)")

    # Hien thi 4 anh dau
    fig, axes = plt.subplots(1, min(4, len(results_zs)), figsize=(16, 4))
    if len(results_zs) == 1:
        axes = [axes]
    for ax, r in zip(axes, results_zs[:4]):
        ax.imshow(r.plot()[:, :, ::-1])
        ax.axis("off")
    plt.suptitle("YOLO-World Zero-Shot Predictions", fontsize=12)
    plt.tight_layout()
    plt.show()

# %% Cell C2 — YOLO-World Fine-tune
from ultralytics import YOLOWorld
print("=" * 55)
print("  [C.2] YOLO-World Fine-tune tren dataset")
print("=" * 55)

model_c_ft = YOLOWorld("yolov8s-worldv2.pt")
results_c = model_c_ft.train(
    data     = DATA_YAML,
    epochs   = min(EPOCHS, 20),   # World hoi tu nhanh
    imgsz    = IMGSZ,
    batch    = BATCH,
    patience = 10,
    plots    = True,
    name     = "exp_C_world_ft",
    project  = f"{HOME}/runs/experiments",
    verbose  = True,
)

map50_c = float(results_c.results_dict.get("metrics/mAP50(B)", 0))
RESULTS["C_world"] = {
    "model":      "yolov8s-world",
    "attention":  "CLIP Text Encoder + Cross-Attention (zero-shot capable)",
    "params_M":   16.0,
    "best_map50": map50_c,
    "weights":    f"{HOME}/runs/experiments/exp_C_world_ft/weights/best.pt",
}
print(f"\n[C] Done! Best mAP@50 (fine-tuned): {map50_c*100:.2f}%")

# %% Cell C3 — Xem ket qua C
from IPython.display import Image as IPyImage, display
import os

d = f"{HOME}/runs/experiments/exp_C_world_ft"
for f in ["results.png", "confusion_matrix.png"]:
    fp = f"{d}/{f}"
    if os.path.exists(fp):
        display(IPyImage(filename=fp, width=700))

# %% [markdown]
# ---
# ## Experiment D — YOLOv8s + SE Channel Attention (tuy chinh)
#
# **Squeeze-and-Excitation (SE)** them vao neck cua YOLOv8:
# ```
# C2f block → AdaptiveAvgPool (squeeze) → FC → Sigmoid (excitation) → scale channels
# ```
# Model tu hoc "channel nao quan trong hon" cho bai toan nay.
#
# **Vi tri them SE:** Head layers 2, 5, 8 (C2f trong neck P5, P4, P3)
# ```
# Backbone (giu nguyen)
#   SPPF → Upsample → [C2f → SE] ← P5 features
#                  → [C2f → SE] ← P4 features
#         Conv → [C2f → SE] ← P3 features
#             → Detect
# ```

# %% Cell D1 — Dinh nghia SEBlock + C2fSE
import torch
import torch.nn as nn
from ultralytics.nn.modules.block import C2f
import ultralytics.nn.tasks as _tasks

class SEBlock(nn.Module):
    """Squeeze-and-Excitation Channel Attention.
    Hoc trong so quan trong cua tung channel feature map.
    """
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        mid = max(channels // reduction, 4)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc   = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.SiLU(),                            # SiLU = Swish, phong cach YOLO
            nn.Linear(mid, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        w = self.pool(x).view(b, c)       # squeeze: (B, C)
        w = self.fc(w).view(b, c, 1, 1)   # excite : (B, C, 1, 1)
        return x * w.expand_as(x)         # scale channels

class C2fSE(C2f):
    """C2f voi SE Channel Attention — drop-in replacement cho C2f.
    Them SE block sau C2f forward, khong thay doi kinh truoc.
    """
    def __init__(self, c1: int, c2: int, n: int = 1,
                 shortcut: bool = False, g: int = 1, e: float = 0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.se = SEBlock(c2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.se(super().forward(x))

# Dang ky vao Ultralytics de YAML parser nhan ra "C2fSE"
_tasks.C2fSE = C2fSE
import ultralytics.nn.modules as _ulm
_ulm.C2fSE = C2fSE

# Kiem tra so params them vao
se_256  = SEBlock(256)
se_512  = SEBlock(512)
params_256 = sum(p.numel() for p in se_256.parameters())
params_512 = sum(p.numel() for p in se_512.parameters())
print(f"SEBlock(256) params: {params_256:,}")
print(f"SEBlock(512) params: {params_512:,}")
print(f"Tong params tang them (3 SE blocks): ~{params_256 + 2*params_512:,}")
print("C2fSE registered OK")

# %% Cell D2 — Tao YAML voi C2fSE trong neck
import yaml
import copy
from pathlib import Path
import ultralytics

# Doc yolov8.yaml goc tu package
pkg_dir  = Path(ultralytics.__file__).parent
yaml_src = pkg_dir / "cfg" / "models" / "v8" / "yolov8.yaml"
with open(yaml_src) as f:
    base_cfg = yaml.safe_load(f)

# Tao ban sao, them C2fSE vao vi tri [2], [5], [8] trong head
se_cfg = copy.deepcopy(base_cfg)
se_cfg["nc"] = NUM_CLASSES

# Head index 2, 5, 8 la C2f trong neck (P5, P4, P3)
neck_c2f_indices = [2, 5, 8]
print("Thay the C2f → C2fSE trong neck:")
for idx in neck_c2f_indices:
    layer = se_cfg["head"][idx]
    assert layer[2] == "C2f", f"head[{idx}] la {layer[2]}, khong phai C2f!"
    se_cfg["head"][idx][2] = "C2fSE"
    print(f"  head[{idx}]: C2f → C2fSE  channels={layer[3]}")

# Ghi YAML tuy chinh
se_yaml_path = f"{HOME}/yolov8s_se.yaml"
with open(se_yaml_path, "w") as f:
    yaml.dump(se_cfg, f, default_flow_style=False, allow_unicode=True)

print(f"\nCustom YAML: {se_yaml_path}")
print("Head layers sau khi sua:")
for i, layer in enumerate(se_cfg["head"]):
    mark = " <-- SE" if i in neck_c2f_indices else ""
    print(f"  [{i}] {layer[2]:10s} {str(layer[3])}{mark}")

# %% Cell D3 — Train YOLOv8s + SE
from ultralytics import YOLO
print("=" * 55)
print("  [D] YOLOv8s + SE Channel Attention")
print("=" * 55)

# Load tu pretrained yolov8s.pt (transfer learning)
# strict=False vi C2fSE co them SEBlock khong co trong yolov8s.pt
model_d = YOLO(se_yaml_path).load("yolov8s.pt")

print(f"\nModel loaded: yolov8s + SE neck")
total_p = sum(p.numel() for p in model_d.model.parameters())
print(f"Total params: {total_p/1e6:.2f}M")

results_d = model_d.train(
    data     = DATA_YAML,
    epochs   = EPOCHS,
    imgsz    = IMGSZ,
    batch    = BATCH,
    patience = PATIENCE,
    plots    = True,
    cos_lr   = True,
    name     = "exp_D_se_yolo8s",
    project  = f"{HOME}/runs/experiments",
    verbose  = True,
)

map50_d = float(results_d.results_dict.get("metrics/mAP50(B)", 0))
RESULTS["D_se_yolo8s"] = {
    "model":      "yolov8s + SE",
    "attention":  "SE Channel Attention (neck C2f → C2fSE, reduction=16)",
    "params_M":   round(total_p / 1e6, 1),
    "best_map50": map50_d,
    "weights":    f"{HOME}/runs/experiments/exp_D_se_yolo8s/weights/best.pt",
}
print(f"\n[D] Done! Best mAP@50: {map50_d*100:.2f}%")

# %% Cell D4 — Xem ket qua D
from IPython.display import Image as IPyImage, display
import os

d = f"{HOME}/runs/experiments/exp_D_se_yolo8s"
for f in ["results.png", "confusion_matrix.png"]:
    fp = f"{d}/{f}"
    if os.path.exists(fp):
        display(IPyImage(filename=fp, width=700))

# %% [markdown]
# ---
# ## Ket qua tong hop — So sanh tat ca experiments

# %% Cell E1 — Bang so sanh
print("=" * 72)
print("  BANG SO SANH EXPERIMENTS — YOLO + Transformer Attention")
print("=" * 72)
print(f"  {'ID':<3} {'Model':<18} {'Attention':<38} {'mAP50':>7}")
print("-" * 72)

if not RESULTS:
    print("  Chua chay experiment nao! Chay cac cell A/B/C/D o tren truoc.")
else:
    sorted_r = sorted(RESULTS.items(), key=lambda x: x[1]["best_map50"], reverse=True)
    for rank, (key, r) in enumerate(sorted_r):
        tag  = " <-- BEST" if rank == 0 else ""
        attn = r["attention"][:36]
        print(f"  {key.split('_')[0]:<3} {r['model']:<18} {attn:<38} {r['best_map50']*100:6.2f}%{tag}")

print("=" * 72)

if RESULTS:
    best_key = max(RESULTS, key=lambda k: RESULTS[k]["best_map50"])
    best_r   = RESULTS[best_key]
    print(f"\nBest model : {best_r['model']}")
    print(f"mAP@50     : {best_r['best_map50']*100:.2f}%")
    print(f"Attention  : {best_r['attention']}")
    print(f"Weights    : {best_r['weights']}")

# %% Cell E2 — Training curves tat ca experiments
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

n = len(RESULTS)
if n == 0:
    print("Khong co ket qua de ve")
else:
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), squeeze=False)
    axes = axes[0]

    for ax, (key, r) in zip(axes, RESULTS.items()):
        exp_name = key.split("_")[0]   # A, B, C, D
        run_dir  = Path(r["weights"]).parent.parent
        csv_file = run_dir / "results.csv"

        if csv_file.exists():
            df = pd.read_csv(csv_file)
            df.columns = [c.strip() for c in df.columns]
            map50_col = [c for c in df.columns
                         if "map50" in c.lower() and "95" not in c.lower()]
            if map50_col:
                ax.plot(df[map50_col[0]], color="steelblue", lw=2)
                ax.axhline(r["best_map50"], color="red", ls="--", lw=1,
                           label=f"Best {r['best_map50']*100:.1f}%")
            ax.set_title(f"[{exp_name}] {r['model']}\n{r['best_map50']*100:.1f}% mAP50",
                         fontsize=9)
            ax.set_xlabel("Epoch", fontsize=8)
            ax.set_ylabel("mAP50", fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=7)
        else:
            ax.text(0.5, 0.5, "No data\n(not run?)", ha="center", va="center",
                    transform=ax.transAxes, fontsize=10, color="gray")
            ax.set_title(f"[{exp_name}] {r['model']}", fontsize=9)

    plt.suptitle("mAP50 Training Curves — All Transformer Experiments", fontsize=12)
    plt.tight_layout()
    plt.show()

# %% Cell E3 — Luu report JSON + tai ve may
import json
from google.colab import files
from pathlib import Path

report = {
    "title":       "YOLO + Transformer Experiments — Drowsy Driver",
    "dataset":     "datio_drowsines_v1",
    "experiments": RESULTS,
    "best_model":  max(RESULTS, key=lambda k: RESULTS[k]["best_map50"]) if RESULTS else None,
}

report_path = f"{HOME}/transformer_experiments_report.json"
with open(report_path, "w") as f:
    json.dump(report, f, indent=2, ensure_ascii=False)

print(f"Report saved: {report_path}")
files.download(report_path)

# Tai best model weights
if RESULTS:
    best_key     = max(RESULTS, key=lambda k: RESULTS[k]["best_map50"])
    best_weights = RESULTS[best_key]["weights"]
    if Path(best_weights).exists():
        print(f"Tai weights tot nhat: {RESULTS[best_key]['model']}")
        files.download(best_weights)
    else:
        print(f"Weights chua ton tai: {best_weights}")
        print("  → Chay Cell tuong ung truoc (A1/B1/C2/D3)")
