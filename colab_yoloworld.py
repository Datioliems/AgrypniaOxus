# %% [markdown]
# # YOLO-World — Zero-Shot Drowsy Detection
# **Phuong phap 4: YOLO + CLIP Text Encoder**
#
# YOLO-World ket hop:
#   - YOLO backbone (trich xuat visual features)
#   - CLIP text encoder (hieu ngu nghia tieng Anh)
#   - Text-Image matching: tim vung anh "khop" voi text prompt
#
# Uu diem:
#   - Zero-shot: khong can train, chi can mo ta bang tieng Anh
#   - Co the detect nhieu class moi ma khong can label lai
#   - Fine-tune voi it du lieu -> accuracy tot hon
#
# Nhuoc diem:
#   - Accuracy thap hon model train chuyen biet
#   - Can prompt engineering de dat ket qua tot
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

# %% Cell 4 — Download dataset (lay test images de demo)
import yaml
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

ORIGINAL_CLASSES = cfg.get("names", [])
test_dir = Path(dataset.location) / "test" / "images"
test_imgs = list(test_dir.glob("*.jpg")) + list(test_dir.glob("*.png"))
print(f"Original classes : {ORIGINAL_CLASSES}")
print(f"Test images      : {len(test_imgs)}")

# %% Cell 5 — YOLO-World Zero-Shot (khong can train)
from ultralytics import YOLOWorld

# YOLO-World models:
#   yolov8s-worldv2.pt : nhanh nhat, 26M params
#   yolov8m-worldv2.pt : can bang
#   yolov8l-worldv2.pt : chinh xac nhat
model = YOLOWorld("yolov8s-worldv2.pt")
print("YOLO-World loaded (zero-shot, khong can train)")

# %% Cell 6 — Thu nghiem nhieu bo TEXT PROMPT khac nhau
# Prompt tot = mo ta chinh xac dac diem hinh anh
# Vi du "drowsy eye" tot hon "drowsy" vì specific hon

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import cv2, numpy as np
from pathlib import Path

PROMPT_SETS = {
    "original_11":    ORIGINAL_CLASSES,                                         # 11 classes goc
    "simplified_4":   ["drowsy eye", "attentive eye", "yawn", "asleep"],        # Gom nhom lai
    "descriptive":    ["closed eye drowsy", "open eye alert", "yawning mouth"], # Mo ta chi tiet
    "binary":         ["drowsy driver", "alert driver"],                         # 2 class don gian
}

# Chay tren 4 anh test
sample_imgs = test_imgs[:4]
fig, axes = plt.subplots(len(PROMPT_SETS), len(sample_imgs),
                          figsize=(4*len(sample_imgs), 4*len(PROMPT_SETS)))

detection_counts = {}
for row, (name, classes) in enumerate(PROMPT_SETS.items()):
    model.set_classes(classes)
    count = 0
    for col, img_path in enumerate(sample_imgs):
        results = model.predict(str(img_path), conf=0.25, verbose=False)
        plotted = results[0].plot()
        axes[row][col].imshow(cv2.cvtColor(plotted, cv2.COLOR_BGR2RGB))
        axes[row][col].axis("off")
        count += len(results[0].boxes)
        if col == 0:
            axes[row][col].set_ylabel(f"{name}\n{len(classes)} classes",
                                       fontsize=8, rotation=0, labelpad=60, va="center")
    detection_counts[name] = count

plt.suptitle("YOLO-World: So sanh cac TEXT PROMPT (conf=0.25)", fontsize=12)
plt.tight_layout()
plt.savefig(f"{HOME}/yoloworld_prompt_compare.png", dpi=100, bbox_inches="tight")
plt.show()

print("\nTong so detection tren 4 anh:")
for name, count in detection_counts.items():
    print(f"  {name:<20}: {count} detections")

# %% Cell 7 — Danh gia dinh luong tren tap test
# Dung 2 bo prompt: original vs simplified
# So sanh precision va recall

from ultralytics import YOLOWorld
import json

EVAL_PROMPTS = {
    "original_11": ORIGINAL_CLASSES,
    "simplified_4": ["drowsy eye", "attentive eye", "yawn", "asleep"],
}

results_summary = {}
for prompt_name, classes in EVAL_PROMPTS.items():
    model_eval = YOLOWorld("yolov8s-worldv2.pt")
    model_eval.set_classes(classes)

    # Phai cap nhat data.yaml cho classes moi
    # NOTE: eval chinh xac hon khi classes khop voi ground truth
    # Dung data.yaml goc (11 classes) cho "original_11"
    if prompt_name == "original_11":
        val_results = model_eval.val(data=DATA_YAML, verbose=False)
    else:
        # Tao data.yaml tam thoi voi classes moi
        new_cfg = dict(cfg)
        new_cfg["names"] = classes
        new_cfg["nc"]    = len(classes)
        tmp_yaml = f"{HOME}/tmp_{prompt_name}.yaml"
        with open(tmp_yaml, "w") as f:
            yaml.dump(new_cfg, f)
        # NOTE: Metrics se thap vi class names khong khop ground truth labels
        print(f"  [{prompt_name}] NOTE: Metrics khong hoan toan chinh xac (class mismatch)")
        continue

    map50 = float(val_results.box.map50)
    results_summary[prompt_name] = {"map50": map50, "classes": classes}
    print(f"  {prompt_name}: mAP50 = {map50*100:.2f}%")

# %% Cell 8 — Fine-tune YOLO-World voi dataset goc
# Fine-tune: adapt text embeddings cho domain drowsiness
# Nhanh hon nhieu so voi train tu dau (chi ~10 epochs)
# LR rat nho de khong lam hong zero-shot ability

from ultralytics import YOLOWorld
import os

print("Fine-tuning YOLO-World voi dataset Datio_drowsines (11 classes)...")
print("  Nay se adapt CLIP embeddings cho drowsy domain")
print()

model_ft = YOLOWorld("yolov8s-worldv2.pt")

results_ft = model_ft.train(
    data=DATA_YAML,
    epochs=10,              # It epochs vì da pretrained
    imgsz=640,
    batch=16,
    patience=5,
    name="drowsy_yoloworld_ft",
    project=f"{HOME}/runs/detect",
    lr0=1e-4,               # LR rat nho khi fine-tune zero-shot model
    lrf=0.01,
    warmup_epochs=1,
    freeze=10,              # Dong bang 10 layers dau (giu pretrained features)
    plots=True,
    degrees=5.0,
    fliplr=0.5,
)

# %% Cell 9 — Validate fine-tuned model
best_ft = f"{HOME}/runs/detect/drowsy_yoloworld_ft/weights/best.pt"
if os.path.exists(best_ft):
    model_val_ft = YOLOWorld(best_ft)
    model_val_ft.set_classes(ORIGINAL_CLASSES)
    val_results_ft = model_val_ft.val(data=DATA_YAML, verbose=True)

    # Doc ket qua
    df_ft = __import__("pandas").read_csv(f"{HOME}/runs/detect/drowsy_yoloworld_ft/results.csv")
    df_ft.columns = [c.strip() for c in df_ft.columns]
    map50_col = [c for c in df_ft.columns if "map50" in c.lower() and "95" not in c.lower()][0]
    best_map50_ft = float(df_ft[map50_col].max())

    print("=" * 55)
    print("  YOLO-World: Zero-shot vs Fine-tuned")
    print("=" * 55)
    if "original_11" in results_summary:
        print(f"  Zero-shot   : {results_summary['original_11']['map50']*100:.2f}%")
    print(f"  Fine-tuned  : {best_map50_ft*100:.2f}%")

    # So sanh voi YOLO26
    yolo26_json = f"{HOME}/summary_yolo26.json"
    if os.path.exists(yolo26_json):
        with open(yolo26_json) as f:
            r26 = json.load(f)
        print(f"  YOLO26m     : {r26['best_map50']*100:.2f}%")

# %% Cell 10 — Demo visual: Zero-shot vs Fine-tuned
import cv2, matplotlib.pyplot as plt

sample_img = str(test_imgs[0]) if test_imgs else None
if sample_img:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Zero-shot
    m_zs = YOLOWorld("yolov8s-worldv2.pt")
    m_zs.set_classes(["drowsy eye", "attentive eye", "yawn", "asleep"])
    r_zs = m_zs.predict(sample_img, conf=0.25, verbose=False)
    axes[0].imshow(cv2.cvtColor(r_zs[0].plot(), cv2.COLOR_BGR2RGB))
    axes[0].set_title(f"Zero-shot\n['drowsy eye', 'attentive eye', ...]", fontsize=9)
    axes[0].axis("off")

    # Fine-tuned
    if os.path.exists(best_ft):
        m_ft = YOLOWorld(best_ft)
        m_ft.set_classes(ORIGINAL_CLASSES)
        r_ft = m_ft.predict(sample_img, conf=0.3, verbose=False)
        axes[1].imshow(cv2.cvtColor(r_ft[0].plot(), cv2.COLOR_BGR2RGB))
        axes[1].set_title(f"Fine-tuned ({len(ORIGINAL_CLASSES)} classes)", fontsize=9)
        axes[1].axis("off")

    plt.suptitle("YOLO-World: Zero-shot vs Fine-tuned Comparison", fontsize=12)
    plt.tight_layout()
    plt.show()

# %% Cell 11 — Download
from google.colab import files
import json

summary = {
    "model":      "yolov8s-worldv2",
    "device":     "colab_t4",
    "zero_shot":  True,
    "finetuned":  os.path.exists(best_ft),
    "classes":    ORIGINAL_CLASSES,
    "note":       "Zero-shot via CLIP text prompts, no label needed",
}
with open(f"{HOME}/summary_yoloworld.json", "w") as f:
    json.dump(summary, f, indent=2)

files.download(f"{HOME}/summary_yoloworld.json")
files.download(f"{HOME}/yoloworld_prompt_compare.png")
if os.path.exists(best_ft):
    files.download(best_ft)
print("Da download xong!")
