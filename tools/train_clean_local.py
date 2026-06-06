# %% [markdown]
# # Train YOLO — Clean Dataset + Experiment Tracking
#
# Workflow:
#   1. Chinh tham so trong  configs/yolo_hparams.yaml
#   2. Chay Cell 1 → Cell 6  (chi can chay Cell 4A/4B/4C 1 lan dau)
#   3. Xem ket qua Cell 7 → so sanh Cell 8
#
# Moi lan train → tu dong luu vao outputs/experiments/run_XXX/

# %% Cell 1 — Load config + setup paths
import os, sys, json, shutil, datetime
from pathlib import Path

import yaml as _yaml

PROJECT_ROOT  = Path(__file__).resolve().parent.parent
CONFIG_FILE   = PROJECT_ROOT / "configs" / "yolo_hparams.yaml"
DATA_DIR      = PROJECT_ROOT / "roboflow_data" / "clean_merged"
DS1_DIR       = PROJECT_ROOT / "roboflow_data" / "ds_augmented"
DS2_DIR       = PROJECT_ROOT / "roboflow_data" / "ds_driveryawn"
EXPERIMENTS   = PROJECT_ROOT / "outputs" / "experiments"
LOG_FILE      = EXPERIMENTS / "experiments_log.json"

EXPERIMENTS.mkdir(parents=True, exist_ok=True)

# Doc config
with open(CONFIG_FILE, encoding="utf-8") as f:
    CFG = _yaml.safe_load(f)

print("=" * 55)
print("CONFIG (tu configs/yolo_hparams.yaml):")
print("=" * 55)
for k, v in CFG.items():
    print(f"  {k:<18} = {v}")
print("=" * 55)

# %% Cell 2 — Kiem tra GPU → tu dong set batch
try:
    import torch
    if torch.cuda.is_available():
        gpu  = torch.cuda.get_device_properties(0)
        vram = gpu.total_memory / 1024**3
        print(f"GPU : {gpu.name}  |  VRAM: {vram:.1f} GB")
        DEVICE = 0
        HALF   = True
        # Tu dong tinh batch neu CFG batch = -1
        if CFG.get("batch", -1) == -1:
            BATCH = 16 if vram >= 6 else (12 if vram >= 4 else 8)
            print(f"Auto batch = {BATCH} (VRAM {vram:.1f}GB)")
        else:
            BATCH = CFG["batch"]
    else:
        print("Khong co GPU CUDA — dung CPU")
        DEVICE, HALF, BATCH = "cpu", False, 4
except ImportError:
    print("PyTorch chua cai!"); sys.exit(1)

# %% Cell 3 — Cai thu vien (chi can chay 1 lan)
import subprocess
for pkg in ["ultralytics", "roboflow"]:
    try:
        __import__(pkg.replace("-", "_"))
    except ImportError:
        print(f"Cai {pkg}...")
        subprocess.run([sys.executable, "-m", "pip", "install", pkg, "-q"], check=True)
import ultralytics
print(f"ultralytics {ultralytics.__version__} san sang")

# %% Cell 4A — Download Dataset 1: Augmented Startups  [chi can chay 1 lan]
# ~2,000 anh | classes: awake / drowsy
YAML1 = DS1_DIR / "data.yaml"

if YAML1.exists():
    print("DS1 da co:", DS1_DIR)
else:
    print("Downloading Dataset 1 (Augmented Startups)...")
    from roboflow import Roboflow
    rf = Roboflow(api_key="qI3lEKlNpIZpNENdk3MH")
    for ver in [1, 2, 3]:
        try:
            ds = rf.workspace("augmented-startups") \
                    .project("drowsiness-detection-cntmz") \
                    .version(ver).download("yolov8", location=str(DS1_DIR))
            print(f"DS1 downloaded (v{ver}):", DS1_DIR)
            break
        except Exception as e:
            print(f"  v{ver}: {e}")

# %% Cell 4B — Download Dataset 2: driver-no-yawn  [chi can chay 1 lan]
# ~2,900 anh | classes: not_drowsy / drowsy
YAML2 = DS2_DIR / "data.yaml"

if YAML2.exists():
    print("DS2 da co:", DS2_DIR)
else:
    print("Downloading Dataset 2 (driver-no-yawn)...")
    from roboflow import Roboflow
    rf = Roboflow(api_key="qI3lEKlNpIZpNENdk3MH")
    for ver in [3, 2, 1]:
        try:
            ds = rf.workspace("driver-no-yawn") \
                    .project("driver-drowsiness1") \
                    .version(ver).download("yolov8", location=str(DS2_DIR))
            print(f"DS2 downloaded (v{ver}):", DS2_DIR)
            break
        except Exception as e:
            print(f"  v{ver}: {e}")

# %% Cell 4C — Merge datasets → awake=0, drowsy=1  [chi can chay 1 lan]
MERGED_YAML = DATA_DIR / "data.yaml"

if MERGED_YAML.exists():
    with open(MERGED_YAML, encoding="utf-8") as f:
        mc = _yaml.safe_load(f)
    counts = {sp: len(list((DATA_DIR/sp/"images").glob("*.*")))
              for sp in ["train","valid","test"]
              if (DATA_DIR/sp/"images").exists()}
    print(f"Merged dataset da co: {sum(counts.values())} anh")
    print(f"  split: {counts}")
    print(f"  classes: {mc.get('names')}")
else:
    ALIAS = {
        "awake": 0, "Awake": 0,
        "not_drowsy": 0, "Not_drowsy": 0, "not drowsy": 0,
        "drowsy": 1, "Drowsy": 1,
    }

    def _read_names(p):
        with open(p, encoding="utf-8") as f:
            return _yaml.safe_load(f).get("names", [])

    def _remap(names):
        r = {}
        for i, n in enumerate(names):
            nid = ALIAS.get(n)
            if nid is not None:
                r[i] = nid
            else:
                print(f"  [WARN] '{n}' khong nhan dang → bo qua")
        return r

    def _copy_split(src, remap, split, prefix):
        ii = Path(src)/split/"images"
        ll = Path(src)/split/"labels"
        oi = DATA_DIR/split/"images"; oi.mkdir(parents=True, exist_ok=True)
        ol = DATA_DIR/split/"labels"; ol.mkdir(parents=True, exist_ok=True)
        if not ii.exists(): return 0
        n = 0
        for img in list(ii.glob("*.jpg")) + list(ii.glob("*.png")):
            nm = f"{prefix}_{img.name}"
            shutil.copy2(img, oi/nm)
            lbl = ll/(img.stem+".txt")
            lines = []
            if lbl.exists():
                for ln in lbl.read_text(encoding="utf-8").strip().splitlines():
                    p = ln.split()
                    if p and int(p[0]) in remap:
                        lines.append(str(remap[int(p[0])]) + " " + " ".join(p[1:]))
            (ol/(Path(nm).stem+".txt")).write_text("\n".join(lines), encoding="utf-8")
            n += 1
        return n

    n1 = _read_names(YAML1); n2 = _read_names(YAML2)
    r1 = _remap(n1);         r2 = _remap(n2)
    print(f"DS1: {n1} → remap {r1}")
    print(f"DS2: {n2} → remap {r2}")

    totals = {}
    for sp in ["train", "valid", "test"]:
        a = _copy_split(DS1_DIR, r1, sp, "aug")
        b = _copy_split(DS2_DIR, r2, sp, "drv")
        totals[sp] = a + b
        print(f"  {sp}: {a}+{b}={a+b}")

    _yaml.dump({"path": str(DATA_DIR).replace("\\","/"),
                "train": "train/images", "val": "valid/images",
                "test": "test/images", "nc": 2,
                "names": ["awake","drowsy"]},
               open(MERGED_YAML,"w",encoding="utf-8"),
               default_flow_style=False)
    print(f"\nMerge xong: {sum(totals.values())} anh | ['awake','drowsy']")

# %% Cell 5 — Xem thong tin dataset
with open(MERGED_YAML, encoding="utf-8") as f:
    ds_info = _yaml.safe_load(f)

print("=" * 50)
print(f"Classes ({ds_info['nc']}): {ds_info['names']}")
for sp in ["train","valid","test"]:
    d = DATA_DIR/sp/"images"
    if d.exists():
        n = len(list(d.glob("*.jpg"))+list(d.glob("*.png")))
        print(f"  {sp:6s}: {n} images")
print("=" * 50)

# %% Cell 6 — TRAIN  (dung tham so tu CFG)
from ultralytics import YOLO

MODEL = CFG["model"]
model = YOLO(f"{MODEL}.pt")

# Tao run folder: run_001, run_002, ...
existing = sorted(EXPERIMENTS.glob("run_*"))
run_id   = f"run_{len(existing)+1:03d}"
run_dir  = EXPERIMENTS / run_id
run_dir.mkdir(parents=True, exist_ok=True)

# Luu config da dung vao run folder ngay truoc khi train
run_cfg = dict(CFG)
run_cfg["batch_actual"] = BATCH
run_cfg["device"]       = str(DEVICE)
run_cfg["half"]         = HALF
run_cfg["run_id"]       = run_id
run_cfg["dataset"]      = "augmented_startups + driver_no_yawn"
run_cfg["timestamp"]    = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

with open(run_dir / "config_used.yaml", "w", encoding="utf-8") as f:
    _yaml.dump(run_cfg, f, default_flow_style=False, allow_unicode=True)

print(f"Run ID  : {run_id}")
print(f"Model   : {MODEL}")
print(f"Epochs  : {CFG['epochs']}  |  batch={BATCH}  |  FP16={HALF}")
print(f"Config  : {run_dir/'config_used.yaml'}")
print()

results = model.train(
    data          = str(MERGED_YAML),
    epochs        = CFG["epochs"],
    imgsz         = CFG["imgsz"],
    batch         = BATCH,
    device        = DEVICE,
    half          = HALF,
    patience      = CFG["patience"],
    optimizer     = CFG["optimizer"],
    lr0           = CFG["lr0"],
    lrf           = CFG["lrf"],
    momentum      = CFG["momentum"],
    weight_decay  = CFG["weight_decay"],
    warmup_epochs = CFG["warmup_epochs"],
    cos_lr        = CFG["cos_lr"],
    box           = CFG["box"],
    cls           = CFG["cls"],
    dfl           = CFG["dfl"],
    hsv_h         = CFG["hsv_h"],
    hsv_s         = CFG["hsv_s"],
    hsv_v         = CFG["hsv_v"],
    degrees       = CFG["degrees"],
    flipud        = CFG["flipud"],
    fliplr        = CFG["fliplr"],
    mosaic        = CFG["mosaic"],
    mixup         = CFG["mixup"],
    copy_paste    = CFG["copy_paste"],
    close_mosaic  = CFG["close_mosaic"],
    dropout       = CFG["dropout"],
    cache         = "ram",
    workers       = 4,
    save          = True,
    exist_ok      = True,
    name          = run_id,
    project       = str(EXPERIMENTS),
    verbose       = True,
)

# %% Cell 7 — Luu ket qua + cap nhat experiments log
import pandas as pd

save_dir = Path(results.save_dir)

# Doc metrics
summary = dict(run_cfg)   # bat dau tu config da dung
summary["save_dir"] = str(save_dir)

results_csv = save_dir / "results.csv"
if results_csv.exists():
    df = pd.read_csv(results_csv)
    df.columns = [c.strip() for c in df.columns]
    map50_col  = [c for c in df.columns if "map50" in c.lower() and "95" not in c.lower()]
    map5095_col= [c for c in df.columns if "map50-95" in c.lower() or "map_50-95" in c.lower()]
    prec_col   = [c for c in df.columns if "precision" in c.lower()]
    rec_col    = [c for c in df.columns if "recall" in c.lower()]

    if map50_col:
        best_idx = df[map50_col[0]].idxmax()
        summary["best_epoch"]    = int(df.get("epoch", df.index).iloc[best_idx])
        summary["epochs_ran"]    = len(df)
        summary["best_map50"]    = round(float(df[map50_col[0]].iloc[best_idx]), 6)
        summary["best_map5095"]  = round(float(df[map5095_col[0]].iloc[best_idx]), 6) if map5095_col else None
        summary["best_precision"]= round(float(df[prec_col[0]].iloc[best_idx]),   6) if prec_col   else None
        summary["best_recall"]   = round(float(df[rec_col[0]].iloc[best_idx]),    6) if rec_col    else None

        # Copy results.csv vao run folder de tham khao sau
        shutil.copy2(results_csv, run_dir / "results.csv")

# Luu results.json vao run folder
with open(run_dir / "results.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

# Cap nhat experiments_log.json
log = []
if LOG_FILE.exists():
    with open(LOG_FILE, encoding="utf-8") as f:
        log = json.load(f)
log.append(summary)
with open(LOG_FILE, "w", encoding="utf-8") as f:
    json.dump(log, f, indent=2, ensure_ascii=False)

# In ket qua
print("\n" + "=" * 55)
print(f"  Run     : {run_id}")
print(f"  Model   : {summary.get('model')}")
print(f"  Note    : {summary.get('note')}")
print(f"  Epoch   : {summary.get('best_epoch')} / {summary.get('epochs_ran')}")
print(f"  mAP50   : {summary.get('best_map50', 0)*100:.2f}%")
print(f"  mAP50-95: {summary.get('best_map5095', 0)*100:.2f}%")
print(f"  Precision: {summary.get('best_precision',0)*100:.2f}%")
print(f"  Recall  : {summary.get('best_recall',0)*100:.2f}%")
print("=" * 55)
print(f"  Config  : {run_dir/'config_used.yaml'}")
print(f"  Weights : {save_dir/'weights'/'best.pt'}")
print(f"  Log     : {LOG_FILE}")

# Kiem tra co phai run tot nhat khong → luu best_config.yaml
best_run = max(log, key=lambda x: x.get("best_map50", 0))
if best_run.get("run_id") == run_id:
    best_out = PROJECT_ROOT / "configs" / "best_config.yaml"
    with open(best_out, "w", encoding="utf-8") as f:
        _yaml.dump(run_cfg, f, default_flow_style=False, allow_unicode=True)
    print(f"\n  NEW BEST! Luu vao: {best_out}")

# %% Cell 8 — So sanh tat ca experiments
import json
from pathlib import Path

LOG_FILE = PROJECT_ROOT / "outputs" / "experiments" / "experiments_log.json"

if not LOG_FILE.exists():
    print("Chua co experiment nao duoc luu.")
else:
    with open(LOG_FILE, encoding="utf-8") as f:
        log = json.load(f)

    if not log:
        print("Log trong.")
    else:
        # Header
        print(f"\n{'Run':<10} {'Model':<12} {'mAP50':>7} {'mAP50-95':>9} "
              f"{'Precision':>10} {'Recall':>8} {'Ep':>4} {'Note'}")
        print("-" * 80)

        best_map = max(r.get("best_map50", 0) for r in log)

        for r in log:
            m50   = r.get("best_map50", 0) or 0
            m5095 = r.get("best_map5095", 0) or 0
            prec  = r.get("best_precision", 0) or 0
            rec   = r.get("best_recall", 0) or 0
            ep    = r.get("best_epoch", "-")
            marker = " <-- BEST" if m50 == best_map else ""
            print(f"{r.get('run_id','?'):<10} "
                  f"{r.get('model','?'):<12} "
                  f"{m50*100:>6.2f}% "
                  f"{m5095*100:>8.2f}% "
                  f"{prec*100:>9.2f}% "
                  f"{rec*100:>7.2f}% "
                  f"{str(ep):>4}  "
                  f"{r.get('note','')}{marker}")

        print("-" * 80)
        print(f"  Tong cong: {len(log)} experiments")
        print(f"  Log file : {LOG_FILE}")
        print(f"  Best cfg : {PROJECT_ROOT/'configs'/'best_config.yaml'}")
