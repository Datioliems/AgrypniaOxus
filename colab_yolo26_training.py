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
ROBOFLOW_PROJECT  = "drowsiness-driver"   # slug dùng dấu - không phải dấu cách
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

# %% [4] Download dataset từ Roboflow — dùng HTTP API trực tiếp
import requests, zipfile, io

def roboflow_download(workspace, project, version, fmt, dest: Path, api_key) -> Path:
    """
    Download dataset từ Roboflow qua HTTP API trực tiếp.
    Tránh lỗi BadZipFile của Roboflow Python client.

    Flow:
      1. GET /workspace/project/version/format?api_key=...
         → trả về JSON có trường "export.link" hoặc "link"
      2. Download zip từ link đó
      3. Giải nén vào dest
    """
    dest.mkdir(parents=True, exist_ok=True)

    # Bước 1: lấy export info
    api_url = f"https://api.roboflow.com/{workspace}/{project}/{version}/{fmt}"
    print(f"   GET {api_url}")
    r = requests.get(api_url, params={"api_key": api_key}, timeout=30)

    if r.status_code != 200:
        raise RuntimeError(f"API error {r.status_code}: {r.text[:300]}")

    info = r.json()

    # Tìm download link trong response
    link = (info.get("export", {}).get("link")
            or info.get("link")
            or info.get("url"))

    if not link:
        # In toàn bộ response để debug
        print("   API response:", json.dumps(info, indent=2)[:500])
        raise RuntimeError(f"Không tìm thấy download link trong response")

    print(f"   Download link: {link[:80]}...")

    # Bước 2: download zip
    r2 = requests.get(link, timeout=120, stream=True)
    if r2.status_code != 200:
        raise RuntimeError(f"Download error {r2.status_code}")

    total = int(r2.headers.get("content-length", 0))
    data  = b""
    for chunk in r2.iter_content(chunk_size=8192):
        data += chunk
    print(f"   Downloaded: {len(data)/1024:.1f} KB")

    if len(data) < 100:
        raise RuntimeError(f"File quá nhỏ ({len(data)} bytes) — không phải zip hợp lệ")

    # Bước 3: giải nén
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        zf.extractall(dest)
        names = zf.namelist()
    print(f"   Extracted {len(names)} files → {dest}")
    return dest


def find_yaml(folder: Path) -> Path | None:
    """Tìm data.yaml hoặc obj.data trong thư mục vừa giải nén"""
    for pattern in ["*.yaml", "*.data", "**/data.yaml", "**/obj.data"]:
        found = list(folder.glob(pattern))
        if found:
            return found[0]
    return None


# ── Chẩn đoán: xem project có gì ─────────────────────────────
print("🔍 Kiểm tra Roboflow project...")
r_check = requests.get(
    f"https://api.roboflow.com/{ROBOFLOW_WORKSPACE}/{ROBOFLOW_PROJECT}",
    params={"api_key": ROBOFLOW_API_KEY}, timeout=15
)
if r_check.status_code == 200:
    proj_info = r_check.json()
    proj = proj_info.get("project", {})
    print(f"   Project : {proj.get('name', '?')}")
    print(f"   Images  : {proj.get('images', '?')}")
    versions = proj_info.get("versions", [])
    print(f"   Versions: {len(versions)}")
    for v in versions:
        print(f"     v{v.get('id','?')}: {v.get('images','?')} images  "
              f"splits={v.get('splits', {})}")
else:
    print(f"   ⚠️  Không lấy được info: {r_check.status_code}")
    print(f"   {r_check.text[:200]}")

# ── Download YOLOv8 format ─────────────────────────────────────
print(f"\n📥 Download YOLOv8 dataset (v{ROBOFLOW_VERSION})...")
yaml_path_v8 = None

try:
    dest_v8 = Path("/content/dataset_yolov8")
    roboflow_download(
        ROBOFLOW_WORKSPACE, ROBOFLOW_PROJECT,
        ROBOFLOW_VERSION, "yolov8",
        dest_v8, ROBOFLOW_API_KEY
    )
    yaml_path_v8 = find_yaml(dest_v8)
    if yaml_path_v8:
        print(f"✅ data.yaml: {yaml_path_v8}")
    else:
        raise RuntimeError("Không tìm thấy data.yaml sau khi giải nén")

except Exception as e:
    print(f"\n❌ Download thất bại: {e}")
    print("""
╔══════════════════════════════════════════════════════╗
║  NGUYÊN NHÂN THƯỜNG GẶP:                            ║
║                                                      ║
║  1. Project chưa có ảnh (0 images)                  ║
║     → Upload ảnh lên roboflow.com trước             ║
║                                                      ║
║  2. Version chưa được Generate/Export               ║
║     → roboflow.com → project → Versions             ║
║       → Generate New Version → Export → YOLOv8      ║
║                                                      ║
║  3. Dùng public dataset thay thế (xem cell dưới)    ║
╚══════════════════════════════════════════════════════╝
""")
    # ── FALLBACK: Dùng public dataset từ Roboflow Universe ──────
    print("⬇️  Thử dùng public drowsiness dataset thay thế...")
    try:
        from roboflow import Roboflow
        rf_pub = Roboflow(api_key=ROBOFLOW_API_KEY)
        # Dataset công khai về drowsiness detection
        pub = rf_pub.workspace("murtazahussain-x7gvk")\
                    .project("drowsiness-detection-ewbtq")\
                    .version(3)\
                    .download("yolov8", location="/content/dataset_yolov8", overwrite=True)
        yaml_path_v8 = Path(pub.location) / "data.yaml"
        print(f"✅ Public dataset: {yaml_path_v8}")
    except Exception as e2:
        print(f"⚠️  Public dataset cũng lỗi: {e2}")
        print("   → Tạo dataset giả để test pipeline, thay dataset thật sau")
        yaml_path_v8 = None

# %% [5] Khám phá dataset
import yaml

# Luôn khởi tạo biến — tránh NameError ở các cell sau
dataset_v8_path = None
cfg = {"nc": 0, "names": []}

if yaml_path_v8 is None or not Path(yaml_path_v8).exists():
    print("⚠️  Chưa có dataset — kiểm tra lại cell [4]")
else:
    dataset_v8_path = Path(yaml_path_v8).parent
    print(f"\n📊 Dataset overview:")
    print(f"   Path: {dataset_v8_path}")

    with open(yaml_path_v8) as f:
        cfg = yaml.safe_load(f)

    print(f"   Classes ({cfg['nc']}): {cfg['names']}")
    for split in ["train", "val", "test"]:
        split_path = dataset_v8_path / split / "images"
        if split_path.exists():
            n = len(list(split_path.glob("*.jpg")) + list(split_path.glob("*.png")))
            print(f"   {split:5s}: {n:5d} images")
        else:
            print(f"   {split:5s}: (không tìm thấy)")
    print("✅ Dataset sẵn sàng")

# %% [6] Visualize mẫu từ dataset
print("\n🖼️  Hiển thị mẫu dataset...")

if dataset_v8_path is None:
    print("⚠️  Bỏ qua — chưa có dataset_v8_path (cell [4] hoặc [5] chưa thành công)")
else:
    train_img_dir = dataset_v8_path / "train" / "images"
    train_imgs = []
    if train_img_dir.exists():
        train_imgs  = list(train_img_dir.glob("*.jpg"))[:6]
        train_imgs += list(train_img_dir.glob("*.png"))[:max(0, 6-len(train_imgs))]
        train_imgs  = train_imgs[:6]

    if not train_imgs:
        print(f"⚠️  Không tìm thấy ảnh trong {train_img_dir}")
    else:
        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        for ax, img_path in zip(axes.flat, train_imgs):
            img = mpimg.imread(str(img_path))
            ax.imshow(img)
            ax.set_title(img_path.stem[:20], fontsize=8)
            ax.axis("off")
        # Tắt các ô thừa nếu ít hơn 6 ảnh
        for ax in axes.flat[len(train_imgs):]:
            ax.axis("off")
        plt.suptitle("Mẫu Dataset — Drowsiness Driver (Roboflow)", fontsize=12)
        plt.tight_layout()
        plt.savefig(str(OUTPUT_DIR / "dataset_samples.png"), dpi=120)
        plt.show()
        print(f"✅ Saved: {OUTPUT_DIR}/dataset_samples.png")

# %% [7] Train với YOLOv8 + Ultralytics
if yaml_path_v8 is None or not Path(yaml_path_v8).exists():
    raise RuntimeError(
        "yaml_path_v8 chưa được set!\n"
        "Quay lại cell [4] — download dataset thành công trước khi train."
    )

print(f"\n{'='*60}")
print(f"TRAINING: {MODEL_BASE} (YOLO26-equivalent so sánh)")
print(f"   T4 GPU, batch={BATCH}, epochs={EPOCHS}")
print(f"   data: {yaml_path_v8}")
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
