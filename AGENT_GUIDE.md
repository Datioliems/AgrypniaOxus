# DrowsyDriverAndroid — Agent Execution Guide
> **Dành cho AI Agent trong IDE (Claude Code, Cursor, Copilot Chat...)**
> Đọc toàn bộ file này trước khi thực hiện bất kỳ task nào.
> Mỗi section có nhãn `[LOCAL]` hoặc `[COLAB]` — KHÔNG chạy nhầm môi trường.

---

## AGENT CONTEXT — Đọc trước

Đây là project AI phát hiện buồn ngủ khi lái xe (IS54A Android).

**Tech stack:**
- Android: Kotlin + CameraX + MediaPipe FaceLandmarker (478 landmarks) + TFLite
- Python training: TensorFlow/Keras (CNN) + Ultralytics (YOLO)
- Local machine: Windows, RTX 4050 6GB, Python 3.12
- Cloud training: Google Colab T4 GPU (15GB VRAM)

**Project root:** `D:\2026.AI\DrowsyDriverAndroid\`

**Constraint cứng — KHÔNG được thay đổi:**
```
CNN Eye  : IMAGE_SIZE=64, normalize /255 NGOAI model, eyes_closed=0, eyes_open=1
CNN Yawn : IMAGE_SIZE=64, normalize /255 NGOAI model, no_yawn=0, yawn=1
TFLite   : Input shape=[1,64,64,3] dtype=float32 range=[0.0,1.0]
Android  : CNN_CONF_THRESHOLD=0.55f (Eye), YAWN_CONF_THRESHOLD=0.60f (Yawn)
```

---

## ENVIRONMENT MAP — Cái nào chạy ở đâu

```
╔══════════════════════════════════════════════════════════════╗
║  LOCAL (VS Code + Python 3.12 + RTX 4050)                   ║
║                                                              ║
║  train_cnn_eye.ipynb          ~20 phút   GPU optional       ║
║  train_cnn_yawn.ipynb         ~15 phút   GPU optional       ║
║  train_cnn_cbam.ipynb         ~25 phút   GPU optional       ║
║  train_temporal_transformer.ipynb  ~10 phút  CPU OK        ║
║  tools/train_clean_local.ipynb    ~12 phút  GPU preferred  ║
║  tune_experiments.ipynb       tùy ý                        ║
╠══════════════════════════════════════════════════════════════╣
║  GOOGLE COLAB (T4 GPU — Upload .ipynb lên drive)            ║
║                                                              ║
║  colab_drowsy_yolo26.ipynb    ~75 phút   T4 BẮT BUỘC       ║
║  colab_yolo11s.ipynb          ~50 phút   T4 BẮT BUỘC       ║
║  colab_rtdetr.ipynb           ~80 phút   T4 BẮT BUỘC       ║
║  colab_yoloworld.ipynb        ~15 phút   T4 (demo nhanh)   ║
╚══════════════════════════════════════════════════════════════╝
```

**Tại sao tách:**
- YOLO models cần T4 (15GB VRAM) — RTX 4050 chỉ có 6GB, sẽ OOM
- CNN models nhỏ (~59K params) — CPU/RTX 4050 đủ dùng
- Colab reset sau ~12h, không nên train CNN nhỏ lên đó

---

## PHASE 1 — LOCAL Training [LOCAL]

> Mở VS Code, đảm bảo `ipykernel` đã cài (`pip install ipykernel`).
> Chạy từng cell bằng nút ▶ hoặc `Shift+Enter`.

### PRE-CHECK trước khi train

```python
# Chạy cell này trước mọi thứ
import tensorflow as tf
from pathlib import Path

print("TF:", tf.__version__)
print("GPU:", tf.config.list_physical_devices("GPU"))

checks = {
    "dataset_mrl/train/eyes_closed": "eyes_closed train",
    "dataset_mrl/train/eyes_open":   "eyes_open train",
    "dataset_mrl/val/eyes_closed":   "eyes_closed val",
    "dataset_mrl/val/eyes_open":     "eyes_open val",
    "rawdata/data/yawn":             "yawn raw data",
    "app/src/main/assets/face_landmarker.task": "MediaPipe model",
}
for path, label in checks.items():
    p = Path(path)
    status = "OK" if p.exists() else "MISSING"
    print(f"  [{status}] {label}: {path}")
```

**Kết quả mong đợi:**
```
TF: 2.x.x
GPU: [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
  [OK] eyes_closed train: dataset_mrl/train/eyes_closed
  [OK] eyes_open train: ...
  [OK] yawn raw data: rawdata/data/yawn
  [OK] MediaPipe model: app/src/main/assets/face_landmarker.task
```

**Nếu `dataset_mrl` MISSING:** Chạy script copy từ rawdata:
```bash
python tools/prepare_yawn_dataset.py  # hoặc xem tools/run_all_checks.py
```

---

### STEP 1A — CNN Eye Baseline [LOCAL]

**File:** `train_cnn_eye.ipynb`
**Input:** `dataset_mrl/` (89,744 ảnh)
**Output:** `app/src/main/assets/drowsiness_model.tflite`

```
Thứ tự chạy:
Cell 1 → Kiểm tra GPU + dataset count
Cell 2 → Config (IMAGE_SIZE=64, EPOCHS=25, PATIENCE=5)
Cell 3 → Load dataset, in CLASS ORDER  ← phải thấy "eyes_closed=0, eyes_open=1"
Cell 4 → Build model (Conv32→Conv64→Conv128→GAP→Dense)
Cell 5 → TRAIN  ← chờ ~20 phút
Cell 6 → Evaluate test set
Cell 7 → Export TFLite  ← OUTPUT QUAN TRỌNG
```

**Verify sau Cell 7:**
```
[AGENT VERIFY] Kết quả Cell 7 PHẢI thấy:
  TFLite Input : shape=[ 1 64 64  3]  dtype=float32
  TFLite Output: shape=[1 2]           dtype=float32
  Classes      : ['eyes_closed', 'eyes_open']
  Android compat: [OK]

[AGENT VERIFY] File phải tồn tại sau khi chạy:
  app/src/main/assets/drowsiness_model.tflite  (~45-55 KB)
  outputs/cnn_eye/summary.json                 (val_accuracy ≥ 90%)
  outputs/cnn_eye/class_names.json             (["eyes_closed","eyes_open"])
```

**Nếu val_accuracy < 85%:**
```python
# Sửa Cell 2, tăng epochs và giảm LR
EPOCHS        = 35      # tăng từ 25
LEARNING_RATE = 5e-4    # giảm từ 1e-3
# Chạy lại Cell 5 (không cần chạy lại Cell 1-4)
```

---

### STEP 1B — CNN Eye CBAM [LOCAL]

**File:** `train_cnn_cbam.ipynb`
**Input:** `dataset_mrl/` + `dataset_yawn/`
**Output:** `app/src/main/assets/drowsiness_cbam.tflite` + `yawn_cbam.tflite`

```
Thứ tự chạy:
Cell 1 → Dataset check
Cell 2 → Config (CBAM_RATIO=8)
Cell 3 → Load Eye dataset
Cell 4 → Định nghĩa CBAM + build model  ← model mới dùng Functional API
Cell 5 → TRAIN CNN Eye CBAM (~25 phút)
Cell 6 → Compare vs baseline
Cell 7 → Export TFLite → drowsiness_cbam.tflite
Cell 8 → Train CNN Yawn CBAM (cần dataset_yawn/ tồn tại)
```

**Verify sau Cell 7:**
```
[AGENT VERIFY] So sánh accuracy:
  Baseline CNN Eye: X.XX%  (từ outputs/cnn_eye/summary.json)
  CBAM CNN Eye    : X.XX%
  Delta           : +X.XX%  ← kỳ vọng +0.5% đến +3%

[AGENT VERIFY] TFLite spec PHẢI giống baseline:
  shape=[1,64,64,3]  dtype=float32  ← KHÔNG ĐƯỢC THAY ĐỔI dù thêm CBAM
```

**CBAM hoạt động thế nào (giải thích cho agent):**
```
Mỗi Conv block: Conv → BN → [Channel Attention] → [Spatial Attention] → MaxPool

Channel Attention: Avg-pool + Max-pool → MLP → sigmoid → nhân với feature map
  "Feature map channel 43 (phát hiện cung mắt) quan trọng hơn channel 12"

Spatial Attention: Avg + Max theo chiều C → Conv7x7 → sigmoid → nhân với feature map
  "Pixel vùng mắt (x=30-50, y=25-40) quan trọng hơn pixel nền"
```

---

### STEP 2A — CNN Yawn Baseline [LOCAL]

**File:** `train_cnn_yawn.ipynb`
**Input:** `rawdata/data/yawn/` (5,119 ảnh)
**Output:** `app/src/main/assets/yawn_model.tflite`

```
Cell 1 → Tự động split rawdata → dataset_yawn/ (80/10/10)  ← TẠO MỚI
Cell 2 → Config (IMAGE_SIZE=64, EPOCHS=20)
Cell 3 → Load, in CLASS ORDER  ← phải thấy "no_yawn=0, yawn=1"
Cell 4 → Build model
Cell 5 → TRAIN (~15 phút)
Cell 6 → Evaluate
Cell 7 → Export yawn_model.tflite
```

**Verify sau Cell 7:**
```
[AGENT VERIFY]
  app/src/main/assets/yawn_model.tflite  (~45-55 KB)
  outputs/cnn_yawn/class_names.json     (["no_yawn","yawn"])
  val_accuracy ≥ 85%
```

> **NOTE cho agent:** Cell 1 sẽ kiểm tra `dataset_yawn/` đã có chưa.
> Nếu đã có thì bỏ qua (không split lại). Nếu chưa có thì tự split.

---

### STEP 3 — YOLO Local (train_clean_local) [LOCAL]

**File:** `tools/train_clean_local.ipynb`
**Input:** Roboflow datasets (download trong notebook)
**Output:** `outputs/experiments/run_001/weights/best.pt`

```
Cell 1 → Load configs/yolo_hparams.yaml
Cell 2 → GPU check (RTX 4050 → BATCH auto ~16, HALF=True)
Cell 3 → Cài ultralytics nếu cần
Cell 4A → Download ds_augmented từ Roboflow (~1-2 phút)  ← CẦN INTERNET
Cell 4B → ds_driveryawn ĐÃ CÓ → bỏ qua
Cell 4C → Merge 2 dataset → roboflow_data/clean_merged/
Cell 5 → Xem thống kê merged dataset
Cell 6 → TRAIN YOLO (~12 phút với yolov8s, 30 epochs, 640px)
Cell 7 → Lưu kết quả → outputs/experiments/experiments_log.json
Cell 8 → So sánh tất cả experiments
```

**Config hiện tại** (`configs/yolo_hparams.yaml`):
```yaml
model: yolov8s
epochs: 50
imgsz: 640
batch: -1   # auto
patience: 15
```

**Để chạy nhanh hơn (5h plan):**
```yaml
# Sửa configs/yolo_hparams.yaml trước Cell 6:
model: yolov8n   # nano thay vì small
epochs: 30
imgsz: 320       # 4x nhanh hơn 640
```

**Verify sau Cell 7:**
```
[AGENT VERIFY]
  outputs/experiments/run_001/weights/best.pt
  outputs/experiments/run_001/results.csv
  outputs/experiments/experiments_log.json  ← có entry mới nhất
  mAP50 ≥ 60%  (với clean 2-class dataset, kỳ vọng 65-80%)
```

---

### STEP 4 — Temporal Transformer [LOCAL]

**File:** `train_temporal_transformer.ipynb`
**Input:** Synthetic EAR sequences (tự tạo, không cần dataset)
**Output:** `app/src/main/assets/temporal_model.tflite`

```
Cell 1 → Setup paths
Cell 2 → Config (SEQ_LEN=20, D_MODEL=16, N_HEADS=4)
Cell 3 → Tạo 28,000 synthetic EAR sequences  ← KHÔNG CẦN DATASET THỰC
Cell 4 → Visualize patterns (awake/drowsy/microsleep)
Cell 5 → Build Transformer model (~4,000 params, rất nhỏ)
Cell 6 → TRAIN (~10 phút, CPU đủ dùng)
Cell 7 → Evaluate, confusion matrix, training curves
Cell 8 → Export TFLite → temporal_model.tflite
```

**Verify sau Cell 8:**
```
[AGENT VERIFY]
  TFLite Input : shape=[ 1 20  1]  dtype=float32  ← 20 EAR values
  TFLite Output: shape=[1 2]        dtype=float32
  app/src/main/assets/temporal_model.tflite  (~5-15 KB, nhỏ hơn CNN nhiều)
  val_accuracy ≥ 85%  (synthetic data, kỳ vọng 88-95%)

[AGENT NOTE] TFLite này cần SELECT_TF_OPS vì dùng MultiHeadAttention.
  Nếu build error trên Android cũ → thêm vào app/build.gradle:
  aaptOptions { noCompress "tflite" }
  Và dùng TFLite runtime có Flex delegate.
```

---

## PHASE 2 — GOOGLE COLAB Training [COLAB]

> **Chuẩn bị trước khi mở Colab:**
> 1. Vào `https://colab.research.google.com`
> 2. Upload file `.ipynb` tương ứng
> 3. `Runtime → Change runtime type → T4 GPU`
> 4. Chờ kết nối (~30 giây)
> 5. `Runtime → Run All` hoặc chạy từng cell

**Lưu ý Colab session:**
- Session tự reset sau ~12 tiếng idle
- Sau khi reset: dataset BIẾN MẤT, cần download lại (Cell 4 tự làm)
- Models trong `/content/runs/` cũng biến mất → Download ngay sau khi train xong (Cell 11/13)

---

### STEP 5 — YOLO26m Baseline [COLAB]

**File:** `colab_drowsy_yolo26.ipynb`
**Input:** `datio_drowsines` dataset (11 classes, tự download)
**Output:** `/content/runs/detect/drowsy_yolo26/weights/best.pt`
**Thời gian:** ~75 phút (10 epochs), ~4-5 giờ (50 epochs)

```
Cell 1 → !nvidia-smi  ← verify T4 GPU
Cell 2 → %pip install ultralytics supervision
Cell 3 → HOME = os.getcwd()
Cell 4 → Download Datio_drowsines từ Roboflow (352,830 files → 19,393 ảnh)
Cell 5 → Xem classes (11 classes: Attentive eye, Drowsy eye, ...)
Cell 6 → Visualize samples
Cell 7 → TRAIN YOLO26m  ← CHẠY LÂU NHẤT (~75 phút/10 epochs)
Cell 8 → Xem results.png + confusion_matrix.png
Cell 9 → Confusion matrix
Cell 10 → Validate  ← Kết quả mAP50
Cell 11 → Predict
Cell 12 → Lưu summary_yolo26.json
Cell 13 → Download best.pt + summary về máy  ← QUAN TRỌNG
```

**Verify sau Cell 10:**
```
[AGENT VERIFY - Colab output]
  all: mAP50 ≈ 50.21%  (đã chạy thành công)
  Attentive eye: ~81.8%
  Drowsy eye   : ~80.7%
  (các class nhỏ thấp hơn là bình thường)

[AGENT ACTION] Sau Cell 13, lưu best.pt vào:
  outputs/training_yolo/yolo26/best.pt  (local)
```

---

### STEP 6 — YOLO11s [COLAB]

**File:** `colab_yolo11s.ipynb`
**Input:** `datio_drowsines` dataset (tự download)
**Output:** `/content/runs/detect/drowsy_yolo11s/weights/best.pt`
**Thời gian:** ~50 phút (50 epochs, 640px)

```
Cell 1-4 → Setup + Download dataset (giống YOLO26)
Cell 5   → In so sánh YOLO11s vs YOLOv8s specs
Cell 6   → TRAIN YOLO11s (model=yolo11s.pt, epochs=50, imgsz=640)
Cell 7   → Validate
Cell 8   → So sánh vs YOLO26 tự động (đọc summary_yolo26.json)
Cell 9   → Xem training curves
Cell 10  → Predict
Cell 11  → Download summary_yolo11s.json + best.pt
```

**Verify:**
```
[AGENT VERIFY]
  YOLO11s mAP50 ≥ YOLO26m mAP50  (kỳ vọng +3-7%)
  Params: 9.4M  (nhỏ hơn YOLO26m 20.4M)

[AGENT ACTION] Lưu best.pt vào:
  outputs/training_yolo/yolo11s/best.pt
```

---

### STEP 7 — RT-DETR [COLAB]

**File:** `colab_rtdetr.ipynb`
**Input:** `datio_drowsines` dataset
**Output:** `/content/runs/detect/drowsy_rtdetr_l/weights/best.pt`
**Thời gian:** ~80 phút (50 epochs)

```
Cell 1-4 → Setup + Download
Cell 5   → In RT-DETR architecture overview
Cell 6   → TRAIN RT-DETR-L
             model=rtdetr-l.pt
             imgsz=640  ← KHÔNG được thay đổi (RT-DETR yêu cầu 640)
             batch=8    ← Nhỏ hơn YOLO vì nặng hơn
             lr0=1e-4   ← LR nhỏ hơn YOLO
Cell 7   → Validate
Cell 8   → So sánh vs YOLO26, YOLO11s
Cell 9-10 → Visualize
Cell 11  → Download summary_rtdetr.json + best.pt
```

**Verify:**
```
[AGENT VERIFY]
  RT-DETR mAP50 ≥ 53%  (kỳ vọng trên COCO baseline)
  Tốc độ inference: ~9.3ms/image trên T4

[AGENT NOTE] RT-DETR không hỗ trợ imgsz=320.
  Nếu OOM: giảm batch=4, KHÔNG giảm imgsz.
  Nếu vẫn OOM: dùng rtdetr-l (không phải rtdetr-x).
```

---

### STEP 8 — YOLO-World [COLAB]

**File:** `colab_yoloworld.ipynb`
**Input:** `datio_drowsines` + text prompts
**Output:** Zero-shot results + optional fine-tuned model
**Thời gian:** ~5 phút (zero-shot) hoặc ~20 phút (với fine-tune)

```
Cell 1-4 → Setup + Download test images
Cell 5   → Load YOLOWorld("yolov8s-worldv2.pt")
Cell 6   → Zero-shot: thử 4 bộ text prompt khác nhau
             "original_11"   : 11 classes gốc từ dataset
             "simplified_4"  : ["drowsy eye", "attentive eye", "yawn", "asleep"]
             "descriptive"   : mô tả chi tiết hơn
             "binary"        : ["drowsy driver", "alert driver"]
Cell 7   → So sánh accuracy các prompt sets
Cell 8   → Fine-tune 10 epochs  ← Optional, làm nếu còn thời gian
Cell 9   → Validate fine-tuned
Cell 10  → Visual demo: zero-shot vs fine-tuned
Cell 11  → Download
```

**Verify:**
```
[AGENT VERIFY]
  Zero-shot mAP50 ≈ 25-40%  (thấp hơn trained model — bình thường)
  Prompt "original_11" thường tốt nhất vì khớp ground truth labels
  Fine-tuned mAP50 ≥ 40%  (nếu có chạy)

[AGENT NOTE] YOLO-World KHÔNG dùng cho production (accuracy thấp).
  Dùng cho: demo, prototyping, detect class mới không cần label.
```

---

## PHASE 3 — Tổng hợp + Android Build

### So sánh tất cả models

```python
# Chạy cell này bất cứ lúc nào để so sánh
import json
from pathlib import Path

ROOT = Path(".")
results = []

# CNN Models
for name, path in [
    ("CNN Eye Baseline",  ROOT/"outputs/cnn_eye/summary.json"),
    ("CNN Eye CBAM",      ROOT/"outputs/cnn_eye_cbam/summary.json"),
    ("CNN Yawn Baseline", ROOT/"outputs/cnn_yawn/summary.json"),
    ("CNN Yawn CBAM",     ROOT/"outputs/cnn_yawn_cbam/summary.json"),
    ("Temporal TF",       ROOT/"outputs/temporal_transformer/summary.json"),
]:
    if path.exists():
        d = json.loads(path.read_text(encoding="utf-8"))
        results.append({
            "model": name,
            "metric": f"val_acc={d.get('best_val_accuracy',0)*100:.2f}%",
            "kb": f"{d.get('tflite_kb',0):.1f} KB",
        })

# YOLO Models (local)
log_path = ROOT/"outputs/experiments/experiments_log.json"
if log_path.exists():
    runs = json.loads(log_path.read_text(encoding="utf-8"))
    for r in runs:
        results.append({
            "model": f"YOLO Local {r['run_id']}",
            "metric": f"mAP50={r.get('best_map50',0)*100:.2f}%",
            "kb": "PT file",
        })

# Colab YOLO summaries
for name, path in [
    ("YOLO26m Colab",  ROOT/"outputs/training_yolo/yolo26/summary_yolo26.json"),
    ("YOLO11s Colab",  ROOT/"outputs/training_yolo/yolo11s/summary_yolo11s.json"),
    ("RT-DETR Colab",  ROOT/"outputs/training_yolo/rtdetr/summary_rtdetr.json"),
]:
    if path.exists():
        d = json.loads(path.read_text(encoding="utf-8"))
        results.append({
            "model": name,
            "metric": f"mAP50={d.get('best_map50',0)*100:.2f}%",
            "kb": "PT file",
        })

print(f"{'Model':<25} {'Metric':<20} {'Size'}")
print("-" * 55)
for r in results:
    print(f"{r['model']:<25} {r['metric']:<20} {r['kb']}")
```

---

### Checklist trước khi build APK

```
[AGENT VERIFY - Tất cả phải OK trước khi build]

□ app/src/main/assets/drowsiness_model.tflite
    → shape=[1,64,64,3]  dtype=float32  size=45-55KB

□ app/src/main/assets/yawn_model.tflite
    → shape=[1,64,64,3]  dtype=float32  size=45-55KB

□ app/src/main/assets/face_landmarker.task
    → size=3,670KB  (đã có sẵn)

□ CNN Eye classes: ["eyes_closed", "eyes_open"]  (không phải ["Open","Close"])
□ CNN Yawn classes: ["no_yawn", "yawn"]

[AGENT OPTIONAL - Nâng cấp nếu muốn]
□ app/src/main/assets/drowsiness_cbam.tflite  (từ train_cnn_cbam.ipynb)
□ app/src/main/assets/yawn_cbam.tflite
□ app/src/main/assets/temporal_model.tflite   (từ train_temporal_transformer.ipynb)
```

### Build APK

```bash
# Trong Android Studio:
# 1. File → Open → chọn thư mục DrowsyDriverAndroid/app
# 2. Gradle sync (tự động)
# 3. Build → Make Project
# 4. Build → Build Bundle(s)/APK(s) → Build APK(s)
# Output: app/build/outputs/apk/debug/app-debug.apk

# Hoặc qua terminal:
cd app
./gradlew assembleDebug
```

### Test APK

```
[AGENT VERIFY - Sau khi install APK]
□ Camera mở, không crash
□ Khuôn mặt detect được (MediaPipe landmark)
□ Nhắm mắt → EYES_CLOSED hiện trên màn hình
□ Nhắm mắt ≥ 1.2 giây → DROWSY + alert
□ Ngáp → YAWNING
□ FPS ≥ 8 fps
□ Latency < 150ms
□ alertSource = "CNN Eye + temporal" | "CNN Yawn" | "EAR/MAR fallback"
```

---

## TROUBLESHOOTING

### Lỗi thường gặp — Local

| Lỗi | Nguyên nhân | Fix |
|---|---|---|
| `OOM` khi train CNN | VRAM đầy | Giảm `BATCH_SIZE=16` |
| `eyes_closed=1` thay vì `=0` | Folder tên sai | Rename `Open/Close` → `eyes_open/eyes_closed` |
| TFLite output `dtype=uint8` | Thiếu `Optimize.DEFAULT` | Thêm `converter.optimizations = [tf.lite.Optimize.DEFAULT]` |
| CBAM TFLite fail convert | Lambda layer không tương thích | Dùng `tf.keras.layers.Lambda(lambda t: tf.reduce_mean(...))` thay vì trực tiếp |
| `dataset_yawn/ not found` | Chưa chạy Cell 1 của yawn | Chạy `train_cnn_yawn.ipynb` Cell 1 trước |
| Temporal TFLite convert error | MultiHeadAttention cần Flex | Thêm `SELECT_TF_OPS` như trong Cell 8 |

### Lỗi thường gặp — Colab

| Lỗi | Nguyên nhân | Fix |
|---|---|---|
| `RuntimeError: CUDA OOM` | Batch quá lớn | Thêm `batch=8` (RT-DETR) hoặc `batch=16` (YOLO) |
| `No module named 'roboflow'` | Chưa cài | `%pip install roboflow` |
| `data.yaml not found` | Download chưa xong | Chạy lại Cell 4 |
| Session reset, mất model | Colab timeout | Luôn chạy Cell 13 (download) ngay sau Cell 10 |
| `yolo26m.pt not found` | Model chưa download | `!pip install ultralytics` rồi Ultralytics sẽ tự download |
| RT-DETR `imgsz` error | Sai size | Giữ nguyên `imgsz=640`, không giảm |

### Nếu accuracy thấp

```
CNN Eye val_acc < 85%:
  → Tăng EPOCHS=35
  → Giảm LEARNING_RATE=5e-4
  → Kiểm tra class order (in CLASS_NAMES trong Cell 3)

CNN Yawn val_acc < 80%:
  → Tăng DROPOUT=0.40 (dataset nhỏ 5k ảnh, dễ overfit)
  → Thêm augmentation trong Cell 5

YOLO mAP50 < 50%:
  → Tăng epochs=50
  → Kiểm tra data.yaml (đường dẫn tuyệt đối)
  → Thử imgsz=640 (thay vì 320)

CBAM không cải thiện accuracy:
  → Giảm CBAM_RATIO=4 (channel attention mạnh hơn)
  → Tăng EPOCHS thêm 5-10 (CBAM cần thêm thời gian hội tụ)
```

---

## CONSTRAINTS REFERENCE — Copy & paste cho agent

### Android Kotlin constants (KHÔNG ĐƯỢC THAY ĐỔI)

```kotlin
// TfliteDrowsinessClassifier.kt
val MODEL_FILE = "drowsiness_model.tflite"
val INPUT_SIZE = 64
val LABELS = arrayOf("eyes_closed", "eyes_open")
val CNN_CONF_THRESHOLD = 0.55f

// YawnClassifier.kt
val MODEL_FILE = "yawn_model.tflite"
val INPUT_SIZE = 64
val LABELS = arrayOf("no_yawn", "yawn")
val YAWN_CONF_THRESHOLD = 0.60f

// DrowsinessAnalyzer.kt
val eyeClosedThreshold = 0.24f   // EAR < 0.24 = mắt nhắm
val yawnThreshold = 0.58f        // MAR > 0.58 = ngáp
val drowsyDurationMs = 1_700L

// MainActivity.kt
val cnnClosedMs >= 1200ms  // → DROWSY state
```

### Python training (PHẢI KHỚP với Android)

```python
# train_cnn_eye.ipynb Cell 2
IMAGE_SIZE = 64        # → Android INPUT_SIZE = 64
# train_cnn_eye.ipynb Cell 3
CLASS_NAMES = ["eyes_closed", "eyes_open"]  # alphabetical, eyes_closed=0
def normalize(x, y):
    return tf.cast(x, tf.float32) / 255.0, y  # NGOÀI model, không trong model

# train_cnn_yawn.ipynb Cell 2
IMAGE_SIZE = 64
# train_cnn_yawn.ipynb Cell 3
CLASS_NAMES = ["no_yawn", "yawn"]  # alphabetical, no_yawn=0

# Cả 2: TFLite export
converter.optimizations = [tf.lite.Optimize.DEFAULT]
# → Input dtype float32 [0,1], Output shape [1,2]
```

### TFLite expected spec

```
drowsiness_model.tflite:
  Input  : name="serving_default_input:0"  shape=[1,64,64,3]  dtype=FLOAT32
  Output : name="StatefulPartitionedCall:0" shape=[1,2]       dtype=FLOAT32
  [0] = P(eyes_closed),  [1] = P(eyes_open)

yawn_model.tflite:
  Input  : shape=[1,64,64,3]  dtype=FLOAT32
  Output : shape=[1,2]        dtype=FLOAT32
  [0] = P(no_yawn),  [1] = P(yawn)

temporal_model.tflite:
  Input  : shape=[1,20,1]   dtype=FLOAT32  (20 EAR values)
  Output : shape=[1,2]      dtype=FLOAT32
  [0] = P(awake),  [1] = P(drowsy)
```

---

## QUICK REFERENCE — Thứ tự chạy tối ưu

```
TERMINAL 1 (Local VS Code)          TERMINAL 2 (Google Colab)
─────────────────────────────────   ──────────────────────────────────
T+0:00  train_cnn_eye.ipynb         T+0:00  colab_drowsy_yolo26.ipynb
        Run All (20 phút)                    Run All (75 phút)

T+0:20  train_cnn_yawn.ipynb
        Run All (15 phút)

T+0:35  tools/train_clean_local     T+1:15  Sau khi YOLO26 xong:
        Run All (12 phút)                    colab_yolo11s.ipynb
                                             Run All (50 phút)

T+0:50  train_cnn_cbam.ipynb
        Cell 1-7 (Eye CBAM, 25 ph)
        Cell 8 (Yawn CBAM, 15 ph)

T+1:30  train_temporal_transformer  T+2:05  colab_rtdetr.ipynb
        Run All (10 phút)                    Run All (80 phút)

T+1:45  ANDROID BUILD               T+3:25  colab_yoloworld.ipynb
        Android Studio → Build APK           Zero-shot demo (5 ph)

T+2:15  TEST APK on phone
T+3:30  Demo recording
T+4:00  Báo cáo IS54A
T+5:00  DEADLINE
```

---

## OUTPUT FILES MAP — Sau khi hoàn thành

```
app/src/main/assets/
├── drowsiness_model.tflite   ← CNN Eye baseline   (train_cnn_eye.ipynb)
├── yawn_model.tflite         ← CNN Yawn baseline  (train_cnn_yawn.ipynb)
├── drowsiness_cbam.tflite    ← CNN Eye CBAM       (train_cnn_cbam.ipynb)
├── yawn_cbam.tflite          ← CNN Yawn CBAM      (train_cnn_cbam.ipynb)
├── temporal_model.tflite     ← Temporal TF        (train_temporal_transformer.ipynb)
└── face_landmarker.task      ← MediaPipe (đã có)

outputs/
├── cnn_eye/
│   ├── best_model.keras
│   ├── summary.json          ← val_accuracy, tflite_kb
│   └── class_names.json      ← ["eyes_closed","eyes_open"]
├── cnn_eye_cbam/
│   └── summary.json
├── cnn_yawn/
│   └── summary.json
├── cnn_yawn_cbam/
│   └── summary.json
├── temporal_transformer/
│   └── summary.json
├── experiments/
│   ├── run_001/              ← YOLO local run
│   │   ├── weights/best.pt
│   │   └── config_used.yaml
│   └── experiments_log.json
└── training_yolo/
    ├── yolo26/best.pt        ← Download từ Colab Cell 13
    ├── yolo11s/best.pt       ← Download từ Colab Cell 11
    └── rtdetr/best.pt        ← Download từ Colab Cell 11
```

---

*Agent guide version: 2026-06-07*
*Dành cho: Claude Code, Cursor, GitHub Copilot Chat, hoặc bất kỳ AI agent nào trong IDE*
