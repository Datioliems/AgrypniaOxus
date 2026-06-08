# EXECUTION GUIDE — LOCAL vs GOOGLE COLAB
> DrowsyDriver Project | Cập nhật: 2026-06-07

---

## ⚡ TÓM TẮT NHANH

```
┌─────────────────────────────────────────────────────────┐
│  LOCAL (máy tính của bạn)      GOOGLE COLAB (T4 GPU)    │
│  ────────────────────────────  ─────────────────────── │
│  ✅ Bước 1–10 (pipeline ảnh)   ✅ YOLOv11s training     │
│  ✅ CNN Eye training            ✅ YOLO26m training       │
│  ✅ CNN Yawn training           ✅ RT-DETR training       │
│  ✅ TFLite conversion           ✅ CBAM + Temporal        │
│  ✅ Android APK build           ✅ Large batch inference  │
│  ✅ Streamlit dashboard                                  │
└─────────────────────────────────────────────────────────┘
```

**Nguyên tắc phân chia:**
- Xử lý dữ liệu (pipeline) → **LOCAL** (CPU đủ)
- CNN nhỏ (64×64) → **LOCAL** (train ~15–30 phút)
- YOLO / model lớn → **COLAB** (cần T4 GPU)

---

## PHẦN 1 — LOCAL ENVIRONMENT

### Setup một lần

```powershell
# 1. Kiểm tra Python version (cần ≥ 3.9)
python --version

# 2. Tạo virtual environment (khuyến nghị)
cd D:\2026.AI\DrowsyDriverAndroid
python -m venv .venv
.venv\Scripts\Activate.ps1

# 3. Cài dependencies
$env:PYTHONUTF8 = "1"
pip install -r requirements_local.txt

# 4. Kiểm tra OpenCV
python -c "import cv2; print('OpenCV:', cv2.__version__)"

# 5. Kiểm tra MediaPipe
python -c "import mediapipe; print('MediaPipe:', mediapipe.__version__)"
```

---

### 📁 Cấu trúc thư mục DATA bạn cần tạo

```
DrowsyDriverAndroid/
└── data/
    └── raw/
        ├── cnn_eye/              ← Dataset mắt từ Roboflow/MRL
        │   ├── train/
        │   │   ├── eyes_closed/  ← ảnh .jpg
        │   │   └── eyes_open/    ← ảnh .jpg
        │   ├── val/
        │   │   ├── eyes_closed/
        │   │   └── eyes_open/
        │   └── test/
        │       ├── eyes_closed/
        │       └── eyes_open/
        ├── cnn_yawn/             ← Dataset miệng/ngáp
        │   ├── train/
        │   │   ├── no_yawn/
        │   │   └── yawn/
        │   ├── val/
        │   └── test/
        └── yolo/                 ← Dataset YOLO (để COLAB xử lý)
            ├── train/
            ├── val/
            └── test/
```

> **Roboflow download:** Vào notebook `01_download_data.ipynb` → chạy cell tải về
> rồi di chuyển vào cấu trúc trên.

---

### 🔄 CHẠY PIPELINE ẢNH (Bước 1–10)

#### Cách 1 — Chạy toàn bộ pipeline tự động

```powershell
$env:PYTHONUTF8 = "1"
cd D:\2026.AI\DrowsyDriverAndroid\tools

# CNN Eye
python run_pipeline.py --dataset cnn_eye

# CNN Yawn
python run_pipeline.py --dataset cnn_yawn

# Cả hai
python run_pipeline.py --dataset all
```

#### Cách 2 — Chạy từng bước thủ công (khuyến nghị lần đầu)

```powershell
$env:PYTHONUTF8 = "1"
cd D:\2026.AI\DrowsyDriverAndroid\tools

# BƯỚC 1: Kiểm kê dataset (không chỉnh sửa dữ liệu)
python step1_ingest.py --dataset cnn_eye
# → Xem: outputs/reports/step1_cnn_eye_inventory.json
# → PASS: đọc được, không quá nhiều hỏng/trùng

# BƯỚC 2: Lọc ảnh kém chất lượng
python step2_quality_filter.py --dataset cnn_eye
# → Xem: outputs/rejected/cnn_eye/step2_quality/ (ảnh bị loại)
# → PASS: < 30% bị loại

# BƯỚC 3: Extract ROI (mắt / miệng)
python step3_extract_roi.py --dataset cnn_eye
# → cnn_eye: already_cropped=True → chỉ validate, không extract
# → cnn_yawn: cần MediaPipe → extract vùng miệng

# BƯỚC 4: Resize về 64×64
python step4_resize.py --dataset cnn_eye
# → Tất cả ảnh thành 64×64×3 uint8

# BƯỚC 5: EDA & Thống kê (không thay đổi dữ liệu)
python step5_eda.py --dataset cnn_eye
# → Xem: outputs/eda/cnn_eye/*.png
# → Xem gợi ý augmentation cần bao nhiêu ảnh

# BƯỚC 6: Augmentation (chỉ train set)
python step6_augment.py --dataset cnn_eye
# → Val/Test KHÔNG bị aug
# → Train augment đến 8000/class

# BƯỚC 7: Split validation
python step7_split.py --dataset cnn_eye
# → Kiểm tra không leakage
# → Tạo data/splits/cnn_eye/

# BƯỚC 8: Verify normalization
python step8_normalize.py --dataset cnn_eye
# → Kiểm tra /255.0 cho ra [0,1]
# → Tạo preprocessing_config.json

# BƯỚC 9: Integrity check TRƯỚC khi train
python step9_integrity_check.py --dataset cnn_eye
# → PHẢI PASS trước khi mở notebook training

# BƯỚC 10: Tạo manifest
python step10_manifest.py --dataset cnn_eye
# → data/splits/cnn_eye/dataset_manifest.json
```

**Thời gian ước tính (cnn_eye ~16k ảnh):**

| Bước | Thời gian | Ghi chú |
|------|-----------|---------|
| 1 | 2–5 phút | MD5 hash toàn bộ |
| 2 | 3–8 phút | Laplacian mỗi ảnh |
| 3 | 5–15 phút | MediaPipe inference |
| 4 | 1–3 phút | OpenCV resize |
| 5 | 2–5 phút | Vẽ biểu đồ |
| 6 | 5–20 phút | Albumentations |
| 7 | 1–2 phút | File copy |
| 8 | 1–2 phút | Verify sample |
| 9 | 3–10 phút | Scan toàn bộ |
| 10 | < 1 phút | JSON tổng hợp |
| **Tổng** | **~25–70 phút** | |

---

### 🧠 TRAIN CNN EYE (LOCAL)

```powershell
# Mở notebook
jupyter notebook notebooks/02_train_cnn_eye.ipynb
```

**Checklist trước khi train:**
```
□ step9_integrity_check.py → PASS
□ data/splits/cnn_eye/dataset_manifest.json tồn tại
□ preprocessing_config.json: normalize=divide_by_255
□ class_to_index: {"eyes_closed": 0, "eyes_open": 1}
□ Input shape sẽ là [1, 64, 64, 3] float32
```

**Thời gian CNN Eye (CPU, không GPU):**
- Máy thường (Intel i5/i7): ~20–40 phút
- Máy có GPU NVIDIA: ~5–10 phút (dùng tensorflow-gpu)

**Output sau train:**
```
outputs/cnn_eye/
├── drowsiness_model.tflite   ← Copy vào assets/
├── drowsiness_model.keras
├── training_history.png
└── summary.json              ← accuracy, loss, tflite_size_kb
```

---

### 🧠 TRAIN CNN YAWN (LOCAL)

```powershell
jupyter notebook notebooks/03_train_cnn_yawn.ipynb
```

Tương tự CNN Eye nhưng:
- Classes: `{"no_yawn": 0, "yawn": 1}`
- Output: `outputs/cnn_yawn/yawn_model.tflite`

---

### 📱 BUILD ANDROID APK (LOCAL)

```powershell
# Sau khi có 2 TFLite models
# Copy models vào assets
copy outputs\cnn_eye\drowsiness_model.tflite app\src\main\assets\
copy outputs\cnn_yawn\yawn_model.tflite app\src\main\assets\

# Build debug APK
cd D:\2026.AI\DrowsyDriverAndroid
.\gradlew assembleDebug

# APK output
# app\build\outputs\apk\debug\app-debug.apk
```

---

## PHẦN 2 — GOOGLE COLAB

### Tại sao dùng Colab cho YOLO?

| Lý do | Chi tiết |
|-------|----------|
| **GPU T4** | YOLO cần CUDA; train trên CPU mất 10–20x lâu hơn |
| **RAM 12GB** | YOLOv11s + batch_size=32 cần ~8GB GPU RAM |
| **Tốc độ** | T4 = ~40 phút; CPU = 8–15 giờ |
| **Ultralytics** | Cài đặt dễ, tích hợp sẵn trên Colab |

---

### Thứ tự chạy 4 Notebook COLAB

```
COLAB Notebook 1: 04_train_yolo11s.ipynb      (~40–60 phút)
COLAB Notebook 2: 05_train_yolo26m.ipynb      (~60–90 phút)
COLAB Notebook 3: 06_train_rtdetr.ipynb       (~50–70 phút)  [optional]
COLAB Notebook 4: 07_evaluate_compare.ipynb   (~15 phút)
```

**Thứ tự PHẢI theo vì:** Notebook 4 cần kết quả từ 1, 2, 3.

---

### Setup Google Colab từng bước

#### Bước C1 — Kết nối T4 GPU

```
1. Vào colab.research.google.com
2. Menu: Runtime → Change runtime type
3. Chọn: Hardware accelerator = T4 GPU
4. Click Save
5. Verify: !nvidia-smi (phải thấy Tesla T4)
```

#### Bước C2 — Mount Google Drive

```python
# Cell đầu tiên của mọi COLAB notebook
from google.colab import drive
drive.mount('/content/drive')

# Tạo thư mục project trên Drive
import os
os.makedirs('/content/drive/MyDrive/DrowsyDriver', exist_ok=True)
PROJECT = '/content/drive/MyDrive/DrowsyDriver'
```

#### Bước C3 — Upload dữ liệu từ LOCAL lên Drive

**Cách 1 (dễ nhất):** Dùng Google Drive desktop app — sync thư mục `data/` lên Drive
```
DrowsyDriverAndroid/data/raw/yolo/ → Google Drive/DrowsyDriver/data/raw/yolo/
```

**Cách 2:** Upload thủ công qua Drive web
```
1. Mở drive.google.com
2. Tạo thư mục: DrowsyDriver/data/raw/yolo/
3. Upload toàn bộ train/val/test
```

**Cách 3:** Tải thẳng từ Roboflow trên Colab (nhanh nhất):
```python
# Trong Colab cell
!pip install roboflow -q
from roboflow import Roboflow
rf = Roboflow(api_key="YOUR_KEY")
project = rf.workspace("YOUR_WS").project("YOUR_PROJECT")
dataset = project.version(1).download("yolov11")
```

#### Bước C4 — Cài packages COLAB

```python
# Cell setup — chạy đầu mỗi session
!pip install ultralytics -q        # YOLO
!pip install roboflow -q           # Dataset download
!pip install albumentations -q

import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
```

#### Bước C5 — Chạy training YOLO (Notebook 4)

```python
from ultralytics import YOLO

# YOLOv11s (nhỏ, nhanh, phù hợp Android)
model = YOLO('yolo11s.pt')
results = model.train(
    data    = f'{PROJECT}/data/raw/yolo/data.yaml',
    epochs  = 100,
    imgsz   = 640,
    batch   = 32,
    workers = 2,
    project = f'{PROJECT}/outputs/yolo',
    name    = 'yolo11s_run1',
    device  = 'cuda',
    patience= 20,
    save    = True,
    plots   = True,
    # Augmentation params
    hsv_h   = 0.015,
    hsv_s   = 0.7,
    hsv_v   = 0.4,
    degrees = 0.0,
    flipud  = 0.0,
    fliplr  = 0.5,
    mosaic  = 1.0,
    mixup   = 0.0,
)
print(f"mAP50: {results.results_dict.get('metrics/mAP50(B)', 0):.4f}")
```

#### Bước C6 — Download kết quả về LOCAL

```python
# Trong Colab — sau khi train xong
from google.colab import files

# Download best model
model_path = f'{PROJECT}/outputs/yolo/yolo11s_run1/weights/best.pt'
files.download(model_path)

# Download metrics
metrics_path = f'{PROJECT}/outputs/yolo/yolo11s_run1/results.csv'
files.download(metrics_path)
```

**Hoặc** dùng Drive sync — file đã tự động có trên Drive.

#### Bước C7 — Convert YOLO → TFLite (nếu cần)

```python
# Vẫn trong Colab
from ultralytics import YOLO
model = YOLO(model_path)

# Export sang TFLite
model.export(format='tflite', imgsz=640, int8=False)
# Output: best_float32.tflite
```

---

### ⚠️ Cảnh báo quan trọng khi dùng Colab

```
⛔ COLAB NGẮT KẾT NỐI sau 12 giờ (free) hoặc 24 giờ (Pro)
   → Luôn lưu checkpoint mỗi 10 epoch lên Drive

⛔ RAM Colab bị giới hạn 12GB
   → Đặt batch_size=16 nếu OOM (Out of Memory)
   → Đặt workers=2 (không tăng hơn)

⛔ Dữ liệu trên /content bị XÓA khi disconnect
   → CHỈ lưu trên /content/drive/MyDrive/...

⛔ T4 GPU có thể không có ngay (hàng đợi)
   → Kết nối lại lúc ít người dùng (sáng sớm, đêm khuya)
```

---

## PHẦN 3 — BẢNG PHÂN CHIA ĐẦY ĐỦ

| Công việc | Môi trường | Thời gian | Lý do |
|-----------|-----------|-----------|-------|
| Bước 1: Inventory | **LOCAL** | 2–5 phút | I/O đơn giản |
| Bước 2: Quality Filter | **LOCAL** | 3–8 phút | OpenCV CPU |
| Bước 3: ROI Extraction | **LOCAL** | 5–15 phút | MediaPipe CPU |
| Bước 4: Resize | **LOCAL** | 1–3 phút | OpenCV trivial |
| Bước 5: EDA | **LOCAL** | 2–5 phút | Matplotlib |
| Bước 6: Augmentation | **LOCAL** | 5–20 phút | Albumentations CPU |
| Bước 7: Split | **LOCAL** | 1–2 phút | File copy |
| Bước 8: Normalize | **LOCAL** | 1–2 phút | Verify numpy |
| Bước 9: Integrity Check | **LOCAL** | 3–10 phút | Scan files |
| Bước 10: Manifest | **LOCAL** | < 1 phút | JSON write |
| **CNN Eye train** | **LOCAL** | 20–40 phút | Model nhỏ 64×64 |
| **CNN Yawn train** | **LOCAL** | 15–30 phút | Model nhỏ 64×64 |
| TFLite conversion | **LOCAL** | 2–5 phút | tensorflow-lite |
| Android APK build | **LOCAL** | 3–10 phút | Gradle |
| **YOLOv11s train** | **COLAB** | 40–60 phút | Cần T4 GPU |
| **YOLO26m train** | **COLAB** | 60–90 phút | Cần T4 GPU |
| RT-DETR train | **COLAB** | 50–70 phút | Cần T4 GPU |
| CBAM/Temporal train | **COLAB** | 30–50 phút | Tùy chọn |
| Model evaluation | **COLAB** | 15 phút | Cùng GPU |
| Streamlit dashboard | **LOCAL** | Liên tục | Chạy local |

**Tổng thời gian:**
- LOCAL: ~3–4 giờ (pipeline + CNN train)
- COLAB: ~3–4 giờ (YOLO models)
- Tổng: **6–8 giờ** (có thể song song hóa)

---

## PHẦN 4 — THỨ TỰ THỰC HIỆN ĐỀ XUẤT

```
NGÀY 1:
  LOCAL:   Bước 1–5 (cnn_eye)          ~1.5 giờ
  LOCAL:   Bước 1–5 (cnn_yawn)         ~1.5 giờ
  LOCAL:   Bước 6–10 (cnn_eye+yawn)    ~1 giờ

NGÀY 2 (buổi sáng):
  LOCAL:   Train CNN Eye                ~30 phút
  LOCAL:   Train CNN Yawn               ~20 phút
  LOCAL:   TFLite conversion            ~5 phút
  LOCAL:   Android APK build + test     ~30 phút

NGÀY 2 (buổi chiều — mở Colab song song):
  COLAB:   Upload data YOLO lên Drive   ~30 phút
  COLAB:   Train YOLOv11s               ~60 phút
  COLAB:   Train YOLO26m                ~90 phút
  COLAB:   Download models + evaluate   ~30 phút

NGÀY 3:
  LOCAL:   Tích hợp models vào Android
  LOCAL:   Test + demo recording
  LOCAL:   Streamlit dashboard
```

---

## PHẦN 5 — DEBUG & TROUBLESHOOTING

### Lỗi thường gặp LOCAL

| Lỗi | Nguyên nhân | Cách sửa |
|-----|------------|---------|
| `'charmap' codec` | Terminal Windows không UTF-8 | `$env:PYTHONUTF8 = "1"` |
| `Cannot read file` | Đường dẫn có ký tự đặc biệt | Dùng `np.fromfile()` fallback |
| `mediapipe not found` | Chưa cài | `pip install mediapipe` |
| `albumentations: var_limit` | Version mới thay đổi API | Xem step6.py: `_make_gauss_noise()` |
| `Laplacian too strict` | Dataset ảnh chất lượng thấp | `--blur-min 30` |

### Lỗi thường gặp COLAB

| Lỗi | Nguyên nhân | Cách sửa |
|-----|------------|---------|
| `CUDA out of memory` | batch_size quá lớn | Giảm `batch=16` hoặc `batch=8` |
| `RuntimeError: session expired` | Colab timeout | Kết nối lại, chạy từ checkpoint |
| `Drive not mounted` | Chưa mount | Chạy `drive.mount()` cell đầu tiên |
| `No module named ultralytics` | Chưa install | `!pip install ultralytics -q` |
| `T4 not available` | Hàng đợi GPU | Chờ hoặc dùng Colab Pro |

---

## PHẦN 6 — LỆNH NHANH

```powershell
# ── LOCAL Quick Commands ──────────────────────
$env:PYTHONUTF8 = "1"
cd D:\2026.AI\DrowsyDriverAndroid\tools

# Full pipeline cho cả 2 dataset (unattended)
python run_pipeline.py --dataset all --no-plots 2>&1 | Tee-Object pipeline.log

# Chỉ integrity check (trước khi train)
python run_pipeline.py --dataset all --only 9

# Resume từ bước 6 (nếu bước 1–5 đã xong)
python run_pipeline.py --dataset cnn_eye --from-step 6

# Xem các report đã tạo
Get-ChildItem ..\outputs\reports\ | Sort-Object LastWriteTime -Descending | Select-Object -First 10
```

```python
# ── COLAB Quick Commands ──────────────────────
# Cell 1: Setup
!pip install ultralytics roboflow -q
from google.colab import drive; drive.mount('/content/drive')

# Cell 2: Verify GPU
import torch; print(torch.cuda.get_device_name(0))

# Cell 3: Train (thay path phù hợp)
from ultralytics import YOLO
YOLO('yolo11s.pt').train(data='/content/drive/MyDrive/DrowsyDriver/data.yaml',
                          epochs=100, batch=32, device='cuda',
                          project='/content/drive/MyDrive/DrowsyDriver/outputs')
```

---

*EXECUTION_GUIDE.md — DrowsyDriver | LOCAL vs COLAB*
