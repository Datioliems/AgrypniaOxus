# Kế hoạch Train 5 Giờ — DrowsyDriverAndroid
> Ngày: 2026-06-07 | Deadline: 5 giờ từ lúc bắt đầu

---

## Trạng thái hiện tại (trước khi bắt đầu)

| Dataset | Trạng thái | Ghi chú |
|---------|-----------|---------|
| `dataset_mrl/` (CNN Eye) | ✅ SẴN SÀNG | 89,744 ảnh đã split |
| `dataset_yawn/` (CNN Yawn) | ⏳ Chưa có | Cell 1 tự tạo từ rawdata |
| `roboflow_data/ds_driveryawn/` | ✅ SẴN SÀNG | 11,143 files đã download |
| `roboflow_data/ds_augmented/` | ⏳ Chưa có | Cell 4A tự download |
| `roboflow_data/clean_merged/` | ⏳ Chưa có | Cell 4C tự merge |

| Output | Trạng thái |
|--------|-----------|
| `app/src/main/assets/drowsiness_model.tflite` | ✅ Có (cũ, cần ghi đè) |
| `app/src/main/assets/yawn_model.tflite` | ❌ Chưa có |
| `outputs/cnn_eye/best_model.keras` | ❌ Chưa có |
| `outputs/cnn_yawn/best_model.keras` | ❌ Chưa có |
| `outputs/experiments/run_001/` (YOLO) | ❌ Chưa có |

---

## Timeline tổng quan (chạy SONG SONG Colab + Local)

```
GIỜ 1         GIỜ 2         GIỜ 3         GIỜ 4         GIỜ 5
├─────────────┼─────────────┼─────────────┼─────────────┤
│[COLAB]──────────YOLO26 (15 epochs ~1h15)─────────────►│
│[LOCAL] CNNEye│CNNYawn│YOLO │[Android APK build ~1.5h] │
│        ~20m  │ ~15m  │~12m │              │Test│Demo  │
└─────────────┴─────────────┴─────────────┴─────────────┘
T+0:00  T+0:25  T+0:40  T+1:00            T+2:30  T+4:00
```

**Chìa khóa:** Bắt Colab chạy TRƯỚC, sau đó làm local trong lúc chờ.

---

## BƯỚC 0 — Chuẩn bị (T+0:00, 5 phút)

### 0A. Mở Colab (làm NGAY ĐẦU TIÊN)
1. Vào `https://colab.research.google.com`
2. Upload `colab_drowsy_yolo26.ipynb`
3. Runtime → Change runtime type → **T4 GPU**
4. **Đổi epochs=15** trong Cell 7 (tránh timeout):
   ```
   epochs=15
   ```
5. Runtime → **Run All** → để chạy tự động

> ⏱️ YOLO26m với 15 epochs ≈ 1 giờ 15 phút trên T4

### 0B. Cấu hình YOLO local
Mở `configs/yolo_hparams.yaml`, đảm bảo đúng như sau:
```yaml
note: "yolov8n fast - 5h plan"
model: yolov8n
epochs: 30
imgsz: 320
batch: -1
```

---

## MODEL 1 — CNN Eye Classifier

| | Chi tiết |
|---|---|
| **File chạy** | `train_cnn_eye.ipynb` (VS Code) |
| **Thời gian** | ~20 phút |
| **Bắt đầu** | T+0:05 |
| **Xong** | T+0:25 |

### Input

| Nguồn | Đường dẫn | Số lượng |
|-------|-----------|---------|
| MRL Eye Dataset (đã có) | `dataset_mrl/train/eyes_closed/` | 26,597 ảnh |
| | `dataset_mrl/train/eyes_open/` | 27,246 ảnh |
| | `dataset_mrl/val/eyes_closed/` | 8,866 ảnh |
| | `dataset_mrl/val/eyes_open/` | 9,084 ảnh |
| | `dataset_mrl/test/eyes_closed/` | 8,867 ảnh |
| | `dataset_mrl/test/eyes_open/` | 9,084 ảnh |
| **Tổng** | | **89,744 ảnh** |

### Biến cấu hình (Cell 2)

| Biến | Giá trị | Ý nghĩa |
|------|---------|---------|
| `IMAGE_SIZE` | `64` | Resize ảnh về 64×64 px |
| `BATCH_SIZE` | `32` | 32 ảnh / batch |
| `EPOCHS` | `25` | Tối đa 25 (EarlyStopping dừng sớm) |
| `LEARNING_RATE` | `1e-3` | Adam LR ban đầu |
| `DROPOUT` | `0.30` | Dropout rate |
| `PATIENCE` | `5` | Dừng nếu 5 epoch không cải thiện |

### Class mapping (quan trọng cho Android)

| Index | Tên class | Ý nghĩa |
|-------|-----------|---------|
| `0` | `eyes_closed` | Mắt nhắm → buồn ngủ |
| `1` | `eyes_open` | Mắt mở → tỉnh táo |

> ⚠️ Thứ tự alphabetical: `eyes_closed` < `eyes_open` — Keras tự sắp xếp.

### Output

| File | Đường dẫn | Dùng để |
|------|-----------|---------|
| TFLite model | `app/src/main/assets/drowsiness_model.tflite` | **Android dùng** |
| Keras model | `outputs/cnn_eye/best_model.keras` | Backup / evaluate |
| Class names | `outputs/cnn_eye/class_names.json` | `["eyes_closed","eyes_open"]` |
| History | `outputs/cnn_eye/training_history.json` | Loss/acc curves |
| Summary | `outputs/cnn_eye/summary.json` | Metrics tổng hợp |

### TFLite spec (Android)

```
Input  : shape=[1, 64, 64, 3]  dtype=float32  range=[0.0, 1.0]
Output : shape=[1, 2]           dtype=float32  (softmax)
         output[0][0] = P(eyes_closed)
         output[0][1] = P(eyes_open)
Threshold: P(eyes_closed) ≥ 0.55 → mắt nhắm  (CNN_CONF_THRESHOLD = 0.55f)
```

### Chạy
```
Mở train_cnn_eye.ipynb
→ Cell 1 (kiểm tra GPU)
→ Cell 2 (config — xem lại nếu muốn đổi)
→ Cell 3 (load dataset)
→ Cell 4 (build model)
→ Cell 5 ← TRAIN (chờ ~20 phút)
→ Cell 6 (evaluate test set)
→ Cell 7 (export TFLite → assets/)
```

---

## MODEL 2 — CNN Yawn Classifier

| | Chi tiết |
|---|---|
| **File chạy** | `train_cnn_yawn.ipynb` (VS Code) |
| **Thời gian** | ~15 phút |
| **Bắt đầu** | T+0:25 (sau CNN Eye xong) |
| **Xong** | T+0:40 |

### Input

| Nguồn | Đường dẫn | Số lượng |
|-------|-----------|---------|
| Raw yawn data (đã có) | `rawdata/data/yawn/no yawn/` | 2,591 ảnh |
| | `rawdata/data/yawn/yawn/` | 2,528 ảnh |
| **Tổng raw** | | **5,119 ảnh** |

> Cell 1 tự động split → `dataset_yawn/` (không cần làm gì thêm)

### Sau khi Cell 1 tạo `dataset_yawn/`

| Split | Class | Số ảnh |
|-------|-------|--------|
| train | no_yawn | ~2,073 (80%) |
| train | yawn | ~2,022 (80%) |
| val | no_yawn | ~259 (10%) |
| val | yawn | ~253 (10%) |
| test | no_yawn | ~259 (10%) |
| test | yawn | ~253 (10%) |

### Biến cấu hình (Cell 2)

| Biến | Giá trị | Ý nghĩa |
|------|---------|---------|
| `IMAGE_SIZE` | `64` | 64×64 px |
| `BATCH_SIZE` | `32` | |
| `EPOCHS` | `20` | Ít hơn Eye vì dataset nhỏ hơn |
| `LEARNING_RATE` | `1e-3` | |
| `DROPOUT` | `0.30` | |
| `PATIENCE` | `5` | |

### Class mapping

| Index | Tên class | Ý nghĩa |
|-------|-----------|---------|
| `0` | `no_yawn` | Không ngáp |
| `1` | `yawn` | Ngáp → buồn ngủ |

### Output

| File | Đường dẫn | Dùng để |
|------|-----------|---------|
| TFLite model | `app/src/main/assets/yawn_model.tflite` | **Android dùng** |
| Keras model | `outputs/cnn_yawn/best_model.keras` | Backup |
| Class names | `outputs/cnn_yawn/class_names.json` | `["no_yawn","yawn"]` |
| Summary | `outputs/cnn_yawn/summary.json` | Metrics |

### TFLite spec (Android)

```
Input  : shape=[1, 64, 64, 3]  dtype=float32  range=[0.0, 1.0]
Output : shape=[1, 2]           dtype=float32  (softmax)
         output[0][0] = P(no_yawn)
         output[0][1] = P(yawn)
Threshold: output[0][1] > 0.6 → đang ngáp
```

### Chạy
```
Mở train_cnn_yawn.ipynb
→ Cell 1 ← tự split dataset (chờ ~30 giây)
→ Cell 2 → Cell 3 → Cell 4
→ Cell 5 ← TRAIN (~15 phút)
→ Cell 6 → Cell 7
```

---

## MODEL 3 — YOLO Object Detection (Local)

| | Chi tiết |
|---|---|
| **File chạy** | `tools/train_clean_local.ipynb` (VS Code) |
| **Thời gian** | ~12 phút |
| **Bắt đầu** | T+0:40 |
| **Xong** | T+0:55 |

### Input — 2 Dataset từ Roboflow

| Dataset | Workspace | Project | Version | Ảnh | Classes |
|---------|-----------|---------|---------|-----|---------|
| Augmented Startups | `augmented-startups` | `drowsiness-detection-cntmz` | 1 | ~2,000 | awake, drowsy |
| driver-no-yawn | `driver-no-yawn` | `driver-drowsiness1` | 3 | ~2,900 | not_drowsy, drowsy |
| **Merged** | | | | **~4,900** | **awake=0, drowsy=1** |

> `ds_driveryawn` đã có (11,143 files). Chỉ cần download `ds_augmented` ở Cell 4A (~1 phút).

### Sau khi merge → `roboflow_data/clean_merged/`

```
clean_merged/
├── data.yaml          ← path: ..., nc: 2, names: [awake, drowsy]
├── train/images/      ← aug_*.jpg + drv_*.jpg
├── train/labels/      ← YOLO format (.txt)
├── valid/images/
└── test/images/
```

### Biến cấu hình (`configs/yolo_hparams.yaml`)

| Biến | Giá trị | Ý nghĩa |
|------|---------|---------|
| `model` | `yolov8n` | Nano — nhanh nhất cho RTX 4050 |
| `epochs` | `30` | |
| `imgsz` | `320` | 320px — nhanh 4× so với 640 |
| `batch` | `-1` | Tự động theo VRAM |
| `patience` | `15` | Early stopping |
| `cos_lr` | `true` | Cosine LR schedule |
| `lr0` | `0.01` | LR ban đầu |

### Biến trong code (Cell 6)

| Biến | Giá trị | Ý nghĩa |
|------|---------|---------|
| `MODEL` | từ CFG | `yolov8n` |
| `DEVICE` | `0` | GPU 0 (RTX 4050) |
| `BATCH` | auto | 16 nếu VRAM ≥ 6GB |
| `HALF` | `True` | FP16 tiết kiệm VRAM |
| `MERGED` | `roboflow_data/clean_merged` | Đường dẫn data |
| `run_id` | `run_001` | Auto-increment |

### Class mapping

| Index | Tên | Ý nghĩa |
|-------|-----|---------|
| `0` | `awake` | Tỉnh táo |
| `1` | `drowsy` | Buồn ngủ |

### Output

| File | Đường dẫn | Dùng để |
|------|-----------|---------|
| Best weights | `outputs/experiments/run_001/weights/best.pt` | Inference / export |
| Config dùng | `outputs/experiments/run_001/config_used.yaml` | Tái tạo experiment |
| Results CSV | `outputs/experiments/run_001/results.csv` | Loss/mAP curves |
| Results JSON | `outputs/experiments/run_001/results.json` | Metrics tổng hợp |
| Experiments log | `outputs/experiments/experiments_log.json` | So sánh runs |
| Best config | `configs/best_config.yaml` | Tự động update nếu run tốt nhất |

### Chạy
```
Mở tools/train_clean_local.ipynb
→ Cell 1 (paths)
→ Cell 2 (GPU check)
→ Cell 3 (install — bỏ qua nếu đã cài)
→ Cell 4A ← download ds_augmented (~1-2 phút)
→ Cell 4B ← ds_driveryawn đã có, bỏ qua
→ Cell 4C ← merge (~30 giây)
→ Cell 5 (xem info)
→ Cell 6 ← TRAIN (~10 phút)
→ Cell 7 (lưu kết quả)
```

---

## MODEL 4 — YOLO26 (Google Colab)

| | Chi tiết |
|---|---|
| **File chạy** | `colab_drowsy_yolo26.ipynb` (Google Colab) |
| **Thời gian** | ~1 giờ 15 phút |
| **Bắt đầu** | T+0:00 (chạy NGAY ĐẦU TIÊN) |
| **Xong** | T+1:15 |

### Input

| Dataset | Workspace | Project | Version | Ảnh | Classes |
|---------|-----------|---------|---------|-----|---------|
| Datio_drowsines | `nguyen-tuan-dat` | `datio_drowsines` | 1 | 9,694 | 11 classes |

> Dataset đã download vào `/content/Datio_drowsines-1/` từ lần trước. Nếu session mới → Cell 4 tự download lại (~1 phút).

### Biến trong Cell 7 (TRAIN)

| Tham số | Giá trị | Ghi chú |
|---------|---------|---------|
| `model` | `yolo26m.pt` | ~22M params |
| `epochs` | **`15`** | ⚠️ Giảm từ 50 → 15 để fit 5h |
| `imgsz` | `640` | |
| `batch` | `16` | T4 có 15GB VRAM |
| `name` | `drowsy_yolo26` | |
| `project` | `{HOME}/runs/detect` | `/content/runs/detect` |

### Biến trong Colab runtime

| Biến | Giá trị |
|------|---------|
| `HOME` | `/content` |
| `dataset.location` | `/content/Datio_drowsines-1` |
| `rf` | `Roboflow(api_key="qI3lEKlNpIZpNENdk3MH")` |

### Output Colab (cần download về máy)

| File Colab | Đường dẫn local | Cách download |
|------------|----------------|---------------|
| `best.pt` | `outputs/training_yolo/yolo26/best.pt` | Cell 13 tự download |
| `summary_yolo26.json` | `outputs/training_yolo/yolo26/summary_yolo26.json` | Cell 13 tự download |
| `results.csv` | `outputs/training_yolo/yolo26/results.csv` | Download thủ công |

### Chạy
```
Colab → Upload colab_drowsy_yolo26.ipynb
→ Runtime → Change runtime type → T4 GPU
→ Mở Cell 7, đổi epochs=15
→ Runtime → Run All (để tự chạy hết)
→ Làm local trong khi chờ
→ Khi xong → Cell 13 download file về
```

---

## Checklist sau khi train xong

### Xác nhận file output

```
□ app/src/main/assets/drowsiness_model.tflite   (CNN Eye)
□ app/src/main/assets/yawn_model.tflite          (CNN Yawn)
□ outputs/cnn_eye/summary.json                   → val_accuracy ≥ 90%
□ outputs/cnn_yawn/summary.json                  → val_accuracy ≥ 85%
□ outputs/experiments/run_001/weights/best.pt    (YOLO local)
□ outputs/training_yolo/yolo26/best.pt           (YOLO26 Colab)
```

### Kiểm tra metrics

| Model | Mục tiêu | Chấp nhận được |
|-------|----------|----------------|
| CNN Eye | val_acc ≥ 95% | ≥ 90% |
| CNN Yawn | val_acc ≥ 90% | ≥ 85% |
| YOLO local (mAP50) | ≥ 82% | ≥ 75% |
| YOLO26 Colab (mAP50) | ≥ 70% | ≥ 60% |

---

## Timeline chi tiết

| Thời điểm | Hành động | Nơi |
|-----------|-----------|-----|
| **T+0:00** | Upload `colab_drowsy_yolo26.ipynb` lên Colab, đổi epochs=15, **Run All** | Colab |
| **T+0:05** | Mở `train_cnn_eye.ipynb` → Run All | VS Code |
| **T+0:25** | CNN Eye xong → mở `train_cnn_yawn.ipynb` → Run All | VS Code |
| **T+0:40** | CNN Yawn xong → mở `tools/train_clean_local.ipynb` → chạy Cell 1→7 | VS Code |
| **T+0:55** | YOLO local xong → mở Android Studio | Android Studio |
| **T+1:00** | Gradle sync + Build APK | Android Studio |
| **T+1:15** | YOLO26 Colab xong → download `best.pt` + `summary_yolo26.json` | Colab |
| **T+1:30** | APK build xong → install lên điện thoại | Điện thoại |
| **T+1:30** | Test các tính năng cơ bản | Điện thoại |
| **T+2:00** | Fix bug nếu có | VS Code / Android Studio |
| **T+3:00** | Quay video demo | Điện thoại + Camera |
| **T+4:00** | Hoàn thiện báo cáo IS54A | VS Code |
| **T+5:00** | **DEADLINE** | |

---

## Nếu xảy ra sự cố

| Sự cố | Giải pháp |
|-------|-----------|
| CNN Eye val_acc < 85% | Tăng `EPOCHS=35`, giảm `LEARNING_RATE=5e-4` → chạy lại Cell 5 |
| CNN Yawn overfitting (gap > 5%) | Tăng `DROPOUT=0.5`, thêm augmentation → chạy lại Cell 5 |
| YOLO local OOM | Giảm `batch: 8` trong `yolo_hparams.yaml` |
| Colab timeout | Session reset → Run All lại, dataset đã cache sẵn |
| APK crash | Kiểm tra `drowsiness_model.tflite` input shape trong Logcat |
| YOLO26 mAP thấp | Dùng kết quả YOLO local thay thế |

---

## So sánh sau khi xong

Chạy cell này để xem kết quả tất cả models:
```python
# Trong bất kỳ notebook nào
import json
from pathlib import Path

ROOT = Path(".")
models = {
    "CNN Eye"  : ROOT/"outputs/cnn_eye/summary.json",
    "CNN Yawn" : ROOT/"outputs/cnn_yawn/summary.json",
}
for name, p in models.items():
    if p.exists():
        d = json.loads(p.read_text())
        print(f"{name}: val_acc={d['best_val_accuracy']*100:.2f}%")

# YOLO
log = ROOT/"outputs/experiments/experiments_log.json"
if log.exists():
    runs = json.loads(log.read_text())
    for r in runs:
        print(f"YOLO {r['run_id']}: mAP50={r['best_map50']*100:.2f}%  note={r['note']}")
```

---

---

## Constraint Verification — Python ↔ Android Alignment

> Bảng này kiểm tra tất cả giá trị quan trọng giữa file train (.ipynb) và Android Kotlin.  
> Tất cả phải KHỚP — sai 1 chỗ là model cho kết quả sai hoàn toàn.

### CNN Eye: `train_cnn_eye.ipynb` ↔ `TfliteDrowsinessClassifier.kt`

| Constraint | Python (.ipynb) | Android (.kt) | Trạng thái |
|---|---|---|---|
| Model file | `TFLITE_OUT = .../drowsiness_model.tflite` | `MODEL_FILE = "drowsiness_model.tflite"` | ✅ KHỚP |
| Input size | `IMAGE_SIZE = 64` | `INPUT_SIZE = 64` | ✅ KHỚP |
| Input shape | `Input(shape=(64, 64, 3))` | `ByteBuffer` với `1 × 64 × 64 × 3 × 4 bytes` | ✅ KHỚP |
| Normalize | `/ 255.0` bên ngoài model | `/ 255f` trong `toModelInput()` | ✅ KHỚP |
| Input dtype | `float32` | `DataType.FLOAT32` | ✅ KHỚP |
| Class index 0 | `eyes_closed` (alphabetical first) | `LABELS[0] = "eyes_closed"` | ✅ KHỚP |
| Class index 1 | `eyes_open` | `LABELS[1] = "eyes_open"` | ✅ KHỚP |
| Threshold | — | `CNN_CONF_THRESHOLD = 0.55f` | ✅ |
| DROWSY trigger | — | `cnnClosedMs ≥ 1200ms` | ✅ |
| Inference mode | left + right eye → average | `predictAverage(bitmaps)` | ✅ KHỚP |

### CNN Yawn: `train_cnn_yawn.ipynb` ↔ `YawnClassifier.kt`

| Constraint | Python (.ipynb) | Android (.kt) | Trạng thái |
|---|---|---|---|
| Model file | `TFLITE_OUT = .../yawn_model.tflite` | `MODEL_FILE = "yawn_model.tflite"` | ✅ KHỚP |
| Input size | `IMAGE_SIZE = 64` | `INPUT_SIZE = 64` | ✅ KHỚP |
| Normalize | `/ 255.0` bên ngoài model | `/ 255f` trong `toModelInput()` | ✅ KHỚP |
| Class index 0 | `no_yawn` | `LABELS[0] = "no_yawn"` | ✅ KHỚP |
| Class index 1 | `yawn` | `LABELS[1] = "yawn"` | ✅ KHỚP |
| Threshold | — | `YAWN_CONF_THRESHOLD = 0.60f` | ✅ |
| Yawn detect | — | `yawnProb = output[0][1] ≥ 0.60f` | ✅ |

### Dataset folder names: `train_cnn_eye.ipynb` ↔ filesystem

| Folder cần có | Nguồn gốc | Trạng thái |
|---|---|---|
| `dataset_mrl/train/eyes_closed/` | Copy từ `rawdata/.../Close/` + rename | ✅ 26,597 ảnh |
| `dataset_mrl/train/eyes_open/` | Copy từ `rawdata/.../Open/` + rename | ✅ 27,246 ảnh |
| `dataset_mrl/val/eyes_closed/` | Đã split | ✅ |
| `dataset_mrl/val/eyes_open/` | Đã split | ✅ |

> **Tại sao tên folder quan trọng:** `image_dataset_from_directory` sắp xếp alphabetical.  
> `eyes_closed` < `eyes_open` → index 0 = closed, 1 = open.  
> Nếu folder tên khác (e.g. `Close`, `Open`) → index bị đảo → model báo ngược!

### Normalize: tại sao phải ngoài model

```
Python train:
  pixel raw (0..255) → / 255.0 trong .map() → TF dataset → model input [0,1]
                       ↑ NGOÀI model, KHÔNG trong tf.keras.layers

TFLite model (sau convert):
  model.input  = float32 [0,1]   ← chờ đầu vào đã normalize
  model.output = float32 softmax

Android:
  Bitmap (0..255) → / 255f trong toModelInput() → ByteBuffer → TFLite.run()
                     ↑ NGOÀI model, tương tự Python

NẾU normalize BÊN TRONG model (Lambda layer):
  → TFLite sẽ tự normalize → Android truyền raw pixel → normalize 2 lần → SAI
```

---

## Android Pipeline — Luồng dữ liệu đầy đủ

```
Camera (CameraX)
    │  90ms interval (MainActivity.kt)
    ▼
MediaPipe FaceLandmarker
    │  face_landmarker.task (3.67 MB, 478 landmarks)
    │  numFaces=1, minDetectionConf=0.5
    ▼
Landmarks (478 điểm, normalized [0,1])
    │
    ├──► Left Eye Crop
    │        landmarks: 33, 133, 160, 158, 153, 144
    │        margin: 90% horizontal, 130% vertical
    │        resize: 64×64, /255f
    │
    ├──► Right Eye Crop
    │        landmarks: 362, 263, 385, 387, 373, 380
    │        resize: 64×64, /255f
    │
    │    predictAverage([left, right])
    │    → output[0][0] = P(eyes_closed)
    │    → output[0][1] = P(eyes_open)
    │    → isClosed = P(eyes_closed) ≥ 0.55f
    │
    ├──► Mouth Crop (classifyMouthRoi)
    │        landmarks: 61, 291, 0, 17, 39, 269, 91, 321, 181, 405
    │        resize: 64×64, /255f
    │
    │    predict(mouthBitmap)
    │    → yawnProb = output[0][1]
    │    → isYawning = yawnProb ≥ 0.60f
    │
    └──► EAR / MAR (DrowsinessAnalyzer.kt — fallback)
             Left EAR:  (33, 160, 158, 133, 153, 144)
             Right EAR: (362, 385, 387, 263, 373, 380)
             EAR threshold: 0.24f → closed
             MAR landmarks: 13,14,82,87,312,317 (vertical) + 78,308 (horizontal)
             MAR threshold: 0.58f → yawning
             drowsyDurationMs: 1700ms


buildHybridStatus() — Kết hợp tất cả:
    ┌──────────────────────────────────────────────────────────────┐
    │  CNN Eye closed ≥ 1200ms              → DROWSY  (alertSource: "CNN Eye + temporal") │
    │  CNN Eye closed  (< 1200ms)           → EYES_CLOSED                                 │
    │  CNN Yawn + MAR                       → YAWNING  (alertSource: "CNN Yawn + MAR")    │
    │  CNN Yawn only                        → YAWNING  (alertSource: "CNN Yawn")          │
    │  MAR only (MAR ≥ 0.58)               → YAWNING  (alertSource: "MAR")               │
    │  EAR/MAR fallback (no CNN result)    → EYES_CLOSED / YAWNING / DROWSY              │
    │  face detected, nothing triggered    → AWAKE                                        │
    │  no face                             → NO_FACE                                      │
    └──────────────────────────────────────────────────────────────┘


DriverStatus → UI:
    state, confidence, ear, mar,
    closedEyeMs, yawningMs,
    fps, latencyMs,
    cnnLabel, cnnConfidence, alertSource
```

---

## Post-Training TFLite Verification

Sau khi chạy Cell 7 của mỗi notebook, PHẢI thấy output như sau:

### CNN Eye (Cell 7 output)
```
TFLite saved: ...app/src/main/assets/drowsiness_model.tflite
  Size: ~45-50 KB
Input : shape=[1 64 64  3]  dtype=float32
Output: shape=[1 2]          dtype=float32
Classes: ['eyes_closed', 'eyes_open']
```

### CNN Yawn (Cell 7 output)
```
TFLite saved: ...app/src/main/assets/yawn_model.tflite
  Size: ~45-50 KB
Input : shape=[1 64 64  3]  dtype=float32
Output: shape=[1 2]          dtype=float32
Classes: ['no_yawn', 'yawn']
```

### Nếu output SAI → dừng ngay, không build APK

| Lỗi | Nguyên nhân | Fix |
|-----|------------|-----|
| dtype=uint8 | Không có `tf.lite.Optimize.DEFAULT` | Thêm `converter.optimizations = [...]` |
| shape=[1, 1] | `Dense(1, sigmoid)` thay vì `Dense(2, softmax)` | Sửa model Cell 4 |
| Classes=['Open','Close'] | Folder tên sai | Rename folder, chạy lại |
| Input range > 1.0 | Normalize trong model | Chuyển normalize ra ngoài `.map()` |

### Quick verify script (chạy bất cứ lúc nào)
```python
# %% Verify TFLite models
import tensorflow as tf
from pathlib import Path

for name, path in [
    ("CNN Eye",  "app/src/main/assets/drowsiness_model.tflite"),
    ("CNN Yawn", "app/src/main/assets/yawn_model.tflite"),
]:
    p = Path(path)
    if not p.exists():
        print(f"[MISSING] {name}: {path}")
        continue
    interp = tf.lite.Interpreter(model_path=str(p))
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]
    ok_shape = inp['shape'].tolist() == [1, 64, 64, 3]
    ok_dtype = inp['dtype'].__name__ == 'float32'
    ok_out   = out['shape'].tolist() == [1, 2]
    status = "[OK]" if (ok_shape and ok_dtype and ok_out) else "[FAIL]"
    print(f"{status} {name}")
    print(f"     Input : {inp['shape']}  {inp['dtype'].__name__}")
    print(f"     Output: {out['shape']}  {out['dtype'].__name__}")
    print(f"     Size  : {p.stat().st_size/1024:.1f} KB")
```

---

## Android Build & Test

### Bước 1 — Mở Android Studio
```
File → Open → chọn thư mục DrowsyDriverAndroid/app
Gradle sync (tự động)
Build → Make Project
```

### Bước 2 — Kiểm tra assets
```
app/src/main/assets/
├── drowsiness_model.tflite   ← CNN Eye (~45-50 KB)
├── yawn_model.tflite         ← CNN Yawn (~45-50 KB)  [tạo sau khi train]
└── face_landmarker.task      ← MediaPipe (~3.67 MB)  [đã có]
```

> ⚠️ Nếu `yawn_model.tflite` chưa có → app build được nhưng yawn detection sẽ crash.  
> Chạy train_cnn_yawn.ipynb Cell 7 trước khi build!

### Bước 3 — Build APK
```
Build → Build Bundle(s) / APK(s) → Build APK(s)
→ app/build/outputs/apk/debug/app-debug.apk
```

### Bước 4 — Test checklist
```
□ Camera mở được, không crash
□ Khuôn mặt được detect (MediaPipe landmark hiện)
□ Mắt nhắm → EYES_CLOSED hiện trên màn hình
□ Nhắm mắt ≥ 1.2 giây → DROWSY + alert
□ Ngáp → YAWNING
□ FPS ≥ 8 fps (90ms interval → ~11 fps lý thuyết)
□ Latency < 150ms
□ alertSource hiện đúng: "CNN Eye + temporal", "CNN Yawn", "MAR", v.v.
```

---

*Cập nhật lần cuối: 2026-06-07*
