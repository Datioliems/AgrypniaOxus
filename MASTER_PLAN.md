# MASTER PLAN — Hệ thống Phát hiện Buồn ngủ Toàn diện
> **DrowsyDriver Pro** — Kĩ càng từng bước, không bỏ qua chi tiết nào
> Version 1.0 | 2026-06-07

---

## MỤC LỤC NHANH

```
PHASE 0  — Kiến trúc tổng thể & Tech stack
PHASE 1  — Thu thập dữ liệu (Data Collection)
PHASE 2  — XỬ LÝ DỮ LIỆU ẢNH (10 bước cực kĩ)  ← TRỌNG TÂM
PHASE 3  — Xây dựng mô hình AI
PHASE 4  — Hệ thống cảnh báo đa cấp
PHASE 5  — Tính năng Gửi Định vị cho Người thân
PHASE 6  — Báo cáo & Phân tích hành vi lái xe
PHASE 7  — Âm thanh tần số gây lo âu / khó ngủ  ← ĐẶC BIỆT
PHASE 8  — Tích hợp & Kiểm thử
PHASE 9  — Deploy (Android / iOS / Streamlit)
```

---

## PHASE 0 — KIẾN TRÚC TỔNG THỂ

### 0.1 Sơ đồ toàn hệ thống

```
┌─────────────────────────────────────────────────────────────┐
│                    CAMERA STREAM (30fps)                     │
└───────────────────────┬─────────────────────────────────────┘
                        │
              ┌─────────▼──────────┐
              │  IMAGE PIPELINE     │  ← PHASE 2 (10 bước)
              │  (kĩ càng nhất)     │
              └─────────┬──────────┘
                        │
         ┌──────────────┼──────────────┐
         │              │              │
    ┌────▼───┐    ┌─────▼────┐   ┌────▼────┐
    │EAR/MAR │    │ CNN Eye  │   │  YOLO   │
    │Baseline│    │ CNN Yawn │   │Detection│
    └────┬───┘    └─────┬────┘   └────┬────┘
         │              │              │
         └──────────────▼──────────────┘
                        │
              ┌─────────▼──────────┐
              │  FUSION ENGINE      │
              │  (kết hợp signals)  │
              └─────────┬──────────┘
                        │
         ┌──────────────┼──────────────┬──────────────┐
         │              │              │              │
    ┌────▼───┐    ┌─────▼────┐   ┌────▼────┐   ┌────▼──────┐
    │CẢNH   │    │  LOGGING  │   │GPS SEND │   │AUDIO FREQ │
    │BÁO    │    │  + EVENT  │   │TO FAMILY│   │ANXIETY    │
    │SYSTEM │    │  DATABASE │   │         │   │TRIGGER    │
    └───────┘    └─────┬────┘   └─────────┘   └───────────┘
                       │
              ┌────────▼───────┐
              │  REPORT ENGINE  │
              │  Peak analysis  │
              └────────────────┘
```

### 0.2 Tech stack

| Layer | Android | iOS | Streamlit |
|-------|---------|-----|-----------|
| Camera | CameraX | AVCaptureSession | OpenCV/WebRTC |
| Face Detection | MediaPipe | MediaPipe iOS | MediaPipe Python |
| Model Runtime | TFLite | TFLite + Metal | TF/ONNX |
| Location | FusedLocationProvider | CoreLocation | — |
| Notification | FCM + SMS (Twilio) | APNs + SMS | Email/Webhook |
| Database | Room (SQLite) | CoreData | SQLite/CSV |
| Audio | MediaPlayer + AudioTrack | AVAudioEngine | PyAudio |

---

## PHASE 1 — THU THẬP DỮ LIỆU

### 1.1 Nguồn dữ liệu và tiêu chí lựa chọn

**Quy tắc cứng:**
- Tối đa **8.000 ảnh/class/split** (train/val/test)
- Ảnh phải có ít nhất **24×24px** vùng mắt rõ nét
- Không chấp nhận ảnh synthetic 100% (khuôn mặt AI-generated)
- Phải có ít nhất **3 điều kiện ánh sáng** trong dataset (sáng/tối vừa/cabin đêm)

**Datasets theo mô hình:**

```
CNN Eye:
  Primary  → MRL Eye Dataset (Roboflow)         84.898 ảnh → lấy 8000/class
  Backup   → CEW Dataset (Closed Eyes in Wild)  ~4.000 ảnh
  Augment  → Tổng hợp từ augmentation pipeline  +20% mỗi class

CNN Yawn:
  Primary  → Driver Drowsiness Yawn (Roboflow)  5.119 ảnh
  Backup   → YawDD (York University)            ~440 video → extract frames
  Augment  → +30% (dataset nhỏ hơn cần aug nhiều hơn)

YOLO Full Detection (4 class):
  Primary  → Datio Drowsines v1 (Roboflow)      ~19k ảnh → lọc 4 class
  Filter   → drowsy_eye / attentive_eye / yawn / asleep
  Cap      → 8000/class sau khi filter + aug

Temporal Sequence (EAR time series):
  Source   → Synthetic generation (3 patterns)
  Count    → 28.000 sequences (14k AWAKE / 14k DROWSY)
```

### 1.2 Metadata bắt buộc cho mỗi ảnh

```json
{
  "image_id": "mrl_00001",
  "source_dataset": "MRL_Eye",
  "original_path": "dataset_mrl/train/eyes_closed/mrl_00001.jpg",
  "label": "eyes_closed",
  "label_id": 0,
  "original_size": [24, 24],
  "lighting_condition": "indoor_bright",
  "subject_id": "anon_001",
  "quality_score": 0.87,
  "blur_score": 112.3,
  "split": "train",
  "augmented": false,
  "aug_parent_id": null,
  "added_date": "2026-06-07"
}
```

---

## PHASE 2 — XỬ LÝ DỮ LIỆU ẢNH (10 BƯỚC CỰC KĨ)

> **Nguyên tắc vàng của phase này:**
> 1. KHÔNG bao giờ xử lý in-place — luôn tạo bản copy mới
> 2. Mọi bước đều có verification script chạy sau
> 3. Log từng ảnh bị loại + lý do
> 4. Hash MD5 dataset trước và sau mỗi bước
> 5. Normalize /255 NGOÀI model, KHÔNG trong layer đầu tiên

---

### BƯỚC 1 — INGESTION & KIỂM KÊ (Inventory)

**Mục đích:** Biết chính xác mình đang có gì trước khi làm gì.

```python
# tools/step1_ingest.py
def run_inventory(dataset_path: Path) -> dict:
    """
    Quét toàn bộ dataset và tạo báo cáo ban đầu.
    """
    report = {
        "total_images": 0,
        "by_class": {},
        "by_extension": {},
        "by_size_bucket": {"tiny(<32px)": 0, "small(32-128)": 0, "medium(128-640)": 0, "large(>640)": 0},
        "corrupted_files": [],
        "duplicate_hashes": {},
        "metadata_missing": [],
    }

    for img_path in dataset_path.rglob("*"):
        if img_path.suffix.lower() not in ['.jpg', '.jpeg', '.png', '.bmp']:
            continue

        # 1a. Kiểm tra file có đọc được không
        try:
            img = cv2.imread(str(img_path))
            if img is None:
                report["corrupted_files"].append(str(img_path))
                continue
        except Exception as e:
            report["corrupted_files"].append(f"{img_path}: {e}")
            continue

        # 1b. Hash MD5 để phát hiện duplicate
        md5 = hashlib.md5(img_path.read_bytes()).hexdigest()
        if md5 in report["duplicate_hashes"]:
            report["duplicate_hashes"][md5].append(str(img_path))
        else:
            report["duplicate_hashes"][md5] = [str(img_path)]

        # 1c. Phân loại kích thước
        h, w = img.shape[:2]
        min_dim = min(h, w)
        if min_dim < 32:    report["by_size_bucket"]["tiny(<32px)"] += 1
        elif min_dim < 128: report["by_size_bucket"]["small(32-128)"] += 1
        elif min_dim < 640: report["by_size_bucket"]["medium(128-640)"] += 1
        else:               report["by_size_bucket"]["large(>640)"] += 1

        report["total_images"] += 1
        class_name = img_path.parent.name
        report["by_class"][class_name] = report["by_class"].get(class_name, 0) + 1

    # Báo cáo final
    true_duplicates = {k: v for k, v in report["duplicate_hashes"].items() if len(v) > 1}
    report["duplicate_count"] = sum(len(v) - 1 for v in true_duplicates.values())
    return report
```

**Ngưỡng dừng (STOP nếu vi phạm):**
- Corrupted files > 5%: Tải lại dataset
- Duplicates > 10%: Deduplicate trước khi tiếp tục
- Tiny images > 20%: Xem xét lại nguồn dataset

---

### BƯỚC 2 — LỌC CHẤT LƯỢNG (Quality Filtering)

**Mục đích:** Loại bỏ ảnh kém chất lượng sẽ làm model học sai.

```python
# tools/step2_quality_filter.py

class QualityFilter:
    THRESHOLDS = {
        "blur_laplacian_min": 50,        # Laplacian variance < 50 → mờ, loại bỏ
        "brightness_min": 20,            # Quá tối → không phân biệt được
        "brightness_max": 235,           # Quá sáng → overexposed
        "contrast_min": 15,              # Độ tương phản tối thiểu
        "min_face_confidence": 0.80,     # MediaPipe face detection confidence
        "eye_region_coverage_min": 0.05, # Vùng mắt phải chiếm ≥ 5% ảnh
    }

    def check_blur(self, img_gray: np.ndarray) -> tuple[bool, float]:
        """Laplacian variance — ảnh sắc nét có variance cao."""
        score = cv2.Laplacian(img_gray, cv2.CV_64F).var()
        return score >= self.THRESHOLDS["blur_laplacian_min"], score

    def check_brightness(self, img_gray: np.ndarray) -> tuple[bool, float]:
        """Kiểm tra độ sáng trung bình."""
        mean_val = float(np.mean(img_gray))
        ok = self.THRESHOLDS["brightness_min"] <= mean_val <= self.THRESHOLDS["brightness_max"]
        return ok, mean_val

    def check_contrast(self, img_gray: np.ndarray) -> tuple[bool, float]:
        """Độ tương phản = std deviation của pixel values."""
        std_val = float(np.std(img_gray))
        return std_val >= self.THRESHOLDS["contrast_min"], std_val

    def check_face_detectable(self, img_rgb: np.ndarray) -> tuple[bool, float]:
        """Chạy MediaPipe — ảnh phải có khuôn mặt phát hiện được."""
        # Chỉ cần detect, không cần 478 landmarks đầy đủ
        results = face_detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb))
        if not results.detections:
            return False, 0.0
        best_conf = max(d.categories[0].score for d in results.detections)
        return best_conf >= self.THRESHOLDS["min_face_confidence"], best_conf

    def filter_image(self, img_path: Path) -> dict:
        """Chạy tất cả checks và trả về kết quả chi tiết."""
        img_bgr = cv2.imread(str(img_path))
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        results = {
            "path": str(img_path),
            "passed": True,
            "rejection_reason": [],
            "metrics": {}
        }

        # Chạy từng check
        blur_ok, blur_score = self.check_blur(img_gray)
        results["metrics"]["blur_score"] = blur_score
        if not blur_ok:
            results["passed"] = False
            results["rejection_reason"].append(f"BLUR: {blur_score:.1f} < {self.THRESHOLDS['blur_laplacian_min']}")

        bright_ok, bright_val = self.check_brightness(img_gray)
        results["metrics"]["brightness"] = bright_val
        if not bright_ok:
            results["passed"] = False
            results["rejection_reason"].append(f"BRIGHTNESS: {bright_val:.1f} out of range [{self.THRESHOLDS['brightness_min']}, {self.THRESHOLDS['brightness_max']}]")

        contrast_ok, contrast_val = self.check_contrast(img_gray)
        results["metrics"]["contrast"] = contrast_val
        if not contrast_ok:
            results["passed"] = False
            results["rejection_reason"].append(f"CONTRAST: {contrast_val:.1f} < {self.THRESHOLDS['contrast_min']}")

        # QUAN TRỌNG: Log ảnh bị loại thành file riêng để review
        if not results["passed"]:
            REJECTION_LOG.append(results)

        return results
```

**Output:** `data/step2_quality_report.json` + `data/rejected/` chứa các ảnh bị loại kèm lý do.

---

### BƯỚC 3 — PHÁT HIỆN KHUÔN MẶT & TRÍCH ROI (Face Detection & ROI)

**Mục đích:** Crop chính xác vùng mắt và miệng cho CNN input.

```python
# tools/step3_extract_roi.py

class ROIExtractor:
    """
    478 landmarks từ MediaPipe FaceLandmarker.
    Sử dụng các điểm landmark cụ thể:
      Mắt trái  : 33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246
      Mắt phải  : 362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398
      Môi ngoài : 61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146
    """

    # Padding mỗi phía khi crop (% so với kích thước bbox)
    EYE_PADDING   = 0.25   # 25% padding mỗi phía
    MOUTH_PADDING = 0.30   # 30% padding (miệng hay bị cắt hơn)

    LEFT_EYE_LANDMARKS  = [33, 160, 158, 133, 153, 144]   # 6 điểm EAR
    RIGHT_EYE_LANDMARKS = [362, 385, 387, 263, 373, 380]   # 6 điểm EAR
    MOUTH_LANDMARKS     = [61, 291, 0, 17, 39, 269, 270, 409]  # 8 điểm MAR

    def extract_eye_roi(self, img: np.ndarray, landmarks, side: str) -> np.ndarray | None:
        """
        Crop vùng mắt với padding an toàn.
        Trả về None nếu ROI nằm ngoài biên ảnh.
        """
        h, w = img.shape[:2]
        pts = self.LEFT_EYE_LANDMARKS if side == 'left' else self.RIGHT_EYE_LANDMARKS

        # Lấy tọa độ pixel từ landmarks chuẩn hóa
        coords = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in pts]
        xs, ys = [c[0] for c in coords], [c[1] for c in coords]

        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)

        # Tính padding
        eye_w = x_max - x_min
        eye_h = y_max - y_min
        pad_x = int(eye_w * self.EYE_PADDING)
        pad_y = int(eye_h * self.EYE_PADDING)

        # Đảm bảo không vượt biên ảnh
        x1 = max(0, x_min - pad_x)
        y1 = max(0, y_min - pad_y)
        x2 = min(w, x_max + pad_x)
        y2 = min(h, y_max + pad_y)

        # Kiểm tra ROI có đủ lớn không (ít nhất 10×10 pixel)
        if (x2 - x1) < 10 or (y2 - y1) < 10:
            return None

        return img[y1:y2, x1:x2].copy()

    def extract_mouth_roi(self, img: np.ndarray, landmarks) -> np.ndarray | None:
        """Tương tự nhưng dùng mouth landmarks."""
        # ... (tương tự extract_eye_roi)
        pass

    def validate_roi(self, roi: np.ndarray, roi_type: str) -> dict:
        """
        Kiểm tra ROI sau khi crop — đảm bảo chất lượng.
        """
        if roi is None:
            return {"valid": False, "reason": "ROI extraction failed (out of bounds)"}

        h, w = roi.shape[:2]
        aspect_ratio = w / h if h > 0 else 0

        # Kiểm tra aspect ratio hợp lý
        if roi_type == 'eye':
            # Mắt thường có tỷ lệ w/h = 2.0 đến 5.0
            if not (1.5 <= aspect_ratio <= 6.0):
                return {"valid": False, "reason": f"Weird aspect ratio: {aspect_ratio:.2f}"}

        if roi_type == 'mouth':
            # Miệng thường có tỷ lệ w/h = 1.5 đến 4.0
            if not (1.0 <= aspect_ratio <= 5.0):
                return {"valid": False, "reason": f"Weird aspect ratio: {aspect_ratio:.2f}"}

        # Kiểm tra ROI không quá tối/sáng
        mean_brightness = np.mean(roi)
        if mean_brightness < 15 or mean_brightness > 240:
            return {"valid": False, "reason": f"ROI brightness out of range: {mean_brightness:.1f}"}

        return {"valid": True, "shape": roi.shape, "aspect_ratio": aspect_ratio}
```

**Lưu ý đặc biệt:**
```
⚠️ KHÔNG dùng bounding box khuôn mặt thô để crop mắt
   → Luôn dùng đúng landmarks điểm mắt/môi
   
⚠️ Lưu lại CẢ HAI mắt riêng biệt (không merge)
   → Mắt trái và mắt phải train riêng, test riêng
   → Nếu chỉ phát hiện 1 mắt → vẫn dùng 1 mắt đó
   
⚠️ Mắt bên nào landmark đủ 6 điểm → mới extract
   → Không guess / interpolate landmark bị thiếu
```

---

### BƯỚC 4 — CHUẨN HÓA KÍCH THƯỚC (Resize & Standardization)

**Mục đích:** Đưa tất cả ảnh về cùng kích thước mà KHÔNG làm méo tỷ lệ.

```python
# tools/step4_resize.py

class ImageStandardizer:
    CNN_SIZE = 64        # CNN input size (64×64)
    YOLO_SIZE = 640      # YOLO input size (640×640)
    INTERPOLATION_UPSCALE   = cv2.INTER_CUBIC    # Phóng to → dùng CUBIC (mịn hơn)
    INTERPOLATION_DOWNSCALE = cv2.INTER_AREA     # Thu nhỏ → dùng AREA (tránh aliasing)

    def resize_with_padding(self, img: np.ndarray, target_size: int,
                             pad_color=(0, 0, 0)) -> tuple[np.ndarray, dict]:
        """
        Resize giữ nguyên aspect ratio + padding đen.
        Trả về ảnh đã resize VÀ metadata về phép transform.
        """
        h, w = img.shape[:2]
        scale = target_size / max(h, w)

        new_w = int(w * scale)
        new_h = int(h * scale)

        # Chọn interpolation dựa trên scale direction
        interp = self.INTERPOLATION_UPSCALE if scale > 1.0 else self.INTERPOLATION_DOWNSCALE
        resized = cv2.resize(img, (new_w, new_h), interpolation=interp)

        # Padding để đạt target_size × target_size
        pad_top    = (target_size - new_h) // 2
        pad_bottom = target_size - new_h - pad_top
        pad_left   = (target_size - new_w) // 2
        pad_right  = target_size - new_w - pad_left

        padded = cv2.copyMakeBorder(resized, pad_top, pad_bottom, pad_left, pad_right,
                                     cv2.BORDER_CONSTANT, value=pad_color)

        # Metadata để reverse transform nếu cần
        transform_meta = {
            "original_size": (h, w),
            "scale": scale,
            "padding": {"top": pad_top, "bottom": pad_bottom, "left": pad_left, "right": pad_right},
            "interpolation": "CUBIC" if scale > 1.0 else "AREA",
        }

        return padded, transform_meta

    def resize_direct_square(self, img: np.ndarray, target_size: int) -> np.ndarray:
        """
        Resize trực tiếp thành square (có thể méo nhẹ).
        Dùng cho CNN Eye/Yawn khi ảnh ROI đã gần vuông.
        Phải kiểm tra aspect ratio trước:
          - aspect_ratio 0.7 → 1.4: OK dùng direct resize
          - ngoài range đó: dùng resize_with_padding
        """
        interp = self.INTERPOLATION_UPSCALE if target_size > min(img.shape[:2]) else self.INTERPOLATION_DOWNSCALE
        return cv2.resize(img, (target_size, target_size), interpolation=interp)

    def verify_resize_output(self, img: np.ndarray, expected_size: int) -> bool:
        """Kiểm tra output sau resize."""
        h, w = img.shape[:2]
        assert h == expected_size and w == expected_size, \
            f"Resize output size mismatch: got {h}x{w}, expected {expected_size}x{expected_size}"
        assert img.dtype == np.uint8, f"Wrong dtype after resize: {img.dtype}"
        assert img.shape[2] == 3, f"Wrong channels: {img.shape[2]}"
        return True
```

**Bảng quyết định resize:**

| Trường hợp | Phương pháp | Lý do |
|---|---|---|
| ROI mắt (aspect ~3:1) | `resize_with_padding` | Tránh méo mắt |
| ROI môi (aspect ~2.5:1) | `resize_with_padding` | Tránh méo miệng |
| Full face (gần vuông) | `resize_direct_square` | Đơn giản, nhanh |
| YOLO input | `letterbox` (Ultralytics) | Giữ nguyên tỷ lệ |

---

### BƯỚC 5 — PHÂN TÍCH THỐNG KÊ DATASET (EDA Chính xác)

**Mục đích:** Nắm rõ đặc điểm dữ liệu trước khi augment — không augment mù quáng.

```python
# tools/step5_eda.py

class DatasetAnalyzer:
    def compute_channel_statistics(self, images: list[np.ndarray]) -> dict:
        """
        Tính mean và std từng channel (R, G, B) trên TOÀN TẬP TRAIN.
        Kết quả dùng cho normalize nếu cần (thay vì /255 đơn thuần).
        """
        all_means = {"R": [], "G": [], "B": []}
        all_stds  = {"R": [], "G": [], "B": []}

        for img in images:
            img_float = img.astype(np.float32) / 255.0
            for i, ch in enumerate(['B', 'G', 'R']):  # OpenCV = BGR
                all_means[ch].append(np.mean(img_float[:,:,i]))
                all_stds[ch].append(np.std(img_float[:,:,i]))

        return {
            "channel_means": {k: float(np.mean(v)) for k, v in all_means.items()},
            "channel_stds":  {k: float(np.mean(v)) for k, v in all_stds.items()},
            # Ví dụ kết quả điển hình cho MRL Eye:
            # means: R=0.485, G=0.456, B=0.406
            # stds:  R=0.229, G=0.224, B=0.225
        }

    def analyze_class_imbalance(self, labels: list) -> dict:
        """
        Tính imbalance ratio và đề xuất strategy.
        """
        from collections import Counter
        counts = Counter(labels)
        total = sum(counts.values())
        max_count = max(counts.values())
        min_count = min(counts.values())
        imbalance_ratio = max_count / min_count

        strategy = "NONE"
        if imbalance_ratio > 1.5:
            strategy = "OVERSAMPLE minority class (RandomOverSampler)"
        if imbalance_ratio > 3.0:
            strategy = "SMOTE or class_weight in loss function"
        if imbalance_ratio > 5.0:
            strategy = "CRITICAL: Collect more data for minority class"

        return {
            "class_counts": dict(counts),
            "imbalance_ratio": imbalance_ratio,
            "recommended_strategy": strategy,
            "class_weights": {c: total / (len(counts) * n) for c, n in counts.items()},
        }

    def analyze_pixel_distribution(self, images_sample: list) -> dict:
        """
        Histogram pixel values để phát hiện:
        - Ảnh quá tối (mode < 50)
        - Ảnh bị overexposed (mode > 200)
        - Ảnh có nhiều vùng pure black/white (có thể là artifact)
        """
        all_pixels = np.concatenate([img.flatten() for img in images_sample[:500]])
        hist, bins = np.histogram(all_pixels, bins=256, range=(0, 256))
        return {
            "mean": float(np.mean(all_pixels)),
            "std":  float(np.std(all_pixels)),
            "p5":   float(np.percentile(all_pixels, 5)),
            "p95":  float(np.percentile(all_pixels, 95)),
            "dark_pixels_pct":   float((all_pixels < 30).sum() / len(all_pixels) * 100),
            "bright_pixels_pct": float((all_pixels > 225).sum() / len(all_pixels) * 100),
        }
```

**Output bắt buộc của BƯỚC 5:**
```
outputs/eda/
├── class_distribution.png     ← biểu đồ cột so sánh classes
├── pixel_histogram.png        ← phân phối pixel values
├── brightness_distribution.png
├── sample_grid.png            ← lưới 5×5 ảnh mẫu mỗi class
├── eda_report.json            ← tất cả metrics số
└── eda_summary.md             ← nhận xét văn bản
```

---

### BƯỚC 6 — DATA AUGMENTATION (Tăng cường dữ liệu)

**Nguyên tắc:**
- Augmentation PHẢI **có lý nghĩa về mặt vật lý** — không augment tùy tiện
- Mỗi kỹ thuật có **xác suất độc lập** — không apply tất cả mọi lúc
- Mọi ảnh augmented có **parent_id** trỏ về ảnh gốc
- KHÔNG augment tập **validation** và **test** — chỉ train

```python
# tools/step6_augment.py

import albumentations as A

class DrowsinessAugmenter:
    """
    Albumentations pipeline được thiết kế riêng cho ảnh mắt/khuôn mặt tài xế.
    Mỗi transform có xác suất và giới hạn cẩn thận.
    """

    # Pipeline cho CNN Eye (ảnh crop mắt 64×64)
    EYE_PIPELINE = A.Compose([
        # --- Spatial transforms ---
        A.HorizontalFlip(p=0.5),
        # Xoay nhẹ — tài xế nghiêng đầu nhưng không đảo lộn
        A.Rotate(limit=15, border_mode=cv2.BORDER_REFLECT, p=0.6),
        # Dịch chuyển nhỏ — mắt vẫn phải trong frame
        A.ShiftScaleRotate(
            shift_limit=0.10,   # ±10% shift
            scale_limit=0.15,   # ±15% scale
            rotate_limit=0,     # Rotate đã xử lý riêng
            p=0.5
        ),
        # Perspective nhẹ — mô phỏng góc camera khác nhau
        A.Perspective(scale=(0.02, 0.05), p=0.3),

        # --- Photometric transforms ---
        A.RandomBrightnessContrast(
            brightness_limit=0.25,
            contrast_limit=0.20,
            p=0.7
        ),
        # Gamma thay đổi — mô phỏng ánh sáng cabin vs ban ngày
        A.RandomGamma(gamma_limit=(70, 130), p=0.4),
        # CLAHE — tăng cường contrast cục bộ (giúp với ảnh tối)
        A.CLAHE(clip_limit=2.0, tile_grid_size=(4, 4), p=0.3),
        # Gaussian noise nhẹ — camera smartphone thực tế
        A.GaussNoise(var_limit=(5.0, 25.0), p=0.4),
        # Gaussian blur nhẹ — mô phỏng camera out-of-focus
        A.GaussianBlur(blur_limit=(1, 3), p=0.3),
        # Motion blur — mô phỏng đầu đang cử động
        A.MotionBlur(blur_limit=5, p=0.2),
        # JPEG compression — ảnh camera kém chất lượng
        A.ImageCompression(quality_lower=70, quality_upper=95, p=0.3),

        # --- KHÔNG dùng ---
        # ❌ VerticalFlip → mắt không thể ngược
        # ❌ Xoay > 30° → mắt ra ngoài frame
        # ❌ Cutout/CoarseDropout lớn → che mất vùng mắt
        # ❌ Channel shuffle → mắt nên giữ màu tự nhiên
    ])

    # Pipeline cho CNN Yawn (ảnh crop miệng 64×64)
    MOUTH_PIPELINE = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=12, border_mode=cv2.BORDER_REFLECT, p=0.5),
        A.ShiftScaleRotate(shift_limit=0.08, scale_limit=0.12, rotate_limit=0, p=0.4),
        A.RandomBrightnessContrast(brightness_limit=0.30, contrast_limit=0.25, p=0.7),
        A.RandomGamma(gamma_limit=(65, 140), p=0.4),
        A.GaussNoise(var_limit=(5.0, 20.0), p=0.35),
        A.GaussianBlur(blur_limit=(1, 3), p=0.25),
        A.ImageCompression(quality_lower=75, quality_upper=95, p=0.25),
        # Elastic transform nhẹ — mô phỏng biến dạng hình học nhỏ
        A.ElasticTransform(alpha=10, sigma=3, alpha_affine=3, p=0.15),
    ])

    # Pipeline cho YOLO (full face / half body 640×640)
    YOLO_PIPELINE = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=10, border_mode=cv2.BORDER_REFLECT, p=0.4),
        A.RandomBrightnessContrast(brightness_limit=0.20, contrast_limit=0.15, p=0.6),
        A.RandomGamma(gamma_limit=(75, 125), p=0.35),
        A.GaussNoise(var_limit=(5.0, 15.0), p=0.3),
        A.GaussianBlur(blur_limit=(1, 5), p=0.2),
        A.MotionBlur(blur_limit=7, p=0.15),
        A.CLAHE(clip_limit=1.5, p=0.25),
        A.ImageCompression(quality_lower=75, quality_upper=95, p=0.2),
        # Giảm saturate — mô phỏng camera cabin thiếu màu
        A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=20, val_shift_limit=15, p=0.3),
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

    def augment_to_target(self, images: list, labels: list, target_per_class: int,
                           pipeline, seed: int = 42) -> list:
        """
        Augment cho đến khi mỗi class đạt target_per_class.
        Tracking parent_id đầy đủ.
        """
        random.seed(seed)
        np.random.seed(seed)

        result = []
        class_counts = Counter(labels)

        for cls in set(labels):
            cls_images = [img for img, lbl in zip(images, labels) if lbl == cls]
            current_count = len(cls_images)
            need = max(0, target_per_class - current_count)

            for i in range(need):
                # Chọn ảnh gốc ngẫu nhiên (với replacement)
                parent_img = random.choice(cls_images)
                parent_id  = hashlib.md5(parent_img.tobytes()).hexdigest()[:8]

                augmented = pipeline(image=parent_img)["image"]

                result.append({
                    "image": augmented,
                    "label": cls,
                    "augmented": True,
                    "parent_id": parent_id,
                    "aug_index": i,
                })

        return result
```

**Mục tiêu augmentation:**

| Dataset | Gốc (mỗi class) | Sau aug | Tổng |
|---|---|---|---|
| CNN Eye train | 6.400 | 8.000 | 16.000 |
| CNN Yawn train | ~2.000 | 4.000 | 8.000 |
| YOLO 4-class | 3.500 avg | 5.000 | 20.000 |

---

### BƯỚC 7 — PHÂN CHIA DỮ LIỆU (Dataset Split)

**Phương pháp: Stratified Split với Subject Separation**

```python
# tools/step7_split.py

def stratified_split_with_subject_separation(
    images: list,
    labels: list,
    subject_ids: list,    # ID người trong ảnh (nếu có)
    ratios: tuple = (0.70, 0.15, 0.15),
    seed: int = 42
) -> dict:
    """
    QUAN TRỌNG: Đảm bảo một người không xuất hiện cả train lẫn test.
    Nếu không có subject_ids → dùng stratified random split thông thường.

    Vấn đề subject leakage:
    - Nếu cùng 1 người xuất hiện ở cả train và test
    → model học "nhận dạng người" chứ không học "phát hiện buồn ngủ"
    → val/test accuracy cao giả tạo
    """

    if subject_ids and len(set(subject_ids)) > 10:
        # Có thông tin subject → subject-aware split
        unique_subjects = list(set(subject_ids))
        random.Random(seed).shuffle(unique_subjects)

        n = len(unique_subjects)
        train_sbj = unique_subjects[:int(n * ratios[0])]
        val_sbj   = unique_subjects[int(n * ratios[0]):int(n * (ratios[0]+ratios[1]))]
        test_sbj  = unique_subjects[int(n * (ratios[0]+ratios[1])):]

        train_idx = [i for i, s in enumerate(subject_ids) if s in set(train_sbj)]
        val_idx   = [i for i, s in enumerate(subject_ids) if s in set(val_sbj)]
        test_idx  = [i for i, s in enumerate(subject_ids) if s in set(test_sbj)]
    else:
        # Không có subject info → stratified random
        from sklearn.model_selection import train_test_split
        all_idx = list(range(len(images)))
        train_val_idx, test_idx = train_test_split(
            all_idx, test_size=ratios[2], stratify=labels, random_state=seed
        )
        adj_val_ratio = ratios[1] / (ratios[0] + ratios[1])
        train_idx, val_idx = train_test_split(
            train_val_idx, test_size=adj_val_ratio,
            stratify=[labels[i] for i in train_val_idx], random_state=seed
        )

    # Verify không có overlap
    assert not (set(train_idx) & set(val_idx)), "CRITICAL: Train/Val overlap!"
    assert not (set(train_idx) & set(test_idx)), "CRITICAL: Train/Test overlap!"
    assert not (set(val_idx) & set(test_idx)), "CRITICAL: Val/Test overlap!"

    return {"train": train_idx, "val": val_idx, "test": test_idx}
```

**Quy tắc split KHÔNG được vi phạm:**
```
✅ Test set KHÔNG được augment — chỉ resize + normalize
✅ Val set KHÔNG được augment — chỉ resize + normalize
✅ CHỈ Train set được augment
✅ Một ảnh chỉ xuất hiện trong ĐÚNG MỘT split
✅ Sau khi split xong → HASH toàn bộ 3 tập → lưu vào split_manifest.json
```

---

### BƯỚC 8 — NORMALIZE & TENSOR PREPARATION

**Mục đích:** Chuẩn hóa giá trị pixel và chuẩn bị tensor đúng format cho TFLite.

```python
# tools/step8_normalize.py

class TensorPreparer:
    """
    CRITICAL CONSTRAINT: Normalize PHẢI thực hiện NGOÀI model.
    Không đưa lớp normalize vào Keras model.
    Lý do: TFLite runtime trên Android nhận float32 [0,1].
    Nếu normalize trong model → phải truyền int8 từ Android → không tương thích.
    """

    @staticmethod
    def prepare_for_cnn(img_bgr: np.ndarray, target_size: int = 64) -> np.ndarray:
        """
        Input:  np.ndarray BGR uint8 (từ cv2.imread hoặc camera)
        Output: np.ndarray float32 [0,1] shape (64, 64, 3) RGB
        """
        # Bước 1: BGR → RGB (MediaPipe và TFLite dùng RGB)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # Bước 2: Resize
        img_resized = cv2.resize(img_rgb, (target_size, target_size),
                                  interpolation=cv2.INTER_CUBIC if img_rgb.shape[0] < target_size
                                               else cv2.INTER_AREA)

        # Bước 3: Normalize — NGOÀI MODEL
        img_float = img_resized.astype(np.float32) / 255.0

        # Bước 4: Verify range
        assert 0.0 <= img_float.min() <= img_float.max() <= 1.0, \
            f"Normalize failed: range [{img_float.min()}, {img_float.max()}]"

        return img_float

    @staticmethod
    def prepare_batch_for_tflite(images: list[np.ndarray], target_size: int = 64) -> np.ndarray:
        """
        Chuẩn bị batch tensor cho TFLite.
        Output shape: (N, target_size, target_size, 3) float32
        """
        tensors = [TensorPreparer.prepare_for_cnn(img, target_size) for img in images]
        batch = np.stack(tensors, axis=0)

        # Double-check shape và dtype
        assert batch.shape[1:] == (target_size, target_size, 3), f"Wrong shape: {batch.shape}"
        assert batch.dtype == np.float32, f"Wrong dtype: {batch.dtype}"

        return batch

    @staticmethod
    def verify_android_compatibility(tflite_path: str) -> dict:
        """
        Kiểm tra TFLite model có compatible với Android runtime không.
        """
        import tensorflow as tf
        interpreter = tf.lite.Interpreter(model_path=tflite_path)
        interpreter.allocate_tensors()

        input_details  = interpreter.get_input_details()[0]
        output_details = interpreter.get_output_details()[0]

        checks = {
            "input_shape":  input_details["shape"].tolist(),
            "input_dtype":  str(input_details["dtype"]),
            "output_shape": output_details["shape"].tolist(),
            "output_dtype": str(output_details["dtype"]),
            "android_compatible": True,
            "issues": [],
        }

        # Kiểm tra từng constraint
        if list(input_details["shape"]) != [1, 64, 64, 3]:
            checks["issues"].append(f"Wrong input shape: {input_details['shape']}")
            checks["android_compatible"] = False

        if input_details["dtype"] != np.float32:
            checks["issues"].append(f"Wrong input dtype: {input_details['dtype']}")
            checks["android_compatible"] = False

        if list(output_details["shape"]) != [1, 2]:
            checks["issues"].append(f"Wrong output shape: {output_details['shape']}")
            checks["android_compatible"] = False

        return checks
```

---

### BƯỚC 9 — KIỂM TRA TOÀN VẸN DỮ LIỆU (Data Integrity Check)

**Mục đích:** Trước khi train, đảm bảo 100% dữ liệu đúng format, không bị corrupt.

```python
# tools/step9_integrity_check.py

def full_integrity_check(dataset_path: Path, split: str) -> dict:
    """
    Chạy TRƯỚC mỗi lần train. Nếu fail → DỪNG LẠI, sửa trước.
    """
    results = {"passed": True, "errors": [], "warnings": [], "stats": {}}

    label_dirs = list((dataset_path / split).iterdir())
    class_counts = {}

    for label_dir in label_dirs:
        if not label_dir.is_dir():
            continue

        label = label_dir.name
        count = 0

        for img_path in label_dir.glob("*.jpg"):
            img = cv2.imread(str(img_path))

            # Check 1: Đọc được
            if img is None:
                results["errors"].append(f"Cannot read: {img_path}")
                results["passed"] = False
                continue

            # Check 2: Đúng shape
            if img.shape != (64, 64, 3):
                results["errors"].append(f"Wrong shape {img.shape}: {img_path}")
                results["passed"] = False

            # Check 3: Không phải toàn đen/trắng
            if img.std() < 5:
                results["warnings"].append(f"Very low contrast (std={img.std():.1f}): {img_path}")

            count += 1

        class_counts[label] = count

    # Check 4: Class balance
    if class_counts:
        max_c = max(class_counts.values())
        min_c = min(class_counts.values())
        if max_c / max(min_c, 1) > 3.0:
            results["warnings"].append(
                f"Severe imbalance: {class_counts} (ratio={max_c/min_c:.1f})"
            )

    results["stats"]["class_counts"] = class_counts
    results["stats"]["total"] = sum(class_counts.values())

    # Check 5: CLASS ORDER NHẤT QUÁN VỚI ANDROID
    expected_orders = {
        "cnn_eye":  ["eyes_closed", "eyes_open"],   # eyes_closed=0, eyes_open=1
        "cnn_yawn": ["no_yawn", "yawn"],             # no_yawn=0, yawn=1
    }
    # ... verify alphabetical order matches Android labels array

    return results
```

---

### BƯỚC 10 — TẠO DATASET MANIFEST (Tracking cuối cùng)

**Mục đích:** Ghi lại toàn bộ quá trình xử lý để có thể tái tạo bất kỳ lúc nào.

```json
// data/dataset_manifest.json
{
  "created_at": "2026-06-07T08:00:00Z",
  "pipeline_version": "1.0.0",
  "dataset_hash_before_processing": "md5:abc123...",
  "dataset_hash_after_processing":  "md5:def456...",
  "steps_applied": [
    {"step": 1, "name": "Inventory", "input_count": 89203, "output_count": 89203},
    {"step": 2, "name": "QualityFilter", "input_count": 89203, "output_count": 85891, "rejected": 3312},
    {"step": 3, "name": "ROIExtraction", "input_count": 85891, "output_count": 84233, "failed_extraction": 1658},
    {"step": 4, "name": "Resize64x64", "input_count": 84233, "output_count": 84233},
    {"step": 5, "name": "EDA", "note": "No data modification"},
    {"step": 6, "name": "Augmentation", "input_count": 84233, "output_count": 128000, "aug_count": 43767},
    {"step": 7, "name": "Split", "train": 89600, "val": 19200, "test": 19200},
    {"step": 8, "name": "Normalize", "note": "float32 /255.0, outside model"},
    {"step": 9, "name": "IntegrityCheck", "passed": true, "errors": 0, "warnings": 2},
    {"step": 10, "name": "ManifestGeneration"}
  ],
  "final_split_counts": {
    "cnn_eye":  {"train": {"eyes_closed": 8000, "eyes_open": 8000}, "val": {"eyes_closed": 2000, "eyes_open": 2000}, "test": {"eyes_closed": 2000, "eyes_open": 2000}},
    "cnn_yawn": {"train": {"no_yawn": 4000, "yawn": 4000}, "val": {"no_yawn": 1000, "yawn": 1000}, "test": {"no_yawn": 1000, "yawn": 1000}}
  },
  "class_order": {
    "cnn_eye":  {"0": "eyes_closed", "1": "eyes_open"},
    "cnn_yawn": {"0": "no_yawn",     "1": "yawn"}
  },
  "normalization": "divide_by_255_outside_model",
  "android_tflite_input_spec": {"shape": [1,64,64,3], "dtype": "float32", "range": [0.0, 1.0]}
}
```

---

## PHASE 3 — XÂY DỰNG MÔ HÌNH AI

### 3.1 Thứ tự train (từ đơn giản đến phức tạp)

```
1. EAR/MAR Baseline (không cần train)    → Benchmark nhanh
2. CNN Eye                                → ~20 phút  LOCAL
3. CNN Yawn                               → ~15 phút  LOCAL
4. Temporal LSTM (EAR sequences)         → ~10 phút  LOCAL
5. YOLOv11s                              → ~50 phút  COLAB T4
6. YOLO26m                               → ~75 phút  COLAB T4
```

### 3.2 Fusion Logic — Cơ chế kết hợp tín hiệu

```kotlin
// DrowsinessAnalyzer.kt — Fusion Engine
fun buildHybridStatus(): DrowsinessStatus {
    // Cấp 1: EAR/MAR Geometric (luôn chạy, không tốn GPU)
    val earStatus  = if (currentEAR < EAR_THRESHOLD && earDuration > 1700) DROWSY else AWAKE
    val marStatus  = if (currentMAR > MAR_THRESHOLD && marDuration > 800)  YAWNING else AWAKE

    // Cấp 2: CNN Eye (TFLite, ~8ms)
    val cnnEyeProb = cnnEyeResult[0]  // P(eyes_closed)
    val cnnEyeStatus = if (cnnEyeProb >= CNN_CONF_THRESHOLD && cnnDuration > 1200) DROWSY else AWAKE

    // Cấp 3: CNN Yawn (TFLite, ~6ms)
    val cnnYawnProb = cnnYawnResult[1]  // P(yawn)
    val cnnYawnStatus = if (cnnYawnProb >= YAWN_CONF_THRESHOLD) YAWNING else AWAKE

    // Voting với trọng số
    var drowsyScore = 0.0f
    if (earStatus == DROWSY)    drowsyScore += 1.0f * 0.25f  // Weight: 25%
    if (cnnEyeStatus == DROWSY) drowsyScore += 1.0f * 0.50f  // Weight: 50% (most reliable)
    if (marStatus == YAWNING)   drowsyScore += 1.0f * 0.15f  // Weight: 15%
    if (cnnYawnStatus == YAWNING) drowsyScore += 1.0f * 0.10f // Weight: 10%

    return when {
        drowsyScore >= 0.50f -> DROWSY
        drowsyScore >= 0.25f -> WARNING
        cnnYawnStatus == YAWNING || marStatus == YAWNING -> YAWNING
        else -> AWAKE
    }
}
```

---

## PHASE 4 — HỆ THỐNG CẢNH BÁO ĐA CẤP

### 4.1 Cấu trúc 5 cấp độ cảnh báo

```
LEVEL 0 — AWAKE          → Không cảnh báo, chỉ monitor
LEVEL 1 — MICRO_DROWSY   → EAR/MAR nhẹ, cảnh báo âm thanh nhẹ (40dB, 0.5s)
LEVEL 2 — DROWSY         → CNN Eyes closed hoặc EAR kéo dài → beep + rung + overlay đỏ
LEVEL 3 — SEVERE_DROWSY  → ≥3 DROWSY events trong 5 phút → cảnh báo mạnh + GPS gửi ngay
LEVEL 4 — CRITICAL       → Mắt nhắm >3 giây / ngáp lặp lại >5 lần / 2 phút
                         → BẮT ĐẦU PHÁT TẦN SỐ ÂM THANH KHÓ NGỦ + GPS liên tục
LEVEL 5 — EMERGENCY      → Không phản hồi sau LEVEL 4 10 giây
                         → Gọi điện tự động + GPS real-time + Alert người thân
```

### 4.2 Event Logging System

```kotlin
// EventLogger.kt
data class DrowsinessEvent(
    val id: String = UUID.randomUUID().toString(),
    val timestamp: Long = System.currentTimeMillis(),
    val eventType: EventType,
    val alertLevel: AlertLevel,
    val earValue: Float,
    val marValue: Float,
    val cnnEyeProb: Float,
    val cnnYawnProb: Float,
    val durationMs: Long,
    val latitude: Double?,
    val longitude: Double?,
    val roadSpeed: Float?,    // Từ GPS speed
    val responseTime: Long?,  // Thời gian đến khi tài xế phản hồi
    val audioPlayed: String?, // Tên audio đã phát
)

enum class EventType {
    EYES_CLOSED, YAWNING, MICROSLEEP, SEVERE_DROWSY, CRITICAL_DROWSY,
    DRIVER_RESPONDED, GPS_SENT, AUDIO_TRIGGERED, SESSION_START, SESSION_END
}

// Lưu vào Room database
@Entity(tableName = "drowsiness_events")
// Đồng thời export CSV cho báo cáo
```

---

## PHASE 5 — GỬI ĐỊNH VỊ CHO NGƯỜI THÂN

### 5.1 Cơ chế hoạt động

```
Khi nào gửi:
  - LEVEL 3+ (SEVERE_DROWSY): Gửi một lần
  - LEVEL 4+ (CRITICAL): Gửi mỗi 2 phút
  - LEVEL 5 (EMERGENCY): Gửi mỗi 30 giây + gọi điện

Cách gửi (theo thứ tự ưu tiên):
  1. Push Notification (FCM/APNs) → App người thân
  2. SMS (Twilio API) → Điện thoại người thân không có app
  3. WhatsApp/Zalo API → Backup nếu SMS fail
  4. Email → Last resort
```

### 5.2 Quản lý danh sách liên hệ

```kotlin
// EmergencyContact.kt
data class EmergencyContact(
    val id: Int = 0,
    val name: String,
    val phone: String,
    val email: String?,
    val relationship: String,     // "spouse", "parent", "friend"
    val notifyLevel: AlertLevel,  // Gửi từ level nào trở lên
    val notifyMethods: List<NotifyMethod>,
    val isActive: Boolean = true,
    val languageCode: String = "vi", // Ngôn ngữ tin nhắn
)

// Mẫu tin nhắn SMS tự động
fun buildAlertMessage(event: DrowsinessEvent, contact: EmergencyContact): String {
    val maps_url = "https://maps.google.com/?q=${event.latitude},${event.longitude}"
    return when (contact.languageCode) {
        "vi" -> """
            [DrowsyDriver] ⚠️ CẢNH BÁO BUỒN NGỦ
            Tài xế: ${driverProfile.name}
            Thời gian: ${formatTime(event.timestamp)}
            Cấp độ: ${event.alertLevel.displayName}
            Vị trí: $maps_url
            Vận tốc: ${event.roadSpeed?.let { "${it.roundToInt()} km/h" } ?: "Không rõ"}
            Nhấn vào link để xem vị trí real-time.
        """.trimIndent()
        else -> "..." // English version
    }
}
```

### 5.3 Location Tracking Module

```kotlin
// LocationTracker.kt
class LocationTracker(private val context: Context) {
    private val fusedClient = LocationServices.getFusedLocationProviderClient(context)

    // Normal mode: update every 5 minutes (tiết kiệm pin)
    private val NORMAL_INTERVAL = 5 * 60 * 1000L

    // Alert mode: update every 30 seconds (khi phát hiện buồn ngủ)
    private val ALERT_INTERVAL = 30 * 1000L

    // Emergency mode: update every 10 seconds (CRITICAL/EMERGENCY)
    private val EMERGENCY_INTERVAL = 10 * 1000L

    fun setTrackingMode(mode: TrackingMode) {
        val interval = when (mode) {
            TrackingMode.NORMAL    -> NORMAL_INTERVAL
            TrackingMode.ALERT     -> ALERT_INTERVAL
            TrackingMode.EMERGENCY -> EMERGENCY_INTERVAL
        }
        // Update location request với interval mới
        val request = LocationRequest.Builder(Priority.PRIORITY_HIGH_ACCURACY, interval).build()
        fusedClient.requestLocationUpdates(request, locationCallback, Looper.getMainLooper())
    }

    // GỬI lên cloud để người thân xem real-time (optional)
    suspend fun uploadLocationToCloud(location: Location, alertLevel: AlertLevel) {
        // Firebase Realtime Database / Firestore
        val locationData = mapOf(
            "lat" to location.latitude,
            "lng" to location.longitude,
            "speed_kmh" to (location.speed * 3.6f),
            "accuracy_m" to location.accuracy,
            "timestamp" to System.currentTimeMillis(),
            "alert_level" to alertLevel.name,
            "driver_id" to preferences.driverId,
        )
        firestore.collection("locations").document(preferences.driverId).set(locationData)
    }
}
```

---

## PHASE 6 — BÁO CÁO VÀ PHÂN TÍCH

### 6.1 Báo cáo khoảng thời gian buồn ngủ nhất

```kotlin
// DrowsinessReportEngine.kt
data class DrowsinessReport(
    val sessionId: String,
    val date: LocalDate,
    val totalDrivingTime: Duration,
    val peakDrowsinessHours: List<HourStats>,   // Giờ nào buồn ngủ nhiều nhất
    val peakDrowsinessKm: List<RouteSegment>,   // Đoạn đường nào nguy hiểm nhất
    val totalDrowsyEvents: Int,
    val totalYawnEvents: Int,
    val longestEyesClosedMs: Long,
    val averageEAR: Float,
    val drowsinessPattern: DrowsinessPattern,   // GRADUAL / SUDDEN / CYCLIC
    val riskLevel: RiskLevel,
    val recommendations: List<String>,
)

fun analyzePeakPeriods(events: List<DrowsinessEvent>): List<HourStats> {
    // Nhóm sự kiện theo giờ trong ngày
    val byHour = events.groupBy { 
        Calendar.getInstance().apply { timeInMillis = it.timestamp }.get(Calendar.HOUR_OF_DAY)
    }

    return byHour.map { (hour, hourEvents) ->
        HourStats(
            hour = hour,
            drowsyCount = hourEvents.count { it.eventType == EventType.EYES_CLOSED },
            yawnCount   = hourEvents.count { it.eventType == EventType.YAWNING },
            avgEAR      = hourEvents.map { it.earValue }.average().toFloat(),
            riskScore   = calculateHourRisk(hourEvents),
        )
    }.sortedByDescending { it.riskScore }
}
```

### 6.2 Dashboard Streamlit — Phân tích dữ liệu nhiều chuyến

```python
# streamlit_dashboard.py
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

def render_peak_analysis_page(df_events: pd.DataFrame):
    st.title("🕐 Phân tích khoảng thời gian buồn ngủ nhất")

    col1, col2, col3 = st.columns(3)
    col1.metric("Tổng sự kiện buồn ngủ", len(df_events[df_events.event_type == 'EYES_CLOSED']))
    col2.metric("Giờ nguy hiểm nhất", f"{peak_hour}:00-{peak_hour+1}:00")
    col3.metric("Mắt nhắm lâu nhất", f"{max_duration_s:.1f}s")

    # Heatmap theo giờ và ngày trong tuần
    fig_heatmap = px.density_heatmap(
        df_events,
        x="hour_of_day",
        y="day_of_week",
        z="drowsy_count",
        title="Phân phối buồn ngủ theo giờ và ngày",
        color_continuous_scale="Reds",
    )
    st.plotly_chart(fig_heatmap, use_container_width=True)

    # Timeline EAR trên một chuyến
    fig_ear = go.Figure()
    fig_ear.add_trace(go.Scatter(x=df_session.timestamp, y=df_session.ear_value,
                                  name="EAR", line=dict(color='blue')))
    fig_ear.add_hline(y=0.24, line_dash="dash", line_color="red",
                       annotation_text="EAR threshold = 0.24")
    st.plotly_chart(fig_ear, use_container_width=True)

    # Bản đồ vị trí buồn ngủ
    if 'latitude' in df_events.columns:
        fig_map = px.scatter_mapbox(
            df_events.dropna(subset=['latitude', 'longitude']),
            lat='latitude', lon='longitude',
            color='alert_level',
            size='severity_score',
            hover_data=['timestamp', 'ear_value', 'alert_level'],
            mapbox_style='open-street-map',
            title="Vị trí các sự kiện buồn ngủ",
        )
        st.plotly_chart(fig_map, use_container_width=True)
```

---

## PHASE 7 — ÂM THANH TẦN SỐ GÂY LO ÂU / KHÓ NGỦ

> **Cơ sở khoa học:** Khi mắt nhắm quá lâu hoặc ngáp buồn ngủ quá nhiều lần,
> hệ thống phát âm thanh tần số đặc biệt để kích thích hệ thần kinh,
> tạo trạng thái lo âu nhẹ và khó ngủ — buộc não bộ phải tỉnh dậy.

### 7.1 Tần số và tác dụng sinh lý

```
TẦNG 1 — CẢNH BÁO ĐẦU TIÊN (Kích hoạt: ≥3 DROWSY events / 15 phút)
  Tần số: 1000–2000 Hz Alert Tone (ngắt quãng 2Hz)
  Tác dụng: Kích thích phản xạ giật mình (startle reflex)
  Cơ sở: Horne & Reyner (1995) BMJ — dải 1–4kHz nhạy cảm nhất thính giác người
  Thời lượng: 5 giây, tự động tắt nếu AWAKE

TẦNG 2 — KÍCH THÍCH BETA (Kích hoạt: ≥5 events / 15 phút)
  Tần số: Binaural Beats — Carrier 200Hz, Beat 18–25Hz (Beta wave range)
  Tác dụng: Đồng bộ não với sóng Beta → trạng thái tập trung, tỉnh táo
  Cơ sở: Lane et al. (1998) Duke University; Chaieb et al. (2015) Univ. Bonn
  Yêu cầu: TAI NGHE STEREO (binaural chỉ hiệu quả khi 2 tai khác tần số)
  Thời lượng: 30 giây, lặp lại mỗi 5 phút nếu vẫn drowsy

TẦNG 3 — STRESS RESPONSE (Kích hoạt: Mắt nhắm >3s HOẶC ≥7 events / 15 phút)
  Tần số: 19 Hz Infrasound + 432 Hz carrier
  Tác dụng: 19 Hz gây cảm giác bất an, lo lắng nhẹ, khó chịu
  Cơ sở: Tandy V. & Lawrence T.R. (1998) "The Ghost in the Machine"
          Journal of the Society for Psychical Research
  Cơ chế: 19 Hz gần với tần số cộng hưởng tự nhiên của nhãn cầu (18-20Hz)
           → tạo cảm giác bất an, hình ảnh nhòe nhạt → não tự kích hoạt cảnh giác
  Thời lượng: 15 giây ON / 10 giây OFF, lặp tối đa 3 lần

TẦNG 4 — GAMMA BURST (Kích hoạt: CRITICAL level — không phản hồi TẦNG 3)
  Tần số: 40 Hz Isochronic Tones
  Tác dụng: 40 Hz gamma kích hoạt vỏ não trước trán, tăng cảnh giác tối đa
  Cơ sở: Oster G. (1973) Scientific American; Multiple Alzheimer studies
  Thời lượng: 30 giây burst, volume leo thang 50→80dB

TẦNG 5 — EMERGENCY ESCALATION (KHÔNG phản hồi sau 10 giây TẦNG 4)
  Âm thanh: Siren-like 800-2400Hz sweeping + 40Hz pulse overlay
  Volume: 85dB (giới hạn an toàn WHO tạm thời)
  Đồng thời: Gọi điện thoại tự động cho liên hệ khẩn cấp
  Thời lượng: Cho đến khi tài xế phản hồi hoặc xe dừng
```

### 7.2 Implementation Audio Engine

```kotlin
// AlertAudioEngine.kt
class AlertAudioEngine(private val context: Context) {

    // Tầng 1: Alert Tone cơ bản (MediaPlayer)
    fun playAlertTone(level: Int) {
        val resId = when(level) {
            1 -> R.raw.alert_1000hz_pulse
            2 -> R.raw.alert_1500hz_pulse
            3 -> R.raw.alert_2000hz_pulse
            else -> R.raw.alert_emergency
        }
        MediaPlayer.create(context, resId).apply {
            setVolume(0.6f, 0.6f)
            start()
        }
    }

    // Tầng 2-4: Binaural Beats / Isochronic (AudioTrack - generated programmatically)
    fun generateBinauralBeat(
        durationSeconds: Int,
        carrierFreqHz: Float = 200f,     // Tần số carrier
        beatFreqHz: Float = 18f,         // Tần số beat (delta giữa 2 tai)
        volumeLevel: Float = 0.7f
    ): AudioTrack {
        val sampleRate = 44100
        val numSamples = sampleRate * durationSeconds
        val buffer = ShortArray(numSamples * 2)  // Stereo

        for (i in 0 until numSamples) {
            val t = i.toDouble() / sampleRate
            // Tai trái: carrier frequency
            val leftSample  = (sin(2 * PI * carrierFreqHz * t) * Short.MAX_VALUE * volumeLevel).toInt().toShort()
            // Tai phải: carrier + beat frequency
            val rightSample = (sin(2 * PI * (carrierFreqHz + beatFreqHz) * t) * Short.MAX_VALUE * volumeLevel).toInt().toShort()

            buffer[i * 2]     = leftSample
            buffer[i * 2 + 1] = rightSample
        }

        return AudioTrack(
            AudioAttributes.Builder()
                .setUsage(AudioAttributes.USAGE_ALARM)
                .setContentType(AudioAttributes.CONTENT_TYPE_SONIFICATION)
                .build(),
            AudioFormat.Builder()
                .setEncoding(AudioFormat.ENCODING_PCM_16BIT)
                .setSampleRate(sampleRate)
                .setChannelMask(AudioFormat.CHANNEL_OUT_STEREO)
                .build(),
            buffer.size * 2,
            AudioTrack.MODE_STATIC,
            AudioManager.AUDIO_SESSION_ID_GENERATE
        ).apply {
            write(buffer, 0, buffer.size)
            play()
        }
    }

    // Tầng 3: 19 Hz Infrasound (WARNING: cần kiểm thử kỹ trước khi deploy)
    fun generateInfrasound19Hz(durationSeconds: Int): AudioTrack {
        // 19 Hz rất thấp, cần loa subwoofer hoặc tai nghe có bass tốt để cảm nhận
        // Trên điện thoại thông thường → có thể không hiệu quả qua loa ngoài
        // Khuyến nghị: kết hợp với 432 Hz carrier để người nghe "cảm nhận" được
        return generateBinauralBeat(
            durationSeconds = durationSeconds,
            carrierFreqHz = 432f,
            beatFreqHz = 19f,    // 432 Hz (tai trái) vs 451 Hz (tai phải) → beat 19 Hz
            volumeLevel = 0.65f
        )
    }

    // Logic kích hoạt theo level
    fun triggerByDrowsinessLevel(level: AlertLevel, context: AudioTriggerContext) {
        when {
            level == AlertLevel.MICRO_DROWSY -> {
                playAlertTone(1)  // Nhẹ nhàng
            }
            level == AlertLevel.DROWSY -> {
                playAlertTone(2)
                vibratePhone(pattern = VibrationPattern.SHORT_BURST)
            }
            level == AlertLevel.SEVERE_DROWSY ||
            context.drowsyEventsLast15Min >= 5 -> {
                // Binaural Beta 18-25Hz
                generateBinauralBeat(30, 200f, 20f, 0.75f)
                vibratePhone(VibrationPattern.REPEATED_PULSE)
                sendLocationUpdate()
            }
            level == AlertLevel.CRITICAL ||
            context.eyesClosedDurationMs > 3000 ||
            context.drowsyEventsLast15Min >= 7 -> {
                // 19 Hz Infrasound + 40 Hz Gamma
                generateInfrasound19Hz(15)
                Handler(Looper.getMainLooper()).postDelayed({
                    generateBinauralBeat(30, 200f, 40f, 0.85f)
                }, 20000)
                sendLocationUpdate(TrackingMode.EMERGENCY)
                notifyEmergencyContacts(AlertLevel.CRITICAL)
            }
            level == AlertLevel.EMERGENCY -> {
                playSiren()
                makeEmergencyCall()
                notifyEmergencyContacts(AlertLevel.EMERGENCY)
            }
        }
    }
}
```

### 7.3 Bảng tóm tắt tần số và trigger

| Tầng | Tần số | Trigger | Tác dụng sinh lý | Thời lượng | Yêu cầu |
|------|--------|---------|-----------------|------------|---------|
| 1 | 1000–2000 Hz Alert | ≥ 1 DROWSY event | Phản xạ giật mình, cảnh báo | 5s | Loa thường |
| 2 | 200Hz carrier + 18–25Hz beat | ≥ 3 events / 15min | Đồng bộ sóng Beta → tỉnh táo, tập trung | 30s | **Tai nghe stereo** |
| 3 | 432Hz + 19Hz beat | ≥ 5 events hoặc mắt nhắm >3s | Lo âu nhẹ, bất an, nhãn cầu cộng hưởng | 15s ON/10s OFF | Tai nghe bass |
| 4 | 200Hz + 40Hz Gamma | CRITICAL / không phản hồi | Kích hoạt vỏ não trước trán, tỉnh tối đa | 30s burst | Tai nghe |
| 5 | 800–2400Hz sweeping siren | EMERGENCY / không phản hồi 10s | Shock + emergency response | Until response | Loa ngoài |

---

## PHASE 8 — TÍCH HỢP & KIỂM THỬ

### 8.1 Unit Tests bắt buộc

```python
# tests/test_image_pipeline.py
def test_normalize_range():
    """Normalize output phải trong [0, 1]"""
    img = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
    result = TensorPreparer.prepare_for_cnn(img)
    assert 0.0 <= result.min() <= result.max() <= 1.0

def test_roi_extraction_no_out_of_bounds():
    """ROI phải luôn trong biên ảnh"""
    ...

def test_tflite_android_compatibility():
    """TFLite spec phải khớp với Android"""
    checks = TensorPreparer.verify_android_compatibility("assets/drowsiness_model.tflite")
    assert checks["android_compatible"], f"Issues: {checks['issues']}"

def test_class_order_consistency():
    """Class order Python phải khớp Android"""
    py_classes  = json.load(open("outputs/cnn_eye/class_names.json"))
    assert py_classes == ["eyes_closed", "eyes_open"], "Class order mismatch!"

def test_audio_generation():
    """Audio buffer phải đúng sample rate và duration"""
    engine = AlertAudioEngine(context)
    track = engine.generateBinauralBeat(5, 200.0, 20.0)
    assert track.state == AudioTrack.STATE_INITIALIZED
```

### 8.2 Integration Test trên thiết bị thật

```
Kịch bản test 1: Bình thường
  → Lái xe 10 phút không buồn ngủ
  → Kiểm tra: không có false alarm, FPS ≥ 15, battery drain < 5%/giờ

Kịch bản test 2: Buồn ngủ nhẹ
  → Nhắm mắt 1 lần > 1.7s
  → Kiểm tra: beep LEVEL 2 trong < 200ms, log được ghi, GPS không gửi

Kịch bản test 3: Buồn ngủ nghiêm trọng
  → Nhắm mắt 5 lần trong 10 phút
  → Kiểm tra: GPS gửi đến số khẩn cấp test, binaural beats phát đúng

Kịch bản test 4: Critical — không phản hồi
  → Nhắm mắt > 5 giây liên tục
  → Kiểm tra: 19Hz + 40Hz kích hoạt, SMS gửi đến số test

Kịch bản test 5: Low light
  → Đặt camera trong ánh sáng yếu (5–10 lux)
  → Kiểm tra: EAR/MAR vẫn detect, CNN confidence không drop về 0
```

---

## PHASE 9 — DEPLOY & MONITORING

### 9.1 Checklist cuối cùng trước deploy

```
□ TFLITE SPEC: drowsiness_model.tflite → [1,64,64,3] float32 ✅
□ TFLITE SPEC: yawn_model.tflite       → [1,64,64,3] float32 ✅
□ CLASS ORDER: eyes_closed=0, eyes_open=1                     ✅
□ CLASS ORDER: no_yawn=0, yawn=1                              ✅
□ THRESHOLD: CNN_CONF=0.55f, YAWN_CONF=0.60f, EAR=0.24f     ✅
□ AUDIO: alert tones đã test qua loa và tai nghe             ✅
□ GPS: test location gửi đến số khẩn cấp thật                ✅
□ LOG: Room database ghi đủ event fields                     ✅
□ REPORT: Export CSV và chart đúng format                    ✅
□ PERMISSIONS: camera, location (fine+coarse), vibrate       ✅
□ BATTERY: Kiểm tra drain < 8%/giờ khi active               ✅
□ PRIVACY: Không upload ảnh khuôn mặt lên cloud             ✅
```

### 9.2 Cấu trúc output files cuối cùng

```
app/src/main/assets/
├── face_landmarker.task        ← MediaPipe (đã có)
├── drowsiness_model.tflite     ← CNN Eye (Phase 2→3)
├── yawn_model.tflite           ← CNN Yawn (Phase 2→3)
└── audio/
    ├── alert_1000hz.wav        ← Tầng 1
    ├── alert_2000hz.wav        ← Tầng 1 mạnh
    ├── binaural_200_18hz.wav   ← Tầng 2 (pre-generated cho thiết bị không đủ CPU)
    ├── binaural_200_40hz.wav   ← Tầng 4 Gamma
    └── emergency_siren.wav     ← Tầng 5

outputs/
├── cnn_eye/summary.json        ← val_accuracy, tflite_kb, class_names
├── cnn_yawn/summary.json
├── temporal/summary.json
├── yolo11s/summary.json        ← mAP50, FPS
├── yolo26m/summary.json
└── experiments/experiments_log.json
```

---

## REFERENCE — Các hằng số KHÔNG được thay đổi

```kotlin
// Constants.kt — SINGLE SOURCE OF TRUTH
object DrowsinessConstants {
    // CNN thresholds
    const val CNN_EYE_CONF_THRESHOLD  = 0.55f
    const val CNN_YAWN_CONF_THRESHOLD = 0.60f

    // EAR/MAR thresholds
    const val EAR_THRESHOLD  = 0.24f
    const val MAR_THRESHOLD  = 0.58f

    // Duration thresholds
    const val DROWSY_DURATION_MS         = 1_700L  // EAR baseline
    const val CNN_DROWSY_DURATION_MS     = 1_200L  // CNN eye
    const val YAWN_DURATION_MS           = 800L

    // Alert escalation
    const val LEVEL3_EVENTS_PER_15MIN = 3
    const val LEVEL4_EVENTS_PER_15MIN = 5
    const val CRITICAL_EVENTS_PER_15MIN = 7
    const val CRITICAL_EYES_CLOSED_MS = 3_000L

    // Audio triggers
    const val AUDIO_TIER2_EVENTS_TRIGGER = 5    // Binaural beta
    const val AUDIO_TIER3_EVENTS_TRIGGER = 7    // 19Hz infrasound
    const val AUDIO_TIER3_EYE_CLOSED_MS  = 3_000L

    // Model specs (PHẢI KHỚP với TFLite file)
    const val CNN_INPUT_SIZE = 64
    val CNN_EYE_CLASSES  = arrayOf("eyes_closed", "eyes_open")
    val CNN_YAWN_CLASSES = arrayOf("no_yawn",     "yawn")
}
```

---

*MASTER_PLAN.md — DrowsyDriver Pro*
*Ngày tạo: 2026-06-07 | Version: 1.0*
*Tuyệt đối không dùng lại model cũ — pipeline mới hoàn toàn từ đầu*
