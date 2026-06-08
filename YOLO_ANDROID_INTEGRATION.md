# Nhúng YOLO vào Android (export TFLite + post-process Kotlin)

> Đây là **track nâng cao**. App hiện tại chạy CNN classifier (64×64) rất nhẹ.
> YOLO là **detector** (640×640, ra nhiều box) → nặng hơn ~10× và phải tự decode + NMS trong Kotlin.

## Vì sao YOLO khó hơn CNN trong Android?

| | CNN (hiện tại) | YOLO (track này) |
|---|---|---|
| Input | 64×64×3 | 640×640×3 |
| Output | 1 nhãn / ảnh | nhiều box (cx,cy,w,h + score) |
| Hậu xử lý | argmax đơn giản | **decode + NMS + gỡ letterbox** |
| Tốc độ điện thoại | rất nhanh | nặng → cân nhắc chạy mỗi vài frame |

---

## Quy trình 5 bước

### Bước 1 — Export YOLO (.pt) → TFLite
Chạy trên Colab/local (có `ultralytics`):
```bash
# Float32 (khớp YoloDetector.kt — KHUYÊN dùng trước cho dễ)
python tools/export_yolo_tflite.py --weights yolo11s_best.pt

# YOLO26 (NMS-free → parse dễ hơn trong Kotlin)
python tools/export_yolo_tflite.py --weights yolo26s_best.pt
```
Kết quả: `app/src/main/assets/yolo_detector.tflite` + in ra mảng `labels` cho Kotlin.

> ⚠️ `--int8` cho model nhẹ/nhanh hơn ~4× nhưng input/output là int8 → phải thêm code dequant. Làm float32 chạy được trước đã.

### Bước 2 — Đưa vào Android
- File `yolo_detector.tflite` đã nằm ở `app/src/main/assets/` (script tự copy).
- Copy mảng `labels` (in ở bước 1) vào `YoloDetector.kt` cho đúng thứ tự index.

### Bước 3 — Dùng `YoloDetector.kt` (đã tạo sẵn)
```kotlin
val detector = YoloDetector(context)            // nạp yolo_detector.tflite
val detections = detector.detect(faceBitmap)    // List<Detection>
for (d in detections) {
    Log.d("YOLO", "${d.label} ${d.score} ${d.box}")
}
```
Class này đã lo: letterbox 640, /255, decode box, **NMS**, gỡ letterbox về toạ độ ảnh gốc.

### Bước 4 — Suy ra trạng thái buồn ngủ từ 6 class
```kotlin
val dets = detector.detect(frameBitmap)
val eyesClosed = dets.count { it.label.startsWith("close_eye") }   // 0,1,2
val eyesOpen   = dets.count { it.label.startsWith("open_eye") }
val yawning    = dets.any  { it.label == "yawn" }

val bothClosed = eyesClosed >= 2
// → đưa vào DrowsinessAnalyzer / PERCLOS temporal như logic EAR/MAR hiện có
```

### Bước 5 — Gắn vào `MainActivity.kt`
2 lựa chọn:
- **Thay** CNN bằng YOLO: trong vòng phân tích frame, gọi `detector.detect()` thay cho `TfliteDrowsinessClassifier`.
- **Chạy song song**: YOLO mỗi N frame (vd 5 frame/lần) để đỡ nặng, CNN/EAR mỗi frame.

```kotlin
// ví dụ throttle YOLO
if (frameCount % 5 == 0) {
    val dets = detector.detect(bitmap)
    updateDrowsyState(dets)
}
```

---

## Lưu ý hiệu năng (quan trọng cho điện thoại)
1. **YOLO 640 nặng** → dùng `yolo11n` (nano) thay `yolo11s` nếu giật.
2. **Chạy throttle** (mỗi 3–5 frame) thay vì mỗi frame.
3. **Bật NNAPI/GPU delegate** trong `Interpreter.Options()` để tăng tốc.
4. **YOLO26 NMS-free** → bỏ được hàm `nms()` trong Kotlin (output đã lọc) → nhẹ hơn.
5. Cân nhắc `--int8` + NNAPI cho bản release (cần thêm code dequant).

## So sánh để viết báo cáo
| Phương án | Ưu | Nhược |
|-----------|-----|-------|
| CNN 64×64 (hiện tại) | siêu nhẹ, nhanh, đủ tốt | chỉ phân loại, không định vị |
| YOLO trong Android | định vị mắt/miệng + 6 class 1 model | nặng, phải NMS, tốn pin |
| **Khuyến nghị** | Giữ CNN làm bản chính, YOLO là **hướng mở rộng** demo | |

## File liên quan
- `tools/export_yolo_tflite.py` — export .pt → .tflite
- `app/src/main/java/com/ai2026/drowsydriver/YoloDetector.kt` — detector + NMS
- weights: `yolo11s_best.pt` / `yolo26s_best.pt` (từ Colab, trên Drive)
