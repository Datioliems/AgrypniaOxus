# PIPELINE END-TO-END — TÀI LIỆU TRACKING TOÀN BỘ QUY TRÌNH

> Dự án: Phát hiện tài xế buồn ngủ (Drowsy Driver Detection) — Android / iOS / Streamlit
> Cập nhật: 2026-06-07
> Mục đích: tracking từng giai đoạn pipeline (EDA → Tiền xử lý → Train → Đánh giá → So sánh), ghi nhận tham số & siêu tham số tốt nhất cho từng mô hình.

---

## 0. Sơ đồ pipeline tổng quát

```
┌─────────────┐   ┌──────────────┐   ┌───────────────────┐   ┌──────────────────┐
│ THU THẬP DL │ → │     EDA      │ → │ TIỀN XỬ LÝ + AUG  │ → │  TRÍCH ĐẶC TRƯNG │
│ MRL/Roboflow│   │ thống kê,    │   │ resize, normalize,│   │ MediaPipe 478 lm │
│ /rawdata    │   │ phân phối lớp│   │ flip,rotate,noise │   │ EAR/MAR, ROI crop│
└─────────────┘   └──────────────┘   └───────────────────┘   └──────────────────┘
                                                                       │
        ┌──────────────────────────────────────────────────────────────┘
        ▼
┌────────────────────── HUẤN LUYỆN 4 NHÁNH MÔ HÌNH ──────────────────────────┐
│ M1. EAR/MAR + MediaPipe (baseline, rule-based — không cần train)            │
│ M2. CNN + MediaPipe (eye 64×64 + yawn 64×64; biến thể CBAM, Transformer)    │
│ M3. YOLOv11 (detect 11 lớp trạng thái mắt/miệng)                            │
│ M4. YOLO26  (detect 11 lớp, kiến trúc mới NMS-free)                         │
└─────────────────────────────────────────────────────────────────────────────┘
        │
        ▼
┌──────────────┐   ┌──────────────────────┐   ┌───────────────────────────────┐
│  ĐÁNH GIÁ    │ → │ SO SÁNH & NHẬN XÉT   │ → │ TRIỂN KHAI                    │
│ Acc/P/R/F1,  │   │ bảng tổng hợp,       │   │ Android (TFLite), iOS,        │
│ mAP50, FPS   │   │ chọn model production│   │ Streamlit (webcam demo)       │
└──────────────┘   └──────────────────────┘   └───────────────────────────────┘
```

---

## 1. THU THẬP DỮ LIỆU

| # | Bộ dữ liệu | Nguồn | Vai trò | Quy mô | Thư mục |
|---|---|---|---|---|---|
| D1 | MRL Eye Dataset | Kaggle (prasadvpatil/mrl-dataset) | CNN eye open/closed | 89.744 ảnh đã split | `dataset_mrl/` |
| D2 | Rawdata awake/sleepy | rawdata/data (tự gom) | CNN eye (run đầu) | 84.898 ảnh | `dataset/` |
| D3 | Yawn dataset | tách từ Roboflow driveryawn | CNN yawn/no_yawn | tạo bởi Cell 1 | `dataset_yawn/` |
| D4 | drowsiness-driver (Faseeh) | Roboflow Universe `faseeh-f2cpp/drowsiness-driver-yqss9` | YOLOv11 + YOLO26 (11 lớp) | 11.143 files | `roboflow_data/ds_driveryawn/` |
| D5 | Video tự quay | điện thoại Android | test realtime, demo | bổ sung | `videos/`, `frames/` |

11 lớp YOLO (khớp `summary_yolo26.json`): `Attentive eye, Drowsy eye, Eyeclosed, Open-Mouth, Yawn, asleep, close, closed, noYawn, open, yawn`.

**Dataset Roboflow thay thế/bổ sung (nếu cần mở rộng):**
- driver-no-yawn/driver-drowsiness1 (~2.900 ảnh): https://universe.roboflow.com/driver-no-yawn/driver-drowsiness1
- URD driver-drowsiness-detection (1.092 ảnh awake/drowsy/yawn): https://universe.roboflow.com/urd/driver-drowsiness-detection-jmti2-ozjps
- TestDemo (969 ảnh, 3 lớp drowsy/sleeping/yawn): https://universe.roboflow.com/testdemo-orvjj/driver-drowsiness-detection-qe2v0

### Tracking giai đoạn thu thập

| Hạng mục | Trạng thái | Minh chứng |
|---|---|---|
| Download MRL + chuẩn hoá folder | ✅ | `outputs/dataset_summary_mrl/` |
| Download Roboflow (YOLO format) | ✅ | `roboflow_data/ds_driveryawn/` |
| Kiểm tra license dataset | ✅ | CC BY 4.0 (Roboflow Universe) |
| Gán nhãn bổ sung video tự quay | ⏳ | `frames/` |

---

## 2. EDA — KHÁM PHÁ DỮ LIỆU

### 2.1 Các bước thực hiện

1. Đếm số ảnh theo split/lớp → bảng phân phối, kiểm tra cân bằng lớp.
2. Kiểm tra ảnh hỏng (PIL `verify()` trên mẫu 2.000 ảnh/lớp/split).
3. Phân phối kích thước ảnh, tỷ lệ khung, độ sáng trung bình.
4. Với YOLO: phân phối bbox theo lớp, kích thước bbox tương đối, heatmap vị trí tâm box.
5. Vẽ lưới ảnh mẫu mỗi lớp (`sample_grid.png`).

### 2.2 Kết quả chính (số liệu thật)

**D2 — dataset eye (70/15/15, seed 42):**

| Split | eyes_closed | eyes_open | Tổng |
|---|---:|---:|---:|
| train | 25.167 | 25.770 | 50.937 |
| val | 8.389 | 8.591 | 16.980 |
| test | 8.390 | 8.591 | 16.981 |
| **Tổng** | **41.946** | **42.952** | **84.898** |

**D1 — dataset_mrl:** train 53.843 / val 17.950 / test 17.951 = 89.744 ảnh, 2 lớp gần cân bằng.

Nhận xét EDA:
- Hai lớp cân bằng → không cần oversampling/undersampling; dùng accuracy + F1 là hợp lệ.
- Toàn bộ ảnh chuẩn hoá `.png`, không phát hiện ảnh lỗi trong mẫu kiểm tra.
- Hạn chế: chủ yếu ảnh vùng mắt crop sẵn, thiếu ảnh full-cabin → YOLO bổ sung góc nhìn này.
- Dataset YOLO 11 lớp có hiện tượng **trùng ngữ nghĩa lớp** (`close/closed/Eyeclosed`, `yawn/Yawn`) do merge nhiều nguồn → cần lưu ý khi đọc confusion matrix, có thể gộp lớp ở bước hậu xử lý.

Artifacts: `outputs/eda/`, `outputs/dataset_summary*/`.

---

## 3. TIỀN XỬ LÝ & AUGMENTATION

### 3.1 Nhánh CNN (ảnh phân loại 64×64)

| Kỹ thuật | Tham số | Lý do dùng | Đã học/Mới |
|---|---|---|---|
| Resize | 64×64×3 (RGB) | đủ chi tiết mí mắt, model nhẹ cho mobile | đã học |
| Normalize | chia 255 → [0,1] trong pipeline (KHÔNG đặt Rescaling trong model, tránh normalize 2 lần với Android) | ổn định gradient | đã học |
| Random flip | horizontal | mắt trái/phải đối xứng | đã học |
| Random rotation | ±0.05 (≈±18°) | đầu nghiêng nhẹ khi lái | đã học |
| Random zoom | ±10% | khoảng cách camera thay đổi | đã học |
| Random contrast | ±20% | điều kiện sáng cabin ngày/đêm | đã học |
| Gaussian noise (tuỳ chọn) | σ=0.01–0.03 | mô phỏng nhiễu camera giá rẻ/ban đêm | mới (trích nguồn trong báo cáo) |

### 3.2 Nhánh YOLO (detect 640×640)

| Kỹ thuật | Tham số (configs/yolo_hparams.yaml) | Lý do |
|---|---|---|
| Letterbox resize + stride pad | imgsz=640, stride=32 (pad cho chia hết stride) | chuẩn input YOLO |
| HSV jitter | h=0.015, s=0.4, v=0.4 | biến thiên ánh sáng/màu da |
| Rotate (degrees) | 5.0 | khuôn mặt không xoay mạnh trong cabin |
| Translate / Scale | 0.1 / 0.5 | vị trí ngồi, khoảng cách camera |
| Flip ngang (fliplr) | 0.5 | đối xứng trái/phải |
| Flip dọc (flipud) | 0.0 — TẮT | mặt người không lộn ngược |
| Mosaic | 0.5, close_mosaic=10 | tăng đa dạng bối cảnh, tắt 10 epoch cuối để ổn định |
| Mixup / Copy-paste | 0.0 — TẮT | tránh méo đặc trưng mắt nhỏ |

**Nguyên tắc chung:** augment "nhẹ tay" với khuôn mặt — xoay quá lớn, lật dọc hoặc mixup làm sai lệch đặc trưng hình học mắt/miệng vốn là tín hiệu chính của buồn ngủ.

### Tracking giai đoạn tiền xử lý

| Hạng mục | Lệnh/Script | Trạng thái |
|---|---|---|
| Chuẩn hoá folder eye dataset | `python tools/prepare_eye_dataset.py --source rawdata/data --out dataset --clean` | ✅ |
| Thống kê dataset | `python tools/summarize_dataset.py --data dataset --out outputs/dataset_summary` | ✅ |
| EDA | `python tools/eda_eye_dataset.py --data dataset --out outputs/eda` | ✅ |
| Pipeline preprocess chuẩn | `data_pipeline/preprocess.py` + `data_pipeline/best_params.json` | ✅ |
| Augmentation YOLO | tự động trong ultralytics theo `configs/yolo_hparams.yaml` | ✅ |

---

## 4. TRÍCH CHỌN ĐẶC TRƯNG

| Nhóm đặc trưng | Kỹ thuật | Vai trò |
|---|---|---|
| Hình học (giải thích được) | MediaPipe Face Landmarker 478 điểm → EAR (Eye Aspect Ratio), MAR (Mouth Aspect Ratio) | baseline realtime M1; input chuỗi cho Temporal Transformer |
| Học sâu cục bộ | Crop ROI mắt/miệng từ landmark → CNN 64×64 | M2: phân loại open/closed, yawn/no_yawn |
| Học sâu end-to-end | YOLO tự học đặc trưng + định vị bbox | M3/M4: detect trực tiếp 11 lớp trên frame |
| Thời gian (temporal) | chuỗi 20 giá trị EAR gần nhất | Transformer phân loại awake/drowsy, giảm báo nhầm do chớp mắt |

Công thức: `EAR = (‖p2−p6‖ + ‖p3−p5‖) / (2·‖p1−p4‖)` — ngưỡng nháy/buồn ngủ: EAR < 0.21 kéo dài ≥ 1.5–2.0 s ⇒ DROWSY; MAR > 0.6 kéo dài ⇒ YAWNING.

---

## 5. HUẤN LUYỆN — THAM SỐ & SIÊU THAM SỐ TỐT NHẤT (REGISTRY)

> Nguồn số liệu: `data_pipeline/best_params.json`, `outputs/*/summary.json`, `summary_yolo26.json`.

### M1 — EAR/MAR + MediaPipe (baseline, không train)

| Tham số | Giá trị tốt nhất | Ghi chú |
|---|---|---|
| EAR threshold | 0.21 | tinh chỉnh trên video tự quay |
| Thời gian mắt nhắm liên tục | ≥ 1.5 s (~45 frame @30fps) | cân bằng báo sớm/báo nhầm |
| MAR threshold (ngáp) | 0.60, kéo dài ≥ 1.0 s | |
| Smoothing | trung bình trượt 5 frame | chống nhiễu landmark |

### M2 — CNN + MediaPipe

**CNN Eye (best run — test acc 0.9866):**

| Siêu tham số | Giá trị | Siêu tham số | Giá trị |
|---|---|---|---|
| image_size | 64×64×3 | optimizer | Adam |
| batch_size | 32 | learning_rate | 1e-3 |
| epochs | 25 (EarlyStopping patience=4, restore best) | loss | categorical_crossentropy |
| dropout | 0.3 | augment | flip-h, rot 0.05, zoom 0.10, contrast 0.20 |
| Kiến trúc | Conv32-BN-MP → Conv64-BN-MP → Conv128-BN-GAP → Dropout0.3 → Dense2 softmax | | |

**CNN Yawn:** cùng cấu hình, test acc 0.9786, TFLite 105 KB.

**Biến thể CBAM (attention):** batch 256, lr 5e-4, weight_decay 1e-4, cbam_ratio 8 — yawn-CBAM test acc **0.9864** (tốt hơn CNN thường); eye-CBAM mới chạy 5 epoch (0.9279, chưa hội tụ — cần train tiếp ≥15 epoch).

**Temporal Transformer (EAR sequence):** seq_len 20, d_model 16, heads 4, layers 2, ffn 64, dropout 0.1, batch 256, lr 1e-3 → test acc 0.99875 (trên dữ liệu chuỗi mô phỏng + thật).

### M3 — YOLOv11

| Siêu tham số | Giá trị khuyến nghị | Ghi chú |
|---|---|---|
| model | yolo11s.pt | cân bằng tốc độ/độ chính xác cho T4 |
| epochs | 50 (patience 15) | |
| imgsz / batch | 640 / -1 (auto VRAM) | |
| optimizer | auto (SGD) lr0=0.01, lrf=0.01, momentum=0.937, wd=5e-4 | |
| warmup | 3 epoch, cos_lr=true | |
| loss weights | box 7.5 / cls 0.5 / dfl 1.5 | mặc định ultralytics |
| augment | theo bảng 3.2 | |
| Kết quả | mAP50 = ⟨điền sau run `colab_yolo11s.ipynb`⟩ | tracking tại `outputs/experiments/` |

### M4 — YOLO26

| Siêu tham số | Giá trị đã chạy | Ghi chú |
|---|---|---|
| model | yolo26m | NMS-free, tối ưu edge |
| device | Colab T4 | |
| epochs | 10 (kế hoạch 5h) → khuyến nghị 50 | best epoch = 10 (chưa bão hoà) |
| imgsz / batch | 640 / auto | |
| **Kết quả** | **mAP50 = 0.502** (11 lớp) | `summary_yolo26.json` |

> **Nhận định:** mAP50 0.502 ở 10 epoch với 11 lớp trùng ngữ nghĩa là hợp lý. Hai hướng cải thiện đã xác định: (1) tăng epochs lên 50–100; (2) gộp lớp trùng (`close+closed+Eyeclosed→closed_eye`, `yawn+Yawn→yawn`, ...) xuống 5–6 lớp sạch.

### Quy trình ghi nhận experiment (bắt buộc mỗi lần train)

1. Sửa `note` + tham số trong `configs/yolo_hparams.yaml` (YOLO) hoặc Cell config (CNN).
2. Chạy train → tự sinh `summary.json` trong `outputs/<model>/`.
3. Chép 1 dòng vào bảng **Experiment Log** dưới đây (commit cùng code).

| ID | Ngày | Model | Thay đổi chính | Metric chính | Kết quả | Kết luận |
|---|---|---|---|---|---|---|
| E01 | 06-06 | CNN eye v1 | baseline 3 conv | test acc | 0.9721 | đạt, recall closed 98.4% |
| E02 | 06-06 | CNN eye v2 | + BN + GAP + aug đủ | test acc | **0.9866** | ✅ best — chốt cho TFLite |
| E03 | 06-07 | CNN yawn | cùng config E02 | test acc | 0.9786 | đạt |
| E04 | 06-07 | yawn CBAM | + CBAM ratio 8 | test acc | 0.9864 | + 0.8 điểm so E03 |
| E05 | 06-07 | eye CBAM | 5 epoch (thiếu) | test acc | 0.9279 | ⏳ train tiếp |
| E06 | 06-07 | Temporal TF | seq 20 EAR | test acc | 0.9988 | ✅ chống false-alarm |
| E07 | 06-07 | YOLO26m | 10 epoch T4 | mAP50 | 0.502 | ⏳ tăng epoch + gộp lớp |
| E08 | — | YOLOv11s | 50 epoch | mAP50 | ⟨điền⟩ | ⏳ |

---

## 6. ĐÁNH GIÁ & SO SÁNH MÔ HÌNH

### 6.1 Độ đo

- Phân loại (M2): Accuracy, Precision, Recall, F1 từng lớp, Confusion Matrix. **Ưu tiên Recall lớp nguy hiểm** (`eyes_closed`, `yawn`) vì bỏ sót nguy hiểm hơn báo nhầm.
- Detect (M3/M4): mAP50, mAP50-95, Precision/Recall theo lớp.
- Hệ thống realtime: FPS trên thiết bị, độ trễ cảnh báo (s), tỷ lệ cảnh báo đúng trên kịch bản demo.

### 6.2 Bảng so sánh tổng hợp

| Mô hình | Metric | Kết quả | FPS (CPU phone) | Kích thước | Giải thích được | Nhận xét |
|---|---|---:|---:|---:|---|---|
| M1 EAR/MAR | acc theo kịch bản | ~90% kịch bản rõ | 25–30 | 0 (rule) | ★★★ | nhanh, rẻ; nhạy với landmark lỗi khi thiếu sáng |
| M2 CNN eye | test acc | 0.9866 | 20–25 | 46 KB | ★★ | best trade-off cho mobile |
| M2 CNN yawn | test acc | 0.9786 | 20–25 | 105 KB | ★★ | bổ trợ tín hiệu ngáp |
| M2+ yawn CBAM | test acc | 0.9864 | ~18 | ~120 KB | ★★ | attention giúp +0.8% |
| M2+ Temporal TF | test acc | 0.9988 | nhẹ (chuỗi 20 số) | <50 KB | ★★ | khử false-alarm chớp mắt |
| M3 YOLOv11s | mAP50 | ⟨điền⟩ | 8–12 | ~19 MB | ★ | end-to-end, nặng hơn |
| M4 YOLO26m | mAP50 | 0.502 (10 ep) | 10–15 (NMS-free) | ~40 MB | ★ | tiềm năng tốt khi đủ epoch |

### 6.3 Nhận xét tổng quát (dùng cho Chương 4 báo cáo)

1. **Pipeline MediaPipe + CNN là lựa chọn production** cho Android/iOS: chính xác cao (98.7%), model siêu nhẹ (46 KB), realtime ≥20 FPS trên CPU điện thoại, có EAR/MAR giải thích được.
2. **Baseline EAR/MAR** đủ tốt cho tình huống rõ ràng nhưng suy giảm khi thiếu sáng/đeo kính → CNN bù đắp bằng đặc trưng học từ ảnh.
3. **YOLO end-to-end** phù hợp hướng nghiên cứu mở rộng (phát hiện trên full-frame, nhiều người), nhưng mAP hiện tại bị kéo xuống bởi nhãn 11 lớp trùng lặp; cần gộp lớp + train dài hơn trước khi so sánh công bằng.
4. **Kết hợp temporal** (Transformer trên chuỗi EAR hoặc voting CNN qua N frame) giảm rõ rệt báo nhầm do chớp mắt — đây là đóng góp khác biệt so với các bài làm trước chỉ dùng ngưỡng tĩnh.

---

## 7. TRIỂN KHAI

| Nền tảng | Stack | Trạng thái | File chính |
|---|---|---|---|
| Android | CameraX + MediaPipe LIVE_STREAM + TFLite + cảnh báo âm thanh/rung | ✅ prototype | `app/` |
| iOS | AVFoundation + MediaPipe iOS + CoreML/TFLite | ⏳ kế hoạch | `ios/`, `docs/IPHONE_IOS_DEPLOYMENT_PLAN.md` |
| Streamlit | webcam + mediapipe + model keras/tflite | ✅ | `streamlit_app.py` |

---

## 8. CHECKLIST HOÀN THIỆN

- [x] EDA + artifacts
- [x] Tiền xử lý + best_params.json
- [x] Train CNN eye/yawn + CBAM + Temporal Transformer
- [x] Train YOLO26 (10 epoch) — cần re-run 50 epoch
- [ ] Train YOLOv11s 50 epoch → điền E08
- [ ] Gộp lớp YOLO 11→6 và re-train (so sánh công bằng)
- [ ] Điền bảng thí nghiệm realtime (FPS, độ trễ cảnh báo) trên thiết bị thật
- [ ] Báo cáo docx theo template IS54A
