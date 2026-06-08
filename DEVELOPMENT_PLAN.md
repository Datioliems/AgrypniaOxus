# 🗺️ KẾ HOẠCH PHÁT TRIỂN HOÀN THIỆN — DrowsyDriver (IS54A)

> Mục tiêu: từ trạng thái hiện tại → sản phẩm hoàn chỉnh nộp được + demo được + (tùy chọn) deploy thật.

---

## 0. HIỆN TRẠNG — Những gì ĐÃ CÓ ✅

| Hạng mục | Trạng thái | Số liệu |
|----------|-----------|---------|
| **Android app** | ✅ Build được (APK 72MB) | CameraX + MediaPipe + TFLite |
| CNN mắt | ✅ `drowsiness_model.tflite` | 97.44% |
| CNN ngáp (CBAM) | ✅ `yawn_cbam.tflite` | 98.64% |
| CNN ngáp (mới train) | ✅ `yawn_model.tflite` | 97.46% |
| Temporal model | ✅ `temporal_model.tflite` | — |
| **YOLO/Transformer (Colab)** | ✅ 4 model train xong | YOLO11s **73.5%** mAP50-95 (cao nhất) |
| Pipeline 10 bước | ⚠️ bước 1-2 chạy, 3 gate (data đã crop) | `tools/step1-10` |
| **Streamlit demo** | ✅ 3 màn hình, đã fix mediapipe | Dashboard/Alert/Analytics |
| Công cụ cắt frame | ✅ `colab_extract_frames.ipynb` | UI + progress |
| Kit nhúng YOLO | ✅ export + `YoloDetector.kt` + doc | chưa wire vào app |
| Dataset | ✅ MRL eye, face-crop eye, yawn, YOLO 6-class | |

---

## 1. ĐỊNH NGHĨA "HOÀN THIỆN" (Definition of Done)

- [ ] **AI**: model chốt + bảng metrics đầy đủ (Acc, Recall, F1, mAP), so sánh CNN vs Transformer
- [ ] **Android**: chạy ổn trên **điện thoại thật**, cảnh báo (âm thanh + overlay) hoạt động đúng
- [ ] **Streamlit**: 3 màn hình chạy + **deploy Cloud** + có link
- [ ] **Dữ liệu thật**: có video/ảnh tự quay → đánh giá điều kiện thực tế (robustness)
- [ ] **Báo cáo**: điền đủ tên/MSSV, kết quả, ảnh, trích nguồn dataset đúng
- [ ] **Demo**: video demo + slide thuyết trình

---

## 2. LỘ TRÌNH 6 GIAI ĐOẠN

### 🟦 Phase A — Chốt AI core (1-2 ngày)
- [ ] Train lại **CNN mắt** trên Colab GPU (`train_cnn_eye.ipynb`) → tải `drowsiness_model.tflite`
- [ ] Chạy `colab_compare_all.ipynb` → bảng mAP công bằng 4 model + biểu đồ
- [ ] (Tùy chọn) chạy `colab_ensemble_transformer.ipynb` → xem ensemble có vượt 73.5%
- [ ] Chốt: model nào dùng cho Android (CNN), model nào để so sánh (YOLO/DETR)
- **Output**: bảng metrics cuối + biểu đồ cho báo cáo

### 🟩 Phase B — Dữ liệu thật + Robustness (1-2 ngày) ⭐ điểm cộng lớn
- [ ] Tự quay **video lái xe / ngồi trước camera** (ngày + tối, có/không kính, ngáp, nhắm mắt)
- [ ] Cắt frame bằng `colab_extract_frames.ipynb` (2-5 fps)
- [ ] Thêm config `real_eye`/`real_yawn` với `already_cropped=False` → pipeline tự crop
- [ ] Chạy pipeline 10 bước trên data thật → train hoặc test chéo
- [ ] So sánh: model trên data lab vs data thật → viết "robustness gap" trong báo cáo
- **Output**: dataset thật + số liệu test thực tế

### 🟨 Phase C — Android hoàn thiện (1-2 ngày)
- [ ] Cài APK lên **điện thoại thật** (USB debugging + `adb install`)
- [ ] Test: overlay EAR/MAR/FPS, cảnh báo DROWSY/YAWNING, âm thanh
- [ ] Tinh chỉnh ngưỡng (`DrowsinessConstants`) cho khớp thực tế
- [ ] (Tùy chọn P2) Nhúng YOLO theo `YOLO_ANDROID_INTEGRATION.md`
- **Output**: APK chạy mượt trên điện thoại + clip quay màn hình

### 🟧 Phase D — Streamlit demo (0.5-1 ngày)
- [ ] Chạy local → chụp 3 màn hình cho báo cáo
- [ ] Deploy Streamlit Cloud (`STREAMLIT_DEPLOY.md`): copy `requirements.txt`, push GitHub
- [ ] Lấy link public để chèn vào báo cáo/slide
- **Output**: link demo web + 3 ảnh màn hình

### 🟥 Phase E — Đánh giá & so sánh (0.5-1 ngày)
- [ ] Bảng so sánh: EAR/MAR rules vs CNN vs CNN+temporal (Acc/Recall/F1/FPS)
- [ ] Bảng CNN vs Transformer (mAP, params, tốc độ, epochs)
- [ ] Confusion matrix + threshold analysis
- [ ] Nhấn mạnh **Recall/F1 lớp nguy hiểm** (an toàn > accuracy)
- **Output**: chương Kết quả của báo cáo

### 🟪 Phase F — Báo cáo + Demo + Slide (1-2 ngày)
- [ ] Điền `BaoCao_IS54A_*.docx`: tên/MSSV, mục lục, kết quả, ảnh, nguồn dataset đúng
- [ ] Quay **video demo** (app chạy + cảnh báo)
- [ ] Làm **slide thuyết trình** (8 phút) — đã có `PRESENTATION_8_MIN_SCRIPT.md`
- **Output**: bộ nộp hoàn chỉnh

---

## 3. ƯU TIÊN

| Mức | Hạng mục | Vì sao |
|-----|----------|--------|
| **P0 — BẮT BUỘC** | APK chạy điện thoại, báo cáo, demo video, bảng metrics | Yêu cầu nộp IS54A |
| **P1 — NÊN CÓ** | Data thật + robustness, Streamlit deploy, so sánh CNN/Transformer | Điểm cộng, phản biện tốt |
| **P2 — MỞ RỘNG** | Nhúng YOLO Android, ensemble, quantize int8, NNAPI | "Hướng phát triển" trong báo cáo |

> Nếu gấp deadline: làm đủ **P0** trước (app + báo cáo + demo), P1/P2 ghi vào "hướng mở rộng".

---

## 4. THỜI GIAN GỢI Ý

| Kịch bản | Cách phân bổ |
|----------|--------------|
| **Gấp (3-4 ngày)** | P0 toàn bộ: A(chốt model có sẵn) → C(test app) → E,F(báo cáo+demo). Bỏ B, D-deploy |
| **Đủ (7-10 ngày)** | A → B → C → D → E → F tuần tự (đúng lộ trình trên) |
| **Thoải mái (2 tuần+)** | Thêm P2: YOLO Android + ensemble + tối ưu mobile |

---

## 5. RỦI RO & GIẢM THIỂU

| Rủi ro | Giảm thiểu |
|--------|-----------|
| Colab GPU hết giờ / ngắt | Lưu checkpoint vào Drive, giảm epochs (đã làm) |
| Mediapipe/thư viện đổi API | Pin version trong requirements; đã fix sang Tasks API |
| App giật trên điện thoại yếu | Throttle inference, dùng model nhẹ (CNN 64 thay YOLO 640) |
| Data thật ít/thiếu đa dạng | Quay nhiều điều kiện; augment; merge với dataset public |
| Console Windows lỗi UTF-8 | Luôn đặt `PYTHONUTF8=1` (đã ghi chú) |

---

## 6. CHECKLIST NGHIỆM THU CUỐI

**AI**
- [ ] CNN mắt + ngáp có metrics (Acc/Recall/F1)
- [ ] So sánh CNN vs Transformer (mAP)
- [ ] (P1) Test trên data thật

**Android**
- [ ] APK cài + chạy điện thoại thật
- [ ] Overlay + cảnh báo hoạt động
- [ ] Ngưỡng tinh chỉnh hợp lý

**Streamlit**
- [ ] 3 màn hình chạy
- [ ] (P1) Deploy Cloud + link

**Tài liệu**
- [ ] Báo cáo điền đủ + nguồn dataset ĐÚNG (MRL ≠ face-crop)
- [ ] Video demo
- [ ] Slide 8 phút

---

## 7. HÀNH ĐỘNG NGAY (Top 5)
1. **Train CNN mắt trên Colab** → tải tflite (Phase A)
2. **Chạy `colab_compare_all.ipynb`** → bảng so sánh (Phase A/E)
3. **Cài APK lên điện thoại** + test cảnh báo (Phase C)
4. **Quay 1-2 video thật** → cắt frame (Phase B)
5. **Chụp 3 màn hình Streamlit** + deploy Cloud (Phase D)

---
*File này là bản đồ tổng. Cập nhật checklist `[x]` khi hoàn thành từng mục.*
