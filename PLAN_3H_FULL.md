# ⏱️ KẾ HOẠCH 3H — PIPELINE HOÀN CHỈNH (8 bước)

> Ràng buộc: **3h tối đa** · commit lên **nhánh mới `Datio-*`** · mọi code Python là **`.ipynb` chạy trong VS Code** · không thiếu phần nào · làm xong phải kiểm tra lại.

## 🎯 Thực tế (đọc trước)
**~70% đã có sẵn** từ các phiên trước → 3h này là **HOÀN THIỆN + TÍCH HỢP + bổ sung phần thiếu**, không train lại từ đầu (train lại 6 model from scratch là bất khả thi trong 3h).

| Đã có ✅ | Cần làm trong 3h ⬜ |
|----------|---------------------|
| CNN eye/yawn/cbam, YOLO11/26, RT-DETR, RF-DETR | Model **kết hợp mắt+ngáp** |
| Streamlit 3 màn, Android app (CNN-primary) | **PWA**, **app thứ 2** (real-data) |
| compare_all, ensemble notebook | **Âm thanh binaural beats** (theo 1.3) |
| Frame extract, real-data train notebook | **Xử lý mặt nghiêng** |
| EAR/MAR baseline | Bảng so sánh + **viết lại lý thuyết** |

---

## 📋 8 BƯỚC ↔ TRẠNG THÁI ↔ CÁCH LÀM

### Bước 1 — Tìm dataset (CNN trước, YOLO sau)
- ✅ CNN: MRL Eye (84k) + Kaggle yawn — đã có
- ✅ YOLO: Roboflow `datio_yolo` 6-class — đã có
- ⬜ **Bổ sung**: tìm dataset có **mặt nghiêng** (side-profile) cho cả CNN + YOLO → notebook `01_datasets.ipynb`

### Bước 2 — Tiền xử lý (chạy được với data tự quay)
- ✅ Pipeline 10 bước + `colab_realdata_train.ipynb` (tự crop từ video thật)
- ⬜ Gộp thành `02_preprocess.ipynb` (VS Code) — hỗ trợ cả dataset public + video tự quay

### Bước 3 — Feature Engineering (tham số, epochs, đánh giá)
- ⬜ `03_feature_engineering.ipynb`: bảng tham số (lr, batch, epochs, augment) + lý do chọn, theo từng model

### Bước 4 — Train 6 mô hình + baseline
| Model | Trạng thái |
|-------|-----------|
| CNN mắt | ✅ 97.44% |
| CNN ngáp | ✅ 98.64% |
| **Kết hợp mắt+ngáp** | ⬜ build `04d_combined.ipynb` (2-head CNN hoặc gộp logic) |
| YOLOv11 | ✅ 73.5% mAP50-95 |
| YOLO26 | ✅ 71.4% |
| **Ensemble (YOLO26+Transformer)** | ✅ notebook có |
| Baseline EAR/MAR (MediaPipe crop) | ✅ |

### Bước 5 — Đánh giá + bảng so sánh
- ✅ `colab_compare_all.ipynb`
- ⬜ `05_evaluation.ipynb`: bảng Accuracy/Recall/F1 (CNN) + mAP (YOLO) + tốc độ + tham số + **kết luận**

### Bước 6 — Tinh chỉnh model tốt nhất + đóng gói
- ⬜ Chọn model tốt nhất → tinh chỉnh ngưỡng → export TFLite → đóng gói `06_finalize.ipynb`

### Bước 7 — Deploy 2 hệ + Android
- ✅ Streamlit (3 màn) · Android (CNN-primary)
- ⬜ **PWA** (progressive web app) — bọc giao diện Stitch
- ⬜ **Âm thanh binaural beats theo CK AI 1.3** (xem dưới) — cả Android + Streamlit/PWA

### Bước 8 — Video thật → train nền → app thứ 2
- ⬜ Quay video → `colab_realdata_train.ipynb` chạy **nền** → tflite mới
- ⬜ Build **app thứ 2** (`app-realdata-debug.apk`) — **KHÔNG ghi đè** bản gốc

---

## 🔊 CƠ CHẾ ÂM THANH (theo CK AI.docx 1.3.1.6 + 1.3.1.7)

**Cơ sở khoa học:** Binaural/monaural beats dải **beta thấp 13–21 Hz** kích thích tỉnh táo (Moessinger 2021, PLOS ONE). Cảnh báo **leo thang đa giác quan** (Beles 2024): thị giác → âm thanh → rung.

**Triển khai (cả Android + web):**
| Tier | Điều kiện | Âm thanh |
|------|-----------|----------|
| 0 | Tỉnh táo | im lặng |
| 1 | EYES_CLOSED ngắn | beep nhẹ + overlay vàng |
| 2 | DROWSY | **binaural beat beta 18Hz** (carrier 200Hz, L/R lệch 18Hz) + rung |
| 3 | DROWSY kéo dài / không phản hồi | còi to + rung mạnh + "Tìm trạm dừng" |
| 4 | Lái > 2h (nghiên cứu Wang 2014) | nhắc nghỉ 15-30 phút |

→ Sinh binaural beat bằng tone L=200Hz, R=218Hz (chênh 18Hz beta). Code trong `AlertController.kt` (Android) + `streamlit_app.py` (web).

---

## 🔄 XỬ LÝ MẶT NGHIÊNG (hạn chế hiện tại)
**Vấn đề:** mặt nghiêng → mất landmark mắt → EAR sai, CNN crop lệch.
**Giải pháp (kết hợp 3 cách):**
1. **Đo head-pose (yaw)** từ landmark → nếu |yaw| > 30° → chế độ "1 mắt" (chỉ mắt nhìn thấy) thay vì trung bình 2 mắt.
2. **Thêm data mặt nghiêng** (tìm dataset side-profile / quay video nghiêng) → train CNN/YOLO robust hơn.
3. **Dùng YOLO** (detect mắt theo pose tốt hơn landmark) làm fallback khi nghiêng nhiều.
4. Khi nghiêng quá → hiện "Chỉnh lại góc camera" thay vì báo động giả.

---

## ⏰ TIMELINE 3H (song song hóa tối đa)

| Thời gian | Việc (nền chạy song song) |
|-----------|---------------------------|
| 0:00–0:10 | Tạo nhánh `Datio-full-pipeline`, commit hiện trạng |
| 0:10–0:40 | **[NỀN]** real-data train · Tôi viết `01-03.ipynb` (dataset/preprocess/FE) |
| 0:40–1:20 | Âm thanh binaural (Android+web) + xử lý mặt nghiêng + build lại app |
| 1:20–2:00 | Model kết hợp mắt+ngáp + `05_evaluation.ipynb` (bảng so sánh) |
| 2:00–2:40 | PWA + tinh chỉnh + đóng gói (`06`, `07`) |
| 2:40–3:00 | App thứ 2 (real-data) + **viết lại lý thuyết** + commit cuối + kiểm tra lại |

**Phân vai:** Tôi viết toàn bộ notebook/code + build APK + commit. Bạn: quay video (cho bước 8), chạy Colab GPU nếu cần train, cắm điện thoại.

---

## ⚠️ CAM KẾT THỰC TẾ
- 3h KHÔNG đủ để train lại 6 model from scratch → **dùng model đã có** + bổ sung phần thiếu.
- Mọi thứ "đóng khung" được trong 3h: notebook đầy đủ 8 bước, âm thanh binaural, mặt nghiêng, PWA scaffold, app thứ 2, bảng so sánh, lý thuyết.
- Phần cần GPU (train real-data) chạy **nền/Colab** — kết quả tích hợp khi xong.
- **Làm xong mỗi phần → kiểm tra lại** (build pass, syntax pass).

---
## ✅ XÁC NHẬN TRƯỚC KHI CHẠY
Tôi sẽ bắt đầu theo timeline trên. Nếu bạn muốn **đổi ưu tiên** (vd bỏ PWA để chắc Android, hay tập trung âm thanh) → báo ngay để tôi điều chỉnh.
