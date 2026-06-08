# LÝ THUYẾT & SO SÁNH MÔ HÌNH — AgrypniaOxus (Drowsy Driver)

> Bản tổng hợp hoàn chỉnh cho báo cáo — kiến trúc, số liệu thật, cơ chế quyết định & cảnh báo.

---

## 1. TỔNG QUAN PHƯƠNG PHÁP

Hệ thống kết hợp **3 tầng** chạy trực tiếp trên thiết bị (on-device, không cloud):

```
Camera → MediaPipe FaceLandmarker (478 điểm)
   ├─ Tầng HÌNH HỌC (baseline): EAR (mắt) + MAR (miệng)  ── giải thích được
   ├─ Tầng HỌC SÂU (chủ đạo):  CNN phân loại mắt nhắm/mở + ngáp
   └─ Tầng QUYẾT ĐỊNH:          Fusion (CNN chủ đạo, EAR/MAR fallback + lưới an toàn)
        → Temporal (PERCLOS/ngưỡng thời gian) → Cảnh báo leo thang (binaural beats)
```

- **MediaPipe** lo phần phát hiện mặt + crop ROI mắt/miệng (nhanh, on-device).
- **CNN** (TFLite 64×64) là phương pháp chính phân loại trạng thái.
- **YOLO / Transformer** là hướng detector end-to-end (so sánh + mở rộng).
- **EAR/MAR** là baseline hình học, đồng thời là cơ chế dự phòng/an toàn.

---

## 2. BẢNG SO SÁNH MÔ HÌNH (số liệu THẬT)

### 2.1. Phân loại CNN (trạng thái mắt / ngáp)
| Model | Nhiệm vụ | Accuracy | Recall (lớp nguy hiểm) | Kích thước | Thiết bị |
|-------|----------|:--------:|:----------------------:|:----------:|----------|
| CNN mắt | eyes_closed/open | **97.44%** | 98.34% (closed) | 46 KB | RTX/Local |
| CNN ngáp (CBAM) | no_yawn/yawn | **98.64%** | 98.03% (yawn) | 131 KB | RTX 4050 |
| CNN ngáp (mới) | no_yawn/yawn | 97.46% | — | 105 KB | CPU local |
| Temporal | chuỗi thời gian | 99.87% | — | 43 KB | Local |

> ⭐ Ưu tiên **Recall lớp nguy hiểm** (bỏ sót tài xế buồn ngủ = tai nạn) hơn Accuracy thuần.

### 2.2. Detector YOLO / Transformer (6-class, tập valid 440 ảnh)
| # | Model | Loại | Epochs | Params | mAP50 | mAP50-95 | Tốc độ T4 |
|---|-------|------|:---:|:---:|:---:|:---:|:---:|
| 1 | YOLO11s | CNN | 60 | 9.4M | 94.1% | **73.5%** 🥇 | 8.4ms |
| 2 | YOLO26s | CNN | 60 | 9.5M | **95.2%** 🥇 | 71.4% | **8.0ms** 🥇 |
| 3 | RT-DETR-L | Transformer | 10 | 32M | 95.1% | 70.7% | 38.2ms |
| 4 | RF-DETR-S | Transformer | 10 | 32M | 93.9% | 67.3% | ~7ms |

**Nhận xét:** CNN nhẹ (YOLO11/26, ~9M) thắng Transformer (~32M) cả độ chính xác lẫn tốc độ TRONG điều kiện này — dù Transformer chỉ train 1/6 số epoch (10 vs 60). Transformer hội tụ nhanh nhờ pretrain (DINOv2) và còn dư địa. **CNN hợp deploy Android** (nhẹ, nhanh ~8ms).

### 2.3. So sánh phương pháp (cho báo cáo)
| Phương pháp | Ưu điểm | Nhược điểm |
|-------------|---------|------------|
| EAR/MAR (baseline) | Nhanh, giải thích được, không cần train | Nhạy với mặt nghiêng, ánh sáng |
| CNN (chủ đạo) | Học đặc trưng thị giác, chính xác cao | Cần dữ liệu khớp điều kiện thật |
| YOLO/Transformer | Detect + định vị 6-class 1 model | Nặng (640px), tốn pin trên mobile |
| **Hybrid (đề xuất)** | CNN chính + EAR/MAR an toàn + temporal | Cần tinh chỉnh ngưỡng |

---

## 3. CƠ CHẾ QUYẾT ĐỊNH (Fusion — CNN chủ đạo)

```
NO_FACE  → không thấy mặt
YAWNING  → CNN Yawn (chính), MAR phụ
CNN mắt đủ tin cậy (≥0.52) → theo CNN (DROWSY/EYES_CLOSED/AWAKE)
   (lưới an toàn: nếu EAR cũng thấy DROWSY → vẫn DROWSY, không bỏ sót)
Còn lại  → fallback EAR/MAR
```
→ `alertSource` hiển thị "CNN Eye (primary)" làm bằng chứng CNN đang dẫn dắt. Tầng temporal: nhắm mắt liên tục ≥1.2s (CNN) hoặc ≥1.7s (EAR) → DROWSY.

---

## 4. CƠ CHẾ CẢNH BÁO (theo CK AI 1.3.1.6 & 1.3.1.7)

**Cơ sở khoa học:** Binaural beats dải **beta thấp 13–21 Hz** (Moessinger 2021, PLOS ONE) kích thích vỏ não dải beta → tăng tỉnh táo. Cảnh báo **leo thang đa giác quan** (Beles 2024) thị giác→âm thanh→rung. Ngưỡng giờ lái (Wang 2014): suy giảm sau 2h.

| Tier | Điều kiện | Cảnh báo |
|------|-----------|----------|
| 1 | EYES_CLOSED ngắn | beep nhẹ + overlay vàng |
| 2 | DROWSY | **binaural beta-beat 18Hz** (L=200Hz, R=218Hz) + rung |
| 3 | DROWSY >4s | beat TO + rung mạnh kép |
| + | Lái >2h | nhắc nghỉ 15-30 phút |

> Tai nghe → hiệu ứng binaural đầy đủ; loa → 2 tần số trộn thành nhịp đập 18Hz nghe được (vẫn cảnh báo mạnh).

---

## 5. XỬ LÝ MẶT NGHIÊNG (khắc phục hạn chế)

**Vấn đề:** mặt nghiêng → mắt phía xa bị "co", EAR trung bình 2 mắt sai.
**Giải pháp đã cài:** đo độ rộng mỗi mắt; mắt nào **rộng hơn (chính diện hơn)** thì đáng tin → ưu tiên dùng EAR mắt đó thay vì trung bình. Khi gần chính diện (tỉ lệ 0.70–1.43) mới trung bình 2 mắt.
**Mở rộng:** thêm dữ liệu mặt nghiêng (quay video nghiêng / dataset side-profile) + dùng YOLO detect mắt theo pose làm fallback.

---

## 6. KẾT LUẬN & LỰA CHỌN

- **Deploy Android:** CNN mắt + CNN ngáp (TFLite, ~9MB tổng, nhanh) + EAR/MAR an toàn + binaural alert.
- **So sánh học thuật:** CNN (YOLO11s 73.5% mAP50-95) > Transformer trong điều kiện 3h; Transformer là hướng mở rộng khi có thêm epoch/data.
- **Hướng phát triển:** train CNN trên ảnh thật của app (đồng nhất crop) → CNN tự tin hơn; ensemble đa model; nhúng YOLO; tối ưu int8/NNAPI.
