# ⏱️ RUNBOOK 3 TIẾNG — Data thật + Deploy điện thoại

> Mục tiêu trong 3h: **(1) model train từ data thật của bạn → (2) cài chạy trên điện thoại thật → (3) quay demo + ghi báo cáo.**
> Phần phức tạp đã tự động hóa trong Colab. Bạn chỉ: quay video → bấm Run → cắm điện thoại.

## 📊 Phân bổ thời gian
| Thời gian | Việc | Ai làm |
|-----------|------|--------|
| 0:00–0:15 | Quay video thật | **Bạn** (điện thoại/webcam) |
| 0:15–0:45 | Colab: cắt+crop+train → tải `.tflite` | **Bạn bấm Run** (AI tự xử lý) |
| 0:45–1:00 | Build lại APK với model mới | **AI / lệnh sẵn** |
| 1:00–1:30 | Cài APK + test điện thoại | **Bạn** (cắm USB) |
| 1:30–3:00 | Quay demo + chụp + ghi báo cáo | **Bạn** + AI hỗ trợ |

---

## 🎥 BƯỚC 1 (0:00–0:15) — Quay video thật

Quay **4 video ngắn ~10-20 giây** bằng điện thoại/webcam, đặt tên theo class:

| Quay gì | Tên file |
|---------|----------|
| Mặt **mở mắt** bình thường | `open_1.mp4` |
| Mặt **nhắm mắt** (giả buồn ngủ) | `closed_1.mp4` |
| **Ngáp** (há miệng to) | `yawn_1.mp4` |
| Miệng **bình thường** | `noyawn_1.mp4` |

> Mẹo: quay đủ sáng, mặt rõ, thử cả có/không kính. Mỗi video 10-20s là đủ (~50-150 ảnh/class sau cắt).

---

## 🤖 BƯỚC 2 (0:15–0:45) — Colab tự train

1. Mở [colab.research.google.com](https://colab.research.google.com) → Upload `colab_realdata_train.ipynb`
2. `Runtime → Change runtime type → T4 GPU`
3. **Cell 2 (rd_003)**: đặt `MODE = 'eye'` (mắt) → Run all
4. Khi hiện nút upload → chọn `open_1.mp4` + `closed_1.mp4`
5. Đợi train (~5-10 phút) → tự tải `drowsiness_model.tflite` về máy
6. **Làm lại cho ngáp**: đổi `MODE = 'yawn'`, upload `yawn_1.mp4` + `noyawn_1.mp4` → tải `yawn_model.tflite`

> Colab tự: cắt frame → dò mặt (MediaPipe) → crop mắt/miệng → train CNN → xuất TFLite. Bạn không phải gán nhãn tay.

---

## 📦 BƯỚC 3 (0:45–1:00) — Build lại APK

Copy 2 file `.tflite` vừa tải vào `app/src/main/assets/` (đè lên file cũ), rồi:
```powershell
cd /d D:\2026.AI\DrowsyDriverAndroid
$env:JAVA_HOME = "C:\Program Files\Android\Android Studio\jbr"
.\gradlew assembleDebug
```
→ APK: `app\build\outputs\apk\debug\app-debug.apk`

> 💬 **Hoặc nhờ AI làm**: gửi 2 file tflite, tôi copy vào assets + build APK cho bạn.

---

## 📱 BƯỚC 4 (1:00–1:30) — Cài điện thoại + test

**Chuẩn bị điện thoại:** Settings → About → bấm "Build number" 7 lần → bật **Developer options** → bật **USB debugging**. Cắm USB.

**Cách A — adb (nhanh nhất):**
```powershell
cd "C:\Program Files\Android\Android Studio\jbr\..\..\platform-tools"   # hoặc nơi có adb
.\adb devices                     # xác nhận thấy điện thoại
.\adb install -r "D:\2026.AI\DrowsyDriverAndroid\app\build\outputs\apk\debug\app-debug.apk"
```

**Cách B — không adb:** copy `app-debug.apk` sang điện thoại (USB/Zalo/Drive) → mở file → "Cài đặt ứng dụng không rõ nguồn" → Install.

**Test trên app:**
- [ ] Camera lên, overlay hiện EAR / MAR / FPS
- [ ] Nhắm mắt vài giây → cảnh báo **DROWSY** + âm thanh
- [ ] Ngáp → cảnh báo **YAWNING**

---

## 🎬 BƯỚC 5 (1:30–3:00) — Demo + Báo cáo

- [ ] **Quay video demo** app chạy (màn hình điện thoại + cảnh báo) — dùng quay màn hình của điện thoại
- [ ] **Chụp** overlay lúc cảnh báo
- [ ] Ghi vào báo cáo: "đã thu thập data thật + train model + chạy trên điện thoại thật" (robustness)
- [ ] (Nếu còn giờ) Chụp 3 màn hình Streamlit (xem `STREAMLIT_DEPLOY.md`)

---

## 🆘 PHƯƠNG ÁN AN TOÀN (nếu hết giờ / model thật lỗi)

App **đã có sẵn model tốt** (mắt 97.44%, ngáp 98.64%) và **build được**. Nếu real-data model gặp trục trặc:
1. **Bỏ qua Bước 2-3**, dùng APK với model có sẵn → vẫn cài + demo được trên điện thoại.
2. Real-data coi như "đã thu thập + thử nghiệm" → ghi vào **hướng phát triển** của báo cáo.

→ Đảm bảo **luôn có APK chạy được** để demo, dù real-data có kịp hay không.

---

## ✅ Checklist 3h
- [ ] Quay 4 video
- [ ] Colab train eye → `drowsiness_model.tflite`
- [ ] Colab train yawn → `yawn_model.tflite`
- [ ] Build APK (model mới HOẶC model có sẵn)
- [ ] Cài điện thoại + test cảnh báo
- [ ] Quay demo + chụp
- [ ] Ghi báo cáo

> ⚡ **Ưu tiên nếu kẹt giờ:** Cài APK (model có sẵn) lên điện thoại + quay demo TRƯỚC (đảm bảo có sản phẩm), real-data làm song song/sau.
