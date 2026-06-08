# Deploy Streamlit — DrowsyDriver (3 màn hình)

App: `streamlit_app.py` — Dashboard / Alert / Analytics, dùng webcam + MediaPipe (EAR/MAR) + TFLite.

---

## A. Chạy LOCAL (thử trước khi deploy)
```powershell
cd D:\2026.AI\DrowsyDriverAndroid
py -3.12 -m pip install -r requirements-streamlit.txt
py -3.12 -m streamlit run streamlit_app.py
```
Mở trình duyệt `http://localhost:8501` → cho phép camera → bật **Kích hoạt giám sát**.

---

## B. Deploy lên Streamlit Community Cloud (miễn phí)

### B1. Đưa code lên GitHub
```powershell
git add streamlit_app.py requirements-streamlit.txt packages.txt
git commit -m "Add Streamlit drowsy demo"
git push
```

### B2. Tạo app
1. Vào [share.streamlit.io](https://share.streamlit.io) → đăng nhập GitHub.
2. **New app** → chọn repo + branch + file `streamlit_app.py`.
3. **Advanced settings** → Python 3.11 (hoặc 3.10/3.12).
4. **Deploy**. Lần đầu cài deps ~3-5 phút.

### B3. File cấu hình cần có trong repo
- `requirements-streamlit.txt` — Streamlit Cloud tự đọc nếu đặt tên `requirements.txt`. **Nếu để tên khác** → đổi tên thành `requirements.txt` hoặc khai báo trong app settings.
- `packages.txt` — system libs cho OpenCV/MediaPipe (tạo file này, nội dung bên dưới).

> ⚠️ **Quan trọng:** Streamlit Cloud chỉ tự nhận `requirements.txt`. Nên copy:
> ```powershell
> Copy-Item requirements-streamlit.txt requirements.txt
> ```

---

## C. `packages.txt` (system deps cho cloud)
Tạo file `packages.txt` ở gốc repo:
```
libgl1
libglib2.0-0
```
(MediaPipe/OpenCV cần `libGL`. Đã dùng `opencv-python-headless` để giảm phụ thuộc.)

---

## D. Camera trên cloud (webrtc)
- Streamlit Cloud chạy **HTTPS** → trình duyệt cho phép webcam ✅.
- `streamlit-webrtc` cần **STUN server** để kết nối — mặc định dùng Google STUN, thường OK.
- Nếu camera không lên (mạng chặn): thêm cấu hình STUN/TURN trong `webrtc_streamer(... rtc_configuration={...})`. Với demo báo cáo, mạng thường thì không cần.

---

## E. 3 màn hình (đáp ứng yêu cầu đề bài)

| Màn hình | Tính năng | Đáp ứng yêu cầu |
|----------|-----------|-----------------|
| **Dashboard** | webcam quét mặt (toggle ẩn/hiện), nút **Kích hoạt giám sát**, thanh trạng thái (TỈNH TÁO/MỆT/BUỒN NGỦ) | ✅ camera AI + nút giám sát + thanh trạng thái |
| **Alert** | màn **đỏ nhấp nháy** (CSS), **âm thanh** lớn (auto-loop), nút to **"Tôi ổn"** + **"Tìm trạm dừng"** | ✅ đỏ nhấp nháy + âm thanh + nút xác nhận |
| **Analytics** | lịch sử sự kiện buồn ngủ, **slider độ nhạy** (EAR/MAR/thời gian), **chọn âm thanh** | ✅ lịch sử + chỉnh độ nhạy + chọn âm thanh |

Tự động chuyển sang màn **Alert** khi phát hiện nhắm mắt quá lâu (lúc đang bật giám sát).

---

## F. Lưu ý cho báo cáo
- Streamlit là **bản demo web/laptop** để trình bày pipeline + UI. **Android native vẫn là bản deploy chính** cho tài xế (camera + cảnh báo chạy thẳng trên điện thoại).
- Streamlit không chạy được trên điện thoại trong xe như app native, nhưng tiện để demo nhanh khi thuyết trình.

---

## G. Lỗi thường gặp
| Lỗi | Cách xử lý |
|-----|-----------|
| `ModuleNotFoundError: streamlit_webrtc` | thiếu `requirements.txt` đúng tên trên cloud |
| Camera đen / không lên | dùng HTTPS (cloud có sẵn); local thì `localhost` được phép |
| `libGL.so.1 not found` | thêm `packages.txt` với `libgl1` |
| MediaPipe cài lâu/fail | chọn Python 3.10–3.11 trên cloud (wheel ổn định hơn) |
