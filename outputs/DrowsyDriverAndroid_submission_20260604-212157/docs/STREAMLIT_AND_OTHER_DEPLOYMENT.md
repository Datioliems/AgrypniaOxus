# Deploy Streamlit Và Các Hệ Thống Khác

Tài liệu này dùng khi cần demo ngoài Android native. Android vẫn là hướng chính của project, Streamlit là fallback/web demo.

## 1. Streamlit Local Demo

Mục tiêu:

- Test nhanh ảnh upload hoặc ảnh chụp webcam.
- Hiển thị kết quả phát hiện landmarks/trạng thái.
- Trình bày metrics, confusion matrix và dataset summary trong báo cáo/demo.

Chạy local:

```bash
pip install -r requirements-streamlit.txt
streamlit run streamlit_app.py
```

Nếu chưa có model `.tflite`, app vẫn có thể chạy phần demo ảnh và trạng thái hướng dẫn. Khi có model, đặt tại:

```text
app/src/main/assets/drowsiness_model.tflite
```

## 2. Streamlit Community Cloud

Các bước:

1. Tạo GitHub repository chứa project.
2. Đảm bảo có `streamlit_app.py` và `requirements-streamlit.txt`.
3. Đăng nhập Streamlit Community Cloud.
4. Chọn New app, chọn repository/branch.
5. Main file chọn `streamlit_app.py`.
6. Deploy và lấy link `.streamlit.app`.

Lưu ý:

- Không nên upload video/khuôn mặt thật của người khác lên cloud.
- Nếu repo public, không đưa dữ liệu nhạy cảm vào repository.
- Với realtime webcam liên tục, Streamlit không ổn định bằng Android native.
- Streamlit phù hợp nhất để demo snapshot, dashboard kết quả và giải thích pipeline.

## 3. FastAPI/Flask Server

Luồng:

```text
Client gửi ảnh/frame
-> FastAPI/Flask nhận request
-> model inference
-> trả JSON state/probability
```

Khi dùng:

- Muốn nhiều client dùng chung một server.
- Muốn triển khai trên máy có GPU.
- Muốn tách app mobile và model backend.

Không nên dùng làm bản chính nếu mục tiêu là cảnh báo tài xế realtime, vì mạng chậm hoặc mất kết nối sẽ làm hệ thống kém an toàn.

## 4. Docker

Docker phù hợp để đóng gói Streamlit/FastAPI:

```text
Docker image
-> cài Python dependencies
-> copy model và code
-> expose port
-> chạy Streamlit/FastAPI
```

Ưu điểm:

- Dễ chạy lại trên máy khác.
- Ít lỗi thiếu thư viện.
- Phù hợp nếu giảng viên muốn xem bản web.

Nhược điểm:

- Cần Docker Desktop.
- Nặng hơn cách chạy local.

## 5. Embedded/Raspberry Pi/Jetson

Luồng embedded:

```text
Camera USB/CSI
-> OpenCV frame
-> MediaPipe/TFLite/ONNX
-> cảnh báo buzzer/LED/speaker
```

Thiết bị gợi ý:

- Raspberry Pi 4/5: phù hợp TFLite model nhỏ, tốc độ vừa phải.
- Jetson Nano/Orin: phù hợp YOLO/ONNX/TensorRT, mạnh hơn nhưng phức tạp hơn.
- Android phone: thực tế nhất cho project hiện tại vì đã có camera, màn hình, loa, rung và pin.

Trong báo cáo nên viết: hệ thống có thể chuyển sang embedded bằng cách thay CameraX bằng OpenCV camera input và thay Android alert bằng buzzer/LED/speaker, còn model TFLite vẫn có thể tái sử dụng.

## 6. Web Browser/TensorFlow.js

Có thể export hoặc chuyển model sang TensorFlow.js để chạy trong browser:

```text
Webcam browser
-> JavaScript preprocessing
-> TensorFlow.js inference
-> UI alert
```

Ưu điểm:

- Không cần cài app.
- Demo tiện trên laptop.

Nhược điểm:

- Tích hợp MediaPipe + custom CNN cần nhiều JavaScript hơn.
- Độ ổn định phụ thuộc browser và thiết bị.
- Không tốt bằng Android native cho cảnh báo thực tế.

## 7. Khuyến Nghị Cuối

Thứ tự triển khai hợp lý:

1. Android native: bản nộp chính.
2. Streamlit: bản demo phụ và dashboard kết quả.
3. FastAPI/Docker: hướng mở rộng nếu cần server.
4. Raspberry Pi/Jetson: hướng embedded thực tế.
5. YOLO/TensorRT/ONNX: hướng nghiên cứu nâng cao khi có dataset bounding box.

Khi bảo vệ, nên nói ngắn gọn:

> Với thời gian của môn học, nhóm chọn Android native + TFLite vì chạy trực tiếp trên điện thoại, không phụ thuộc mạng và phù hợp bài toán cảnh báo tài xế. Streamlit được dùng như bản demo web để minh họa thuật toán và kết quả đánh giá. Các hướng FastAPI, Docker, Raspberry Pi/Jetson là hướng phát triển sau khi hệ thống ổn định hơn.
