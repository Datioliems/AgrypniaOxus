# Hướng Dẫn Xây Mô Hình AI Và Deploy Toàn Hệ Thống

Tài liệu này là đường đi chính cho project “Phát hiện và cảnh báo buồn ngủ tài xế sử dụng MediaPipe và CNN”. Mục tiêu là có một mô hình AI có số liệu đánh giá thật, một app Android native demo được, và một bản Streamlit/web fallback để trình bày khi cần.

## 1. Chọn Phạm Vi Để Kịp Nộp

Phương án nên chọn làm bản chính:

```text
Android CameraX
-> MediaPipe Face Landmarker
-> EAR/MAR baseline
-> crop vùng mắt
-> CNN phân loại eyes_open / eyes_closed
-> TFLite
-> smoothing theo thời gian
-> cảnh báo âm thanh/rung
```

Lý do chọn hướng này:

- MediaPipe giải quyết câu hỏi “mặt ở đâu, mắt ở đâu”.
- CNN giải quyết câu hỏi “mắt đang mở hay nhắm”.
- EAR/MAR giúp demo realtime ổn định ngay cả khi CNN chưa hoàn hảo.
- TFLite phù hợp Android phone, không cần server liên tục.
- Dễ giải thích trong Chương 3 và dễ bảo vệ trước giảng viên.

YOLO end-to-end có thể ghi trong hướng mở rộng. Chỉ nên chuyển sang YOLO làm bản chính nếu đã có dataset bounding box tốt và đủ thời gian xử lý export/post-processing trên Android.

## 2. Chuẩn Bị Dataset

Dataset chính nên dùng:

| Nguồn | Vai trò | Ghi vào báo cáo |
|---|---|---|
| `prasadvpatil/mrl-dataset` | Train CNN mắt mở/mắt nhắm | Dataset chính cho eye-state classification |
| `dheerajperumandla/drowsiness-dataset` | Khảo sát/mở rộng yawning/drowsy | Dataset phụ, dùng để so sánh hoặc minh họa |
| Video tự quay bằng Android phone | Test thực tế | Dữ liệu kiểm thử ngoài phân phối train |

Cấu trúc thư mục sau khi chuẩn hóa:

```text
dataset/
  train/
    eyes_open/
    eyes_closed/
  val/
    eyes_open/
    eyes_closed/
  test/
    eyes_open/
    eyes_closed/
```

Việc cần làm cho Chương 2:

1. Nêu nguồn dataset, link Kaggle, ngày tải.
2. Thống kê số ảnh từng lớp trong train/val/test.
3. Mô tả cách gán nhãn: thư mục `eyes_open` là nhãn mở mắt, `eyes_closed` là nhãn nhắm mắt.
4. Nêu cách tiền xử lý: resize ảnh về `64x64`, chuẩn hóa pixel về `[0,1]`, chia train/val/test, loại ảnh lỗi nếu có.
5. Chụp 4-6 ảnh ví dụ trước/sau tiền xử lý để đưa vào báo cáo.

Lệnh cần chạy:

```bash
python tools/summarize_dataset.py --data dataset --out outputs/dataset_summary
```

Kết quả cần giữ:

```text
outputs/dataset_summary/dataset_summary.csv
outputs/dataset_summary/dataset_summary.json
```

## 3. Train CNN Tự Xây

Mô hình v1:

| Thành phần | Cấu hình |
|---|---|
| Input | ảnh mắt `64x64x3` |
| Output | `eyes_closed`, `eyes_open` |
| Kiến trúc | Conv2D, MaxPooling, GlobalAveragePooling, Dropout, Dense Softmax |
| Loss | categorical/binary cross entropy tùy script |
| Metrics | accuracy, precision, recall, F1, confusion matrix |
| Export | `app/src/main/assets/drowsiness_model.tflite` |

Lệnh train:

```bash
python tools/train_eye_classifier.py --data dataset --out app/src/main/assets/drowsiness_model.tflite --epochs 12
```

Lệnh evaluate:

```bash
python tools/evaluate_eye_classifier.py --data dataset --tflite app/src/main/assets/drowsiness_model.tflite
```

Kết quả cần đưa vào Chương 4:

```text
outputs/evaluation/metrics.json
outputs/evaluation/confusion_matrix.csv
```

Cách trình bày kết quả:

- Accuracy: mô hình đúng bao nhiêu phần trăm trên test set.
- Precision lớp `eyes_closed`: khi mô hình báo buồn ngủ/nhắm mắt thì đúng bao nhiêu.
- Recall lớp `eyes_closed`: trong các ảnh nhắm mắt thật, mô hình phát hiện được bao nhiêu.
- F1-score: cân bằng giữa precision và recall.
- Confusion matrix: bảng nhầm lẫn giữa mở mắt và nhắm mắt.

Với bài toán cảnh báo an toàn, recall lớp `eyes_closed` quan trọng hơn accuracy, vì bỏ sót tài xế nhắm mắt nguy hiểm hơn báo nhầm nhẹ.

## 4. Deploy Android Native

Android là sản phẩm chính nên demo theo luồng này:

```text
CameraX preview
-> ImageAnalysis frame
-> MediaPipe Face Landmarker
-> tính EAR/MAR
-> crop eye ROI
-> TFLite classifier
-> cập nhật overlay
-> cảnh báo DROWSY/YAWNING
-> log event tối thiểu
```

Các file quan trọng:

| File | Vai trò |
|---|---|
| `MainActivity.kt` | Camera, MediaPipe, crop ROI, gọi classifier |
| `DrowsinessAnalyzer.kt` | EAR/MAR, smoothing trạng thái |
| `TfliteDrowsinessClassifier.kt` | Load và chạy model `.tflite` |
| `StatusOverlayView.kt` | Hiển thị EAR/MAR/FPS/CNN/state |
| `EventLogger.kt` | Ghi sự kiện DROWSY/YAWNING |
| `app/src/main/assets/face_landmarker.task` | Model pretrained MediaPipe |
| `app/src/main/assets/drowsiness_model.tflite` | Model CNN tự train |

Các bước chạy Android:

1. Mở `D:\2026.AI\DrowsyDriverAndroid` bằng Android Studio.
2. Đợi Gradle sync.
3. Cắm điện thoại Android, bật Developer options và USB debugging.
4. Chọn thiết bị thật, bấm Run.
5. Cho phép camera permission.
6. Kiểm tra overlay có `EAR`, `MAR`, `FPS`, `Drowsy alert`, `Yawning`, và nếu có model thì có dòng CNN.

Nếu muốn cài app vào điện thoại:

1. Build debug APK trong Android Studio hoặc chạy `tools/build_android.ps1` khi môi trường Android đã đủ.
2. File thường nằm ở `app/build/outputs/apk/debug/app-debug.apk`.
3. Cài bằng Android Studio, hoặc chuyển APK sang điện thoại và bật “Install unknown apps”.
4. Với bản nộp, chỉ cần debug APK là đủ nếu giảng viên không yêu cầu Play Store.

## 5. Deploy Streamlit Fallback

Streamlit dùng để demo web/laptop, không thay thế Android native trong báo cáo. Nó phù hợp để trình bày pipeline AI, dataset, ảnh test, metrics và demo snapshot từ webcam.

Luồng Streamlit nên trình bày:

```text
camera snapshot / uploaded image
-> MediaPipe FaceMesh hoặc OpenCV fallback
-> tính EAR/MAR nếu có landmarks
-> optional TFLite/CNN inference nếu model tồn tại
-> hiển thị trạng thái và ảnh kết quả
```

Chạy local:

```bash
pip install -r requirements-streamlit.txt
streamlit run streamlit_app.py
```

Deploy Streamlit Community Cloud:

1. Đưa project lên GitHub.
2. Đảm bảo có `streamlit_app.py` và `requirements-streamlit.txt`.
3. Vào Streamlit Community Cloud, chọn repository, branch và file chính `streamlit_app.py`.
4. Bấm Deploy, sau đó lấy link dạng `https://ten-app.streamlit.app`.

Theo tài liệu chính thức của Streamlit, Community Cloud deploy app từ GitHub và mỗi app có một subdomain trên `streamlit.app`.

Điểm cần nói khi bảo vệ:

- Streamlit là bản demo thuật toán trên web.
- Android native mới là bản deploy thực tế cho tài xế vì camera và cảnh báo chạy trực tiếp trên điện thoại.
- Streamlit có độ trễ và phụ thuộc browser/laptop, nên không phải hướng realtime chính.

## 6. Các Hệ Thống Deploy Khác

| Hệ thống | Cách deploy | Khi nào dùng |
|---|---|---|
| Android native | TFLite on-device | Bản chính, chạy trên điện thoại tài xế |
| Streamlit | Python web app | Fallback demo, dashboard metrics |
| Flask/FastAPI | REST API nhận ảnh/video frame | Khi muốn client-server hoặc tích hợp camera IP |
| Docker | Đóng gói Streamlit/FastAPI | Khi nộp môi trường chạy ổn định |
| Raspberry Pi/Jetson | TFLite/ONNX runtime | Hướng embedded thật, gắn camera riêng |
| Web frontend TensorFlow.js | Chạy model trong browser | Demo web không cần server inference |
| Flutter/React Native | App cross-platform | Khi muốn Android/iOS, nhưng tích hợp MediaPipe/TFLite phức tạp hơn native |

Khuyến nghị cho deadline:

1. Bản chính: Android native.
2. Bản phụ: Streamlit.
3. Hướng mở rộng trong báo cáo: YOLO/TFLite, FastAPI, Raspberry Pi/Jetson.

## 7. Checklist Nộp Bài

Tối thiểu để qua yêu cầu:

- App Android mở camera và hiển thị overlay.
- Có MediaPipe Face Landmarker.
- Có EAR/MAR baseline.
- Có CNN tự train hoặc ít nhất có script train/evaluate rõ ràng.
- Có dataset summary.
- Có metrics và confusion matrix.
- Có demo video hoặc ảnh minh họa chức năng.
- Có báo cáo Word điền theo Chương 1-4.
- Có script thuyết trình 8 phút.

Mạnh hơn:

- Android app load được `drowsiness_model.tflite`.
- Có debug APK.
- Có Streamlit fallback chạy được.
- Có event log CSV.
- Có bảng so sánh MediaPipe vs YOLO.

## 8. Nguồn Chính Thức Nên Trích

- MediaPipe Face Landmarker Android: https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker/android
- Android CameraX ImageAnalysis: https://developer.android.com/training/camerax/analyze
- TensorFlow Lite Android inference: https://www.tensorflow.org/lite/guide/inference
- Streamlit deploy: https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy
- Ultralytics YOLO export: https://docs.ultralytics.com/modes/export/
