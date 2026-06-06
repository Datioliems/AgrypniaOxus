# 8-Minute Presentation Script

## Goal

Present the project clearly within 8 minutes and prioritize what the rubric
cares about: problem, data, model, evaluation, demo, limitations, and practical
deployment.

## Timeline

| Time | Content | Key message |
|---:|---|---|
| 0:00-0:45 | Problem and motivation | Drowsy driving is dangerous; Android phone can be a low-cost edge device |
| 0:45-1:30 | Input/output and scope | Camera stream -> driver state + alert; prototype scope is one face |
| 1:30-2:20 | Related work and gaps | Offline/laptop demos, weak deployment evidence, accuracy-only evaluation |
| 2:20-3:10 | Data | Public datasets + self-collected Android phone data |
| 3:10-4:20 | Pipeline/model | CameraX -> MediaPipe -> EAR/MAR + eye ROI -> CNN/TFLite -> smoothing -> alert |
| 4:20-5:20 | Evaluation | Accuracy, recall, F1, confusion matrix, FPS; recall closed/drowsy is important |
| 5:20-6:50 | Demo | Awake -> eyes closed -> drowsy alert -> yawning -> CNN line if model exists |
| 6:50-7:30 | Limitations | Lighting, sunglasses, occlusion, dataset size, not road-certified |
| 7:30-8:00 | Practical value and future work | Low-cost driver aid; add PERCLOS/head pose/logging/personalized thresholds |

## Spoken Script

### Opening

> Nhóm em xây dựng hệ thống phát hiện và cảnh báo dấu hiệu buồn ngủ của tài xế
> trên thiết bị Android. Điểm chính của đề tài là xử lý trực tiếp trên điện
> thoại, không gửi video lên cloud, nên phù hợp hơn với bối cảnh xe đang di
> chuyển và có chi phí thấp.

### Problem

> Input của hệ thống là luồng camera realtime. Output là trạng thái tài xế:
> không thấy mặt, tỉnh táo, mắt nhắm, ngáp hoặc cảnh báo buồn ngủ. Khi mắt nhắm
> liên tục vượt ngưỡng thời gian, app phát âm thanh và rung.

### Research Gap

> Khi khảo sát, nhóm thấy nhiều bài làm chỉ demo trên laptop hoặc đánh giá
> offline. Khoảng trống là thiếu triển khai on-device, thiếu đo FPS/latency và
> thường chỉ báo accuracy, trong khi bài toán an toàn cần quan tâm recall của
> trạng thái buồn ngủ. Vì vậy nhóm chọn Android phone làm thiết bị edge và dùng
> smoothing theo thời gian để tránh báo nhầm do chớp mắt tự nhiên.

### Data

> Dữ liệu gồm dataset public như NTHU-DDD, YawDD, DROZY để khảo sát và dataset
> mắt mở/mắt nhắm để train CNN nhanh. Ngoài ra nhóm tự quay thêm bằng Android
> phone để kiểm tra điều kiện camera gần với lúc demo.

### Model

> Pipeline gồm CameraX lấy frame, MediaPipe Face Landmarker trích landmark,
> EAR/MAR làm baseline có thể giải thích, sau đó crop vùng mắt từ landmark và
> đưa vào CNN/TFLite phân loại mắt mở hoặc mắt nhắm. Quyết định cảnh báo cuối
> cùng dùng smoothing theo thời gian, ví dụ mắt nhắm liên tục khoảng 1.7 giây
> mới cảnh báo.

### Evaluation

> Nhóm đánh giá bằng accuracy, precision, recall, F1-score và confusion matrix.
> Với bài toán này, recall của lớp eyes_closed/drowsy quan trọng vì bỏ sót tài
> xế buồn ngủ nguy hiểm hơn báo nhầm.

### Demo

Say while demoing:

> Đây là trạng thái tỉnh táo. Trên overlay có EAR, MAR, FPS và trạng thái CNN.
> Khi em nhắm mắt liên tục, bộ đếm thời gian mắt nhắm tăng lên và hệ thống
> chuyển sang drowsy alert. Khi mở miệng/ngáp, MAR tăng và trạng thái yawning
> được hiển thị.

### Limitations

> Hạn chế là hệ thống còn phụ thuộc ánh sáng, góc camera, kính râm và che khuôn
> mặt. Dataset tự thu còn nhỏ, nên trước khi dùng ngoài đời cần kiểm thử an toàn
> trên nhiều người, nhiều thiết bị và nhiều điều kiện ánh sáng.

### Closing

> Giá trị thực tế của đề tài là chứng minh một hướng hỗ trợ cảnh báo buồn ngủ
> chi phí thấp bằng Android phone. Hướng phát triển tiếp theo là thêm PERCLOS,
> head pose, gaze, cá nhân hóa ngưỡng và lưu log cho cá nhân hoặc doanh nghiệp
> vận tải.

## Demo Preparation

Open before presentation:

- Android Studio project.
- Phone connected and app already installed if possible.
- Final demo video backup.
- Source files:
  - `MainActivity.kt`
  - `DrowsinessAnalyzer.kt`
  - `TfliteDrowsinessClassifier.kt`
- Report document.
- Metrics/confusion matrix if available.

## If Live Demo Fails

Say:

> Do môi trường trình chiếu/camera có thể không ổn định, nhóm có video demo đã
> quay trước trên thiết bị Android. Em sẽ dùng video này để minh họa luồng xử lý
> và vẫn mở source code để giải thích pipeline.

Then show the recorded video and source code.

## Do Not Overclaim

Avoid saying:

- "Hệ thống dùng được ngay trên đường thật."
- "Độ chính xác tuyệt đối."
- "Thay thế hệ thống an toàn trên xe."

Say instead:

- "Prototype."
- "Có thể mở rộng."
- "Cần kiểm thử thực tế có kiểm soát."
- "Hỗ trợ cảnh báo, không thay thế tài xế."
