# Đối Chiếu Yêu Cầu IS54A Với Project

Tài liệu này bám theo file `IS54A - Hướng dẫn bài tập lớn cuối kỳ.xlsx`. Mục tiêu là biết phần nào đã có, phần nào cần bổ sung bằng số liệu thật, và phần nào nên nhấn mạnh khi bảo vệ.

## 1. Báo Cáo - 4.5 Điểm

| Mục | Điểm | Cách đáp ứng trong project | Artifact cần có |
|---|---:|---|---|
| Chương 1 | 0.5 | Giới thiệu bài toán phát hiện buồn ngủ, input là camera frame/ảnh mắt/landmarks, output là trạng thái Awake/Drowsy/Yawning, phạm vi Android phone, khảo sát MediaPipe/CNN/YOLO | `docs/DE_CUONG_BAO_CAO.md`, `docs/BAO_CAO_NHAP.md`, `research/SOURCES_FOR_REPORT.md` |
| Chương 2 | 1.0 | Nêu dataset MRL, drowsiness/yawning phụ, dữ liệu tự quay; thống kê số lượng; mô tả resize, crop ROI, normalize, split train/val/test; trích nguồn | `tools/summarize_dataset.py`, `outputs/dataset_summary/*`, `tools/DATASET_PREP.md` |
| Chương 3 | 1.5 | Giải thích đặc trưng EAR/MAR, MediaPipe landmarks, CNN eye-state, TFLite, pipeline Android; phân biệt pretrained và model tự xây | `docs/MEDIAPIPE_LANDMARK_PIPELINE.md`, `docs/AI_MODEL_AND_DEPLOYMENT_FULL_GUIDE.md`, source Kotlin |
| Chương 4 | 1.0 | Nêu accuracy/precision/recall/F1/confusion matrix, thí nghiệm, công nghệ prototype, ảnh app và demo | `tools/evaluate_eye_classifier.py`, `outputs/evaluation/*`, `docs/THI_NGHIEM_VA_BANG_KET_QUA.md` |
| Kết luận và format | 0.5 | Tổng kết đã làm, hạn chế, hướng phát triển, format báo cáo sạch | `outputs/Bao_cao_nhap_DrowsyDriverAndroid.docx` |

Phần còn cần số liệu thật:

- `outputs/dataset_summary/dataset_summary.csv`
- `outputs/evaluation/metrics.json`
- `outputs/evaluation/confusion_matrix.csv`
- Ảnh/screenshot app Android thật
- Video demo cuối
- Nếu kịp: `app/src/main/assets/drowsiness_model.tflite`

## 2. Sản Phẩm - 1.5 Điểm

| Mục | Điểm | Cách đáp ứng |
|---|---:|---|
| Demo đúng bài toán, giao diện/luồng xử lý phù hợp, mã nguồn khớp báo cáo | 1.0 | App Android dùng camera, hiển thị trạng thái, cảnh báo khi nhắm mắt/ngáp, báo cáo mô tả đúng source |
| Project tổ chức hợp lý, có commit, code sạch/comment | 0.5 | Cấu trúc `app/`, `tools/`, `docs/`, `research/`, `outputs/`; nên tạo git repository và commit theo từng mốc |

Checklist sản phẩm:

- Android app chạy được trên điện thoại.
- Overlay hiển thị EAR/MAR/FPS/state.
- Có cảnh báo âm thanh/rung khi DROWSY.
- Có event log tối thiểu, không lưu video khuôn mặt.
- Có Streamlit fallback nếu Android gặp lỗi sát giờ.
- Có zip nộp bằng `tools/package_submission.ps1`.

Nếu chưa có git:

```bash
git init
git add .
git commit -m "Initial drowsiness detection prototype"
git commit -m "Add dataset training and evaluation tools"
git commit -m "Add Android MediaPipe and TFLite integration"
git commit -m "Add report docs and deployment guides"
```

## 3. Thuyết Trình - 1.0 Điểm

| Mục | Điểm | Cách đáp ứng |
|---|---:|---|
| Không quá 8 phút | 0.2 | Dùng script 8 phút, luyện trước với timer |
| Trình bày điểm nổi bật và demo trọn vẹn | 0.8 | Mở sẵn Android Studio/app/video demo/source/report; demo nhắm mắt 2 giây và ngáp |

Thứ tự trình bày nên dùng:

1. Bài toán và ứng dụng thực tế.
2. Dataset và tiền xử lý.
3. Pipeline MediaPipe + EAR/MAR + CNN/TFLite.
4. Kết quả đánh giá.
5. Demo Android.
6. Hạn chế và hướng phát triển YOLO/embedded.

## 4. Cá Nhân - 3.0 Điểm

| Mục | Điểm | Cách chuẩn bị |
|---|---:|---|
| Trả lời tốt phần công việc của mình | 2.0 | Nắm rõ phần mình làm: dataset, train CNN, Android, Streamlit, báo cáo |
| Trả lời tốt dự án chung | 0.5 | Hiểu toàn bộ pipeline và vì sao chọn MediaPipe + CNN |
| Tỷ lệ đóng góp | 0.5 | Có bảng phân công, commit/log công việc, ảnh demo cá nhân |

Câu hỏi dễ gặp:

- Vì sao không dùng YOLO làm chính?
- EAR là gì, MAR là gì?
- MediaPipe là pretrained hay tự train?
- CNN tự xây ở đâu?
- Tại sao dùng recall/F1 thay vì chỉ accuracy?
- Nếu ánh sáng yếu, đeo kính, nghiêng mặt thì sao?
- Android deploy khác Streamlit ở điểm nào?
- Hệ thống có lưu dữ liệu khuôn mặt không?

## 5. Việc Cần Làm Theo Mức Ưu Tiên

Ưu tiên 1, bắt buộc:

1. Train/evaluate model hoặc ít nhất chuẩn bị dataset summary thật.
2. Chạy Android app và quay demo.
3. Điền số liệu thật vào báo cáo Word.
4. Luyện script 8 phút.

Ưu tiên 2, tăng điểm:

1. Export `drowsiness_model.tflite` và đặt vào Android assets.
2. Build APK debug.
3. Chạy Streamlit fallback.
4. Thêm confusion matrix và ảnh kết quả vào báo cáo.

Ưu tiên 3, hướng mở rộng:

1. Thử YOLO end-to-end với dataset có bounding box.
2. Docker hóa Streamlit/FastAPI.
3. Đề xuất Raspberry Pi/Jetson hoặc Android edge deployment.

## 6. Câu Kết Luận Nên Dùng Khi Bảo Vệ

Project chọn MediaPipe + CNN vì đây là hướng phù hợp với thời gian và thiết bị Android. MediaPipe giúp định vị khuôn mặt, mắt và miệng bằng mô hình pretrained; CNN tự xây phân loại trạng thái mắt; EAR/MAR và smoothing giúp cảnh báo ổn định theo thời gian. YOLO end-to-end là hướng nghiên cứu mở rộng có thể định vị và phân loại trực tiếp, nhưng cần bounding box labels và tích hợp Android phức tạp hơn.
