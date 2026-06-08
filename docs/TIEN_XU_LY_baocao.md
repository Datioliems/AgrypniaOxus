# TIỀN XỬ LÝ DỮ LIỆU (phần viết cho báo cáo Word)

> Dán/định dạng lại vào báo cáo. Văn phong học thuật, số liệu đúng với thực tế đã triển khai.

---

## X.1. Nguồn dữ liệu

Hệ thống sử dụng hai nhóm dữ liệu bổ trợ nhau nhằm vừa đảm bảo quy mô huấn luyện, vừa kiểm chứng khả năng hoạt động trong điều kiện triển khai thực tế:

**a) Dữ liệu công khai (huấn luyện chính):**
- **MRL Eye Dataset** (Đại học Kỹ thuật Ostrava): 84.898 ảnh mắt hồng ngoại của 37 đối tượng, đa dạng điều kiện ánh sáng, cảm biến và có/không đeo kính — dùng huấn luyện CNN phân loại trạng thái mắt (eyes_open / eyes_closed).
- **Drowsiness Dataset** (Kaggle, *dheerajperumandla*): ảnh vùng miệng — huấn luyện CNN phân loại ngáp (no_yawn / yawn).
- **Bộ 6-class trên Roboflow** (close_eyeL/R, open_eyeL/R, yawn, no_yawn): huấn luyện các detector YOLOv11/YOLO26 và Transformer (RT-DETR, RF-DETR).

**b) Dữ liệu tự thu thập (kiểm chứng & thích nghi miền):**
Nhóm tự quay video bằng webcam/điện thoại ở nhiều điều kiện ánh sáng và góc camera, gồm các đoạn có chủ đích cho từng trạng thái (mắt mở, nhắm mắt, ngáp). Dữ liệu này dùng để (i) kiểm thử hệ thống trong điều kiện thật và (ii) tinh chỉnh (fine-tune) mô hình theo đúng kiểu ảnh mà ứng dụng thu được — giải quyết khoảng cách miền (domain gap) giữa dữ liệu phòng thí nghiệm và camera thực tế.

---

## X.2. Quy trình tiền xử lý

Toàn bộ quy trình được tổ chức thành các bước có kiểm soát chất lượng:

1. **Trích khung hình (frame extraction):** với dữ liệu dạng video, tách khung theo tần suất cố định (ví dụ 2–10 khung/giây) để tránh các khung gần như trùng nhau.
2. **Phát hiện khuôn mặt & trích vùng quan tâm (ROI):** dùng **MediaPipe Face Landmarker** (478 điểm mốc) để định vị khuôn mặt, sau đó cắt riêng **vùng mắt** (cho mô hình mắt) và **vùng miệng** (cho mô hình ngáp). Việc cắt ROI giúp mô hình tập trung vào đặc trưng cần thiết, loại bỏ nền nhiễu.
3. **Chuẩn hóa kích thước:** đưa mọi ảnh ROI về **64×64×3** (đầu vào CNN) hoặc **640×640** (đầu vào YOLO).
4. **Chuẩn hóa pixel:** đưa giá trị điểm ảnh về khoảng **[0, 1]** (chia 255). Bước chuẩn hóa được thực hiện **bên ngoài mô hình** để khớp với cách nạp ảnh trên Android (TFLite).
5. **Lọc chất lượng:** loại bỏ ảnh hỏng, mờ, hoặc không phát hiện được khuôn mặt.
6. **Tăng cường dữ liệu (augmentation):** lật ngang, xoay nhẹ, thay đổi độ sáng, phóng to nhẹ — tăng tính bất biến của mô hình với điều kiện thực tế.
7. **Chia tập:** train / validation / test theo tỉ lệ **60/20/20** (hoặc 80/20 train/val cho dữ liệu nhỏ), dùng `random_state` cố định để tái lập.

> **Lưu ý về dữ liệu đã cắt sẵn:** Một số bộ công khai (như MRL) vốn đã là ảnh mắt cắt sẵn, nên bước trích ROI được bỏ qua (chỉ kiểm tra kích thước/định dạng). Ngược lại, với **video tự quay là ảnh khuôn mặt đầy đủ**, bước trích ROI bằng MediaPipe được kích hoạt để tự động cắt mắt/miệng — đây chính là cơ chế cho phép hệ thống xử lý được dữ liệu người dùng tự cung cấp.

---

## X.3. Tiền xử lý cho hai mô hình CNN

| | CNN Mắt | CNN Ngáp |
|---|---------|----------|
| Nguồn | MRL + video tự quay (awake/closed) | Kaggle + video tự quay (yawn) |
| ROI | vùng mắt (trái + phải) | vùng miệng |
| Lớp | eyes_closed (0), eyes_open (1) | no_yawn (0), yawn (1) |
| Kích thước | 64×64×3 | 64×64×3 |
| Chuẩn hóa | /255 → [0,1] | /255 → [0,1] |

Thứ tự lớp được giữ **đúng thứ tự alphabet** để khớp với hằng số phía Android (tránh đảo nhãn khi suy luận).

---

## X.4. Tiền xử lý dữ liệu thật (quy trình tự thu thập)

Để mô hình thích nghi với chính kiểu ảnh mà ứng dụng thu được, nhóm xây dựng quy trình bán tự động:

1. Quay các video ngắn có chủ đích cho từng trạng thái (mắt mở, nhắm mắt, ngáp).
2. Trích khung hình và dùng MediaPipe **tự động cắt vùng mắt/miệng** từ khuôn mặt — không cần gán nhãn thủ công từng khung (nhãn suy ra từ chủ đích của video).
3. Huấn luyện lại (fine-tune) CNN trên tập dữ liệu thật này, xuất ra mô hình TFLite riêng.
4. Đóng gói thành một phiên bản ứng dụng độc lập để so sánh với mô hình huấn luyện trên dữ liệu công khai.

Quy trình này thể hiện vòng lặp **thu thập → tiền xử lý → huấn luyện → triển khai** khép kín ngay trên thiết bị người dùng, đồng thời là minh chứng cho khả năng mở rộng và thích nghi miền của hệ thống.
