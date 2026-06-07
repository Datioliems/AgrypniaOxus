# Tài liệu Pipeline — So sánh chia dữ liệu RANDOM vs THEO SUBJECT

Tài liệu này mô tả chi tiết toàn bộ pipeline trong notebook `SoSanh_Split_Colab.ipynb`,
mục đích là **so sánh hai cách chia tập dữ liệu** cho mô hình CNN phân loại trạng thái
mắt (eyes_closed / eyes_open), từ đó chứng minh và đo lường hiện tượng **rò rỉ dữ liệu
(data leakage)**.

---

## 1. Mục tiêu

Huấn luyện **cùng một mô hình CNN** với **cùng cấu hình**, chỉ khác **cách chia
train/val/test (70/20/10)**, theo hai góc nhìn:

| Góc nhìn | Cách chia | Hệ quả |
|---|---|---|
| 1. RANDOM theo ảnh | trộn tất cả ảnh rồi cắt 70/20/10 | cùng một người có thể ở cả train+test → **rò rỉ** → metric ảo cao |
| 2. THEO SUBJECT | chia theo người (1 người chỉ ở 1 tập) | **trung thực** → phản ánh khả năng tổng quát thật |

> Điều kiện so sánh công bằng: hai thí nghiệm dùng **chung** CONFIG, kiến trúc, seed,
> class_weight, augmentation; **chỉ khác đúng một biến** là hàm chia.

---

## 2. Sơ đồ luồng pipeline

```text
dataset.zip (Drive)
   │  giải nén
   ▼
/content/dataset/{train,val,test}/{eyes_closed,eyes_open}/*.png
   │  gom toàn bộ + trích subject (s####)
   ▼
all_items = [(lop, subject, path), ...]   (84.898 phần tử, 37 subject)
   │
   ├──► split_random()  ──┐
   │                      ├──► make_ds() ─► build_model() ─► train+EarlyStopping ─► đánh giá test
   └──► split_subject() ──┘
   │
   ▼
Bảng so sánh + 2 confusion matrix + mức rò rỉ (chênh lệch test acc)
```

---

## 3. Giải thích từng bước (theo cell)

| Cell | Bước | Vai trò |
|---|---|---|
| 1 | Cấu hình + GPU + seed | Khai báo CONFIG dùng chung; cố định seed cho tái lập + công bằng |
| 2 | Mount Drive + giải nén | Lấy dữ liệu từ `dataset.zip` ra `/content/dataset` |
| 3 | Gom ảnh + trích subject | Tạo `all_items`; parse `s####` từ tên file (MRL) |
| 4 | Hai hàm chia | `split_random` (theo ảnh) và `split_subject` (theo người) — **điểm khác biệt duy nhất** |
| 5 | tf.data + augmentation | Đọc ảnh trực tiếp từ đường dẫn (không copy), chuẩn hoá /255, augment cho train |
| 6 | build_model + run_experiment | Dựng CNN; hàm chạy trọn 1 thí nghiệm (chia → train → đánh giá) |
| 7 | Chạy góc nhìn 1 | `run_experiment(split_random, ...)` |
| 8 | Chạy góc nhìn 2 | `run_experiment(split_subject, ...)` |
| 9 | So sánh | Bảng số liệu + 2 confusion matrix + mức rò rỉ; lưu CSV + PNG |

### Các điểm kỹ thuật quan trọng
- **Đọc ảnh bằng tf.data từ danh sách file** (`from_tensor_slices`) → không copy 84k ảnh
  mỗi lần → chạy nhanh, tiết kiệm bộ nhớ.
- **Chuẩn hoá /255 đúng một lần** trong pipeline (không đặt lớp Rescaling trong model →
  tránh chuẩn hoá hai lần).
- **Tập test KHÔNG shuffle** → đảm bảo `y_true` khớp thứ tự `y_pred`.
- **`clear_session()` + `set_seed()` đầu mỗi thí nghiệm** → khởi tạo trọng số giống nhau,
  so sánh công bằng.
- **class_weight** ưu tiên `eyes_closed` (hệ số 1,3) → tăng recall lớp an toàn.
- **EarlyStopping(restore_best_weights)** + **ReduceLROnPlateau** → chống overfit, hội tụ ổn định.

---

## 4. Cách đọc & diễn giải kết quả

Cell 9 in ra bảng dạng:

| Cách chia | Test acc | Macro F1 | Recall closed | Gap train-val |
|---|---|---|---|---|
| RANDOM theo ảnh | cao hơn (≈0,98–0,99) | cao | cao | nhỏ |
| THEO SUBJECT | thấp hơn (≈0,96) | thấp hơn | 0,99 | nhỏ |

**Diễn giải (câu chốt cho báo cáo):**
> "Cách chia random cho test accuracy cao hơn **không phải vì mô hình tốt hơn**, mà vì nó
> được kiểm thử trên dữ liệu đã thấy (cùng người ở train+test). Khoảng chênh lệch chính là
> mức metric bị **thổi phồng do rò rỉ dữ liệu**. Chia theo subject tuy cho số thấp hơn nhưng
> **trung thực** vì đánh giá trên người hoàn toàn mới."

---

## 5. Lưu ý / Hạn chế (nên ghi vào báo cáo)

- Bộ MRL Eye Dataset chỉ có **37 người** (84.898 ảnh là nhiều khung hình của 37 người).
  → tập test theo subject chỉ gồm vài người → **kết quả có phương sai**; nên cân nhắc
  **K-fold theo subject** để có ước lượng vững hơn.
- "84k ảnh" là độ đa dạng nội bộ (ánh sáng/góc/kính), không phải 84k người khác nhau.

---

## 6. Cách chạy

1. Upload `dataset.zip` (chứa `train/val/test/<lớp>`) lên **Google Drive → My Drive**.
2. Mở `SoSanh_Split_Colab.ipynb` trên Colab → **Runtime → GPU (T4)**.
3. Sửa biến `ZIP` ở Cell 2 cho đúng đường dẫn.
4. Chạy lần lượt Cell 1 → 9 (train 2 lần, mỗi lần ~10 phút trên T4).
5. Lấy `so_sanh_split.csv` và `so_sanh_confusion.png` để chèn vào báo cáo.
