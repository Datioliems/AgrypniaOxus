# Ngay 2: Train CNN Va Tao Ket Qua Danh Gia

Muc tieu ngay 2 la co bang ket qua dinh luong, confusion matrix, va file
`drowsiness_model.tflite`.

## 1. Chon Scope Model

Trong 4 ngay, uu tien model:

```text
eyes_open vs eyes_closed
```

Ly do:

- Lien quan truc tiep den buon ngu.
- De cat ROI mat.
- De demo: nham mat 2 giay la canh bao.
- De giai thich trong bao cao.

Neu con thoi gian moi them:

```text
yawning vs not_yawning
```

## 2. Cau Truc Dataset

Sap xep du lieu theo cau truc:

```text
dataset/
  train/
    eyes_closed/
    eyes_open/
  val/
    eyes_closed/
    eyes_open/
  test/
    eyes_closed/
    eyes_open/
```

Neu dataset chua co test rieng, dung validation de bao cao tam thoi, nhung phai
ghi ro han che.

## 3. Train Model

Truoc khi train, tao thong ke dataset:

```bash
python tools/summarize_dataset.py --data dataset --out outputs/dataset_summary
```

Dung `outputs/dataset_summary/dataset_summary.csv` de dien Chuong 2.

Chay:

```bash
python tools/train_eye_classifier.py --data dataset --out app/src/main/assets/drowsiness_model.tflite --epochs 12
```

Class order trong Android hien tai:

```text
eyes_closed
eyes_open
```

Neu script in class order khac, sua file:

```text
app/src/main/java/com/ai2026/drowsydriver/TfliteDrowsinessClassifier.kt
```

## 4. Ket Qua Can Luu

Ghi vao bao cao:

| Muc | Gia tri |
|---|---|
| So anh train eyes_open | TBD |
| So anh train eyes_closed | TBD |
| So anh val eyes_open | TBD |
| So anh val eyes_closed | TBD |
| Image size | 64x64 |
| Epoch | 12 |
| Optimizer | Adam |
| Loss | Categorical crossentropy |

## 5. Danh Gia Model

Sau khi train va co model, chay danh gia tren tap `test/`:

```bash
python tools/evaluate_eye_classifier.py --data dataset --tflite app/src/main/assets/drowsiness_model.tflite
```

Neu ban luu model Keras rieng:

```bash
python tools/evaluate_eye_classifier.py --data dataset --keras model.keras
```

Ket qua duoc luu vao:

```text
outputs/evaluation/metrics.json
outputs/evaluation/confusion_matrix.csv
outputs/evaluation/predictions.csv
```

Dung `metrics.json` va `confusion_matrix.csv` de dien vao Chuong 4.

## 6. Bang Danh Gia

Can co it nhat:

| Model | Accuracy | Precision closed | Recall closed | F1 closed |
|---|---:|---:|---:|---:|
| CNN eye classifier | TBD | TBD | TBD | TBD |

## 7. Cach Giai Thich Ket Qua

Khong nen chi noi accuracy cao. Hay noi:

- Recall cua `eyes_closed` quan trong vi bo sot mat nham lau la nguy hiem.
- Precision quan trong de tranh bao nham qua nhieu.
- Smoothing theo thoi gian giup phan biet chop mat tu nhien voi buon ngu.

## 8. Dua Vao Android

Sau khi co model:

```text
app/src/main/assets/drowsiness_model.tflite
```

Chay app lai. Neu overlay/message hien `CNN ready`, nghia la model da load duoc.
Overlay se hien them dong `CNN eyes_closed` hoac `CNN eyes_open` kem confidence
neu classifier chay thanh cong tren vung mat duoc crop tu MediaPipe landmark.

## 9. Neu Chua Kip Tich Hop CNN Vao Frame

Van co the bao ve theo cach trung thuc:

- Android app da co pipeline realtime MediaPipe + EAR/MAR + alert.
- CNN da duoc train va export sang TFLite.
- Android da co `TfliteDrowsinessClassifier` load/predict model va crop eye ROI.
- Neu chua tune tot, dung CNN line nhu bang chung tich hop, con canh bao cuoi
  van dua vao smoothing EAR/MAR de demo on dinh.

Tuy nhien, neu kip, nen tich hop classify ROI mat de san pham manh hon.
