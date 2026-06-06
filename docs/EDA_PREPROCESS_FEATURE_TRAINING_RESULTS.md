# EDA, Tien Xu Ly, Feature Engineering Va Huan Luyen Mo Hinh

Tai lieu nay tom tat ket qua thuc nghiem that sau khi chuan hoa dataset va train CNN eye-state model.

## 1. EDA

Nguon du lieu chinh: `rawdata/data`.

Mapping nhan:

| Nhan goc | Nhan chuan | Y nghia |
|---|---|---|
| `awake` | `eyes_open` | Mat mo |
| `sleepy` | `eyes_closed` | Mat nham |

Thong ke sau khi chuan hoa:

| Split | eyes_closed | eyes_open | Total |
|---|---:|---:|---:|
| train | 25,167 | 25,770 | 50,937 |
| val | 8,389 | 8,591 | 16,980 |
| test | 8,390 | 8,591 | 16,981 |
| total | 41,946 | 42,952 | 84,898 |

Kiem tra nhanh EDA:

- Tat ca file trong dataset chuan la anh `.png`.
- Kiem tra mau 2,000 anh moi split/class, khong phat hien anh loi trong mau kiem tra.
- Hai lop can bang tuong doi, phu hop binary image classification.
- Han che: dataset chu yeu la anh mat, chua phai full cabin/dashcam frame.

Artifacts:

```text
outputs/dataset_summary/dataset_summary.csv
outputs/dataset_summary/dataset_summary.json
outputs/eda/eda_summary.csv
outputs/eda/eda_summary.json
outputs/eda/class_distribution.png
outputs/eda/sample_grid.png
```

## 2. Tien Xu Ly

Cau truc du lieu chuan:

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

Lenh da dung:

```bat
python tools\prepare_eye_dataset.py --source rawdata\data --out dataset --clean
python tools\summarize_dataset.py --data dataset --out outputs\dataset_summary
python tools\eda_eye_dataset.py --data dataset --out outputs\eda --max-open-per-class 2000
```

Tien xu ly trong train:

- Resize anh ve `64x64`.
- Doc anh thanh 3 kenh RGB.
- Normalize pixel ve `[0,1]`.
- Khong dat lop `Rescaling(1/255)` trong TFLite model, vi Android app da dua input ve `[0,1]`.

## 3. Feature Engineering

Feature engineering trong de tai gom hai phan:

| Nhom dac trung | Ky thuat | Vai tro |
|---|---|---|
| Dac trung hinh hoc | MediaPipe landmarks, EAR, MAR | Giai thich duoc, lam baseline realtime |
| Dac trung hoc sau | CNN tren anh mat `64x64x3` | Tu hoc canh, texture, mi mat, vung sang/toi |

Cau giai thich ngan:

```text
MediaPipe giai quyet "mat o dau".
EAR/MAR giai thich duoc trang thai mat/mieng theo hinh hoc.
CNN giai quyet "mat dang mo hay nham" bang dac trung hoc tu anh.
```

YOLO khong duoc dung trong model v1 vi can bounding box labels va Android post-processing phuc tap hon.

## 4. Huan Luyen Mo Hinh

Framework:

```text
Python + TensorFlow/Keras -> TensorFlow Lite
```

Kien truc CNN:

```text
Input 64x64x3
-> Conv2D(24, relu)
-> MaxPooling2D
-> Conv2D(48, relu)
-> MaxPooling2D
-> Conv2D(64, relu)
-> GlobalAveragePooling2D
-> Dropout(0.25)
-> Dense(2, softmax)
```

Lenh train:

```bat
python tools\train_eye_classifier.py --data dataset --out app\src\main\assets\drowsiness_model.tflite --epochs 12 --early-stop-patience 3
```

Ket qua validation cuoi:

| Metric | Gia tri |
|---|---:|
| train accuracy | 0.9635 |
| train loss | 0.1032 |
| val accuracy | 0.9744 |
| val loss | 0.0720 |

Class order:

```text
['eyes_closed', 'eyes_open']
```

Class order nay khop voi Android classifier:

```text
eyes_closed, eyes_open
```

Artifacts:

```text
app/src/main/assets/drowsiness_model.tflite
outputs/training/drowsiness_model.keras
outputs/training/training_history.json
outputs/training/model_summary.txt
outputs/training/class_names.json
outputs/training/training_curves.png
```

## 5. Danh Gia Model

Lenh evaluate:

```bat
python tools\evaluate_eye_classifier.py --data dataset --tflite app\src\main\assets\drowsiness_model.tflite
```

Ket qua test:

| Metric | Gia tri |
|---|---:|
| Accuracy | 0.9721 |
| Precision `eyes_closed` | 0.9608 |
| Recall `eyes_closed` | 0.9838 |
| F1 `eyes_closed` | 0.9721 |
| Precision `eyes_open` | 0.9838 |
| Recall `eyes_open` | 0.9608 |
| F1 `eyes_open` | 0.9721 |

Confusion matrix:

| Actual / Predicted | eyes_closed | eyes_open |
|---|---:|---:|
| eyes_closed | 8,254 | 136 |
| eyes_open | 337 | 8,254 |

Nhan xet:

- Model dat accuracy test 97.21%.
- Recall lop `eyes_closed` dat 98.38%, phu hop bai toan canh bao vi bo sot mat nham nguy hiem hon bao nham nhe.
- Model da export TFLite va san sang tich hop Android.

Artifacts:

```text
outputs/evaluation/metrics.json
outputs/evaluation/confusion_matrix.csv
outputs/evaluation/confusion_matrix.png
outputs/evaluation/predictions.csv
```
