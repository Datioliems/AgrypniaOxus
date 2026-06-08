# -*- coding: utf-8 -*-
"""
Sinh notebook ALL-IN-ONE chạy toàn bộ pipeline dự án trong VS Code:
CNN-mắt, CNN-ngáp, YOLOv11, YOLO26, RT-DETR, RF-DETR -> tổng hợp -> model cuối (TFLite + fusion).
Chạy: py -3.12 tools/build_allinone_notebook.py  ->  notebooks/AllInOne_Pipeline.ipynb
"""
import nbformat as nbf
import os

nb = nbf.v4.new_notebook()
cells = []
def md(t): cells.append(nbf.v4.new_markdown_cell(t))
def code(t): cells.append(nbf.v4.new_code_cell(t))

md("""# Pipeline toàn bộ dự án — Phát hiện buồn ngủ khi lái xe
**Môn IS54A · Chạy trong Visual Studio Code (kernel Python 3.12 có TensorFlow).**

Notebook gom toàn bộ quy trình của đề tài thành một mạch chạy:
1. Chuẩn bị môi trường và dữ liệu
2. Huấn luyện **CNN phân loại trạng thái mắt**
3. Huấn luyện **CNN phân loại ngáp**
4. Huấn luyện **YOLOv11** (phát hiện 6 lớp)
5. Huấn luyện **YOLO26** (NMS-free)
6. Huấn luyện **RT-DETR** (Transformer)
7. Huấn luyện **RF-DETR** (Transformer, backbone DINOv2)
8. **Tổng hợp & so sánh** các mô hình
9. **Mô hình cuối**: hợp nhất CNN + EAR/MAR, xuất TFLite cho Android

> Các phần CNN (mục 2–3, 8–9) chạy được trực tiếp trên máy. Các phần phát hiện vật thể
> (mục 4–7) cần GPU; có thể chạy trên Google Colab hoặc máy có CUDA — mã giữ nguyên.""")

md("""## 0. Chuẩn bị môi trường
Kiểm tra TensorFlow (cho CNN) và PyTorch/Ultralytics (cho phát hiện vật thể).""")
code("""import sys, os, glob, json
import numpy as np
print("Python:", sys.version.split()[0])
try:
    import tensorflow as tf
    print("TensorFlow:", tf.__version__, "| GPU:", tf.config.list_physical_devices('GPU'))
except Exception as e:
    print("TensorFlow chưa sẵn sàng:", e)
try:
    import cv2; print("OpenCV:", cv2.__version__)
except Exception as e:
    print("OpenCV:", e)""")

md("""## 1. Dữ liệu sử dụng
- **CNN mắt/ngáp**: bộ ảnh đã cắt vùng mắt/miệng (MRL Eye, Kaggle Yawn) + bộ RGB *Datio_yolo*.
- **Phát hiện vật thể**: bộ RGB *Datio_yolo* 6 lớp (close_eyeL/R, open_eyeL/R, yawn, no_yawn) tải từ Roboflow.

Cấu hình đường dẫn (sửa cho đúng máy của bạn).""")
code("""# Thư mục ảnh đã cắt sẵn cho CNN (mỗi lớp một thư mục con)
EYE_DIR  = "dataset"        # dataset/train/{eyes_open,eyes_closed}
YAWN_DIR = "dataset_yawn"   # dataset_yawn/train/{no_yawn,yawn}
IMG_SIZE = 64
ASSETS   = "app/src/main/assets"   # nơi lưu TFLite cho Android
os.makedirs(ASSETS, exist_ok=True)
print("EYE_DIR tồn tại:", os.path.isdir(EYE_DIR), "| YAWN_DIR:", os.path.isdir(YAWN_DIR))""")

md("""## 2. CNN phân loại trạng thái mắt
**Lý thuyết:** Mạng tích chập (CNN) tự học đặc trưng phân cấp qua các bộ lọc (kernel) — đây là
kiến thức Thị giác máy tính đã học. Mô hình gọn nhẹ gồm ba khối tích chập tăng dần số bộ lọc,
kết hợp gộp cực đại (MaxPool) và gộp trung bình toàn cục (GAP), thêm Dropout chống quá khớp.""")
code("""from tensorflow.keras import layers, models

def build_cnn(n_classes=2, img=IMG_SIZE):
    return models.Sequential([
        layers.Input((img, img, 3)),
        layers.Conv2D(32, 3, activation="relu", padding="same"), layers.BatchNormalization(), layers.MaxPool2D(),
        layers.Conv2D(64, 3, activation="relu", padding="same"), layers.BatchNormalization(), layers.MaxPool2D(),
        layers.Conv2D(128, 3, activation="relu", padding="same"), layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation="relu"), layers.Dropout(0.4),
        layers.Dense(n_classes, activation="softmax"),
    ])

def make_ds(data_dir, classes):
    tr = tf.keras.utils.image_dataset_from_directory(f"{data_dir}/train", image_size=(IMG_SIZE, IMG_SIZE),
            batch_size=32, label_mode="categorical", class_names=classes, validation_split=0.2, subset="training", seed=42)
    va = tf.keras.utils.image_dataset_from_directory(f"{data_dir}/train", image_size=(IMG_SIZE, IMG_SIZE),
            batch_size=32, label_mode="categorical", class_names=classes, validation_split=0.2, subset="validation", seed=42)
    aug = models.Sequential([layers.RandomFlip("horizontal"), layers.RandomRotation(0.08), layers.RandomBrightness(0.2)])
    norm = layers.Rescaling(1./255)
    tr = tr.map(lambda x, y: (norm(aug(x)), y)).prefetch(2)
    va = va.map(lambda x, y: (norm(x), y)).prefetch(2)
    return tr, va

if os.path.isdir(f"{EYE_DIR}/train"):
    eye_classes = ["eyes_closed", "eyes_open"]
    tr, va = make_ds(EYE_DIR, eye_classes)
    eye_model = build_cnn(2)
    eye_model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    h_eye = eye_model.fit(tr, validation_data=va, epochs=15,
        callbacks=[tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)], verbose=2)
    print("Eye val_acc tốt nhất:", max(h_eye.history["val_accuracy"]))
else:
    print("Bỏ qua: chưa có", EYE_DIR)""")

md("""## 3. CNN phân loại ngáp
Cùng kiến trúc CNN, huấn luyện trên ảnh vùng miệng để phân biệt **ngáp / không ngáp**.""")
code("""if os.path.isdir(f"{YAWN_DIR}/train"):
    yawn_classes = ["no_yawn", "yawn"]
    tr, va = make_ds(YAWN_DIR, yawn_classes)
    yawn_model = build_cnn(2)
    yawn_model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    h_yawn = yawn_model.fit(tr, validation_data=va, epochs=15,
        callbacks=[tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)], verbose=2)
    print("Yawn val_acc tốt nhất:", max(h_yawn.history["val_accuracy"]))
else:
    print("Bỏ qua: chưa có", YAWN_DIR)""")

md("""## 4. YOLOv11 — phát hiện vật thể 6 lớp
**Lý thuyết:** YOLO chia ảnh thành lưới và dự đoán trực tiếp khung bao + lớp trong một lượt
truyền (one-stage), nhờ đó nhanh. Mô hình nạp trọng số tiền huấn luyện COCO rồi tinh chỉnh.
> Cần GPU. Tải dữ liệu từ Roboflow (định dạng YOLO).""")
code("""# !pip install -q ultralytics roboflow
RUN_DETECT = False   # đặt True khi chạy trên GPU
if RUN_DETECT:
    from roboflow import Roboflow
    rf = Roboflow(api_key="qI3lEKlNpIZpNENdk3MH")
    ds = rf.workspace("nguyen-tuan-dat").project("datio_yolo").version(1).download("yolov11")
    from ultralytics import YOLO
    m = YOLO("yolo11s.pt")
    m.train(data=f"{ds.location}/data.yaml", epochs=60, imgsz=640, batch=16, name="yolo11_datio")
    print(m.val().box.map, m.val().box.map50)
else:
    print("RUN_DETECT=False — kết quả đã có: YOLOv11s mAP@50-95 = 73.5%, mAP@50 = 97.0%")""")

md("""## 5. YOLO26 — phiên bản NMS-free
YOLO26 dùng đầu dự đoán một-một (end-to-end), loại bỏ bước hậu xử lý NMS và tối ưu MuSGD.""")
code("""if RUN_DETECT:
    rf = Roboflow(api_key="qI3lEKlNpIZpNENdk3MH")
    ds26 = rf.workspace("nguyen-tuan-dat").project("datio_yolo").version(1).download("yolo26")
    from ultralytics import YOLO
    m26 = YOLO("yolo26s.pt")
    m26.train(data=f"{ds26.location}/data.yaml", epochs=60, imgsz=640, batch=16, name="yolo26_datio")
else:
    print("RUN_DETECT=False — kết quả đã có: YOLO26s mAP@50-95 = 71.4%")""")

md("""## 6. RT-DETR — Transformer thời gian thực
RT-DETR dùng bộ giải mã Transformer với cơ chế chú ý toàn cục, không cần anchor/NMS.""")
code("""if RUN_DETECT:
    from ultralytics import RTDETR
    rt = RTDETR("rtdetr-l.pt")
    rt.train(data=f"{ds.location}/data.yaml", epochs=10, imgsz=640, batch=8, name="rtdetr_datio")
else:
    print("RUN_DETECT=False — kết quả đã có: RT-DETR mAP@50-95 = 70.7% (10 epoch)")""")

md("""## 7. RF-DETR — Transformer backbone DINOv2
RF-DETR (Roboflow) dùng backbone DINOv2 tiền huấn luyện, hội tụ nhanh với ít epoch.""")
code("""# !pip install -q "rfdetr[train]"
if RUN_DETECT:
    from rfdetr import RFDETRSmall
    rfm = RFDETRSmall()
    rfm.train(dataset_dir="<COCO_FORMAT_DIR>", epochs=10, batch_size=4, grad_accum_steps=4)
else:
    print("RUN_DETECT=False — kết quả đã có: RF-DETR mAP@50-95 = 67.3% (10 epoch)")""")

md("""## 8. Tổng hợp & so sánh các mô hình
Gom kết quả đã đo của tất cả mô hình để so sánh trực quan.""")
code("""import matplotlib.pyplot as plt
results = {
    "CNN mắt (phân loại)":  97.44,
    "CNN ngáp (phân loại)": 98.64,
    "YOLOv11s (mAP50-95)":  73.5,
    "YOLO26s (mAP50-95)":   71.4,
    "RT-DETR (mAP50-95)":   70.7,
    "RF-DETR (mAP50-95)":   67.3,
}
fig, ax = plt.subplots(figsize=(9, 4))
colors = ["#4CAF50", "#4CAF50", "#2196F3", "#2196F3", "#FF9800", "#FF9800"]
ax.barh(list(results.keys())[::-1], list(results.values())[::-1], color=colors[::-1])
for i, v in enumerate(list(results.values())[::-1]): ax.text(v + 0.5, i, f"{v}", va="center")
ax.set_xlabel("Điểm số (%)"); ax.set_xlim(0, 105)
ax.set_title("Tổng hợp kết quả các mô hình trong dự án", fontweight="bold")
plt.tight_layout(); plt.savefig("outputs/allinone_compare.png", dpi=120, bbox_inches="tight"); plt.show()
print("Phân loại dùng Accuracy; phát hiện vật thể dùng mAP@50-95 (không so sánh trực tiếp giữa hai nhóm).")""")

md("""## 9. Mô hình cuối — hợp nhất và xuất TFLite cho Android
**Mô hình cuối** của hệ thống không phải một mạng duy nhất mà là **cơ chế hợp nhất lấy CNN làm chủ đạo**:
CNN quyết định khi đủ tin cậy, EAR/MAR (đặc trưng hình học từ MediaPipe) làm lớp an toàn dự phòng.
Hai CNN được xuất sang TensorFlow Lite để chạy on-device.""")
code("""# Xuất TFLite (nếu đã huấn luyện ở mục 2-3)
def export_tflite(model, path):
    open(path, "wb").write(tf.lite.TFLiteConverter.from_keras_model(model).convert())
    print("Đã lưu:", path, "-", os.path.getsize(path), "bytes")

try:
    export_tflite(eye_model,  f"{ASSETS}/drowsiness_model.tflite")
    export_tflite(yawn_model, f"{ASSETS}/yawn_model.tflite")
except NameError:
    print("Chưa có eye_model/yawn_model — hãy chạy mục 2 và 3 trước.")""")
code('''# Mô tả logic hợp nhất CNN-primary (triển khai thực tế trong app Android - Kotlin)
fusion_pseudocode = """
trạng_thái_buồn_ngủ(frame):
    landmark = MediaPipe(frame)              # 478 điểm
    mắt_crop, miệng_crop = cắt_ROI(landmark)
    p_eye  = CNN_mắt(mắt_crop)               # xác suất nhắm mắt
    p_yawn = CNN_ngáp(miệng_crop)            # xác suất ngáp
    ear, mar = tính_EAR_MAR(landmark)        # đặc trưng hình học (an toàn)
    nếu max(p_eye) >= 0.52:                   # CNN đủ tin cậy -> CHỦ ĐẠO
        nhắm = (argmax(p_eye) == 'closed')
    ngược lại:                                # CNN không chắc -> dự phòng
        nhắm = (ear < ngưỡng_EAR)
    buồn_ngủ = duy_trì(nhắm) hoặc (p_yawn cao) hoặc (mar > ngưỡng_MAR)  # lớp an toàn
    return buồn_ngủ
"""
print(fusion_pseudocode)''')

md("""## Kết luận
Notebook đã đi qua toàn bộ pipeline: từ dữ liệu, huấn luyện sáu mô hình (hai CNN phân loại và
bốn mô hình phát hiện vật thể), tổng hợp so sánh, đến mô hình cuối hợp nhất CNN với đặc trưng
hình học và xuất TFLite cho Android. Kết quả và phân tích trung thực được trình bày chi tiết
trong báo cáo (Chương 2–4).""")

nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"display_name": "Python 3.12", "language": "python", "name": "python3"},
                  "language_info": {"name": "python", "version": "3.12"}}
os.makedirs("notebooks", exist_ok=True)
out = "notebooks/AllInOne_Pipeline.ipynb"
nbf.write(nb, out)
print("OK ->", out, "|", len(cells), "cells")
