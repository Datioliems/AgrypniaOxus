# -*- coding: utf-8 -*-
"""
STEP 8 — Train model từ VIDEO THẬT CHỦ ĐÍCH (awake / closed / yawn).
Tự: copy file (tránh path Unicode) → cắt frame → crop mắt/miệng (MediaPipe)
    → train CNN mắt + CNN ngáp → xuất TFLite vào assets cho APP THỨ 2 (realdata).

Chạy: py -3.12 tools/step8_realdata_train.py
"""
import os, shutil, zipfile, math, glob
import cv2, numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mpp
from mediapipe.tasks.python import vision
import tensorflow as tf
from tensorflow.keras import layers, models

# File đã copy sẵn bằng bash sang path ASCII (tránh lỗi Unicode cv2/os trên Windows):
#   /tmp/s8prep/{awake,closed_1,yawn_1}.mp4 + /tmp/s8prep/closed_zip/
WORK = "/tmp/s8prep"
CLOSED_EXTRA = f"{WORK}/closed_zip"
shutil.rmtree(f"{WORK}/eye", ignore_errors=True)
shutil.rmtree(f"{WORK}/yawn", ignore_errors=True)

# ── MediaPipe FaceLandmarker ──
if not os.path.exists("face_landmarker.task"):
    import urllib.request
    urllib.request.urlretrieve("https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task", "face_landmarker.task")
opts = vision.FaceLandmarkerOptions(base_options=mpp.BaseOptions(model_asset_path="face_landmarker.task"),
                                    running_mode=vision.RunningMode.IMAGE, num_faces=1)
LMK = vision.FaceLandmarker.create_from_options(opts)

LEFT = [33, 133, 160, 158, 153, 144]; RIGHT = [362, 263, 385, 387, 373, 380]
MOUTH = [61, 291, 0, 17, 13, 14]

def box(img, P, idxs, padx=0.4, pady=0.6):
    xs = [P(i)[0] for i in idxs]; ys = [P(i)[1] for i in idxs]
    x1, x2, y1, y2 = min(xs), max(xs), min(ys), max(ys); w = x2 - x1; h = y2 - y1
    x1, x2 = int(x1 - w*padx), int(x2 + w*padx); y1, y2 = int(y1 - h*pady), int(y2 + h*pady)
    x1, y1 = max(0, x1), max(0, y1); c = img[y1:y2, x1:x2]
    return cv2.resize(c, (64, 64)) if c.size else None

def imread_u(p):
    """Đọc ảnh chịu được path Unicode trên Windows."""
    try:
        return cv2.imdecode(np.fromfile(p, dtype=np.uint8), cv2.IMREAD_COLOR)
    except Exception:
        return None

def landmarks(img):
    if img is None or img.size == 0:
        return None
    r = LMK.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
    if not r.face_landmarks: return None
    lm = r.face_landmarks[0]; h, w = img.shape[:2]
    return lambda i: (lm[i].x * w, lm[i].y * h)

def frames_of_video(path, step=1):
    cap = cv2.VideoCapture(path); i = 0
    while True:
        ok, fr = cap.read()
        if not ok: break
        if i % step == 0: yield fr
        i += 1
    cap.release()

def save_eye_crops(frames, out_dir):
    os.makedirs(out_dir, exist_ok=True); n = 0
    for fr in frames:
        P = landmarks(fr)
        if P is None: continue
        for idxs in (LEFT, RIGHT):
            c = box(fr, P, idxs)
            if c is not None: cv2.imwrite(f"{out_dir}/e{n}.jpg", c); n += 1
    return n

def save_mouth_crops(frames, out_dir):
    os.makedirs(out_dir, exist_ok=True); n = 0
    for fr in frames:
        P = landmarks(fr)
        if P is None: continue
        c = box(fr, P, MOUTH, padx=0.5, pady=0.5)
        if c is not None: cv2.imwrite(f"{out_dir}/m{n}.jpg", c); n += 1
    return n

# ── Dataset MẮT: open (awake) vs closed (closed_1 + closed.zip) ──
EYE = f"{WORK}/eye"
print("Cropping eyes...")
no = save_eye_crops(frames_of_video(f"{WORK}/awake.mp4"), f"{EYE}/eyes_open")
nc = save_eye_crops(frames_of_video(f"{WORK}/closed_1.mp4"), f"{EYE}/eyes_closed")
nz = save_eye_crops((imread_u(p) for p in glob.glob(f"{CLOSED_EXTRA}/**/*.jpg", recursive=True)), f"{EYE}/eyes_closed")
print(f"  eyes_open={no}  eyes_closed={nc+nz}")

# ── Dataset NGÁP: no_yawn (awake) vs yawn (yawn_1) ──
YAWN = f"{WORK}/yawn"
print("Cropping mouths...")
nn = save_mouth_crops(frames_of_video(f"{WORK}/awake.mp4"), f"{YAWN}/no_yawn")
ny = save_mouth_crops(frames_of_video(f"{WORK}/yawn_1.mp4"), f"{YAWN}/yawn")
print(f"  no_yawn={nn}  yawn={ny}")

def train_cnn(data_dir, classes, out_tflite, epochs=25):
    tr = tf.keras.utils.image_dataset_from_directory(data_dir, image_size=(64,64), batch_size=16,
            label_mode="categorical", class_names=classes, validation_split=0.2, subset="training", seed=42)
    va = tf.keras.utils.image_dataset_from_directory(data_dir, image_size=(64,64), batch_size=16,
            label_mode="categorical", class_names=classes, validation_split=0.2, subset="validation", seed=42)
    norm = layers.Rescaling(1./255)
    tr = tr.map(lambda x,y:(norm(x),y)).cache().prefetch(2); va = va.map(lambda x,y:(norm(x),y)).cache().prefetch(2)
    m = models.Sequential([layers.Input((64,64,3)),
        layers.Conv2D(32,3,activation="relu",padding="same"), layers.BatchNormalization(), layers.MaxPool2D(),
        layers.Conv2D(64,3,activation="relu",padding="same"), layers.BatchNormalization(), layers.GlobalAveragePooling2D(),
        layers.Dense(64,activation="relu"), layers.Dropout(0.4), layers.Dense(len(classes),activation="softmax")])
    m.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    h = m.fit(tr, validation_data=va, epochs=epochs,
              callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=6, restore_best_weights=True)], verbose=2)
    open(out_tflite, "wb").write(tf.lite.TFLiteConverter.from_keras_model(m).convert())
    return max(h.history["val_accuracy"])

# Output cho APP THỨ 2 (flavor realdata) — KHÔNG ghi đè assets gốc
RD = "app/src/realdata/assets"; os.makedirs(RD, exist_ok=True)
print("\nTrain CNN eye...");  acc_e = train_cnn(EYE,  ["eyes_closed","eyes_open"], f"{RD}/drowsiness_model.tflite")
print("Train CNN yawn...");   acc_y = train_cnn(YAWN, ["no_yawn","yawn"],         f"{RD}/yawn_model.tflite")
print(f"\n✅ XONG. Eye val_acc={acc_e*100:.1f}%  Yawn val_acc={acc_y*100:.1f}%")
print(f"   TFLite → {RD}/ (dùng cho app thứ 2 flavor 'realdata')")
