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
WORK = "_s8tmp"   # path tương đối project — bash & python cùng hiểu (tránh /tmp khác namespace)
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

# ── TRỘN dữ liệu GỐC (MRL/Kaggle) để model RD ROBUST với mọi góc mặt ──
# (real-data 1 người/trực diện → dễ false-alarm khi nghiêng; trộn data gốc đa dạng để khắc phục)
import random as _rnd
def add_samples(src, dst, n):
    fs = glob.glob(f"{src}/*.*")
    if not fs: return 0
    k = 0
    for p in _rnd.sample(fs, min(n, len(fs))):
        im = imread_u(p)
        if im is not None and im.size:
            cv2.imwrite(f"{dst}/orig_{k}.jpg", cv2.resize(im, (64, 64))); k += 1
    return k
me1 = add_samples("dataset/train/eyes_open",   f"{EYE}/eyes_open",   1500)
me2 = add_samples("dataset/train/eyes_closed", f"{EYE}/eyes_closed", 1500)
my1 = add_samples("dataset_yawn/train/no_yawn", f"{YAWN}/no_yawn", 1000)
my2 = add_samples("dataset_yawn/train/yawn",    f"{YAWN}/yawn",    1000)
print(f"  + trộn data GỐC: eye(+{me1}/+{me2}) yawn(+{my1}/+{my2}) → robust mọi góc mặt")

# ── TRỘN bộ RGB Datio_yolo (crop theo NHÃN BOX sẵn — ảnh màu khớp camera) ──
# Class: 0=close_eyeL 1=close_eyeR 2=no_yawn 3=open_eyeL 4=open_eyeR 5=yawn
RGB_IMG = "_rgbdata/train/images"; RGB_LBL = "_rgbdata/train/labels"
def crop_yolo(img, parts, pad=0.15):
    cx, cy, w, h = map(float, parts[1:5]); Hh, Ww = img.shape[:2]
    x1 = int((cx - w/2 - w*pad)*Ww); x2 = int((cx + w/2 + w*pad)*Ww)
    y1 = int((cy - h/2 - h*pad)*Hh); y2 = int((cy + h/2 + h*pad)*Hh)
    x1, y1 = max(0, x1), max(0, y1); c = img[y1:y2, x1:x2]
    return cv2.resize(c, (64, 64)) if c.size else None
re_ = ro_ = ry_ = rn_ = 0
for ip in glob.glob(f"{RGB_IMG}/*.jpg"):
    lp = f"{RGB_LBL}/{os.path.splitext(os.path.basename(ip))[0]}.txt"
    if not os.path.exists(lp): continue
    img = imread_u(ip)
    if img is None: continue
    for line in open(lp):
        parts = line.split()
        if len(parts) < 5: continue
        cls = int(parts[0]); c = crop_yolo(img, parts)
        if c is None: continue
        if cls in (0, 1):   cv2.imwrite(f"{EYE}/eyes_closed/rgb_{re_}.jpg", c); re_ += 1
        elif cls in (3, 4): cv2.imwrite(f"{EYE}/eyes_open/rgb_{ro_}.jpg", c);  ro_ += 1
        elif cls == 5:      cv2.imwrite(f"{YAWN}/yawn/rgb_{ry_}.jpg", c);      ry_ += 1
        elif cls == 2:      cv2.imwrite(f"{YAWN}/no_yawn/rgb_{rn_}.jpg", c);   rn_ += 1
print(f"  + RGB Datio_yolo: eye(closed+{re_}/open+{ro_}) yawn(yawn+{ry_}/no+{rn_}) → ảnh màu khớp camera")

def train_cnn(data_dir, classes, out_tflite, epochs=20):
    # Data ít + frame gần trùng → KHÔNG dùng BatchNorm (gây gap train/val), thêm augmentation
    # + class_weight (cân bằng lớp) + dropout cao để chống overfit.
    tr = tf.keras.utils.image_dataset_from_directory(data_dir, image_size=(64,64), batch_size=16,
            label_mode="categorical", class_names=classes, validation_split=0.25, subset="training", seed=123, shuffle=True)
    va = tf.keras.utils.image_dataset_from_directory(data_dir, image_size=(64,64), batch_size=16,
            label_mode="categorical", class_names=classes, validation_split=0.25, subset="validation", seed=123, shuffle=True)
    cnt = [len(glob.glob(f"{data_dir}/{c}/*.jpg")) for c in classes]
    tot = sum(cnt); cw = {i: tot/(len(classes)*max(cnt[i],1)) for i in range(len(classes))}
    aug = models.Sequential([layers.RandomFlip("horizontal"), layers.RandomRotation(0.08),
                             layers.RandomBrightness(0.2), layers.RandomZoom(0.1)])
    norm = layers.Rescaling(1./255)
    tr = tr.map(lambda x,y:(norm(aug(x)),y)).cache().prefetch(2); va = va.map(lambda x,y:(norm(x),y)).cache().prefetch(2)
    m = models.Sequential([layers.Input((64,64,3)),
        layers.Conv2D(16,3,activation="relu",padding="same"), layers.MaxPool2D(),
        layers.Conv2D(32,3,activation="relu",padding="same"), layers.MaxPool2D(),
        layers.Conv2D(64,3,activation="relu",padding="same"), layers.GlobalAveragePooling2D(),
        layers.Dropout(0.5), layers.Dense(len(classes),activation="softmax")])
    m.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    h = m.fit(tr, validation_data=va, epochs=epochs, class_weight=cw,
              callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=8, restore_best_weights=True)], verbose=2)
    open(out_tflite, "wb").write(tf.lite.TFLiteConverter.from_keras_model(m).convert())
    return max(h.history["val_accuracy"])

# Output cho APP THỨ 2 (flavor realdata) — KHÔNG ghi đè assets gốc
RD = "app/src/realdata/assets"; os.makedirs(RD, exist_ok=True)
print("\nTrain CNN eye...");  acc_e = train_cnn(EYE,  ["eyes_closed","eyes_open"], f"{RD}/drowsiness_model.tflite")
print("Train CNN yawn...");   acc_y = train_cnn(YAWN, ["no_yawn","yawn"],         f"{RD}/yawn_model.tflite")
print(f"\n✅ XONG. Eye val_acc={acc_e*100:.1f}%  Yawn val_acc={acc_y*100:.1f}%")
print(f"   TFLite → {RD}/ (dùng cho app thứ 2 flavor 'realdata')")
