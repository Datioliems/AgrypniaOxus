# -*- coding: utf-8 -*-
"""
Chạy pipeline AgrypniaOxus (MediaPipe + EAR/MAR + CNN-primary) lên VIDEO THẬT
→ xuất video có overlay (state, EAR, MAR, alertSource) làm demo "robustness" cho báo cáo.

Chạy:  py -3.12 tools/process_real_video.py "đường_dẫn_video.mp4"
"""
import sys, os, math
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mpp
from mediapipe.tasks.python import vision
import tensorflow as tf

VIDEO = sys.argv[1] if len(sys.argv) > 1 else "C:/Users/ADMIN/Videos/Video Project 8.mp4"
OUT_DIR = "outputs/real_video"; os.makedirs(OUT_DIR, exist_ok=True)
OUT = f"{OUT_DIR}/{os.path.splitext(os.path.basename(VIDEO))[0]}_annotated.mp4"

# ── MediaPipe FaceLandmarker (VIDEO mode) ──
opts = vision.FaceLandmarkerOptions(
    base_options=mpp.BaseOptions(model_asset_path="face_landmarker.task"),
    running_mode=vision.RunningMode.VIDEO, num_faces=1)
LMK = vision.FaceLandmarker.create_from_options(opts)

# ── CNN mắt (TFLite) ──
eye = tf.lite.Interpreter("app/src/main/assets/drowsiness_model.tflite"); eye.allocate_tensors()
ei, eo = eye.get_input_details()[0], eye.get_output_details()[0]
EYE = ["eyes_closed", "eyes_open"]   # index 0,1

LEFT = [33, 160, 158, 133, 153, 144]; RIGHT = [362, 385, 387, 263, 373, 380]

def _ear(p):
    v = math.dist(p[1], p[5]) + math.dist(p[2], p[4]); h = 2 * math.dist(p[0], p[3]) + 1e-6
    return v / h

def crop_eye(img, P, idxs):
    xs = [P(i)[0] for i in idxs]; ys = [P(i)[1] for i in idxs]
    x1, x2, y1, y2 = min(xs), max(xs), min(ys), max(ys); w = x2 - x1; h = y2 - y1
    x1, x2, y1, y2 = int(x1 - w*0.4), int(x2 + w*0.4), int(y1 - h*0.6), int(y2 + h*0.6)
    x1, y1 = max(0, x1), max(0, y1); c = img[y1:y2, x1:x2]
    return cv2.resize(c, (64, 64)) if c.size else None

def cnn_eye(crop):
    x = (cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype("float32") / 255.0)[None]
    eye.set_tensor(ei["index"], x); eye.invoke()
    p = eye.get_tensor(eo["index"])[0]; k = int(np.argmax(p))
    return EYE[k], float(p[k])

cap = cv2.VideoCapture(VIDEO)
fps = cap.get(cv2.CAP_PROP_FPS) or 30
W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
vw = cv2.VideoWriter(OUT, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

closed_frames = 0; drowsy_n = yawn_n = cnn_used = total = 0; idx = 0
DROWSY_FRAMES = int(1.2 * fps)   # nhắm mắt ≥1.2s → DROWSY
while True:
    ok, fr = cap.read()
    if not ok: break
    ts = int(idx * 1000 / fps)
    res = LMK.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB,
                                        data=cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)), ts)
    state, ear_v, mar_v, src = "NO_FACE", 0.0, 0.0, "-"
    if res.face_landmarks:
        total += 1
        lm = res.face_landmarks[0]; P = lambda i: (lm[i].x * W, lm[i].y * H)
        ear_v = (_ear([P(i) for i in LEFT]) + _ear([P(i) for i in RIGHT])) / 2
        mar_v = math.dist(P(13), P(14)) / (math.dist(P(78), P(308)) + 1e-6)
        # CNN mắt (chủ đạo)
        cl = crop_eye(fr, P, LEFT); label, conf = (None, 0.0)
        if cl is not None: label, conf = cnn_eye(cl)
        cnn_closed = label == "eyes_closed" and conf >= 0.52
        if cnn_closed or (label is None and ear_v < 0.24):
            closed_frames += 1
        else:
            closed_frames = 0
        src = "CNN Eye (primary)" if conf >= 0.52 else "EAR/MAR (fallback)"
        if conf >= 0.52: cnn_used += 1
        is_yawn = mar_v > 0.58
        if closed_frames >= DROWSY_FRAMES:
            state = "DROWSY"; drowsy_n += 1
        elif is_yawn:
            state = "YAWNING"; yawn_n += 1
        elif closed_frames > 0:
            state = "EYES_CLOSED"
        else:
            state = "AWAKE"
    # overlay
    col = {"AWAKE": (0, 200, 0), "EYES_CLOSED": (0, 200, 230), "YAWNING": (0, 140, 255),
           "DROWSY": (40, 40, 255), "NO_FACE": (150, 150, 150)}[state]
    cv2.rectangle(fr, (0, 0), (560, 130), (19, 19, 19), -1)
    cv2.putText(fr, f"{state}", (15, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.1, col, 3)
    cv2.putText(fr, f"EAR {ear_v:.2f}  MAR {mar_v:.2f}", (15, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230, 230, 230), 1)
    cv2.putText(fr, f"src: {src}", (15, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (252, 28, 70), 2)
    vw.write(fr); idx += 1

cap.release(); vw.release()
print(f"OK -> {OUT}")
print(f"frames={idx} face={total} CNN-primary={cnn_used} DROWSY={drowsy_n} YAWNING={yawn_n}")
