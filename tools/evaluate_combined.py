# -*- coding: utf-8 -*-
"""
ĐÁNH GIÁ ĐẦY ĐỦ CHỈ SỐ — so sánh model GỐC (public) vs COMBINED (real+MRL+RGB).
Test trên RGB Datio_yolo VALID (held-out, model CHƯA train) → công bằng + đúng modality camera.
Xuất: Accuracy/Precision/Recall/F1 per-class + Confusion Matrix.

Chạy: py -3.12 tools/evaluate_combined.py
"""
import glob, os, random
import cv2, numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

OUT = "outputs/evaluation"; os.makedirs(OUT, exist_ok=True)
RGB_VAL_IMG = "_rgbdata/valid/images"; RGB_VAL_LBL = "_rgbdata/valid/labels"

def imread_u(p): return cv2.imdecode(np.fromfile(p, np.uint8), cv2.IMREAD_COLOR)
def crop_yolo(img, parts, pad=0.15):
    cx, cy, w, h = map(float, parts[1:5]); H, W = img.shape[:2]
    x1 = int((cx-w/2-w*pad)*W); x2 = int((cx+w/2+w*pad)*W); y1 = int((cy-h/2-h*pad)*H); y2 = int((cy+h/2+h*pad)*H)
    x1, y1 = max(0, x1), max(0, y1); c = img[y1:y2, x1:x2]
    return cv2.resize(c, (64, 64)) if c.size else None

# Build test set RGB held-out: crop theo nhãn box (0,1=closed 3,4=open | 2=no_yawn 5=yawn)
eye_X, eye_y, yawn_X, yawn_y = [], [], [], []
for ip in glob.glob(f"{RGB_VAL_IMG}/*.jpg"):
    lp = f"{RGB_VAL_LBL}/{os.path.splitext(os.path.basename(ip))[0]}.txt"
    if not os.path.exists(lp): continue
    img = imread_u(ip)
    if img is None: continue
    for line in open(lp):
        p = line.split()
        if len(p) < 5: continue
        cls = int(p[0]); c = crop_yolo(img, p)
        if c is None: continue
        x = cv2.cvtColor(c, cv2.COLOR_BGR2RGB).astype("float32")/255.0
        if cls in (0, 1):   eye_X.append(x); eye_y.append(0)   # closed
        elif cls in (3, 4): eye_X.append(x); eye_y.append(1)   # open
        elif cls == 2:      yawn_X.append(x); yawn_y.append(0) # no_yawn
        elif cls == 5:      yawn_X.append(x); yawn_y.append(1) # yawn
eye_X = np.array(eye_X, "float32"); eye_y = np.array(eye_y)
yawn_X = np.array(yawn_X, "float32"); yawn_y = np.array(yawn_y)
print(f"Test RGB held-out: eye={len(eye_y)} (closed={sum(eye_y==0)}/open={sum(eye_y==1)}) yawn={len(yawn_y)}")

def predict(tflite, X):
    it = tf.lite.Interpreter(tflite); it.allocate_tensors()
    i = it.get_input_details()[0]; o = it.get_output_details()[0]
    out = []
    for x in X:
        it.set_tensor(i["index"], x[None]); it.invoke()
        out.append(int(np.argmax(it.get_tensor(o["index"])[0])))
    return np.array(out)

def report(name, tflite, X, y, classes):
    if not os.path.exists(tflite) or len(y) == 0:
        print(f"\n[{name}] skip (thiếu model/data)"); return None
    yp = predict(tflite, X)
    acc = accuracy_score(y, yp)
    print(f"\n========== {name}  |  Accuracy = {acc*100:.2f}% ==========")
    print(classification_report(y, yp, target_names=classes, digits=4, zero_division=0))
    return confusion_matrix(y, yp), acc

print("\n############ MẮT (eyes_closed / eyes_open) ############")
r1 = report("MẮT - GỐC (public, MRL/IR)",        "app/src/main/assets/drowsiness_model.tflite",     eye_X, eye_y, ["closed","open"])
r2 = report("MẮT - COMBINED (real+MRL+RGB)",     "app/src/realdata/assets/drowsiness_model.tflite", eye_X, eye_y, ["closed","open"])
print("\n############ NGÁP (no_yawn / yawn) ############")
r3 = report("NGÁP - GỐC (public)",               "app/src/main/assets/yawn_model.tflite",           yawn_X, yawn_y, ["no_yawn","yawn"])
r4 = report("NGÁP - COMBINED (real+Kaggle+RGB)", "app/src/realdata/assets/yawn_model.tflite",       yawn_X, yawn_y, ["no_yawn","yawn"])

# Confusion matrix 2x2 cho 4 model
fig, ax = plt.subplots(2, 2, figsize=(11, 10))
for a, r, t, cls in zip(ax.flat, [r1, r2, r3, r4],
        ["MẮT GỐC", "MẮT COMBINED", "NGÁP GỐC", "NGÁP COMBINED"],
        [["closed","open"]]*2 + [["no_yawn","yawn"]]*2):
    if r is None: a.axis("off"); continue
    cm = r[0]
    a.imshow(cm, cmap="Blues"); a.set_title(f"{t}  (acc {r[1]*100:.1f}%)", fontweight="bold")
    a.set_xticks([0,1]); a.set_xticklabels(cls); a.set_yticks([0,1]); a.set_yticklabels(cls)
    a.set_xlabel("Dự đoán"); a.set_ylabel("Thực tế")
    for ii in range(2):
        for jj in range(2): a.text(jj, ii, cm[ii,jj], ha="center", va="center", fontsize=16,
                                    color="white" if cm[ii,jj] > cm.max()/2 else "black")
fig.suptitle("Confusion Matrix — Gốc vs Combined (test RGB held-out)", fontsize=14, fontweight="bold")
plt.tight_layout(); plt.savefig(f"{OUT}/confusion_compare.png", dpi=120, bbox_inches="tight")
print(f"\n✅ Confusion matrix → {OUT}/confusion_compare.png")
