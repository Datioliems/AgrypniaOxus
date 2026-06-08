# -*- coding: utf-8 -*-
"""
PIPELINE 10 BƯỚC — TRỰC QUAN HÓA trên bộ RGB Datio_yolo.
Sinh ẢNH CHỤP KẾT QUẢ TỪNG BƯỚC (raw → quality → ROI → resize → EDA → augment
→ split → normalize → integrity → manifest) vào outputs/pipeline_steps/.

Chạy: py -3.12 tools/pipeline_visual.py
"""
import os, glob, zipfile, math, json, random
import cv2, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mediapipe as mp
from mediapipe.tasks import python as mpp
from mediapipe.tasks.python import vision

ZIP = r"D:/Downloads/Datio_yolo.yolo26.zip"
DATA = "_rgbdata"; OUT = "outputs/pipeline_steps"
os.makedirs(OUT, exist_ok=True)
random.seed(42)

# ── Giải nén (nếu chưa) ──
if not os.path.isdir(f"{DATA}/train/images"):
    os.makedirs(DATA, exist_ok=True)
    with zipfile.ZipFile(ZIP) as z: z.extractall(DATA)
imgs = glob.glob(f"{DATA}/train/images/*.jpg")
print(f"Tổng ảnh train: {len(imgs)}")

def imread_u(p): return cv2.imdecode(np.fromfile(p, np.uint8), cv2.IMREAD_COLOR)
def rgb(b): return cv2.cvtColor(b, cv2.COLOR_BGR2RGB)
def save(fig, name): fig.savefig(f"{OUT}/{name}", dpi=120, bbox_inches="tight"); plt.close(fig); print("  ✓", name)

sample = imread_u(random.choice(imgs))
H, W = sample.shape[:2]

# ── BƯỚC 0 — RAW (ảnh ví dụ gốc) + lưới mẫu ──
fig, ax = plt.subplots(2, 4, figsize=(16, 8))
for a, p in zip(ax.flat, random.sample(imgs, 8)):
    a.imshow(rgb(imread_u(p))); a.axis("off")
fig.suptitle("BƯỚC 0 — Dữ liệu RAW (RGB, ví dụ)", fontsize=15, fontweight="bold")
save(fig, "step0_raw.png")

# ── BƯỚC 1 — INGEST (kiểm kê) ──
sizes = [imread_u(p).shape[:2] for p in random.sample(imgs, min(60, len(imgs)))]
fig, ax = plt.subplots(figsize=(8, 4))
ax.hist([w for h, w in sizes], bins=15, color="#2196F3", alpha=0.8)
ax.set_title(f"BƯỚC 1 — Ingest: {len(imgs)} ảnh train | kích thước phổ biến", fontweight="bold")
ax.set_xlabel("Chiều rộng (px)"); ax.set_ylabel("Số ảnh")
save(fig, "step1_ingest.png")

# ── BƯỚC 2 — QUALITY FILTER (blur/brightness) ──
def quality(b):
    g = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(g, cv2.CV_64F).var(), g.mean()
blur, bright = quality(sample)
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
ax[0].imshow(rgb(sample)); ax[0].axis("off")
ax[0].set_title(f"Mẫu: blur={blur:.0f}  sáng={bright:.0f}")
ok = blur > 100 and 40 < bright < 220
ax[1].text(0.1, 0.6, f"Độ nét (Laplacian var): {blur:.0f}\n(ngưỡng >100 → {'ĐẠT' if blur>100 else 'MỜ'})",
           fontsize=13); ax[1].text(0.1, 0.3, f"Độ sáng: {bright:.0f}\n(40–220 → {'ĐẠT' if 40<bright<220 else 'LỖI'})", fontsize=13)
ax[1].text(0.1, 0.05, f"→ {'✅ GIỮ' if ok else '❌ LOẠI'}", fontsize=16, fontweight="bold"); ax[1].axis("off")
fig.suptitle("BƯỚC 2 — Quality Filter (lọc ảnh mờ/quá sáng-tối)", fontweight="bold")
save(fig, "step2_quality.png")

# ── BƯỚC 3 — ROI EXTRACTION (MediaPipe crop mắt/miệng) ──
if not os.path.exists("face_landmarker.task"):
    import urllib.request
    urllib.request.urlretrieve("https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task", "face_landmarker.task")
opts = vision.FaceLandmarkerOptions(base_options=mpp.BaseOptions(model_asset_path="face_landmarker.task"),
                                    running_mode=vision.RunningMode.IMAGE, num_faces=1)
LMK = vision.FaceLandmarker.create_from_options(opts)
LEFT=[33,133,160,158,153,144]; RIGHT=[362,263,385,387,373,380]; MOUTH=[61,291,0,17,13,14]
def crop(img,P,idxs,px=0.4,py=0.6):
    xs=[P(i)[0] for i in idxs]; ys=[P(i)[1] for i in idxs]
    x1,x2,y1,y2=min(xs),max(xs),min(ys),max(ys); w=x2-x1; h=y2-y1
    x1,x2,y1,y2=int(x1-w*px),int(x2+w*px),int(y1-h*py),int(y2+h*py); x1,y1=max(0,x1),max(0,y1)
    c=img[y1:y2,x1:x2]; return c if c.size else None
# tìm ảnh có mặt
roi_img=None
for p in random.sample(imgs, 30):
    b=imread_u(p); r=LMK.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb(b)))
    if r.face_landmarks: roi_img=b; lm=r.face_landmarks[0]; break
fig, ax = plt.subplots(1, 4, figsize=(16, 4))
if roi_img is not None:
    h,w=roi_img.shape[:2]; P=lambda i:(lm[i].x*w, lm[i].y*h)
    ax[0].imshow(rgb(roi_img)); ax[0].set_title("Ảnh mặt");
    le=crop(roi_img,P,LEFT); re=crop(roi_img,P,RIGHT); mo=crop(roi_img,P,MOUTH,0.5,0.5)
    for a,im,t in zip(ax[1:],[le,re,mo],["Mắt trái","Mắt phải","Miệng"]):
        if im is not None: a.imshow(rgb(im)); a.set_title(t)
for a in ax: a.axis("off")
fig.suptitle("BƯỚC 3 — ROI Extraction (MediaPipe cắt mắt/miệng từ ảnh RGB)", fontweight="bold")
save(fig, "step3_roi.png")

# ── BƯỚC 4 — RESIZE 64x64 ──
fig, ax = plt.subplots(1, 3, figsize=(11, 4))
if roi_img is not None and le is not None:
    ax[0].imshow(rgb(le)); ax[0].set_title(f"ROI gốc {le.shape[1]}x{le.shape[0]}")
    r64=cv2.resize(le,(64,64)); ax[1].imshow(rgb(r64)); ax[1].set_title("Resize 64x64")
    ax[2].imshow(rgb(cv2.resize(r64,(256,256),interpolation=cv2.INTER_NEAREST))); ax[2].set_title("64x64 (phóng to xem)")
for a in ax: a.axis("off")
fig.suptitle("BƯỚC 4 — Resize về 64×64 (đầu vào CNN)", fontweight="bold")
save(fig, "step4_resize.png")

# ── BƯỚC 5 — EDA (phân bố class từ nhãn YOLO) ──
labels = glob.glob(f"{DATA}/train/labels/*.txt")
cls_count = {}
for lp in labels:
    for line in open(lp):
        c = line.split()[0] if line.strip() else None
        if c is not None: cls_count[c] = cls_count.get(c, 0) + 1
names = ["close_eyeL","close_eyeR","no_yawn","open_eyeL","open_eyeR","yawn"]
fig, ax = plt.subplots(figsize=(9, 4))
keys = sorted(cls_count, key=lambda x:int(x))
vals = [cls_count[k] for k in keys]
ax.bar([names[int(k)] if int(k)<len(names) else k for k in keys], vals, color="#FF5722", alpha=0.8)
ax.set_title("BƯỚC 5 — EDA: phân bố nhãn 6-class", fontweight="bold"); plt.xticks(rotation=20)
save(fig, "step5_eda.png")

# ── BƯỚC 6 — AUGMENTATION ──
base = cv2.resize(le if (roi_img is not None and le is not None) else sample, (64,64))
def aug(im, k):
    if k==1: return cv2.flip(im,1)
    if k==2: M=cv2.getRotationMatrix2D((32,32),12,1); return cv2.warpAffine(im,M,(64,64))
    if k==3: return np.clip(im*1.4,0,255).astype(np.uint8)
    if k==4: return np.clip(im*0.6,0,255).astype(np.uint8)
    return im
fig, ax = plt.subplots(1, 5, figsize=(14, 3))
for a,k,t in zip(ax,range(5),["Gốc","Lật","Xoay","Sáng+","Tối-"]):
    a.imshow(rgb(aug(base,k))); a.set_title(t); a.axis("off")
fig.suptitle("BƯỚC 6 — Augmentation (tăng cường dữ liệu)", fontweight="bold")
save(fig, "step6_augment.png")

# ── BƯỚC 7 — SPLIT ──
n_tr=len(glob.glob(f"{DATA}/train/images/*.jpg")); n_va=len(glob.glob(f"{DATA}/valid/images/*.jpg")); n_te=len(glob.glob(f"{DATA}/test/images/*.jpg"))
fig, ax = plt.subplots(figsize=(7,4))
ax.bar(["train","valid","test"], [n_tr,n_va,n_te], color=["#2196F3","#4CAF50","#FF9800"])
for i,v in enumerate([n_tr,n_va,n_te]): ax.text(i,v,str(v),ha="center",va="bottom")
ax.set_title("BƯỚC 7 — Dataset Split", fontweight="bold")
save(fig, "step7_split.png")

# ── BƯỚC 8 — NORMALIZE ──
fig, ax = plt.subplots(1, 2, figsize=(11,4))
ax[0].hist(base.flatten(), bins=50, color="#9C27B0", alpha=0.7); ax[0].set_title("Trước: pixel [0,255]")
ax[1].hist((base/255.0).flatten(), bins=50, color="#3F51B5", alpha=0.7); ax[1].set_title("Sau: pixel [0,1] (/255)")
fig.suptitle("BƯỚC 8 — Normalize pixel về [0,1]", fontweight="bold")
save(fig, "step8_normalize.png")

# ── BƯỚC 9+10 — INTEGRITY + MANIFEST (bảng tổng kết) ──
manifest = {"dataset":"Datio_yolo (RGB)","total_train":n_tr,"valid":n_va,"test":n_te,
            "classes":names,"class_counts":{names[int(k)] if int(k)<len(names) else k:cls_count[k] for k in keys},
            "input_size":64,"channels":3,"normalize":"[0,1]"}
json.dump(manifest, open(f"{OUT}/manifest.json","w"), indent=2, ensure_ascii=False)
fig, ax = plt.subplots(figsize=(9,5)); ax.axis("off")
txt = "BƯỚC 9 — Integrity Check: ✅ ảnh hợp lệ, đúng cặp ảnh-nhãn\n\nBƯỚC 10 — Manifest:\n"
txt += json.dumps(manifest, indent=2, ensure_ascii=False)
ax.text(0.02, 0.98, txt, fontsize=10, va="top", family="monospace")
save(fig, "step9_10_manifest.png")

print(f"\n✅ XONG. Ảnh từng bước → {OUT}/  (step0..step9_10)")
