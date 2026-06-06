# De Cuong Bao Cao Theo Template Mon Hoc

## Ten De Tai De Xuat

**He thong phat hien va canh bao dau hieu buon ngu cua tai xe tren thiet bi
Android su dung MediaPipe va mang no-ron tich chap**

Ten de tai co du 3 yeu to:

- Bai toan: phat hien va canh bao buon ngu cua tai xe.
- Pham vi: thiet bi Android, camera trong cabin/phone dat truoc mat.
- Ky thuat: MediaPipe Face Landmarker, CNN/TensorFlow Lite, smoothing theo
  thoi gian.

## Chuong 1: Gioi Thieu Bai Toan

### 1.1. Phat Bieu Bai Toan

Input:

- Luong anh realtime tu camera Android phone.
- Doi tuong chinh la khuon mat tai xe trong cabin.

Output:

- Trang thai: `No face`, `Awake`, `Eyes closed`, `Yawning`, `Drowsy alert`.
- Canh bao bang am thanh/rung khi mat nham lien tuc qua nguong thoi gian.

Pham vi:

- Mot tai xe trong khung hinh.
- Camera dat gan truc chinh dien voi mat.
- Prototype uu tien dieu kien anh sang vua du; anh sang yeu, kinh ram, che mat
  la cac han che can danh gia va neu ro.

### 1.2. Ung Dung Va Gia Tri Thuc Te

Gia tri cho ca nhan:

- Canh bao som khi tai xe co dau hieu met moi.
- Co the dung dien thoai Android san co, khong can thiet bi dac biet.

Gia tri cho to chuc:

- Cong ty van tai, taxi, giao hang co the dung nhu ban mau de giam rui ro tai
  nan do met moi.
- Truong hoc/nhom nghien cuu co demo edge AI thuc te ve computer vision.

Gia tri cong dong:

- Gop phan nang cao nhan thuc ve lai xe khi buon ngu.
- Huong toi he thong ho tro lai xe chi phi thap.

### 1.3. Khao Sat Bai Lam Lien Quan

Nhom cong trinh nen trinh bay:

1. He thong EAR/MAR rule-based dung landmark mat va mieng.
2. CNN phan loai trang thai mat hoac trang thai buon ngu.
3. He thong hybrid MediaPipe + CNN/MAR tren embedded hoac mobile.
4. Cac bo du lieu NTHU-DDD, YawDD, DROZY.

Bang so sanh goi y:

| Huong tiep can | Uu diem | Han che | Cach de tai nay cai thien |
|---|---|---|---|
| EAR/MAR rule-based | Nhe, de giai thich | Phu thuoc nguong | Them CNN va smoothing |
| CNN anh mat/mieng | Hoc duoc dac trung anh | Can du lieu, co do tre | Chay tren ROI nho + TFLite |
| Laptop webcam demo | De lam | Kho deploy trong xe | Dung Android phone on-device |
| Cloud/API | De quan ly model | Do tre, phu thuoc mang | Xu ly tren thiet bi |

## Chuong 2: Chuan Bi Du Lieu

### 2.1. Thu Thap Du Lieu

Nguon du lieu chinh:

- NTHU Driver Drowsiness Detection Dataset: de mo ta va neu trong khao sat, xin
  tai neu kip.
- YawDD hoac EyeState/YawnDrowsiness: dung de train nhanh mat/mo va ngap.
- Du lieu tu thu bang Android phone: dung de kiem tra domain deployment.

Bang thong ke nen co:

| Nguon | So anh/video | Nhan | Vai tro |
|---|---:|---|---|
| Dataset public | TBD | eyes_open/eyes_closed/yawning | Train/validation |
| Android phone tu quay | TBD | awake/closed/yawn | Test/demo |

### 2.2. Tien Xu Ly Du Lieu

Pipeline tien xu ly:

```text
Video/anh
-> phat hien mat bang MediaPipe/OpenCV
-> lay landmark mat va mieng
-> crop ROI mat/mieng
-> resize 64x64 hoac 96x96
-> normalize pixel ve [0, 1]
-> chia train/validation/test
```

Bien phap can giai thich:

- Crop ROI de giam tinh toan va tap trung vao dau hieu buon ngu.
- Resize de phu hop CNN nhe va TFLite.
- Data augmentation nhe: thay doi do sang, xoay nho, zoom nho.
- Chia tap theo subject/video neu co the, de tranh cung mot nguoi xuat hien ca
  train va test.

## Chuong 3: Xay Dung Mo Hinh

### 3.1. Trich Chon Dac Trung

Dac trung hinh hoc:

- EAR: Eye Aspect Ratio, do mo cua mat.
- MAR: Mouth Aspect Ratio, do mo cua mieng.
- Thoi gian lien tuc mat nham/yawning.

Dac trung anh:

- ROI mat trai/phai hoac mat tong hop.
- CNN hoc dac trung texture/hinh dang de phan loai mat mo/mat nham.

### 3.2. Lua Chon Mo Hinh

Baseline:

- MediaPipe landmark + EAR/MAR threshold.

Mo hinh AI:

- CNN nhe cho `eyes_open` va `eyes_closed`.
- Sau khi train, convert sang TensorFlow Lite de chay tren Android.

Mo hinh cuoi:

```text
CameraX
-> MediaPipe Face Landmarker
-> EAR/MAR baseline
-> CNN/TFLite eye classifier
-> temporal smoothing
-> alert
```

### 3.3. Cau Hinh Huan Luyen

Can ghi:

- Image size: 64x64 hoac 96x96.
- Optimizer: Adam.
- Loss: categorical crossentropy.
- Epoch: 10-20.
- Batch size: 32.
- Hardware train: laptop/Google Colab.
- Deployment: Android phone, CameraX, MediaPipe Tasks, TensorFlow Lite.

## Chuong 4: Danh Gia Mo Hinh Va Demo

### 4.1. Do Do Danh Gia

Can co:

- Accuracy.
- Precision.
- Recall.
- F1-score.
- Confusion matrix.
- FPS/latency khi chay demo.

Luu y an toan:

- Recall cua lop `drowsy`/`eyes_closed` quan trong hon accuracy tong, vi bo sot
  tai xe buon ngu nguy hiem hon bao nham.

### 4.2. Ket Qua Danh Gia

Nen tao ket qua bang `tools/evaluate_eye_classifier.py` de co file
`metrics.json`, `confusion_matrix.csv` va `predictions.csv`.

Bang so sanh nen co:

| Phuong phap | Accuracy | Recall drowsy | F1 | FPS | Nhan xet |
|---|---:|---:|---:|---:|---|
| EAR/MAR | TBD | TBD | TBD | TBD | Nhanh, de giai thich |
| CNN eye classifier | TBD | TBD | TBD | TBD | Hoc dac trung anh |
| Hybrid + smoothing | TBD | TBD | TBD | TBD | On dinh hon khi demo |

### 4.3. Demo

Demo can the hien:

- App Android mo camera.
- Trang thai hien tren man hinh.
- Mat nham hon 1.7 giay thi canh bao.
- MAR tang khi ngap/mo mieng.
- Co FPS va confidence.

## Ket Luan

Diem da lam duoc:

- Xay dung pipeline Android on-device.
- Su dung MediaPipe de trich landmark realtime.
- Co baseline EAR/MAR va huong tich hop CNN/TFLite.
- Co canh bao bang am thanh/rung.

Han che:

- Chua dam bao moi dieu kien anh sang.
- Kinh ram va che mat co the lam landmark sai.
- Dataset tu thu nho, can mo rong theo nhieu nguoi/thiet bi.
- Can danh gia tren tinh huong lai xe thuc te co kiem soat an toan.

Huong phat trien:

- Tich hop model CNN/TFLite da train.
- Ca nhan hoa nguong EAR/MAR theo tung tai xe.
- Them head pose, gaze, PERCLOS.
- Luu log su kien va xuat bao cao cho doi xe/tai xe.
