# He thong phat hien va canh bao dau hieu buon ngu cua tai xe tren thiet bi Android su dung MediaPipe va mang no-ron tich chap

## Bang Phan Cong Cong Viec

| Thanh vien | Quyen bao cao | San pham | Ty le dong gop |
|---|---|---|---:|
| Thanh vien 1 | Chuong 1, Chuong 3 | Pipeline MediaPipe, Android app | TBD |
| Thanh vien 2 | Chuong 2, Chuong 4 | Dataset, train CNN, danh gia | TBD |
| Thanh vien 3 | Ket luan, tai lieu tham khao | Demo, video, Q&A | TBD |

## Chuong 1: Gioi Thieu Bai Toan

### 1.1. Phat bieu bai toan

Lai xe trong trang thai buon ngu la mot trong nhung nguyen nhan gay tai nan
giao thong nghiem trong. Cac dau hieu nhu mat nham lau, ngap lien tuc, cui dau
hoac mat tap trung co the duoc quan sat thong qua camera dat trong cabin. De tai
nay xay dung mot prototype phat hien va canh bao dau hieu buon ngu cua tai xe
bang thiet bi Android phone.

Input cua he thong la luong hinh anh realtime tu camera Android. Output la trang
thai cua tai xe, gom `No face`, `Awake`, `Eyes closed`, `Yawning` va `Drowsy
alert`. Khi he thong phat hien mat nham lien tuc qua nguong thoi gian, ung dung
se phat canh bao bang am thanh va rung.

Pham vi cua de tai la mot tai xe trong khung hinh, camera dat gan chinh dien
khuon mat, xu ly tren thiet bi Android. He thong chua huong toi thay the cac he
thong an toan tren xe thuong mai, ma la prototype AI co tinh ung dung, co the
demo va co lo trinh phat trien thanh san pham thuc te.

### 1.2. Ung dung cua bai toan

He thong co the ho tro ca nhan lai xe duong dai, tai xe taxi, xe cong nghe, xe
giao hang, xe khach hoac xe tai. Diem manh cua huong Android la tan dung thiet
bi pho bien, co san camera, loa, rung, CPU/GPU va man hinh. Toan bo xu ly dien
ra tren thiet bi, khong phu thuoc vao internet, nen phu hop hon cac cach gui
video len cloud trong boi canh xe dang di chuyen.

Gia tri thuc te cua de tai gom:

- Canh bao som cho ca nhan khi co dau hieu buon ngu.
- Giam chi phi so voi viec lap camera cabin va bo xu ly rieng.
- Lam nen tang prototype cho doanh nghiep van tai hoac nhom nghien cuu edge AI.
- Nang cao nhan thuc cong dong ve nguy co lai xe khi met moi.

### 1.3. Khao sat cac bai lam lien quan

Cac he thong phat hien buon ngu thuong duoc chia thanh ba nhom. Nhom thu nhat
dung dac trung hinh hoc nhu EAR, MAR, PERCLOS va head pose. Nhom nay nhanh, de
giai thich, nhung phu thuoc nguong va bi anh huong boi goc camera, kinh ram,
anh sang yeu. Nhom thu hai dung CNN hoac cac mo hinh hoc sau de phan loai trang
thai mat, mieng hoac khuon mat. Nhom nay hoc duoc dac trung anh, nhung can du
lieu da dang va phai toi uu de chay realtime. Nhom thu ba la huong hybrid, ket
hop landmark de dinh vi vung quan tam voi CNN/TFLite de phan loai.

De tai nay chon huong hybrid: MediaPipe Face Landmarker trich xuat landmark,
EAR/MAR la baseline co the giai thich, CNN/TFLite la thanh phan hoc sau, va
temporal smoothing giup giam bao nham do chop mat tu nhien.

Khoang trong nghien cuu ma de tai tap trung:

- Nhieu demo chi danh gia offline, it quan tam latency/FPS tren thiet bi thuc.
- Nhieu bai laptop webcam kho trien khai vao xe.
- Nhieu bai chi dung accuracy, trong khi bai toan an toan can chu trong recall
  cua lop buon ngu.
- CNN co the kho giai thich, can ket hop voi dac trung landmark ro rang.

## Chuong 2: Chuan Bi Du Lieu

### 2.1. Thu thap du lieu

De tai su dung ba nhom du lieu:

1. Dataset public lien quan den tai xe buon ngu, nhu NTHU Driver Drowsiness
   Detection Dataset, YawDD va DROZY.
2. Dataset de train nhanh trang thai mat hoac ngap, vi du cac dataset
   eye-state/yawning co cau truc nhan ro rang.
3. Du lieu tu thu bang Android phone dung de kiem tra dieu kien camera thuc te.

Bang thong ke du lieu se duoc dien sau khi tai va tien xu ly:

| Nguon du lieu | So mau | Nhan | Vai tro |
|---|---:|---|---|
| Dataset public | TBD | eyes_open/eyes_closed/yawning | Train/validation |
| Video tu quay bang Android | TBD | awake/closed/yawn | Test/demo |

### 2.2. Tien xu ly du lieu

Du lieu anh/video duoc tien xu ly theo cac buoc:

```text
Anh/video
-> phat hien khuon mat
-> trich landmark bang MediaPipe
-> crop ROI mat va/hoac mieng
-> resize ve 64x64 hoac 96x96
-> normalize pixel ve [0, 1]
-> chia train/validation/test
```

Viec crop ROI giup mo hinh CNN tap trung vao vung co dau hieu buon ngu va giam
chi phi tinh toan. Neu du lieu it, de tai su dung augmentation nhe nhu thay doi
do sang, xoay nho va zoom nho de tang kha nang tong quat.

## Chuong 3: Xay Dung Mo Hinh

### 3.1. Trich chon dac trung

He thong su dung hai loai dac trung:

- Dac trung hinh hoc tu landmark: EAR cho mat, MAR cho mieng, thoi gian mat nham
  lien tuc va thoi gian ngap lien tuc.
- Dac trung anh tu ROI: anh mat hoac mieng da crop duoc dua vao CNN de phan loai
  trang thai.

EAR duoc tinh bang ty le khoang cach doc va ngang cua mat. MAR duoc tinh tu do
mo cua mieng. Hai chi so nay co y nghia giai thich ro rang, phu hop de lam
baseline va de thuyet trinh.

### 3.2. Lua chon thuat toan/mo hinh

Pipeline cuoi cua he thong:

```text
CameraX Android
-> MediaPipe Face Landmarker LIVE_STREAM
-> tinh EAR/MAR va crop ROI vung mat
-> CNN/TFLite phan loai mat mo/mat nham tren eye ROI
-> temporal smoothing
-> canh bao am thanh/rung
```

Baseline la MediaPipe + EAR/MAR threshold. Mo hinh hoc sau la CNN nhe, duoc
train tren anh ROI va export sang TensorFlow Lite de chay tren Android. Trong
prototype hien tai, phan Android da co CameraX, MediaPipe, EAR/MAR, crop eye
band tu landmark, smoothing, alert va TFLite classifier. Khi co file
`drowsiness_model.tflite`, app se hien nhan CNN va confidence tren overlay.

### 3.3. Cau hinh huan luyen

Cau hinh de xuat:

- Input CNN: anh ROI 64x64x3.
- Lop: `eyes_open`, `eyes_closed`; co the mo rong `yawning`, `not_yawning`.
- Optimizer: Adam.
- Loss: categorical crossentropy.
- Epoch: 10-20.
- Batch size: 32.
- Deployment: TensorFlow Lite tren Android.

## Chuong 4: Danh Gia Mo Hinh Va Demo

### 4.1. Do do danh gia

Do do danh gia gom accuracy, precision, recall, F1-score, confusion matrix va
FPS/latency khi chay demo. Trong bai toan an toan, recall cua lop buon ngu hoac
mat nham quan trong hon accuracy tong, vi bo sot tai xe buon ngu nguy hiem hon
bao nham.

### 4.2. Ket qua danh gia

Ket qua duoc tao bang script `tools/evaluate_eye_classifier.py`. Script nay doc
tap `test/`, chay model Keras hoac TensorFlow Lite, sau do xuat:

- `outputs/evaluation/metrics.json`
- `outputs/evaluation/confusion_matrix.csv`
- `outputs/evaluation/predictions.csv`

Bang ket qua du kien:

| Phuong phap | Accuracy | Recall drowsy/closed | F1-score | FPS | Nhan xet |
|---|---:|---:|---:|---:|---|
| EAR/MAR threshold | TBD | TBD | TBD | TBD | Nhanh, de giai thich |
| CNN eye classifier | TBD | TBD | TBD | TBD | Hoc dac trung anh |
| Hybrid + smoothing | TBD | TBD | TBD | TBD | On dinh hon khi demo |

### 4.3. Demo

Demo Android can the hien:

- Ung dung mo camera phone.
- Overlay hien trang thai, EAR, MAR, FPS va confidence.
- Khi mat nham lien tuc hon 1.7 giay, he thong chuyen sang `Drowsy alert`.
- Khi mieng mo/ngap trong thoi gian ngan, he thong hien `Yawning`.
- Canh bao bang am thanh va rung.

## Ket Luan

De tai da xay dung duoc pipeline prototype cho bai toan phat hien dau hieu buon
ngu cua tai xe tren Android. Huong tiep can ket hop MediaPipe landmark, baseline
EAR/MAR va CNN/TFLite giup he thong vua co kha nang giai thich, vua co lo trinh
trien khai AI tren thiet bi bien.

Han che hien tai:

- Do chinh xac phu thuoc anh sang, goc camera va kinh/che mat.
- Dataset tu thu con nho.
- Can kiem thu tren nhieu nguoi va nhieu thiet bi Android.
- Chua duoc thu nghiem trong dieu kien lai xe thuc te co kiem soat an toan.

Huong phat trien:

- Tich hop day du CNN/TFLite vao app.
- Them PERCLOS, head pose va gaze.
- Ca nhan hoa nguong theo tung tai xe.
- Luu log va thong ke su kien cho ca nhan hoac doanh nghiep van tai.

### Khia canh rieng tu va an toan

Vi he thong su dung camera huong vao khuon mat tai xe, nhom uu tien xu ly truc
tiep tren thiet bi va khong luu video khuon mat mac dinh. Neu trien khai trong
to chuc, he thong can co thong bao ro rang, su dong y cua nguoi dung, chinh sach
luu tru du lieu va chi nen luu log su kien toi thieu thay vi video lien tuc.

He thong chi la cong cu ho tro canh bao som, khong thay the trach nhiem nghi
ngoi cua tai xe va khong phai he thong an toan da duoc chung nhan cho xe thuong
mai. Truoc khi dung thuc te can kiem thu co kiem soat trong nhieu dieu kien anh
sang, goc camera va nhom nguoi dung khac nhau.

## Tai Lieu Tham Khao Goi Y

Xem danh sach chi tiet tai `research/SOURCES_FOR_REPORT.md`.
