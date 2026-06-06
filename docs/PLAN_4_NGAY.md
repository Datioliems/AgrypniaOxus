# Phuong An 4 Ngay

## Muc Tieu San Pham

San pham cuoi cung nen la mot prototype Android native co camera realtime,
phat hien khuon mat bang MediaPipe, phan tich dau hieu buon ngu bang landmark
va co moc tich hop CNN/TFLite.

Muc tieu khong phai la he thong xe hoi thuong mai hoan hao, ma la prototype
co demo tot, co danh gia dinh luong, va co lo trinh deploy thuc te.

## Ngay 1: Chot Du Lieu Va Baseline

- Tai dataset de dung: EyeState/YawnDrowsiness hoac YawDD.
- Xin NTHU-DDD neu con thoi gian, dua vao bao cao neu chua kip tai.
- Tu quay them video bang Android phone: tinh tao, nham mat lau, ngap, quay dau.
- Chay app Android CameraX + MediaPipe.
- Kiem tra EAR/MAR va canh bao sau 1.5-2 giay.

Ket qua can co:

- Video demo baseline.
- Anh chup giao dien Android.
- Bang mo ta input/output.

## Ngay 2: CNN Nhe

- Uu tien bai toan mat mo/mat nham.
- Cat ROI mat tu anh/video bang MediaPipe/OpenCV.
- Train CNN nho hoac MobileNetV2 transfer learning.
- Luu ket qua danh gia: accuracy, precision, recall, F1, confusion matrix.
- Export model sang `.tflite`.

Ket qua can co:

- Model `.tflite`.
- Confusion matrix.
- Bang so sanh baseline EAR voi CNN.

## Ngay 3: Tich Hop Android

- Dat `drowsiness_model.tflite` vao `app/src/main/assets`.
- Ket noi output CNN voi logic canh bao.
- Giu xu ly 8-12 FPS de tranh nghen.
- Dung smoothing theo thoi gian thay vi canh bao tung frame.
- Quay demo cuoi cung tren Android phone.

Ket qua can co:

- App demo duoc tren dien thoai.
- Video demo backup.
- Log FPS/confidence/trang thai.

## Ngay 4: Bao Cao Va Thuyet Trinh

- Viet bao cao theo template:
  - Chuong 1: bai toan, ung dung, pham vi, lien quan.
  - Chuong 2: du lieu, tien xu ly, gan nhan.
  - Chuong 3: MediaPipe, EAR/MAR, CNN, pipeline Android.
  - Chuong 4: do do, ket qua, demo, han che.
- Chuan bi cau tra loi:
  - CNN nam o dau?
  - Vi sao dung MediaPipe?
  - Vi sao Android phone hop ly hon webcam laptop?
  - Han che khi dung ngoai doi la gi?

## Scope An Toan

Neu thieu thoi gian, uu tien:

- Android camera + MediaPipe landmark.
- EAR/MAR + smoothing + alert.
- CNN da train va danh gia, co the chua tich hop hoan hao.
- Bao cao noi ro CNN/TFLite la huong tich hop trong pipeline.

Neu moi thu on, tich hop them `.tflite` vao app.
