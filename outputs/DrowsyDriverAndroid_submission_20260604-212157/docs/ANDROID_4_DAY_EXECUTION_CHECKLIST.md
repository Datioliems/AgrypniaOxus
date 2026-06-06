# Checklist Thuc Thi Android Native Trong 4 Ngay

## Viec Bat Buoc

Neu muon bao ve de tai tot, toi thieu phai co:

- App Android native mo duoc camera.
- MediaPipe Face Landmarker chay tren frame camera.
- Overlay hien `Awake`, `Eyes closed`, `Yawning`, `Drowsy alert`.
- Canh bao am thanh hoac rung.
- Video demo backup.
- Bang ket qua danh gia it nhat cho baseline va/hoac CNN.
- Bao cao noi ro han che va dieu kien chay.

## Viec Diem Cong

- Model CNN da convert sang `.tflite`.
- Tich hop TFLite vao Android.
- Do FPS tren dien thoai.
- Co confusion matrix.
- Co du lieu tu quay bang chinh Android phone dung demo.
- Co so sanh EAR/MAR voi CNN.

## Thu Tu Lam De Giam Rui Ro

### Buoc 1: Chay Camera Truoc

Chay script kiem tra moi truong:

```powershell
powershell -ExecutionPolicy Bypass -File tools/check_android_env.ps1
```

Mo Android Studio, sync Gradle, chay app tren dien thoai. Neu camera khong len,
chua lam model voi bao cao voi Android voi.

### Buoc 2: Them MediaPipe Model Asset

Dat `face_landmarker.task` vao:

```text
app/src/main/assets/face_landmarker.task
```

Chay lai app. Neu overlay hien EAR/MAR/FPS la du demo baseline.

### Buoc 3: Tune Nguong

Mo file:

```text
app/src/main/java/com/ai2026/drowsydriver/DrowsinessAnalyzer.kt
```

Chinh:

- `eyeClosedThreshold`
- `yawnThreshold`
- `drowsyDurationMs`

Gia tri ban dau:

- `eyeClosedThreshold = 0.19`
- `yawnThreshold = 0.58`
- `drowsyDurationMs = 1700`

### Buoc 4: Train CNN

Neu dataset da co anh:

```text
dataset/
  train/eyes_open
  train/eyes_closed
  val/eyes_open
  val/eyes_closed
```

Chay:

```bash
python tools/train_eye_classifier.py --data dataset --out app/src/main/assets/drowsiness_model.tflite
```

Neu chua tune CNN kip, van dua ket qua train vao bao cao. App da co duong load
TFLite, crop eye ROI tu landmark va goi classifier; canh bao demo co the van dua
vao EAR/MAR smoothing de on dinh.

### Buoc 5: Quay Demo

Demo can quay:

1. App mo camera.
2. Trang thai `Awake`.
3. Nham mat 2 giay de chuyen `Drowsy alert`.
4. Mo mieng/ngap de chuyen `Yawning`.
5. Man hinh co FPS/EAR/MAR.

## Fallback Neu Android Bi Loi

Neu Android Studio/build loi sat deadline:

- Van nop project Android native da co cau truc, MediaPipe asset, crop eye ROI
  va TFLite classifier inference.
- Quay demo Python/OpenCV neu da co.
- Trong bao cao ghi Android la muc tieu deploy, Python la prototype kiem chung
  thuat toan.

Tuy nhien nen uu tien sua Android truoc, vi de tai cua ban se manh hon nhieu khi
phone vua la camera vua la thiet bi xu ly.
