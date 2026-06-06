# Cau Hoi Co The Bi Hoi Khi Thuyet Trinh

## 1. AI nam o dau trong bai?

AI nam o hai diem:

- MediaPipe Face Landmarker su dung mo hinh hoc may de trich xuat landmark
  khuon mat realtime.
- CNN/TensorFlow Lite phan loai trang thai mat/mo-mat nham hoac ngap/khong ngap
  tu vung anh da crop.

EAR/MAR la baseline giai thich duoc, khong phai thanh phan AI chinh duy nhat.

## 2. Vi sao khong chi dung EAR/MAR?

EAR/MAR nhanh va de giai thich, nhung phu thuoc nguong. Nguoi khac nhau, goc
camera khac nhau, anh sang khac nhau se lam nguong thay doi. CNN giup hoc dac
trung anh tu du lieu, con smoothing giup tranh bao nham khi chi chop mat tu
nhien.

## 3. Vi sao dung Android phone?

Android phone co san camera, CPU/GPU/NPU, loa, rung va man hinh. He thong chay
on-device nen khong phu thuoc internet, do tre thap hon cloud/API va chi phi
thap hon camera + embedded board rieng.

## 4. Neu khong co model CNN kip tich hop thi co bi yeu khong?

Prototype van co san pham demo bang MediaPipe + EAR/MAR + smoothing. Bao cao can
trinh bay CNN da train/danh gia hoac it nhat pipeline TFLite ro rang. Tuy nhien,
de diem tot nen co bang ket qua CNN va model `.tflite` trong assets.

## 5. Tai sao dung recall/F1 thay vi chi accuracy?

Trong bai toan an toan, bo sot tai xe buon ngu nguy hiem hon bao nham. Vi vay
recall cua lop buon ngu/mat nham la do do quan trong. F1 giup can bang precision
va recall.

## 6. He thong co chay realtime that khong?

Muc tieu prototype la realtime gan dung tren Android. App khong nhat thiet xu ly
30 FPS bang CNN; chi can 8-12 FPS cho AI la du vi buon ngu la trang thai keo dai
theo thoi gian. CameraX dung chien luoc giu frame moi nhat de tranh tre hang doi.

## 7. Han che lon nhat la gi?

- Kinh ram, che mat, anh sang yeu.
- Camera dat lech qua nhieu.
- Dataset chua du da dang.
- Can thu nghiem an toan trong dieu kien gan thuc te truoc khi dung ngoai duong.

## 8. Khoang trong nghien cuu cua de tai la gi?

De tai tap trung vao khoang trong giua demo offline va deploy thuc te:

- Do latency/FPS tren thiet bi.
- Chay on-device bang Android phone.
- Ket hop baseline giai thich duoc voi CNN.
- Uu tien recall va smoothing thay vi accuracy tung frame.
