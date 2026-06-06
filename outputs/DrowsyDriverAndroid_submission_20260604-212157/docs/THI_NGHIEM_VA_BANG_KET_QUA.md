# Thiet Ke Thi Nghiem Va Bang Ket Qua

## Muc Tieu Thi Nghiem

Danh gia xem he thong co phat hien duoc dau hieu buon ngu trong cac dieu kien
co ban hay khong, dong thoi so sanh baseline giai thich duoc voi CNN.

## Thi Nghiem 1: Baseline EAR/MAR

Input:

- Video tu camera Android hoac webcam.
- Cac hanh dong: tinh tao, nham mat, ngap.

Output:

- EAR, MAR, trang thai he thong.

Bang ghi nhan:

| Tinh huong | EAR trung binh | MAR trung binh | Trang thai dung? | Ghi chu |
|---|---:|---:|---|---|
| Tinh tao | TBD | TBD | TBD | TBD |
| Nham mat 2 giay | TBD | TBD | TBD | TBD |
| Ngap/mo mieng | TBD | TBD | TBD | TBD |

## Thi Nghiem 2: CNN Eye Classifier

Input:

- Anh ROI mat trong tap validation/test.

Output:

- `eyes_open` hoac `eyes_closed`.

Bang ket qua:

| Lop | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| eyes_open | TBD | TBD | TBD | TBD |
| eyes_closed | TBD | TBD | TBD | TBD |

Confusion matrix:

| Actual / Predict | eyes_open | eyes_closed |
|---|---:|---:|
| eyes_open | TBD | TBD |
| eyes_closed | TBD | TBD |

## Thi Nghiem 3: Realtime Android Demo

Input:

- Camera phone.

Metric:

- FPS hien tren overlay.
- Thoi gian canh bao sau khi nham mat.
- Ty le lan demo dung/truot.

Bang ghi nhan:

| Lan demo | FPS trung binh | Thoi gian nham mat | Co canh bao? | Ghi chu |
|---|---:|---:|---|---|
| 1 | TBD | TBD | TBD | TBD |
| 2 | TBD | TBD | TBD | TBD |
| 3 | TBD | TBD | TBD | TBD |

## Ket Luan Thi Nghiem Can Viet

Mau cau:

> Ket qua cho thay baseline EAR/MAR co the phat hien nhanh cac dau hieu ro rang
> nhu mat nham lau va ngap. CNN eye classifier bo sung kha nang hoc dac trung
> anh cua vung mat. Khi ket hop voi smoothing theo thoi gian, he thong giam bao
> nham do chop mat ngan va phu hop hon voi demo realtime tren Android.
