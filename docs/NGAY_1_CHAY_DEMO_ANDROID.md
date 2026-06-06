# Ngay 1: Chay Demo Android Baseline

Muc tieu ngay 1 la co video demo dau tien. Chua can CNN hoan hao.

## 1. Mo Project

Truoc khi mo project, chay kiem tra moi truong:

```powershell
powershell -ExecutionPolicy Bypass -File tools/check_android_env.ps1
```

Mo Android Studio, chon:

```text
D:\2026.AI\DrowsyDriverAndroid
```

Cho Gradle sync xong. Neu Android Studio hoi cai SDK/Gradle/Kotlin, chon cai
dat theo goi y cua IDE.

## 2. Ket Noi Dien Thoai

- Bat Developer Options.
- Bat USB debugging.
- Cam cap USB.
- Chon device trong Android Studio.
- Bam Run.

## 3. Kiem Tra App

App mong doi:

- Camera preview hien len.
- Overlay tren cung hien trang thai.
- EAR/MAR/FPS thay doi khi di chuyen mat/mieng.
- Nham mat khoang 2 giay thi co canh bao.
- Mo mieng/ngap thi hien `Yawning`.

## 4. Neu MediaPipe Khong Chay

Kiem tra file nay co ton tai:

```text
app/src/main/assets/face_landmarker.task
```

File nay da duoc tai san trong project, kich thuoc khoang 3.6 MB.

## 5. Quay Video Demo Backup

Dung tinh nang quay man hinh cua dien thoai hoac quay bang dien thoai khac.

Video toi thieu nen co:

1. Mo app.
2. Trang thai `Awake`.
3. Nham mat 2 giay.
4. App canh bao `Drowsy alert`.
5. Mo mieng/ngap de hien `Yawning`.

## 6. Ket Qua Ngay 1 Can Luu

- Video demo baseline.
- Anh chup man hinh app.
- Ghi lai FPS trung binh.
- Ghi lai nguong EAR/MAR ban da tune neu co.

Neu ngay 1 lam xong cac muc nay, de tai da co san pham demo toi thieu de bao ve.
