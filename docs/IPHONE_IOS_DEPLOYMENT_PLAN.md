# Ke hoach trien khai tren iPhone/iOS

## Ket luan ngan gon

Khong the chi tao file APK de chay tren iPhone. APK la goi cai dat cua Android.
iPhone/iOS can mot trong cac huong sau:

1. Ung dung iOS native bang Swift/SwiftUI va Xcode.
2. Ung dung cross-platform bang Flutter hoac React Native.
3. Web/PWA/Streamlit chay tren trinh duyet Safari cua iPhone.
4. Xu ly o server/laptop, iPhone chi dong vai tro camera/client.

Voi project hien tai, Android native van la san pham chinh. Neu muon iPhone dung
duoc nhanh, huong kha thi nhat la web/PWA hoac Streamlit. Neu muon app iPhone
that su, can xay dung them mot nhanh iOS rieng.

## APK, IPA va TestFlight

| Nen tang | Goi cai dat | Cong cu build | Cach cai/thử nghiem |
|---|---|---|---|
| Android | `.apk` hoac `.aab` | Android Studio/Gradle | Cai truc tiep APK, Google Play, internal testing |
| iPhone/iOS | `.ipa` | Xcode tren macOS | Xcode device run, TestFlight, App Store |
| Web/PWA | URL | Web framework | Mo bang Safari/Chrome tren iPhone |

Theo tai lieu Apple, de phan phoi app cho beta testers bang TestFlight hoac dua
len App Store, can tao project iOS, archive build bang Xcode va upload len App
Store Connect. APK khong phai dinh dang hop le tren iOS.

## Phan nao cua project hien tai tai su dung duoc

Tai su dung duoc:

- Mo hinh da train: `drowsiness_model.tflite`.
- Pipeline AI logic: phat hien mat, crop ROI, CNN, smoothing, event logging.
- Dataset, EDA, metrics, confusion matrix.
- Bao cao, dashboard, Streamlit fallback.
- Cac nguong logic nhu `drowsyDurationMs`, `eyeClosedThreshold`, yawn threshold.

Can viet lai neu lam iOS native:

- CameraX phai thay bang AVFoundation.
- Kotlin/Android Activity phai thay bang Swift/SwiftUI hoac UIKit.
- Android permission/camera lifecycle phai thay bang iOS permission/lifecycle.
- TFLite interpreter can dung ban TensorFlow Lite cho iOS.
- MediaPipe Face Landmarker can dung SDK/huong tich hop phu hop cho iOS.
- Vibration/sound/location/contact notification phai dung API iOS rieng.

## Ba phuong an cho bai cua minh

### Phuong an A - Khuyen nghi cho deadline

Giữ Android native là bản chính. iPhone dùng trang web hoặc Streamlit demo:

```text
iPhone Safari -> Web/Streamlit dashboard -> upload image/video hoặc camera snapshot
Android phone -> native realtime app -> demo chính
```

Uu diem:

- Nhanh nhat.
- Khong can Mac/Xcode ngay.
- Van chung minh duoc he thong co the mo rong sang iPhone qua web.

Han che:

- Web tren iPhone khong on dinh bang native app cho realtime camera.
- Streamlit khong phai app cai dat truc tiep tren iPhone.

Scaffold da them trong project:

```text
web/iphone-pwa/
tools/run_iphone_pwa_server.py
```

Ban PWA hien co cac chuc nang chinh:

- camera realtime qua HTTPS/ngrok;
- snapshot fallback neu Safari chan camera;
- MediaPipe Face Landmarker JS de tinh EAR/MAR;
- canh bao am thanh, event log, dashboard khung gio;
- lay vi tri va chia se canh bao cho lien he tin cay;
- Add to Home Screen nhu mot PWA.

Chay thu tren laptop:

```bat
python tools\run_iphone_pwa_server.py --host 0.0.0.0 --port 8502
```

Sau do iPhone mo Safari:

```text
https://NGROK_URL/web/iphone-pwa/index.html?v=4
```

### Phuong an B - iOS native that su

Xay app iOS rieng:

```text
iPhone camera -> AVFoundation -> Face landmark / TFLite -> smoothing -> alert
```

Can co:

- May Mac hoac cloud Mac.
- Xcode.
- Tai khoan Apple Developer neu muon TestFlight/App Store.
- Thoi gian rewrite UI/camera/inference.

Phu hop neu co them thoi gian sau khi nop bai hoac muon phat trien san pham that.

Scaffold da them trong project:

```text
ios/
ios/DrowsyDriverIOS/
ios/Info.plist.snippet
```

Thu muc nay gom cac file SwiftUI mau: camera preview bang AVFoundation, overlay
trang thai, placeholder inference service, am thanh canh bao va service vi tri.
Can dua sang Mac/Xcode de bien thanh app iOS that.

### Phuong an C - Cross-platform

Viet lai bang Flutter hoac React Native:

```text
Mot codebase -> Android + iOS
```

Uu diem:

- Ve lau dai co the chay ca Android va iPhone.

Han che:

- Realtime camera + MediaPipe/TFLite co the phuc tap.
- Van can Xcode/macOS de build iOS.
- Khong nen doi sang huong nay khi deadline gan.

## Kien truc iOS native de dua vao huong phat trien

```text
iPhone Camera
-> AVFoundation frame stream
-> Face landmark / face detector
-> eye ROI + mouth ROI
-> TFLite CNN / YOLO model
-> temporal smoothing
-> audio/vibration/location notification
-> dashboard/event log
```

## Cau tra loi nen dung khi bao ve

De tai hien tai khong the bien APK Android thanh ung dung iPhone. iOS yeu cau
mot artifact va toolchain rieng, thuong la project Xcode va file `.ipa`.
Tuy nhien, thanh phan AI nhu dataset, mo hinh `.tflite`, logic smoothing,
metrics va dashboard co the tai su dung. Vi gioi han thoi gian, de tai chon
Android native lam san pham chinh, con iPhone duoc de xuat theo hai huong mo
rong: web/PWA de demo nhanh hoac iOS native bang Swift/AVFoundation neu phat
trien tiep.

## Tai lieu tham khao

- Apple Developer - Distributing your app for beta testing and releases:
  https://developer.apple.com/documentation/Xcode/distributing-your-app-for-beta-testing-and-releases
- Apple Developer - Preparing your app for distribution:
  https://developer.apple.com/documentation/Xcode/preparing-your-app-for-distribution
- Apple Developer - TestFlight:
  https://developer.apple.com/testflight
- Apple Developer Program:
  https://developer.apple.com/programs/
