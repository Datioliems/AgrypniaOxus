# DrowsyDriverIOS scaffold

Day la nhanh mau cho huong iPhone/iOS cua project. APK Android khong chay duoc
tren iPhone, nen neu muon co app iOS native can tao project Xcode rieng.

Thu muc nay khong anh huong den app Android hien tai. No dong vai tro scaffold
de dua sang may Mac va tao project iOS.

## Yeu cau

- macOS
- Xcode
- iPhone that de test camera
- Apple Developer account neu muon TestFlight/App Store

## Cach dung nhanh

Tren Mac:

1. Mo Xcode.
2. Tao project moi:
   - iOS App
   - Product Name: `DrowsyDriverIOS`
   - Interface: SwiftUI
   - Language: Swift
3. Copy cac file trong `ios/DrowsyDriverIOS/` vao project Xcode.
4. Them permission camera vao `Info.plist`:

```xml
<key>NSCameraUsageDescription</key>
<string>Ung dung can camera de phat hien dau hieu buon ngu cua tai xe.</string>
```

5. Them model TFLite neu da co:

```text
DrowsyDriverIOS/Resources/drowsiness_model.tflite
```

## Kien truc iOS du kien

```text
iPhone Camera
-> AVFoundation frame stream
-> Face landmark / face detector
-> eye ROI + mouth ROI
-> TFLite CNN / YOLO model
-> temporal smoothing
-> sound/vibration/location alert
-> event log/dashboard
```

## File scaffold

- `DrowsyDriverIOSApp.swift`: entry point SwiftUI.
- `ContentView.swift`: UI overlay giong Android.
- `CameraPreview.swift`: cau noi SwiftUI voi camera layer.
- `CameraViewModel.swift`: placeholder cho AVFoundation + pipeline.
- `DriverStatus.swift`: state model.
- `DrowsinessInferenceService.swift`: placeholder TFLite/MediaPipe inference.
- `AlertSoundService.swift`: canh bao am thanh theo cap do.
- `LocationAlertService.swift`: placeholder gui vi tri cho nguoi than.

## Ghi chu cho bao cao

Trong pham vi deadline, iOS native la huong mo rong. Thanh phan AI nhu dataset,
metrics, `.tflite` model va logic smoothing co the tai su dung, nhung camera,
permission, UI va packaging phai viet lai bang iOS APIs.

