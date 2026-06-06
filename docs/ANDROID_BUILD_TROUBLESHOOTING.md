# Android Build Troubleshooting

## Quick Check

From project root:

```powershell
powershell -ExecutionPolicy Bypass -File tools/check_android_env.ps1
```

This checks:

- Java/JDK.
- Gradle wrapper or system Gradle.
- Android SDK path.
- `local.properties`.

## Recommended Path

The fastest path for the 4-day deadline is Android Studio:

1. Open `D:\2026.AI\DrowsyDriverAndroid`.
2. Let Android Studio install missing SDK, Gradle, and Kotlin components.
3. Check that `local.properties` contains `sdk.dir=...`.
4. Connect Android phone with USB debugging.
5. Press Run.

## If `local.properties` Is Missing

Copy:

```text
local.properties.template
```

to:

```text
local.properties
```

Then edit `sdk.dir`.

Example:

```properties
sdk.dir=C\:\\Users\\ADMIN\\AppData\\Local\\Android\\Sdk
```

## If `face_landmarker.task` Is Missing

The file should already exist:

```text
app/src/main/assets/face_landmarker.task
```

If it is missing, download:

```text
https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
```

## If CNN Shows `CNN model pending`

This is expected until you train and copy:

```text
app/src/main/assets/drowsiness_model.tflite
```

The app can still demo the MediaPipe + EAR/MAR + smoothing baseline.

## If Camera Preview Works But No Face Is Detected

- Use the front camera.
- Keep the face centered and close enough.
- Improve lighting.
- Avoid sunglasses during the first demo.
- Check that `face_landmarker.task` exists in assets.

## If App Is Slow

- Keep AI around 8-12 FPS.
- Use the current frame throttle in `MainActivity.kt`.
- Prefer a small CNN input size such as `64x64`.
- Demo with a single face in frame.

## Build From Terminal

When Java/SDK/Gradle are ready:

```powershell
powershell -ExecutionPolicy Bypass -File tools/build_android.ps1
```

Expected output APK:

```text
app/build/outputs/apk/debug/app-debug.apk
```
