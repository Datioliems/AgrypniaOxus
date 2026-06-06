# Code Audit Errors

Generated: 2026-06-07

Scope: manual review of the project code folders, excluding large image datasets from file listing noise. CodeRabbit was not used.

## Checks Run

| Check | Result |
|---|---|
| `python tools\run_all_checks.py` | PASS: 33 Python files pass syntax/light run checks |
| Notebook JSON scan | PASS: 8 notebooks parse, 0 saved error outputs |
| `node --check web\iphone-pwa\app.js` | PASS |
| `node --check web\iphone-pwa\service-worker.js` | PASS |
| Android XML parse | PASS |
| `gradlew.bat assembleDebug` with Android Studio JBR | PASS |
| `gradlew.bat testDebugUnitTest` with Android Studio JBR | PASS, but `NO-SOURCE` tests |
| `gradlew.bat lintDebug` | PASS with warnings |
| `npx pyright` | FAIL: 111 type-check reports, many caused by noisy include paths |

## High Priority

### 1. Hard-coded Roboflow API key in source

Files:

- `colab_yolo26_training.py:63`
- `tools/train_yolov8_local.py:81`
- `tools/train_clean_local.py:79`
- `tools/train_clean_local.py:99`
- `notebook_multi_model.py:164`
- `colab_drowsy_yolo26.ipynb`
- `colab_yolo26_training.ipynb`
- `tools/train_clean_local.ipynb`

Impact:

The Roboflow key is committed directly in code/notebooks. This can leak credentials when sharing the repo, screenshots, reports, or submissions.

Suggested fix:

Move the key to an environment variable or Colab secret:

```python
import os
api_key = os.environ["ROBOFLOW_API_KEY"]
rf = Roboflow(api_key=api_key)
```

Also rotate the exposed key if it is real.

### 2. `dataset_yolo26` is used before it is defined

File:

- `colab_yolo26_training.py:331`
- `colab_yolo26_training.py:337`

Current code:

```python
if dataset_yolo26 is not None and DATASET_FORMAT == "yolo26":
```

Impact:

Running the Colab notebook sequentially can fail with:

```text
NameError: name 'dataset_yolo26' is not defined
```

Suggested fix:

Initialize it before the download cells:

```python
dataset_yolo26 = None
```

or implement the missing YOLO26 download step that assigns this variable.

### 3. Local YOLO clean training can fail because Dataset 1 is missing

Files:

- `tools/train_clean_local.py:20`
- `tools/train_clean_local.py:72`
- `tools/train_clean_local.py:164`

Current workspace state:

```text
roboflow_data/ds_augmented/data.yaml      missing
roboflow_data/ds_driveryawn/data.yaml     exists
roboflow_data/clean_merged/data.yaml      missing
```

Impact:

If Dataset 1 download fails, the merge step still reaches `_read_names(YAML1)` and can crash with `FileNotFoundError`.

Suggested fix:

After all Dataset 1 download attempts, add a hard guard:

```python
if not YAML1.exists():
    raise FileNotFoundError(f"Missing Dataset 1 YAML: {YAML1}")
```

Do the same for `YAML2` before merge.

## Medium Priority

### 4. CNN Yawn pipeline is incomplete in the current workspace

Files:

- `train_cnn_yawn.py:18`
- `app/src/main/java/com/ai2026/drowsydriver/YawnClassifier.kt:30`
- `app/src/main/java/com/ai2026/drowsydriver/YawnClassifier.kt:49`

Current workspace state:

```text
dataset_yawn                         missing
app/src/main/assets/yawn_model.tflite missing
rawdata/data/yawn/no yawn             2591 files
rawdata/data/yawn/yawn                2528 files
```

Impact:

Android build still succeeds because `YawnClassifier` falls back to MAR, but CNN Yawn does not run. Model 2 is not deploy-ready.

Suggested fix:

Run:

```powershell
python tools\prepare_yawn_dataset.py
```

Then train/export:

```powershell
python train_cnn_yawn.py
```

Verify `app/src/main/assets/yawn_model.tflite` exists afterward.

### 5. Android build fails by default if `JAVA_HOME` is not set

Command that failed:

```powershell
.\gradlew.bat assembleDebug
```

Error:

```text
ERROR: JAVA_HOME is not set and no 'java' command could be found in your PATH.
```

Observed workaround:

```powershell
$env:JAVA_HOME="$env:ProgramFiles\Android\Android Studio\jbr"
$env:PATH="$env:JAVA_HOME\bin;$env:PATH"
.\gradlew.bat assembleDebug
```

With that workaround, `assembleDebug` passes.

Suggested fix:

Set `JAVA_HOME` permanently to Android Studio JBR or an installed JDK 17/21 path.

### 6. Android Gradle Plugin is older than the declared `compileSdk`

Files:

- `build.gradle:2`
- `app/build.gradle:8`

Current state:

```gradle
id "com.android.application" version "8.5.2"
compileSdk 35
```

Gradle warning:

```text
This Android Gradle plugin (8.5.2) was tested up to compileSdk = 34.
```

Impact:

Build passes, but the project has a compatibility warning and may break with newer SDK tooling.

Suggested fix:

Upgrade Android Gradle Plugin to a version tested with SDK 35, or add `android.suppressUnsupportedCompileSdk=35` only if intentionally staying on AGP 8.5.2.

### 7. Pyright scans generated `outputs/` copies and reports noisy errors

File:

- `pyrightconfig.json:7`

Current exclude only contains:

```json
[
  "colab_yolo26_training.py",
  "colab_drowsy_yolo26.ipynb"
]
```

Impact:

`npx pyright` reports 111 errors, including files under `outputs/DrowsyDriverAndroid_submission_*`, which are generated submission copies rather than active source.

Suggested fix:

Exclude generated/heavy folders:

```json
"exclude": [
  "outputs",
  "output",
  "dataset",
  "dataset_mrl",
  "mrleyedataset",
  "rawdata",
  ".gradle",
  "build",
  "app/build"
]
```

### 8. Real source still has pyright type issues after excluding generated files

Examples:

- `streamlit_app.py:42` and `streamlit_app.py:44`: `analyze_snapshot` returns `dict[str, object]`, so `result["confidence"]` is typed as `object`, not `float`.
- `tools/train_clean_local.py:268`: `results.save_dir` is treated as possibly missing/None.
- `tools/train_yolov8_local.py:197`: same `results.save_dir` optional-member issue.
- `tune_experiments.py:425` and `tune_experiments.py:427`: optional metric values are used in arithmetic.

Impact:

Most of these are not syntax errors, but they make static checking unreliable and can hide real runtime bugs.

Suggested fix:

Use typed dicts/casts for Streamlit results, guard YOLO training results before accessing `save_dir`, and validate metrics are not `None` before arithmetic.

## Low Priority / Warnings

### 9. Android lint warnings

Lint result:

```text
0 errors, 12 warnings
```

Notable warnings:

- `StatusOverlayView.kt:43`: allocates `RectF` inside `onDraw`.
- `AlertController.kt:33`: SDK check for `O` is obsolete because `minSdk` is 26.
- `MainActivity.kt:91`: hard-coded UI string `"Preparing camera"`.
- `app/build.gradle:35-43`: dependencies have newer versions available.

Suggested fix:

Preallocate/reuse `RectF`, remove obsolete SDK check, move UI strings to `strings.xml`, and update dependencies deliberately.

### 10. Unit test task has no test source

Command:

```powershell
.\gradlew.bat testDebugUnitTest
```

Result:

```text
BUILD SUCCESSFUL
testDebugUnitTest NO-SOURCE
```

Impact:

The app currently has no Android unit tests protecting logic like EAR/MAR thresholds, alert state transitions, or TFLite fallback behavior.

Suggested fix:

Add focused tests for `DrowsinessAnalyzer`, `AlertController` throttling logic if extracted, and model availability fallback paths.

### 11. Documentation mismatch about missing Yawn model

File:

- `TRAINING_PLAN.md`

Issue:

The plan says missing `yawn_model.tflite` can make the app crash, but current Kotlin code catches missing asset and uses MAR fallback.

Impact:

Docs can confuse demo/troubleshooting.

Suggested fix:

Update the documentation to say: missing `yawn_model.tflite` disables CNN Yawn and falls back to MAR, but does not crash the current app.

## Passed Areas

- Python syntax/light checks pass for all 33 `.py` files scanned by `tools/run_all_checks.py`.
- 8 notebooks parse as valid JSON and contain no saved error outputs.
- Android `assembleDebug` passes when Java is configured.
- Android XML files parse successfully.
- PWA JavaScript files pass `node --check`.
- Android lint has no errors, only warnings.

## Recommended Fix Order

1. Remove hard-coded Roboflow API keys and rotate the exposed key.
2. Fix `dataset_yolo26` undefined variable in `colab_yolo26_training.py`.
3. Add missing dataset guards in `tools/train_clean_local.py`.
4. Prepare/train/export CNN Yawn if Model 2 is required.
5. Set permanent `JAVA_HOME` for Android builds.
6. Clean `pyrightconfig.json` excludes, then fix remaining active-source type issues.
7. Address Android lint warnings and add unit tests.

