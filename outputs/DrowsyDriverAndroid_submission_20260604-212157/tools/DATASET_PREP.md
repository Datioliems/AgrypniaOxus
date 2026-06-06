# Dataset Preparation

Fastest path for the 4-day deadline:

1. Download an easy eye/yawn dataset first.
2. Extract frames from videos if the dataset is video-based.
3. Use MediaPipe/OpenCV or manual filtering to crop eye regions.
4. Arrange images as:

```text
dataset/
  train/
    eyes_open/
    eyes_closed/
  val/
    eyes_open/
    eyes_closed/
```

5. Train:

```bash
python tools/summarize_dataset.py --data dataset --out outputs/dataset_summary
```

```bash
python tools/train_eye_classifier.py --data dataset --out app/src/main/assets/drowsiness_model.tflite
```

6. Evaluate:

```bash
python tools/evaluate_eye_classifier.py --data dataset --tflite app/src/main/assets/drowsiness_model.tflite
```

For the report, keep a table with:

- number of images per class;
- train/validation/test split;
- preprocessing size;
- augmentation;
- metrics for each method.
