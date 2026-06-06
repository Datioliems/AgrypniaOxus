# Datasets, Related Work, And Research Gaps

## Recommended Datasets

| Dataset | Use | Strength | Risk |
|---|---|---|---|
| NTHU Driver Drowsiness Detection Dataset | Main driver drowsiness dataset | Driver-specific, includes day/night, glasses, sunglasses, yawning, nodding | Requires request/license |
| YawDD | Yawning detection | In-vehicle yawning videos | Access may require IEEE DataPort |
| DROZY | Research-grade drowsiness monitoring | Video, physiological signals, sleepiness scores | Larger and more complex |
| EyeState/YawnDrowsiness Kaggle variants | Fast prototype training | Easy to download and train quickly | Less rigorous provenance |
| Self-collected Android videos | Domain adaptation and demo validation | Matches final camera/device | Small and biased dataset |

## Research Gaps

1. **Realtime gap**

Many projects report accuracy on static images or offline video, but do not
measure latency, FPS, or behavior under a realtime camera stream. This project
addresses the gap by using Android CameraX, MediaPipe live stream processing,
and frame throttling.

2. **Deployment gap**

Laptop demos are common, but real driving use needs on-device processing. This
project uses Android phones as edge devices, reducing cost and avoiding network
latency.

3. **Robustness gap**

Drowsiness systems often fail with glasses, sunglasses, night lighting, head
rotation, partial occlusion, and different phone mounting positions. This
project documents these limits and collects small Android-phone samples to test
the actual deployment condition.

4. **Safety-metric gap**

Accuracy alone is not enough. Missing a drowsy driver is more dangerous than a
false alarm. This project prioritizes recall and F1-score for the drowsy class,
plus temporal smoothing to reduce false alarms from natural blinking.

5. **Explainability gap**

Pure CNN outputs can be hard to explain during presentation. This project keeps
MediaPipe landmark features such as EAR and MAR as interpretable signals, then
uses CNN/TFLite as a learnable component.

## How This Project Answers The Gaps

| Gap | Project Response | Value |
|---|---|---|
| Realtime | CameraX + MediaPipe LIVE_STREAM + limited FPS inference | More realistic demo |
| Deployment | Android phone as camera and compute device | Low-cost practical setup |
| Robustness | Dataset mix plus self-collected Android videos | Better domain match |
| Safety metrics | Recall/F1 for drowsy class and confusion matrix | Better safety argument |
| Explainability | EAR/MAR baseline plus CNN comparison | Easier to defend in Q&A |

## Practical Value

- **Individuals:** low-cost warning aid for drivers using a normal Android phone.
- **Transport companies:** prototype for fatigue monitoring in buses, taxis, and
  delivery fleets.
- **Schools/research groups:** reusable Android edge-AI demo for computer vision
  and mobile AI courses.
- **Community safety:** helps raise awareness of fatigue-related driving risk and
  promotes accessible driver-assistance tools.

## Suggested Evaluation Table

| Method | Accuracy | Drowsy Recall | F1-score | FPS | Notes |
|---|---:|---:|---:|---:|---|
| EAR/MAR rules | TBD | TBD | TBD | TBD | Fast and interpretable |
| CNN eye classifier | TBD | TBD | TBD | TBD | Learns visual features |
| MediaPipe + CNN + smoothing | TBD | TBD | TBD | TBD | Final proposed method |
