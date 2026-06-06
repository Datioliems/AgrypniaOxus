# Data Collection With Android Phone

## Goal

Collect a small domain-matched dataset using the same Android phone or similar
phone that will run the demo. This helps answer the research gap that many
drowsiness projects train offline but do not test on the actual deployment
camera.

## What To Record

Record short clips, 10-20 seconds each:

| Scenario | Clips | Notes |
|---|---:|---|
| Awake, normal blinking | 3-5 | Look forward, normal face |
| Eyes closed | 3-5 | Close eyes for 2-3 seconds |
| Slow blink / sleepy eyes | 3-5 | Half-closed eyes, slow reopen |
| Yawning / open mouth | 3-5 | Natural yawn or simulated yawn |
| Looking left/right | 2-3 | For limitation testing |
| Low light | 2-3 | Only if safe and possible |
| Glasses | 2-3 | If available |

Keep the phone in the same position expected for demo:

- front camera or phone mounted near dashboard;
- face centered;
- one person only;
- stable lighting for first demo.

## Suggested Labels

For a 4-day project, keep labels simple:

```text
eyes_open
eyes_closed
yawning
not_yawning
```

For the CNN, prioritize:

```text
eyes_open
eyes_closed
```

## Dataset Folder Structure

Use:

```text
dataset/
  train/
    eyes_closed/
    eyes_open/
  val/
    eyes_closed/
    eyes_open/
  test/
    eyes_closed/
    eyes_open/
```

Recommended split:

- train: 70%;
- val: 15%;
- test: 15%.

If data is very small, still create all three folders, even if test only has a
few images per class. In the report, honestly state that dataset size is a
limitation.

## Frame Extraction

If using video, extract frames manually or with any video tool. Do not keep every
frame because adjacent frames are almost identical.

Recommended:

- sample 2-5 frames per second;
- delete blurry frames;
- avoid duplicate frames;
- keep class counts roughly balanced.

## What To Put In Chapter 2

Fill this table:

| Split | eyes_open | eyes_closed | Total |
|---|---:|---:|---:|
| train | TBD | TBD | TBD |
| val | TBD | TBD | TBD |
| test | TBD | TBD | TBD |

Explain:

- source of data;
- how labels were assigned;
- how ROI images were prepared;
- why the dataset is limited;
- why self-collected phone data matters for Android deployment.

## Safety Note

Do not collect data while actually driving. Record while parked or sitting in a
safe indoor setup. The demo is a prototype, not a road-tested safety product.
