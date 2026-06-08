# -*- coding: utf-8 -*-
"""
BƯỚC 3 — TRÍCH XUẤT ROI (Face → Eye / Mouth)
Dùng MediaPipe FaceLandmarker (478 landmarks) để crop chính xác:
  - already_cropped=True  (cnn_eye): Bỏ qua extract, chỉ validate aspect ratio
  - already_cropped=False (cnn_yawn): Crop vùng miệng từ ảnh khuôn mặt
  - roi_type=full         (yolo)   : Giữ nguyên ảnh gốc

Fallback: OpenCV Haar Cascade nếu MediaPipe không có model file.

Chạy:
    python step3_extract_roi.py --dataset cnn_eye
    python step3_extract_roi.py --dataset cnn_yawn
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    PROCESSED_DIR, REPORTS_DIR, LOGS_DIR, REJECTED_DIR, DATASETS,
    LEFT_EYE_LANDMARKS, RIGHT_EYE_LANDMARKS, MOUTH_LANDMARKS,
    EYE_PAD_RATIO, MOUTH_PAD_RATIO, FACE_LANDMARKER_PATH,
    EYE_AR_MIN, EYE_AR_MAX, MOUTH_AR_MIN, MOUTH_AR_MAX
)
from utils import (
    setup_logging, save_json, safe_imread, save_image, copy_image,
    print_banner, print_stat, print_ok, print_warn, print_fail,
    build_report, create_dirs, list_images
)

STEP = 3
STEP_NAME = "roi_extraction"


# ─────────────────────────────────────────────
# MEDIAPIPE SETUP (lazy load)
# ─────────────────────────────────────────────

_face_landmarker = None
_use_fallback    = False   # True nếu không có MediaPipe

def _init_mediapipe():
    global _face_landmarker, _use_fallback
    if _face_landmarker is not None:
        return

    try:
        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision

        if not FACE_LANDMARKER_PATH.exists():
            raise FileNotFoundError(f"face_landmarker.task không tìm thấy: {FACE_LANDMARKER_PATH}")

        opts = vision.FaceLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=str(FACE_LANDMARKER_PATH)),
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_score=0.5,
        )
        _face_landmarker = vision.FaceLandmarker.create_from_options(opts)
        print_ok("MediaPipe FaceLandmarker khởi tạo thành công")
    except Exception as e:
        print_warn(f"MediaPipe unavailable ({e}) → fallback sang OpenCV Haar Cascade")
        _use_fallback = True


# ─────────────────────────────────────────────
# MEDIAPIPE LANDMARK DETECTION
# ─────────────────────────────────────────────

def _detect_landmarks(img_bgr: np.ndarray):
    """Trả về list 478 landmarks hoặc None nếu không detect được."""
    try:
        import mediapipe as mp
        img_rgb  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        result   = _face_landmarker.detect(mp_image)
        if result.face_landmarks:
            return result.face_landmarks[0]   # List 478 NormalizedLandmark
        return None
    except Exception:
        return None


def _landmarks_to_pixels(landmarks, img_h: int, img_w: int,
                           indices: list) -> list[tuple[int, int]]:
    """Chuyển NormalizedLandmark → tọa độ pixel."""
    return [
        (int(landmarks[i].x * img_w), int(landmarks[i].y * img_h))
        for i in indices
    ]


# ─────────────────────────────────────────────
# CROP ROI
# ─────────────────────────────────────────────

def crop_roi(img_bgr: np.ndarray, points: list[tuple[int, int]],
             pad_ratio: float, label: str) -> np.ndarray | None:
    """
    Crop bbox từ list điểm + padding.
    Đảm bảo không vượt biên ảnh. Trả về None nếu ROI quá nhỏ.
    """
    h, w = img_bgr.shape[:2]
    xs   = [p[0] for p in points]
    ys   = [p[1] for p in points]

    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    roi_w = x_max - x_min
    roi_h = y_max - y_min

    if roi_w < 5 or roi_h < 5:
        return None

    pad_x = int(roi_w * pad_ratio)
    pad_y = int(roi_h * pad_ratio)

    x1 = max(0, x_min - pad_x)
    y1 = max(0, y_min - pad_y)
    x2 = min(w, x_max + pad_x)
    y2 = min(h, y_max + pad_y)

    if (x2 - x1) < 10 or (y2 - y1) < 10:
        return None

    return img_bgr[y1:y2, x1:x2].copy()


def validate_roi(roi: np.ndarray, roi_type: str) -> tuple[bool, str]:
    """Kiểm tra ROI sau khi crop."""
    if roi is None:
        return False, "ROI extraction failed (None)"

    h, w = roi.shape[:2]
    if h == 0 or w == 0:
        return False, "ROI rỗng (0-size)"

    ar = w / h

    if roi_type == "eye":
        if not (EYE_AR_MIN <= ar <= EYE_AR_MAX):
            return False, f"Eye aspect ratio không hợp lệ: {ar:.2f}"
    elif roi_type == "mouth":
        if not (MOUTH_AR_MIN <= ar <= MOUTH_AR_MAX):
            return False, f"Mouth aspect ratio không hợp lệ: {ar:.2f}"

    # Kiểm tra brightness
    mean_b = float(np.mean(roi))
    if mean_b < 10 or mean_b > 245:
        return False, f"ROI brightness bất thường: {mean_b:.1f}"

    return True, ""


# ─────────────────────────────────────────────
# FALLBACK: OpenCV Haar Cascade
# ─────────────────────────────────────────────

_eye_cascade   = None
_face_cascade  = None

def _init_fallback():
    global _eye_cascade, _face_cascade
    _face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    _eye_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_eye.xml"
    )


def crop_eye_fallback(img_bgr: np.ndarray) -> list[np.ndarray]:
    """Haar Cascade backup: detect mắt không cần MediaPipe."""
    if _face_cascade is None:
        _init_fallback()

    gray  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    faces = _face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(50, 50))
    rois  = []
    for (fx, fy, fw, fh) in faces:
        face_roi  = gray[fy:fy+fh, fx:fx+fw]
        face_bgr  = img_bgr[fy:fy+fh, fx:fx+fw]
        eyes = _eye_cascade.detectMultiScale(face_roi, 1.05, 3)
        for (ex, ey, ew, eh) in eyes:
            pad = int(max(ew, eh) * 0.25)
            x1  = max(0, ex - pad)
            y1  = max(0, ey - pad)
            x2  = min(fw, ex + ew + pad)
            y2  = min(fh, ey + eh + pad)
            rois.append(face_bgr[y1:y2, x1:x2].copy())
    return rois


def crop_mouth_fallback(img_bgr: np.ndarray) -> np.ndarray | None:
    """Haar fallback: ước lượng vùng miệng từ bottom 1/3 của face bbox."""
    if _face_cascade is None:
        _init_fallback()

    gray  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    faces = _face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(50, 50))
    if len(faces) == 0:
        return None
    fx, fy, fw, fh = faces[0]
    # Miệng nằm ở bottom 40% của khuôn mặt, giữa 25-75% chiều ngang
    y1 = fy + int(fh * 0.6)
    y2 = fy + fh
    x1 = fx + int(fw * 0.2)
    x2 = fx + int(fw * 0.8)
    roi = img_bgr[y1:y2, x1:x2]
    return roi if roi.size > 0 else None


# ─────────────────────────────────────────────
# XỬ LÝ THEO ROI TYPE
# ─────────────────────────────────────────────

def process_image(img_path: Path, roi_type: str,
                  already_cropped: bool) -> list[dict]:
    """
    Trả về list kết quả (có thể > 1 nếu cả 2 mắt được extract).
    Mỗi kết quả: {"roi": np.ndarray, "side": "left"|"right"|"mouth"|"full"}
    """
    img = safe_imread(img_path)
    if img is None:
        return []

    # ── Full images (YOLO) ──────────────────────
    if roi_type == "full":
        return [{"roi": img, "side": "full"}]

    # ── Already cropped (CNN Eye từ MRL) ────────
    if already_cropped:
        # Chỉ validate, không extract
        ok, msg = validate_roi(img, roi_type)
        if ok:
            return [{"roi": img, "side": "already_cropped"}]
        return []

    # ── Extract ROI bằng MediaPipe ──────────────
    if not _use_fallback:
        landmarks = _detect_landmarks(img)
        if landmarks:
            h, w = img.shape[:2]
            results = []

            if roi_type == "eye":
                # Mắt trái
                pts_l = _landmarks_to_pixels(landmarks, h, w, LEFT_EYE_LANDMARKS)
                roi_l = crop_roi(img, pts_l, EYE_PAD_RATIO, "left_eye")
                ok_l, _ = validate_roi(roi_l, "eye")
                if ok_l:
                    results.append({"roi": roi_l, "side": "left"})

                # Mắt phải
                pts_r = _landmarks_to_pixels(landmarks, h, w, RIGHT_EYE_LANDMARKS)
                roi_r = crop_roi(img, pts_r, EYE_PAD_RATIO, "right_eye")
                ok_r, _ = validate_roi(roi_r, "eye")
                if ok_r:
                    results.append({"roi": roi_r, "side": "right"})

                return results

            elif roi_type == "mouth":
                pts_m = _landmarks_to_pixels(landmarks, h, w, MOUTH_LANDMARKS)
                roi_m = crop_roi(img, pts_m, MOUTH_PAD_RATIO, "mouth")
                ok_m, _ = validate_roi(roi_m, "mouth")
                if ok_m:
                    return [{"roi": roi_m, "side": "mouth"}]
                return []

    # ── Fallback OpenCV ─────────────────────────
    if roi_type == "eye":
        rois = crop_eye_fallback(img)
        return [{"roi": r, "side": f"fallback_{i}"} for i, r in enumerate(rois)]
    elif roi_type == "mouth":
        roi = crop_mouth_fallback(img)
        if roi is not None:
            return [{"roi": roi, "side": "fallback_mouth"}]

    return []


# ─────────────────────────────────────────────
# MAIN RUNNER
# ─────────────────────────────────────────────

def run_roi_extraction(dataset_name: str) -> dict:
    print_banner("TRÍCH XUẤT ROI (Eye / Mouth)", STEP)
    logger = setup_logging(f"step{STEP}_{dataset_name}", LOGS_DIR)

    cfg             = DATASETS[dataset_name]
    roi_type        = cfg["roi_type"]
    already_cropped = cfg["already_cropped"]
    src_path        = PROCESSED_DIR / cfg["raw_subdir"] / "quality_filtered"
    out_path        = PROCESSED_DIR / cfg["raw_subdir"] / "roi_extracted"
    rej_path        = REJECTED_DIR / dataset_name / "step3_roi"

    create_dirs(out_path, rej_path, REPORTS_DIR)

    if not src_path.exists():
        print_fail(f"Nguồn không tồn tại: {src_path}")
        print("  → Hãy chạy step2_quality_filter.py trước")
        sys.exit(1)

    print_stat("Dataset", dataset_name)
    print_stat("ROI type", roi_type)
    print_stat("Already cropped", already_cropped)

    if roi_type in ("eye", "mouth") and not already_cropped:
        _init_mediapipe()

    total   = 0
    saved   = 0
    failed  = 0

    for split in ["train", "val", "test"]:
        split_src = src_path / split
        if not split_src.exists():
            continue

        for class_dir in sorted(split_src.iterdir()):
            if not class_dir.is_dir():
                continue
            class_name = class_dir.name
            print(f"\n  [{split}/{class_name}]")

            split_saved = 0
            split_fail  = 0

            for img_path in list_images(class_dir):
                total += 1
                rois = process_image(img_path, roi_type, already_cropped)

                if not rois:
                    failed += 1
                    split_fail += 1
                    # Copy bản gốc sang rejected
                    copy_image(img_path, rej_path / split / class_name / img_path.name)
                    logger.debug(f"ROI FAIL: {img_path.name}")
                    continue

                # Lưu tất cả ROI tìm được
                for r_info in rois:
                    stem = img_path.stem
                    side = r_info["side"]
                    out_name = f"{stem}_{side}.jpg"
                    dst = out_path / split / class_name / out_name
                    if save_image(r_info["roi"], dst):
                        saved += 1
                        split_saved += 1
                    else:
                        failed += 1
                        split_fail += 1

            print_stat("    Saved ROIs", split_saved)
            print_stat("    Failed", split_fail)

    fail_pct = failed / max(total, 1) * 100
    status   = "PASS"
    warnings, errors = [], []

    if fail_pct > 20:
        errors.append(f"Quá nhiều ảnh không extract được ROI: {fail_pct:.1f}%")
        status = "FAIL"
    elif fail_pct > 10:
        warnings.append(f"Tỷ lệ ROI fail cao: {fail_pct:.1f}%")
        status = "WARNING"

    print()
    print_stat("Tổng ảnh đầu vào", total)
    print_stat("ROI saved", saved)
    print_stat("Thất bại", f"{failed} ({fail_pct:.1f}%)")

    for w in warnings:
        print_warn(w)
    for e in errors:
        print_fail(e)
    if status == "PASS":
        print_ok(f"Bước 3 hoàn thành — {saved} ROI từ {total} ảnh")

    report = build_report(
        step=STEP, name=STEP_NAME, dataset=dataset_name,
        input_count=total, output_count=saved,
        status=status, warnings=warnings, errors=errors,
        metrics={
            "roi_type": roi_type,
            "already_cropped": already_cropped,
            "mediapipe_used": not _use_fallback,
            "fail_pct": round(fail_pct, 2),
            "output_dir": str(out_path),
        }
    )
    save_json(report, REPORTS_DIR / f"step{STEP}_{dataset_name}_{STEP_NAME}.json")

    if status == "FAIL":
        sys.exit(1)
    return report


def main():
    parser = argparse.ArgumentParser(description="Bước 3: ROI Extraction")
    parser.add_argument("--dataset", default="cnn_eye",
                        choices=list(DATASETS.keys()) + ["all"])
    args = parser.parse_args()

    if args.dataset == "all":
        for ds in DATASETS:
            run_roi_extraction(ds)
    else:
        run_roi_extraction(args.dataset)


if __name__ == "__main__":
    main()
