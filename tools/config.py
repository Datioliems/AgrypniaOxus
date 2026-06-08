# -*- coding: utf-8 -*-
"""
DrowsyDriver — Shared Configuration
Tất cả hằng số và đường dẫn dùng chung cho toàn bộ pipeline.
QUAN TRỌNG: Giá trị ở đây PHẢI khớp với Constants.kt trong Android app.
"""
from pathlib import Path

# ═══════════════════════════════════════════════════════════
# ROOT PATHS
# ═══════════════════════════════════════════════════════════
PROJECT_ROOT  = Path(__file__).parent.parent   # DrowsyDriverAndroid/
DATA_DIR      = PROJECT_ROOT / "data"
TOOLS_DIR     = PROJECT_ROOT / "tools"
OUTPUTS_DIR   = PROJECT_ROOT / "outputs"
LOGS_DIR      = PROJECT_ROOT / "logs"
REPORTS_DIR   = OUTPUTS_DIR / "reports"

# Data pipeline dirs
PROCESSED_DIR  = DATA_DIR / "processed"  # Sau bước 1–4
SPLITS_DIR     = DATA_DIR / "splits"     # Sau bước 7 (train/val/test)
TENSORS_DIR    = DATA_DIR / "tensors"    # Sau bước 8 (npy arrays, optional)
EDA_DIR        = OUTPUTS_DIR / "eda"
REJECTED_DIR   = OUTPUTS_DIR / "rejected"

# ═══════════════════════════════════════════════════════════
# ĐÃ TÌM THẤY DATASET THỰC TẾ TRÊN MÁY
# (Cập nhật 2026-06-07 — quét thư mục project)
#
# CNN EYE (nhiều nguồn, cả đã split và chưa split):
#   dataset/           eyes_closed/eyes_open — 84,899 ảnh, ĐÃ split train/val/test ✅
#   mrleyedataset/     Close-Eyes/Open-Eyes  — 84,898 ảnh, CHƯA split
#   rawdata/data/eyes/ Close/Open            — 90,643 ảnh, ĐÃ split (train/val/test)
#   rawdata/closed_eye + rawdata/open_eye    — 48,000 ảnh, CHƯA split
#   → Dùng: dataset/ (đã split, class names đúng format)
#
# CNN YAWN (1 nguồn chính):
#   dataset_yawn/      no_yawn/yawn          —  5,119 ảnh, ĐÃ split train/val/test ✅
#   rawdata/data/yawn/ no yawn/yawn          —  5,119 ảnh, CHƯA split (cùng nguồn)
#   → Dùng: dataset_yawn/ (đã split, class names đúng format)
#
# YOLO (format images+labels):
#   roboflow_data/ds_driveryawn/             —  5,570 ảnh YOLO format, train/val/test
#   → Dùng trên COLAB cho YOLO training
# ═══════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════
# DATASET CONFIGURATIONS — ĐÃ CẬP NHẬT ĐƯỜNG DẪN THỰC TẾ
# ═══════════════════════════════════════════════════════════
DATASETS = {
    "cnn_eye": {
        # Đường dẫn TUYỆT ĐỐI đến dataset thực tế trên máy
        "raw_path": PROJECT_ROOT / "dataset",
        # Tên subdirectory dùng cho processed/splits output
        "raw_subdir": "cnn_eye",
        # THỨ TỰ PHẢI KHỚP VỚI Android DrowsinessConstants.CNN_EYE_CLASSES
        "classes":   ["eyes_closed", "eyes_open"],   # index 0=eyes_closed, 1=eyes_open
        # Tên class trong thư mục nguồn (có thể khác với target classes)
        "source_class_map": {
            "eyes_closed": "eyes_closed",   # folder name → target class
            "eyes_open":   "eyes_open",
        },
        "roi_type":       "eye",
        "already_cropped": True,   # dataset/ đã là ảnh crop vùng mắt
        "target_size": 64,
        # Mục tiêu TỐI ĐA mỗi class (Roboflow cap: 8000/class/split)
        "train_per_class": 8000,
        "val_per_class":   2000,
        "test_per_class":  2000,
        # Số ảnh thực tế có trên máy (train split)
        "actual_train_counts": {"eyes_closed": 25167, "eyes_open": 25770},
        "note": "Có sẵn 25k/class — cần SAMPLE xuống 8000 thay vì aug lên",
    },
    "cnn_yawn": {
        "raw_path": PROJECT_ROOT / "dataset_yawn",
        "raw_subdir": "cnn_yawn",
        # THỨ TỰ PHẢI KHỚP VỚI Android DrowsinessConstants.CNN_YAWN_CLASSES
        "classes":   ["no_yawn", "yawn"],            # index 0=no_yawn, 1=yawn
        "source_class_map": {
            "no_yawn": "no_yawn",
            "yawn":    "yawn",
        },
        "roi_type":        "mouth",
        "already_cropped": True,   # dataset_yawn/ ảnh đã crop vùng miệng/khuôn mặt
        "target_size": 64,
        "train_per_class": 2000,   # Dataset nhỏ: chỉ có ~2k/class → aug lên 4k
        "val_per_class":   252,    # Giữ nguyên val (252 yawn, 259 no_yawn)
        "test_per_class":  254,    # Giữ nguyên test (254 yawn, 260 no_yawn)
        "actual_train_counts": {"no_yawn": 2072, "yawn": 2022},
        "note": "Dataset nhỏ ~2k/class — cần aug nhiều hơn (aug lên 4k)",
    },
    "yolo": {
        "raw_path": PROJECT_ROOT / "roboflow_data" / "ds_driveryawn",
        "raw_subdir": "yolo",
        "classes": ["drowsy"],   # ds_driveryawn có thể chỉ 1 class — xem data.yaml
        "roi_type":        "full",
        "already_cropped": False,
        "target_size": 640,
        "train_per_class": 4005,  # Thực tế: 4,005 ảnh train
        "note": "YOLO format — chạy trên COLAB, không qua pipeline bước 1-10",
    },
    # Extra: MRL Eye raw (chưa split) — dùng nếu cần thêm data
    "mrl_eye_raw": {
        "raw_path": PROJECT_ROOT / "mrleyedataset",
        "raw_subdir": "mrl_eye_raw",
        "classes":   ["eyes_closed", "eyes_open"],
        "source_class_map": {
            "Close-Eyes": "eyes_closed",   # Cần rename khi copy
            "Open-Eyes":  "eyes_open",
        },
        "roi_type":        "eye",
        "already_cropped": True,
        "target_size": 64,
        "train_per_class": 8000,
        "actual_train_counts": {"Close-Eyes": 41946, "Open-Eyes": 42952},
        "note": "Nguồn backup — chưa split, cần bước 7 split từ đầu",
    },
}

# ═══════════════════════════════════════════════════════════
# IMAGE SIZE
# ═══════════════════════════════════════════════════════════
CNN_INPUT_SIZE  = 64    # CNN: 64×64×3
YOLO_INPUT_SIZE = 640   # YOLO: 640×640×3

# ═══════════════════════════════════════════════════════════
# BƯỚC 2 — QUALITY FILTER THRESHOLDS
# ═══════════════════════════════════════════════════════════
BLUR_LAPLACIAN_MIN    = 50.0    # Laplacian variance < 50 → mờ → loại
BRIGHTNESS_MIN        = 20.0    # Quá tối
BRIGHTNESS_MAX        = 235.0   # Quá sáng (overexposed)
CONTRAST_STD_MIN      = 15.0    # std pixel < 15 → không có tương phản
FACE_CONF_MIN         = 0.75    # MediaPipe face detection confidence tối thiểu
MIN_IMAGE_DIM         = 20      # Chiều nhỏ nhất chấp nhận (pixel)
EYE_AR_MIN            = 1.2     # Aspect ratio tối thiểu eye ROI
EYE_AR_MAX            = 7.0     # Aspect ratio tối đa eye ROI
MOUTH_AR_MIN          = 1.0     # Aspect ratio tối thiểu mouth ROI
MOUTH_AR_MAX          = 5.0     # Aspect ratio tối đa mouth ROI

# ═══════════════════════════════════════════════════════════
# BƯỚC 3 — ROI EXTRACTION (MediaPipe landmarks)
# ═══════════════════════════════════════════════════════════
# 6 điểm EAR cho mỗi mắt
LEFT_EYE_LANDMARKS  = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_LANDMARKS = [362, 385, 387, 263, 373, 380]
# 8 điểm MAR cho miệng
MOUTH_LANDMARKS     = [61, 291, 0, 17, 39, 269, 270, 409]
# Padding khi crop (tỷ lệ so với kích thước bbox)
EYE_PAD_RATIO   = 0.25   # 25% mỗi phía
MOUTH_PAD_RATIO = 0.30   # 30% mỗi phía

# MediaPipe model path
FACE_LANDMARKER_PATH = PROJECT_ROOT / "app" / "src" / "main" / "assets" / "face_landmarker.task"

# ═══════════════════════════════════════════════════════════
# BƯỚC 6 — AUGMENTATION TARGETS
# ═══════════════════════════════════════════════════════════
AUG_SEED = 42   # Random seed để tái tạo

# ═══════════════════════════════════════════════════════════
# BƯỚC 7 — SPLIT RATIOS
# ═══════════════════════════════════════════════════════════
SPLIT_TRAIN = 0.70
SPLIT_VAL   = 0.15
SPLIT_TEST  = 0.15
SPLIT_SEED  = 42

# ═══════════════════════════════════════════════════════════
# ANDROID / TFLITE CONSTANTS — PHẢI KHỚP VỚI Constants.kt
# ═══════════════════════════════════════════════════════════
CNN_EYE_CONF_THRESHOLD  = 0.55   # DrowsinessConstants.CNN_EYE_CONF_THRESHOLD
CNN_YAWN_CONF_THRESHOLD = 0.60   # DrowsinessConstants.CNN_YAWN_CONF_THRESHOLD
EAR_THRESHOLD           = 0.24   # DrowsinessConstants.EAR_THRESHOLD
MAR_THRESHOLD           = 0.58   # DrowsinessConstants.MAR_THRESHOLD
DROWSY_DURATION_MS      = 1700   # DrowsinessConstants.DROWSY_DURATION_MS
CNN_DROWSY_DURATION_MS  = 1200   # DrowsinessConstants.CNN_DROWSY_DURATION_MS
YAWN_DURATION_MS        = 800    # DrowsinessConstants.YAWN_DURATION_MS

# TFLite spec bắt buộc
TFLITE_INPUT_SHAPE = [1, CNN_INPUT_SIZE, CNN_INPUT_SIZE, 3]
TFLITE_INPUT_DTYPE = "float32"
TFLITE_VALUE_RANGE = (0.0, 1.0)   # Normalize /255.0 NGOÀI model

# ═══════════════════════════════════════════════════════════
# ALERT AUDIO THRESHOLDS
# ═══════════════════════════════════════════════════════════
AUDIO_TIER1_MIN_LEVEL   = 1     # ≥ LEVEL 1
AUDIO_TIER2_EVENTS      = 3     # ≥ 3 events/15min → Binaural Beta
AUDIO_TIER3_EVENTS      = 5     # ≥ 5 events/15min → 19Hz Infrasound
AUDIO_TIER4_EVENTS      = 7     # ≥ 7 events/15min → 40Hz Gamma
AUDIO_TIER3_EYE_MS      = 3000  # Mắt nhắm > 3s → Tier 3
AUDIO_TIER4_NO_RESPONSE = 10000 # Không phản hồi 10s → Tier 4→5

# ═══════════════════════════════════════════════════════════
# REPORT STEP KEYS (dùng trong step reports)
# ═══════════════════════════════════════════════════════════
STEP_KEYS = {
    1: "inventory",
    2: "quality_filter",
    3: "roi_extraction",
    4: "resize",
    5: "eda",
    6: "augmentation",
    7: "split",
    8: "normalize",
    9: "integrity_check",
    10: "manifest",
}
