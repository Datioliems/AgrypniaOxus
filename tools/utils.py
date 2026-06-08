# -*- coding: utf-8 -*-
"""
DrowsyDriver — Shared Utilities
Các hàm tiện ích dùng chung cho tất cả step scripts.
"""
import json
import logging
import sys
import hashlib
import shutil
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np


# ─────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────

def setup_logging(step_name: str, log_dir: Path) -> logging.Logger:
    """Tạo logger ghi ra cả file lẫn console."""
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(step_name)
    logger.setLevel(logging.DEBUG)
    if logger.handlers:
        logger.handlers.clear()

    ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
    fh  = logging.FileHandler(log_dir / f"{step_name}_{ts}.log", encoding="utf-8")
    ch  = logging.StreamHandler(sys.stdout)
    fh.setLevel(logging.DEBUG)
    ch.setLevel(logging.INFO)

    fmt = logging.Formatter("%(asctime)s [%(levelname)-7s] %(message)s", "%H:%M:%S")
    fh.setFormatter(fmt)
    ch.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


# ─────────────────────────────────────────────
# JSON HELPERS
# ─────────────────────────────────────────────

def save_json(data: dict, path: Path, indent: int = 2):
    """Ghi dict ra file JSON (UTF-8)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)


def load_json(path: Path) -> dict:
    """Đọc JSON từ file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ─────────────────────────────────────────────
# HASH
# ─────────────────────────────────────────────

def md5_file(path: Path) -> str:
    """Tính MD5 hash của file."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def md5_dir(directory: Path, extensions: tuple = (".jpg", ".jpeg", ".png", ".bmp")) -> str:
    """
    Tính MD5 tổng hợp của toàn bộ ảnh trong thư mục (sorted để reproducible).
    """
    h = hashlib.md5()
    for fp in sorted(directory.rglob("*")):
        if fp.suffix.lower() in extensions and fp.is_file():
            h.update(fp.name.encode())
            h.update(md5_file(fp).encode())
    return h.hexdigest()


# ─────────────────────────────────────────────
# DIRECTORY HELPERS
# ─────────────────────────────────────────────

def create_dirs(*paths: Path):
    """Tạo nhiều thư mục cùng lúc (bao gồm parent)."""
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)


def copy_image(src: Path, dst: Path):
    """Copy ảnh, tự tạo thư mục đích nếu chưa có."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def list_images(directory: Path, extensions: tuple = (".jpg", ".jpeg", ".png", ".bmp")) -> list:
    """Liệt kê tất cả ảnh trong thư mục (đệ quy)."""
    return [p for p in directory.rglob("*") if p.suffix.lower() in extensions and p.is_file()]


# ─────────────────────────────────────────────
# IMAGE HELPERS
# ─────────────────────────────────────────────

def safe_imread(path: Path) -> np.ndarray | None:
    """
    Đọc ảnh an toàn. Trả về None nếu không đọc được.
    Tự xử lý đường dẫn Unicode trên Windows.
    """
    try:
        img = cv2.imread(str(path))
        if img is None:
            # Thử lại với numpy fromfile (handles Unicode paths on Windows)
            arr = np.fromfile(str(path), dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return img
    except Exception:
        return None


def save_image(img: np.ndarray, path: Path) -> bool:
    """Lưu ảnh an toàn. Trả về True nếu thành công."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        ext = path.suffix.lower()
        if ext in (".jpg", ".jpeg"):
            params = [cv2.IMWRITE_JPEG_QUALITY, 95]
        elif ext == ".png":
            params = [cv2.IMWRITE_PNG_COMPRESSION, 3]
        else:
            params = []
        # Encode rồi write thay vì imwrite để handle Unicode paths
        success, buf = cv2.imencode(ext, img, params)
        if success:
            with open(path, "wb") as f:
                f.write(buf.tobytes())
            return True
        return False
    except Exception:
        return False


# ─────────────────────────────────────────────
# CONSOLE OUTPUT
# ─────────────────────────────────────────────

def print_banner(title: str, step_num: int = None, width: int = 62):
    step_str = f"BƯỚC {step_num:2d} — " if step_num else ""
    border = "═" * width
    print(f"\n{border}")
    print(f"  {step_str}{title}")
    print(f"{border}")


def print_stat(label: str, value, indent: int = 2):
    prefix = " " * indent
    print(f"{prefix}{'·'} {label:<35} {value}")


def print_ok(msg: str):
    print(f"  ✓ {msg}")


def print_warn(msg: str):
    print(f"  ⚠ {msg}")


def print_fail(msg: str):
    print(f"  ✗ {msg}")


# ─────────────────────────────────────────────
# REPORT BUILDER
# ─────────────────────────────────────────────

def build_report(step: int, name: str, dataset: str,
                 input_count: int, output_count: int,
                 status: str = "PASS",
                 warnings: list = None,
                 errors: list = None,
                 metrics: dict = None) -> dict:
    """Tạo report dict chuẩn để lưu vào JSON."""
    return {
        "step": step,
        "name": name,
        "dataset": dataset,
        "started_at": datetime.now().isoformat(),
        "input_count": input_count,
        "output_count": output_count,
        "rejected_count": input_count - output_count,
        "status": status,          # "PASS" | "WARNING" | "FAIL"
        "warnings": warnings or [],
        "errors": errors or [],
        "metrics": metrics or {},
    }
