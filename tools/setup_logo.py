# -*- coding: utf-8 -*-
"""
Chạy script này SAU KHI bạn lưu logo gốc (cả icon + chữ) vào:
    D:/2026.AI/DrowsyDriverAndroid/tools/logo_source.png

Script sẽ:
  1. Crop phần trên (icon) — bỏ phần text "AgrypniaOxus" ở đáy
  2. Thêm padding trong suốt xung quanh
  3. Xuất ra 4 kích cỡ density cho Android

Nếu KHÔNG có PIL:  pip install pillow
"""
import os, sys
from pathlib import Path

try:
    from PIL import Image
except ImportError:
    sys.exit("Cần cài Pillow:  pip install pillow")

SRC   = Path(__file__).parent / "logo_source.png"
OUT   = Path(__file__).parent.parent / "app/src/main/res"
SIZES = {
    "drawable-hdpi":    72,
    "drawable-xhdpi":   96,
    "drawable-xxhdpi":  144,
    "drawable-xxxhdpi": 192,
    "drawable":         144,   # fallback
}

if not SRC.exists():
    sys.exit(f"❌ Không tìm thấy: {SRC}\n   Lưu logo gốc vào đó rồi chạy lại.")

img = Image.open(SRC).convert("RGBA")
w, h = img.size

# Ảnh logo gốc: phần icon chiếm ~72% chiều cao trên cùng, chữ 28% cuối
# → crop bỏ 30% đáy để lấy icon sạch
crop_h = int(h * 0.70)
icon   = img.crop((0, 0, w, crop_h))

# Thêm padding 8%
pad = int(crop_h * 0.08)
canvas_size = crop_h + pad * 2
canvas = Image.new("RGBA", (canvas_size, canvas_size), (0, 0, 0, 0))
offset_x = (canvas_size - w) // 2
canvas.paste(icon, (offset_x, pad), icon)

print(f"✅ Icon size gốc: {w}×{crop_h}px, canvas: {canvas_size}×{canvas_size}px")

for folder, size in SIZES.items():
    out_dir = OUT / folder
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "logo_icon.png"
    canvas.resize((size, size), Image.LANCZOS).save(out_file, "PNG")
    print(f"   → {out_file} ({size}×{size})")

print("\n🎉 Xong! Build lại app là splash screen sẽ hiển thị logo mới.")
