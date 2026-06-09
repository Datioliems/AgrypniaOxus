# -*- coding: utf-8 -*-
"""Tạo icon adaptive cho Android từ logo AgrypniaOxus."""
from PIL import Image
import os

src = 'D:/2026.AI/DrowsyDriverAndroid/icon/image-1780965096748.webp'
img = Image.open(src).convert('RGBA')
w, h = img.size

# Crop phần icon (bỏ text ~38% đáy)
icon_h = int(h * 0.62)
icon   = img.crop((0, 0, w, icon_h))

# Adaptive icon foreground: padding 17% mỗi chiều để vào safe zone
PAD   = int(w * 0.17)
total = w + PAD * 2
fg    = Image.new('RGBA', (total, total), (0, 0, 0, 0))
y_off = (total - icon_h) // 2
fg.paste(icon, (PAD, y_off), icon)

SIZES = [
    ('drawable',         432),   # foreground PNG cho adaptive icon XML
    ('mipmap-mdpi',      108),
    ('mipmap-hdpi',      162),
    ('mipmap-xhdpi',     216),
    ('mipmap-xxhdpi',    324),
    ('mipmap-xxxhdpi',   432),
]
base = 'D:/2026.AI/DrowsyDriverAndroid/app/src/main/res'

for folder, sz in SIZES:
    d = f'{base}/{folder}'
    os.makedirs(d, exist_ok=True)
    r = fg.resize((sz, sz), Image.LANCZOS)
    # foreground PNG (dùng bởi adaptive icon XML)
    r.save(f'{d}/ic_launcher_foreground.png')
    # launcher PNG thường (fallback + splash icon)
    if 'mipmap' in folder:
        r.save(f'{d}/ic_launcher.png')
        r.save(f'{d}/ic_launcher_round.png')
    print(f'  -> {folder}: {sz}px OK')

print('ALL DONE')
