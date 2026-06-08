# -*- coding: utf-8 -*-
"""
Export YOLO (.pt) → TFLite cho Android.

Chạy (Colab hoặc local có ultralytics):
    python tools/export_yolo_tflite.py --weights yolo11s_best.pt --int8
    python tools/export_yolo_tflite.py --weights yolo26s_best.pt   # YOLO26 = NMS-free, parse dễ hơn

Kết quả: app/src/main/assets/yolo_detector.tflite  +  in ra LABELS để dán vào Kotlin.

LƯU Ý:
- YOLO11  → output [1, 4+nc, 8400] → Kotlin PHẢI tự NMS (xem YoloDetector.kt).
- YOLO26  → end-to-end (NMS-free) → output đã lọc sẵn, Kotlin chỉ cần lọc theo conf.
- --int8 : lượng tử hóa 8-bit → model nhỏ hơn ~4×, nhanh hơn trên điện thoại (nên dùng).
"""
import argparse
import shutil
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="đường dẫn .pt (yolo11s_best.pt / yolo26s_best.pt)")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--int8", action="store_true", help="lượng tử hóa 8-bit (khuyên dùng cho mobile)")
    ap.add_argument("--out", default="app/src/main/assets/yolo_detector.tflite")
    args = ap.parse_args()

    from ultralytics import YOLO
    model = YOLO(args.weights)

    print(f"  Exporting {args.weights} → TFLite (imgsz={args.imgsz}, int8={args.int8})...")
    exported = model.export(format="tflite", imgsz=args.imgsz, int8=args.int8)

    # ultralytics trả về path (file hoặc thư mục saved_model)
    ep = Path(exported)
    tfls = list(ep.glob("*.tflite")) if ep.is_dir() else [ep]
    if not tfls:
        tfls = list(ep.parent.rglob("*.tflite"))
    if not tfls:
        raise SystemExit(f"❌ Không tìm thấy .tflite trong {exported}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(tfls[0], out)
    size_mb = out.stat().st_size / 1e6
    print(f"  ✅  Lưu {out}  ({size_mb:.1f} MB)")

    # In labels theo đúng index để dán vào Kotlin
    names = model.names
    print("\n  LABELS cho Kotlin (đúng thứ tự index):")
    print("  val labels = arrayOf(")
    for i in sorted(names):
        print(f'      "{names[i]}",   // {i}')
    print("  )")
    print("\n  → Copy mảng trên + yolo_detector.tflite vào Android, dùng YoloDetector.kt")


if __name__ == "__main__":
    main()
