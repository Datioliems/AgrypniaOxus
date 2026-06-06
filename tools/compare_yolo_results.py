"""
So sánh kết quả YOLOv8 (local RTX 4050) vs YOLO26 (Colab T4)

Đọc các file summary JSON từ cả 2 training runs và tạo:
  1. Bảng so sánh in terminal
  2. Bar chart PNG
  3. JSON tổng hợp

Usage:
  # Sau khi cả 2 training xong:
  cd D:\\2026.AI\\DrowsyDriverAndroid
  python tools\\compare_yolo_results.py

  # Chỉ đọc 1 file (khi chỉ 1 bên xong):
  python tools\\compare_yolo_results.py --only-local
  python tools\\compare_yolo_results.py --only-colab
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


PROJECT_ROOT = Path(__file__).parent.parent

# Vị trí summary JSON
LOCAL_SUMMARIES = [
    PROJECT_ROOT / "outputs" / "training_yolo" / "yolov8" / "summary_yolov8n.json",
    PROJECT_ROOT / "outputs" / "training_yolo" / "yolov8" / "summary_yolov8s.json",
    PROJECT_ROOT / "outputs" / "training_yolo" / "yolov8" / "summary_yolov8m.json",
]
COLAB_SUMMARY = PROJECT_ROOT / "outputs" / "training_yolo" / "yolo26" / "summary_yolo26.json"

OUT_DIR = PROJECT_ROOT / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_summary(path: Path) -> dict | None:
    """Load JSON summary, return None nếu không tồn tại"""
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception as e:
        print(f"[!]  Cannot read {path}: {e}")
        return None


def load_from_csv(csv_path: Path, model_name: str, device: str) -> dict | None:
    """Fallback: load từ results.csv của Ultralytics nếu không có summary JSON"""
    if not csv_path.exists():
        return None
    try:
        import pandas as pd
        df = pd.read_csv(csv_path)
        df.columns = [c.strip() for c in df.columns]

        map50_cols   = [c for c in df.columns if "map50" in c.lower() and "95" not in c.lower()]
        map5095_cols = [c for c in df.columns if "map50-95" in c.lower() or "map50_95" in c.lower()]
        prec_cols    = [c for c in df.columns if "precision" in c.lower()]
        rec_cols     = [c for c in df.columns if "recall" in c.lower()]

        def get_best(cols):
            if not cols: return 0.0, 0
            col = cols[0]
            idx = df[col].idxmax()
            ep  = int(df["epoch"].iloc[idx]) if "epoch" in df.columns else idx
            return float(df[col].iloc[idx]), ep

        best_map50, ep = get_best(map50_cols)
        best_5095, _   = get_best(map5095_cols)
        best_prec, _   = get_best(prec_cols)
        best_rec, _    = get_best(rec_cols)

        return {
            "model":          model_name,
            "device":         device,
            "epochs_ran":     len(df),
            "best_epoch":     ep,
            "best_map50":     best_map50,
            "best_map50_95":  best_5095,
            "best_precision": best_prec,
            "best_recall":    best_rec,
        }
    except Exception as e:
        print(f"[!]  Cannot parse CSV {csv_path}: {e}")
        return None


def scan_for_results(search_root: Path) -> list[dict]:
    """Tìm tất cả results.csv trong thư mục training"""
    found = []
    for csv in search_root.rglob("results.csv"):
        run_name = csv.parent.name
        device   = "local_rtx4050" if "yolov8" in str(csv) else "colab_t4"
        s = load_from_csv(csv, run_name, device)
        if s:
            found.append(s)
    return found


def make_comparison_table(summaries: list[dict]) -> str:
    """Tạo bảng so sánh text"""
    headers = ["Run", "Device", "Epochs", "mAP@50", "mAP@50-95", "Prec", "Recall"]
    rows    = []

    for s in summaries:
        rows.append([
            s.get("model", "?")[:20],
            s.get("device", "?"),
            str(s.get("epochs_ran", "?")),
            f"{s.get('best_map50', 0)*100:.2f}%",
            f"{s.get('best_map50_95', 0)*100:.2f}%",
            f"{s.get('best_precision', 0)*100:.2f}%",
            f"{s.get('best_recall', 0)*100:.2f}%",
        ])

    col_widths = [max(len(h), max((len(r[i]) for r in rows), default=0))
                  for i, h in enumerate(headers)]

    def fmt_row(row):
        return "│ " + " │ ".join(
            cell.ljust(w) for cell, w in zip(row, col_widths)
        ) + " │"

    sep = "├─" + "─┼─".join("─" * w for w in col_widths) + "─┤"
    top = "┌─" + "─┬─".join("─" * w for w in col_widths) + "─┐"
    bot = "└─" + "─┴─".join("─" * w for w in col_widths) + "─┘"

    lines = [top, fmt_row(headers), sep]
    for row in rows:
        lines.append(fmt_row(row))
    lines.append(bot)
    return "\n".join(lines)


def make_bar_chart(summaries: list[dict], out_path: Path):
    """Vẽ bar chart so sánh metrics"""
    if not summaries:
        print("Không có data để vẽ")
        return

    labels = [s.get("model", "?").replace("drowsy_", "") for s in summaries]
    metrics = {
        "mAP@50 (%)":    [s.get("best_map50", 0) * 100 for s in summaries],
        "Precision (%)": [s.get("best_precision", 0) * 100 for s in summaries],
        "Recall (%)":    [s.get("best_recall", 0) * 100 for s in summaries],
        "mAP@50-95 (%)": [s.get("best_map50_95", 0) * 100 for s in summaries],
    }

    # Màu theo device
    colors = []
    for s in summaries:
        dev = s.get("device", "")
        if "colab" in dev:
            colors.append("#FF6B6B")   # đỏ = Colab
        elif "local" in dev or "rtx" in dev:
            colors.append("#4ECDC4")   # xanh = Local
        else:
            colors.append("#95A5A6")   # xám = unknown

    x        = np.arange(len(summaries))
    n_metric = len(metrics)
    width    = 0.8 / n_metric

    fig, ax  = plt.subplots(figsize=(max(10, len(summaries) * 3), 7))

    for i, (metric_name, vals) in enumerate(metrics.items()):
        offset  = (i - n_metric / 2 + 0.5) * width
        bars    = ax.bar(x + offset, vals, width, label=metric_name, alpha=0.85)
        for bar, val in zip(bars, vals):
            if val > 1:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                        f"{val:.1f}", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=9)
    ax.set_ylabel("Score (%)", fontsize=11)
    ax.set_title("[SCAN] So Sánh YOLOv8 (Local RTX 4050) vs YOLO26 (Colab T4)", fontsize=13)
    ax.set_ylim(0, 115)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)

    # Legend thiết bị
    local_patch = mpatches.Patch(color="#4ECDC4", label="Local RTX 4050")
    colab_patch = mpatches.Patch(color="#FF6B6B", label="Colab T4")
    ax.legend(
        handles=list(ax.get_legend_handles_labels()[0]) + [local_patch, colab_patch],
        labels=list(ax.get_legend_handles_labels()[1]) + ["Local RTX 4050", "Colab T4"],
        loc="upper right", fontsize=8, ncol=2,
    )

    plt.tight_layout()
    plt.savefig(str(out_path), dpi=130, bbox_inches="tight")
    plt.show()
    print(f"[OK] Bar chart saved: {out_path}")


def print_winner(summaries: list[dict]):
    """In ra model thắng cuộc"""
    if len(summaries) < 2:
        return

    best = max(summaries, key=lambda s: s.get("best_map50", 0))
    print(f"\n[BEST] WINNER: {best.get('model','?')} [{best.get('device','?')}]")
    print(f"   mAP@50    : {best.get('best_map50', 0)*100:.2f}%")
    print(f"   mAP@50-95 : {best.get('best_map50_95', 0)*100:.2f}%")
    print(f"   Precision : {best.get('best_precision', 0)*100:.2f}%")
    print(f"   Recall    : {best.get('best_recall', 0)*100:.2f}%")

    # Inference speed nếu có
    if "inference_ms_colab" in best:
        ms = best["inference_ms_colab"]
        fps = best.get("fps_colab", 1000/ms if ms > 0 else 0)
        print(f"   Inference  : {ms:.1f} ms/frame ({fps:.1f} FPS)")

    # Khuyến nghị
    map50 = best.get("best_map50", 0)
    print("\n[Android] Khuyến nghị cho Android:")
    if map50 >= 0.85:
        print("   [OK] Model đủ tốt cho production (mAP50 ≥ 85%)")
    elif map50 >= 0.70:
        print("   [!]  Tạm dùng được, nên train thêm (mAP50 70-85%)")
    else:
        print("   [X] Cần cải thiện thêm (mAP50 < 70%)")

    print("\n[NOTE] Lưu ý triển khai:")
    print("   - YOLOv8 export TFLite: model.export(format='tflite', imgsz=320)")
    print("   - TFLite model ~ 3-10MB (tùy size)")
    print("   - Android real-time cần < 100ms/frame")
    print("   - Nếu chậm: dùng yolov8n với imgsz=320 thay vì 640")


def main():
    parser = argparse.ArgumentParser(description="So sánh kết quả training YOLO")
    parser.add_argument("--only-local", action="store_true",
                        help="Chỉ đọc kết quả local")
    parser.add_argument("--only-colab", action="store_true",
                        help="Chỉ đọc kết quả Colab")
    parser.add_argument("--scan", action="store_true",
                        help="Tự động quét thư mục outputs/ tìm results.csv")
    args = parser.parse_args()

    print("=" * 60)
    print("[CHART] YOLO COMPARISON REPORT")
    print("=" * 60)

    summaries = []

    if args.scan:
        # Tự động tìm tất cả results
        found = scan_for_results(PROJECT_ROOT / "outputs")
        summaries.extend(found)
        print(f"[SCAN] Auto-scan: tìm thấy {len(found)} run(s)")

    else:
        # ── Local results ────────────────────────────────────────
        if not args.only_colab:
            print("\n[1] Đọc kết quả LOCAL (RTX 4050):")
            for path in LOCAL_SUMMARIES:
                s = load_summary(path)
                if s:
                    summaries.append(s)
                    print(f"   [OK] {path.name}: mAP50={s.get('best_map50',0)*100:.2f}%")
                else:
                    # Fallback: tìm results.csv trong outputs/training_yolo/yolov8/
                    model_name = path.stem.replace("summary_", "")
                    yolo_dir   = PROJECT_ROOT / "outputs" / "training_yolo" / "yolov8"
                    for run_dir in yolo_dir.glob(f"*{model_name}*"):
                        csv = run_dir / "results.csv"
                        s2  = load_from_csv(csv, model_name, "local_rtx4050")
                        if s2:
                            summaries.append(s2)
                            print(f"   [OK] {model_name} (từ CSV): mAP50={s2.get('best_map50',0)*100:.2f}%")
                            break
                    else:
                        print(f"   [...] {path.name} chưa có (training chưa xong?)")

        # ── Colab results ─────────────────────────────────────────
        if not args.only_local:
            print("\n[2] Đọc kết quả COLAB (T4):")
            s = load_summary(COLAB_SUMMARY)
            if s:
                summaries.append(s)
                print(f"   [OK] {COLAB_SUMMARY.name}: mAP50={s.get('best_map50',0)*100:.2f}%")
            else:
                colab_dir = PROJECT_ROOT / "outputs" / "training_yolo" / "yolo26"
                found_csv = list(colab_dir.rglob("results.csv"))
                if found_csv:
                    s2 = load_from_csv(found_csv[0], "yolo26_colab", "colab_t4")
                    if s2:
                        summaries.append(s2)
                        print(f"   [OK] YOLO26 (từ CSV): mAP50={s2.get('best_map50',0)*100:.2f}%")
                else:
                    print(f"   [...] {COLAB_SUMMARY.name} chưa có")
                    print(f"      → Download từ Colab và copy vào: {COLAB_SUMMARY.parent}")

    if not summaries:
        print("\n[!]  Chưa có kết quả nào!")
        print("   Đợi training xong rồi chạy lại script này.")
        print("\n   Checklist:")
        print("   □ Local RTX 4050: chạy tools/train_yolov8_local.py")
        print("   □ Colab T4: upload và chạy colab_yolo26_training.py")
        print("   □ Download summary_yolo26.json từ Colab")
        print(f"   □ Copy vào: {COLAB_SUMMARY.parent}")
        sys.exit(0)

    # ── In bảng so sánh ────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  KẾT QUẢ SO SÁNH ({len(summaries)} runs):")
    print(f"{'='*60}")
    print(make_comparison_table(summaries))

    # ── Vẽ biểu đồ ────────────────────────────────────────────
    chart_path = OUT_DIR / "yolo_comparison_report.png"
    make_bar_chart(summaries, chart_path)

    # ── Winner ────────────────────────────────────────────────
    print_winner(summaries)

    # ── Lưu JSON tổng hợp ─────────────────────────────────────
    comparison_json = OUT_DIR / "yolo_comparison.json"
    comparison_json.write_text(json.dumps(summaries, indent=2))
    print(f"\n[OK] Full comparison saved: {comparison_json}")
    print(f"   Chart: {chart_path}")


if __name__ == "__main__":
    main()
