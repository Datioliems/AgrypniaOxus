# -*- coding: utf-8 -*-
"""
MASTER RUNNER — Chạy toàn bộ pipeline bước 1–10
Tự động bỏ qua bước đã có output (trừ khi --force).

Cách dùng:
    # Chạy toàn bộ từ đầu
    python run_pipeline.py --dataset cnn_eye

    # Chỉ chạy từ bước 5 trở đi (đã có bước 1–4)
    python run_pipeline.py --dataset cnn_eye --from-step 5

    # Chạy bước cụ thể
    python run_pipeline.py --dataset cnn_eye --only 9

    # Bắt buộc chạy lại dù output đã có
    python run_pipeline.py --dataset cnn_eye --force

    # Chạy cho cả hai dataset
    python run_pipeline.py --dataset all
"""
import argparse
import sys
import time
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))
from config import REPORTS_DIR, DATASETS, STEP_KEYS
from utils import setup_logging, load_json, save_json, print_banner, print_ok, print_warn, print_fail, create_dirs

# ─── Import tất cả step runners ───────────────
from step1_ingest        import run_inventory
from step2_quality_filter import run_quality_filter
from step3_extract_roi   import run_roi_extraction
from step4_resize        import run_resize
from step5_eda           import run_eda
from step6_augment       import run_augmentation
from step7_split         import run_split
from step8_normalize     import run_normalize
from step9_integrity_check import run_integrity_check
from step10_manifest     import run_manifest

STEP_RUNNERS = {
    1:  run_inventory,
    2:  run_quality_filter,
    3:  run_roi_extraction,
    4:  run_resize,
    5:  run_eda,
    6:  run_augmentation,
    7:  run_split,
    8:  run_normalize,
    9:  run_integrity_check,
    10: run_manifest,
}

STEP_NAMES = {
    1:  "Inventory & Kiểm kê",
    2:  "Quality Filter",
    3:  "ROI Extraction",
    4:  "Resize & Standardize",
    5:  "EDA & Thống kê",
    6:  "Augmentation",
    7:  "Dataset Split",
    8:  "Normalize Verification",
    9:  "Integrity Check",
    10: "Dataset Manifest",
}

STEP_ENV = {
    1:  "LOCAL",
    2:  "LOCAL",
    3:  "LOCAL",
    4:  "LOCAL",
    5:  "LOCAL",
    6:  "LOCAL",
    7:  "LOCAL",
    8:  "LOCAL",
    9:  "LOCAL",
    10: "LOCAL",
}


def step_has_output(step: int, dataset: str) -> bool:
    """Kiểm tra bước này đã có output chưa."""
    key  = STEP_KEYS.get(step, f"step{step}")
    path = REPORTS_DIR / f"step{step}_{dataset}_{key}.json"
    if not path.exists():
        return False
    try:
        report = load_json(path)
        return report.get("status") in ("PASS", "WARNING")
    except Exception:
        return False


def run_pipeline(dataset_name: str,
                 from_step: int = 1,
                 to_step: int = 10,
                 only_step: int = None,
                 force: bool = False,
                 no_plots: bool = False):

    print_banner(f"DROWSYDRIVER PIPELINE — {dataset_name.upper()}")
    print(f"  Bắt đầu: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Bước: {from_step}–{to_step}{f' (chỉ bước {only_step})' if only_step else ''}")
    print(f"  Force: {force}")

    create_dirs(REPORTS_DIR)
    logger = setup_logging(f"pipeline_{dataset_name}", REPORTS_DIR.parent / "logs")

    steps  = [only_step] if only_step else list(range(from_step, to_step + 1))
    start_total = time.time()
    results = {}

    print()
    print("  Kế hoạch thực thi:")
    for s in range(1, 11):
        if s in steps:
            has_out = step_has_output(s, dataset_name)
            skip = has_out and not force
            icon = "→" if not skip else "✓ (skip)"
            print(f"    Bước {s:2d}  [{STEP_ENV[s]}]  {STEP_NAMES[s]:<30} {icon}")
        else:
            print(f"    Bước {s:2d}  [{STEP_ENV[s]}]  {STEP_NAMES[s]:<30} --")

    print()

    for step_num in steps:
        runner = STEP_RUNNERS.get(step_num)
        if runner is None:
            print_warn(f"Bước {step_num} không tìm thấy runner — bỏ qua")
            continue

        # Kiểm tra xem có bỏ qua không
        if not force and step_has_output(step_num, dataset_name):
            print(f"  [Bước {step_num}] {STEP_NAMES[step_num]} — ✓ Đã có output, bỏ qua "
                  f"(dùng --force để chạy lại)")
            results[step_num] = "SKIPPED"
            continue

        # Chạy bước
        print(f"\n{'─' * 60}")
        t0 = time.time()
        try:
            # Bước 5 (EDA) có tham số no_plots
            if step_num == 5:
                runner(dataset_name, make_plots=not no_plots)
            else:
                runner(dataset_name)
            elapsed = time.time() - t0
            results[step_num] = "PASS"
            print(f"\n  [Bước {step_num}] hoàn thành trong {elapsed:.1f}s")
            logger.info(f"Step {step_num} PASS ({elapsed:.1f}s)")
        except SystemExit as e:
            elapsed = time.time() - t0
            results[step_num] = "FAIL"
            print_fail(f"\n[Bước {step_num}] FAIL sau {elapsed:.1f}s — pipeline dừng lại")
            logger.error(f"Step {step_num} FAIL")
            break
        except Exception as e:
            results[step_num] = "ERROR"
            print_fail(f"\n[Bước {step_num}] ERROR: {e}")
            logger.exception(f"Step {step_num} error")
            break

    # ─── Tổng kết ─────────────────────────────
    total_time = time.time() - start_total
    print()
    print("=" * 60)
    print(f"  KẾT QUẢ PIPELINE — {dataset_name}")
    print("=" * 60)
    for s, status in results.items():
        icon = "✓" if status in ("PASS", "SKIPPED") else "✗"
        print(f"  {icon} Bước {s:2d} — {STEP_NAMES[s]:<30} {status}")

    passed = sum(1 for s in results.values() if s in ("PASS", "SKIPPED"))
    failed = sum(1 for s in results.values() if s in ("FAIL", "ERROR"))

    print()
    print(f"  Tổng thời gian: {total_time:.1f}s ({total_time/60:.1f} phút)")
    print(f"  Bước hoàn thành: {passed}/{len(results)}")

    if failed == 0:
        print_ok("Pipeline hoàn thành thành công!")
        print()
        print("  Bước tiếp theo:")
        print("  → Mở Google Colab và upload data/splits/ lên Drive")
        print("  → Chạy notebook CNN Eye training (LOCAL hoặc COLAB)")
        print("  → Chạy YOLO notebooks trên COLAB (cần T4 GPU)")
    else:
        print_fail(f"Pipeline có {failed} bước thất bại — xem log tại logs/")

    save_json({
        "dataset": dataset_name,
        "steps": {str(k): v for k, v in results.items()},
        "total_seconds": round(total_time, 1),
        "completed_at": datetime.now().isoformat(),
    }, REPORTS_DIR / f"pipeline_run_{dataset_name}.json")

    return failed == 0


def main():
    parser = argparse.ArgumentParser(
        description="DrowsyDriver Pipeline Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ví dụ:
  python run_pipeline.py --dataset cnn_eye            # Chạy toàn bộ
  python run_pipeline.py --dataset cnn_yawn --from-step 3
  python run_pipeline.py --dataset all --only 9       # Chỉ integrity check
  python run_pipeline.py --dataset cnn_eye --force    # Chạy lại từ đầu
        """
    )
    parser.add_argument("--dataset", default="cnn_eye",
                        choices=list(DATASETS.keys()) + ["all"],
                        help="Dataset cần xử lý")
    parser.add_argument("--from-step", type=int, default=1, dest="from_step",
                        help="Bắt đầu từ bước (1–10)")
    parser.add_argument("--to-step", type=int, default=10, dest="to_step",
                        help="Kết thúc ở bước (1–10)")
    parser.add_argument("--only", type=int, default=None,
                        help="Chỉ chạy bước này")
    parser.add_argument("--force", action="store_true",
                        help="Chạy lại dù output đã có")
    parser.add_argument("--no-plots", action="store_true",
                        help="Bỏ qua vẽ biểu đồ EDA (headless)")
    args = parser.parse_args()

    datasets = list(DATASETS.keys()) if args.dataset == "all" else [args.dataset]

    all_ok = True
    for ds in datasets:
        ok = run_pipeline(
            ds,
            from_step  = args.from_step,
            to_step    = args.to_step,
            only_step  = args.only,
            force      = args.force,
            no_plots   = args.no_plots,
        )
        if not ok:
            all_ok = False

    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
