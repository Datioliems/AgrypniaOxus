"""
Chay syntax check + import check tren tat ca .py files trong du an.
Ghi ket qua vao error_log.txt o root project.

Usage:
    python tools\run_all_checks.py
"""
import subprocess, sys, datetime, os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOG  = ROOT / "error_log.txt"

# ── Phan loai file ────────────────────────────────────────────────────────────
# Nhung file can nhieu gio / GPU / dataset dac biet → chi check syntax
SKIP_RUN = {
    "train_eye_classifier.py",    # can GPU + dataset
    "train_yolov8_local.py",      # can GPU + dataset
    "train_clean_local.py",       # can GPU + dataset
    "tune_experiments.py",        # can GPU + dataset
    "notebook_full_pipeline.py",  # can GPU + dataset
    "notebook_multi_model.py",    # can GPU + dataset
    "colab_yolo26_training.py",   # Colab-only
    "scan_dashcam_subnet.py",     # can network
    "probe_dashcam_stream.py",    # can network
    "test_dashcam_rtsp.py",       # can network
    "describe_dashcam_rtsp.py",   # can network
    "run_iphone_pwa_server.py",   # can server
    "streamlit_app.py",           # can server
    "build_report_docx.py",       # can dataset + docx
    "build_is54a_report_docx.py", # can dataset + docx
    "add_alertness_research_section.py",  # can docx file
    "add_continuous_arousal_section.py",  # can docx file
    "check_yolo_dataset.py",      # can --dataset arg + dataset folder
    "summarize_dataset.py",       # can --data arg
    "compare_yolo_results.py",    # can training results files
    "run_all_checks.py",          # chinh no
}

# File se thu RUN that su (nhanh, khong can GPU/dataset lon)
RUN_SCRIPT = {
    "verify_project.py",
    "summarize_dataset.py",
    "convert_to_ipynb.py",
    "compare_yolo_results.py",
    "check_yolo_dataset.py",
    "plot_model_results.py",
}

def py_compile_check(path: Path) -> tuple[bool, str]:
    """Kiem tra syntax bang py_compile."""
    r = subprocess.run(
        [sys.executable, "-m", "py_compile", str(path)],
        capture_output=True, text=True, cwd=str(ROOT)
    )
    ok  = r.returncode == 0
    msg = (r.stderr or r.stdout).strip()
    return ok, msg

def run_script(path: Path, timeout=30) -> tuple[bool, str]:
    """Thu chay script, timeout 30s."""
    r = subprocess.run(
        [sys.executable, str(path)],
        capture_output=True, text=True,
        cwd=str(ROOT), timeout=timeout
    )
    ok  = r.returncode == 0
    msg = (r.stderr + r.stdout).strip()
    return ok, msg[:800]   # gioi han do dai

# ── Tim tat ca .py (bo qua outputs/ va rawdata/) ─────────────────────────────
all_py = [
    p for p in ROOT.rglob("*.py")
    if "outputs" not in p.parts
    and "rawdata" not in p.parts
    and ".idea" not in p.parts
]
all_py.sort()

# ── Chay checks ───────────────────────────────────────────────────────────────
results = []
errors  = []
now     = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

print(f"=== DrowsyDriver Check | {now} ===")
print(f"Tim thay {len(all_py)} file .py\n")

for path in all_py:
    rel   = path.relative_to(ROOT)
    fname = path.name

    # 1. Syntax check
    ok_syn, msg_syn = py_compile_check(path)
    status = "OK" if ok_syn else "SYNTAX_ERR"

    # 2. Neu syntax OK, co trong RUN_SCRIPT, va KHONG trong SKIP_RUN → thu chay
    run_msg = ""
    if ok_syn and fname in RUN_SCRIPT and fname not in SKIP_RUN:
        try:
            ok_run, run_out = run_script(path, timeout=30)
            if not ok_run:
                status  = "RUNTIME_ERR"
                run_msg = run_out
        except subprocess.TimeoutExpired:
            status  = "TIMEOUT"
            run_msg = "(chay qua 30s — co the can dataset, bo qua)"
        except Exception as e:
            status  = "RUNTIME_ERR"
            run_msg = str(e)

    cat = ("SKIP_RUN" if fname in SKIP_RUN
           else "CHECK_ONLY" if fname not in RUN_SCRIPT
           else "RAN")

    icon = "OK" if "ERR" not in status and status != "TIMEOUT" else "!!"
    line = f"  [{icon}] {str(rel):<55} {status:<14} ({cat})"
    print(line)

    entry = {
        "file": str(rel),
        "status": status,
        "category": cat,
        "syntax_err": msg_syn if not ok_syn else "",
        "run_err": run_msg,
    }
    results.append(entry)

    if "ERR" in status or status == "TIMEOUT":
        errors.append(entry)

# ── Ghi error_log.txt ─────────────────────────────────────────────────────────
with open(LOG, "w", encoding="utf-8") as f:
    f.write(f"DrowsyDriverAndroid — Error Log\n")
    f.write(f"Generated : {now}\n")
    f.write(f"Python    : {sys.version}\n")
    f.write(f"Total files: {len(all_py)}\n")
    f.write(f"Errors     : {len(errors)}\n")
    f.write("=" * 70 + "\n\n")

    if not errors:
        f.write("Khong co loi nao! Tat ca files deu pass.\n")
    else:
        for e in errors:
            f.write(f"FILE   : {e['file']}\n")
            f.write(f"STATUS : {e['status']}  |  {e['category']}\n")
            if e["syntax_err"]:
                f.write(f"SYNTAX :\n{e['syntax_err']}\n")
            if e["run_err"]:
                f.write(f"RUNTIME:\n{e['run_err']}\n")
            f.write("-" * 60 + "\n")

    f.write("\n\n=== TAT CA FILES ===\n")
    for r in results:
        f.write(f"[{r['status']:<14}] {r['file']}\n")

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n{'='*55}")
print(f"Tong: {len(all_py)} files")
print(f"LOI : {len(errors)}")
print(f"Log : {LOG}")
if errors:
    print("\nCac file co loi:")
    for e in errors:
        print(f"  !! {e['file']}  [{e['status']}]")
else:
    print("Tat ca files deu OK!")
