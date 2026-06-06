"""
Chuyển file .py có # %% cell markers sang .ipynb

Hỗ trợ:
  # %%              → code cell
  # %% [markdown]   → markdown cell (tự strip dấu # ở đầu dòng)
  # %% [N] Tên      → code cell có tên (bỏ qua trong ipynb)

Usage:
  # Convert 1 file:
  python tools/convert_to_ipynb.py colab_yolo26_training.py

  # Convert nhiều file:
  python tools/convert_to_ipynb.py colab_yolo26_training.py tune_experiments.py

  # Convert tất cả file notebook trong project:
  python tools/convert_to_ipynb.py --all
"""

import argparse
import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

# Các file notebook-style cần convert
NOTEBOOK_FILES = [
    "train_cnn_eye.py",
    "train_cnn_yawn.py",
    "tools/train_clean_local.py",
    "tune_experiments.py",
    "notebook_full_pipeline.py",
    "notebook_multi_model.py",
    "colab_yolo26_training.py",
]


def make_metadata(is_colab: bool = False) -> dict:
    meta = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "version": "3.12.0",
        },
    }
    if is_colab:
        meta["colab"] = {"provenance": []}
        meta["accelerator"] = "GPU"
        meta["gpuClass"] = "standard"
    return meta


def parse_cells(py_text: str) -> list[dict]:
    """
    Phân tích file .py thành danh sách cell cho ipynb.

    Quy tắc:
    - Dòng bắt đầu bằng '# %%' → ranh giới cell mới
    - '# %% [markdown]'        → markdown cell
    - Còn lại                  → code cell
    - Markdown cell: strip '# ' ở đầu mỗi dòng nội dung
    """
    CELL_RE = re.compile(r"^# %%(.*)$")
    lines = py_text.splitlines(keepends=True)

    cells: list[dict] = []
    current_lines: list[str] = []
    current_is_md: bool = False
    started: bool = False  # bỏ qua nội dung trước cell đầu tiên

    def flush():
        if not started or not current_lines:
            return
        # Bỏ blank lines ở cuối cell
        src = current_lines.copy()
        while src and src[-1].strip() == "":
            src.pop()
        if not src:
            return

        if current_is_md:
            # Strip '# ' hoặc '#' ở đầu mỗi dòng
            md_src = []
            for line in src:
                s = line.rstrip("\n")
                if s.startswith("# "):
                    md_src.append(s[2:] + "\n")
                elif s == "#":
                    md_src.append("\n")
                else:
                    md_src.append(line)
            cells.append({
                "cell_type": "markdown",
                "metadata": {},
                "source": md_src,
            })
        else:
            cells.append({
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": src,
            })

    for line in lines:
        m = CELL_RE.match(line.rstrip("\n"))
        if m:
            flush()
            current_lines = []
            tag = m.group(1).lower()
            current_is_md = "[markdown]" in tag
            started = True
            # Bỏ dòng marker (không đưa vào nội dung cell)
        else:
            if started:
                current_lines.append(line)

    flush()
    return cells


def py_to_ipynb(py_path: Path) -> dict:
    text = py_path.read_text(encoding="utf-8")
    is_colab = "colab" in py_path.name.lower() or "google.colab" in text

    cells = parse_cells(text)

    # Thêm cell đầu nếu là colab: badge + hướng dẫn
    if is_colab:
        badge_src = [
            f"# {py_path.stem}\n",
            "# Chạy trên Google Colab với T4 GPU\n",
        ]
        cells.insert(0, {
            "cell_type": "markdown",
            "metadata": {},
            "source": badge_src,
        })

    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": make_metadata(is_colab=is_colab),
        "cells": cells,
    }


def convert(py_path: Path, out_dir: Path | None = None) -> Path:
    if not py_path.exists():
        raise FileNotFoundError(f"Không tìm thấy: {py_path}")

    nb = py_to_ipynb(py_path)

    dest_dir = out_dir or py_path.parent
    out_path = dest_dir / (py_path.stem + ".ipynb")

    out_path.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")

    n_code = sum(1 for c in nb["cells"] if c["cell_type"] == "code")
    n_md   = sum(1 for c in nb["cells"] if c["cell_type"] == "markdown")
    size_kb = out_path.stat().st_size / 1024

    print(f"  OK  {out_path.name:<45} "
          f"{n_code} code cells, {n_md} md cells  "
          f"({size_kb:.1f} KB)")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Convert .py → .ipynb")
    parser.add_argument("files", nargs="*", help="File .py cần convert")
    parser.add_argument("--all", action="store_true",
                        help="Convert tất cả NOTEBOOK_FILES")
    parser.add_argument("--out-dir", default=None,
                        help="Thư mục output (mặc định: cùng thư mục với file .py)")
    args = parser.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else None

    targets: list[Path] = []
    if args.all:
        targets = [PROJECT_ROOT / f for f in NOTEBOOK_FILES]
    elif args.files:
        targets = [Path(f) if Path(f).is_absolute() else PROJECT_ROOT / f
                   for f in args.files]
    else:
        # Không có args → convert tất cả
        targets = [PROJECT_ROOT / f for f in NOTEBOOK_FILES]

    print(f"\n[ipynb] Converting {len(targets)} file(s)...\n")

    ok, fail = 0, 0
    for p in targets:
        try:
            convert(p, out_dir)
            ok += 1
        except FileNotFoundError as e:
            print(f"  ⚠️  {e}")
            fail += 1
        except Exception as e:
            print(f"  ❌ {p.name}: {e}")
            fail += 1

    print(f"\n{'='*55}")
    print(f"  Done: {ok} converted, {fail} skipped")
    if ok:
        print(f"\n  --> Mo .ipynb trong VS Code hoac upload len Colab")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()
