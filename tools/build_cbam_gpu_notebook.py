"""Create a VS Code friendly notebook for fast RTX 4050 CBAM training."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "train_cbam_fast_gpu.ipynb"


def md(text: str) -> dict:
    clean = textwrap.dedent(text).strip()
    return {"cell_type": "markdown", "metadata": {}, "source": [line + "\n" for line in clean.splitlines()]}


def code(text: str) -> dict:
    clean = textwrap.dedent(text).strip()
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in clean.splitlines()],
    }


cells = [
    md(
        """
        # Fast CBAM Training on RTX 4050

        Notebook này chạy CBAM bằng **PyTorch CUDA** để dùng RTX 4050. Lý do không dùng TensorFlow trực tiếp: TensorFlow 2.11+ trên native Windows không còn CUDA GPU support, còn PyTorch hiện đang nhận CUDA (`torch.cuda=True`).

        Sau khi train bằng GPU, script sẽ copy weight sang Keras model tương đương và export `.tflite` cho Android.
        """
    ),
    code(
        """
        from pathlib import Path
        import json, subprocess, sys

        ROOT = Path.cwd()
        print(ROOT)

        # Bật các cờ này khi muốn chạy thật trong VS Code.
        # Mặc định False để bạn chạy từng cell kiểm tra trước, không vô tình train lại model lớn.
        RUN_TRAIN_YAWN = False
        RUN_TRAIN_EYE = False
        RUN_FINALIZE_EYE = False

        YAWN_EPOCHS = 3
        EYE_EPOCHS = 5
        BATCH_SIZE = 256
        # Eye quick-run controls:
        # 0 means use all images. Start capped, then remove caps for final metrics.
        EYE_TRAIN_FRACTION = 1.0
        EYE_VAL_FRACTION = 1.0
        EYE_TEST_FRACTION = 1.0
        EYE_MAX_TRAIN_PER_CLASS = 8000
        EYE_MAX_EVAL_PER_CLASS = 0
        WORKERS = 0  # Windows notebook ổn định hơn với 0; tăng 2-4 nếu muốn thử nhanh hơn.
        """
    ),
    md("## 1. Kiểm tra RTX 4050 và process GPU"),
    code(
        """
        import torch, subprocess
        print("Torch:", torch.__version__)
        print("CUDA available:", torch.cuda.is_available())
        if torch.cuda.is_available():
            print("GPU:", torch.cuda.get_device_name(0))
            print("VRAM GB:", round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2))
        subprocess.run(["nvidia-smi"], check=False)
        """
    ),
    md("## 2. Đọc phương án của best_params.json"),
    code(
        """
        params = json.loads((ROOT / "data_pipeline" / "best_params.json").read_text(encoding="utf-8"))
        print(json.dumps(params, indent=2, ensure_ascii=False))
        """
    ),
    md(
        """
        ## 3. Vì sao precision và recall trước đó giống nhau?

        Trong Keras, `tf.keras.metrics.Precision()` và `Recall()` khi dùng với one-hot + softmax thường tính theo dạng global/micro trên toàn bộ vector output. Với bài toán 2 class, mỗi mẫu có đúng một nhãn đúng và model chọn một class, micro precision và micro recall rất dễ bằng accuracy, nên nhìn giống nhau kỳ lạ.

        Script đã được sửa để lưu `confusion_matrix` và `per_class` metrics: precision/recall riêng cho `eyes_closed`, `eyes_open`, `no_yawn`, `yawn`.
        """
    ),
    md("## 4. Train Yawn CBAM nhanh bằng RTX 4050"),
    code(
        """
        # Đã chạy thành công trước đó: 3 epochs, test_acc khoảng 98.64%.
        cmd = [
            sys.executable, "tools/train_cbam_fast_gpu.py",
            "--tasks", "yawn",
            "--epochs", str(YAWN_EPOCHS),
            "--batch-size", str(BATCH_SIZE),
            "--workers", str(WORKERS),
            "--target-acc", "0.985",
        ]
        print("COMMAND:", " ".join(cmd))
        if RUN_TRAIN_YAWN:
            subprocess.run(cmd, cwd=ROOT, check=True)
        else:
            print("RUN_TRAIN_YAWN=False -> không retrain. Đổi thành True ở cell setup nếu muốn chạy.")
            summary = ROOT / "outputs/cnn_yawn_cbam/summary.json"
            if summary.exists():
                print(summary.read_text(encoding="utf-8"))
        """
    ),
    md("## 5. Train Eye CBAM nhanh bằng RTX 4050"),
    code(
        """
        # Eye dataset lớn hơn nhiều, nên bắt đầu 2-3 epoch trước.
        # training_history.json và summary_running.json được ghi sau mỗi epoch.
        cmd = [
            sys.executable, "tools/train_cbam_fast_gpu.py",
            "--tasks", "eye",
            "--epochs", str(EYE_EPOCHS),
            "--batch-size", str(BATCH_SIZE),
            "--workers", str(WORKERS),
            "--target-acc", "0.985",
            "--train-fraction", str(EYE_TRAIN_FRACTION),
            "--val-fraction", str(EYE_VAL_FRACTION),
            "--test-fraction", str(EYE_TEST_FRACTION),
            "--max-train-per-class", str(EYE_MAX_TRAIN_PER_CLASS),
            "--max-eval-per-class", str(EYE_MAX_EVAL_PER_CLASS),
            "--skip-baseline-init",
            "--progress-every", "10",
        ]
        print("COMMAND:", " ".join(cmd))
        if RUN_TRAIN_EYE:
            subprocess.run(cmd, cwd=ROOT, check=True)
        else:
            print("RUN_TRAIN_EYE=False -> không train Eye. Đổi thành True ở cell setup nếu muốn chạy.")
            for name in ["summary_running.json", "summary_best_checkpoint.json", "training_history.json"]:
                p = ROOT / "outputs/cnn_eye_cbam" / name
                if p.exists():
                    print("\\n--", name, "--")
                    print(p.read_text(encoding="utf-8")[:2000])
        """
    ),
    md("## 6. Nếu bị ngắt giữa chừng: xem epoch đã lưu"),
    code(
        """
        for folder in ["outputs/cnn_eye_cbam", "outputs/cnn_yawn_cbam"]:
            print("\\n==", folder, "==")
            for name in ["training_history.json", "summary_running.json", "summary_best_checkpoint.json", "summary.json"]:
                p = ROOT / folder / name
                if p.exists():
                    print("\\n--", name, "--")
                    print(p.read_text(encoding="utf-8")[:4000])
        """
    ),
    md("## 7. Finalize checkpoint bị ngắt thành TFLite"),
    code(
        """
        # Dùng khi có best_cbam_torch_state.pt nhưng chưa có summary/tflite.
        # Lưu ý: finalize Eye có thể lâu vì evaluate val+test lớn và convert TFLite bằng TensorFlow CPU.
        cmd = [
            sys.executable, "tools/finalize_cbam_checkpoint.py",
            "--task", "eye",
            "--batch-size", str(BATCH_SIZE),
            "--workers", str(WORKERS),
            "--note", "finalized from notebook after interrupted run",
        ]
        print("COMMAND:", " ".join(cmd))
        if RUN_FINALIZE_EYE:
            subprocess.run(cmd, cwd=ROOT, check=True)
        else:
            print("RUN_FINALIZE_EYE=False -> không finalize vì bước này có thể lâu.")
        """
    ),
    md("## 8. Verify TFLite assets"),
    code(
        """
        import tensorflow as tf
        for rel in [
            "app/src/main/assets/drowsiness_model.tflite",
            "app/src/main/assets/yawn_model.tflite",
            "app/src/main/assets/yawn_cbam.tflite",
            "app/src/main/assets/drowsiness_cbam.tflite",
            "app/src/main/assets/temporal_model.tflite",
        ]:
            p = ROOT / rel
            print("\\n", rel, "exists=", p.exists(), "KB=", round(p.stat().st_size/1024,1) if p.exists() else None)
            if p.exists():
                interp = tf.lite.Interpreter(model_path=str(p))
                interp.allocate_tensors()
                inp = interp.get_input_details()[0]
                out = interp.get_output_details()[0]
                print(" input ", inp["shape"].tolist(), inp["dtype"].__name__)
                print(" output", out["shape"].tolist(), out["dtype"].__name__)
        """
    ),
]

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUT.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print(OUT)
