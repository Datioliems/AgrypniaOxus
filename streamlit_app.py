from __future__ import annotations

from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "app" / "src" / "main" / "assets" / "drowsiness_model.tflite"


def main() -> None:
    st.set_page_config(page_title="Drowsy Driver Demo", layout="wide")
    st.title("Drowsy Driver Detection - Streamlit Demo")
    st.caption("Web fallback for explaining the AI pipeline. Android native remains the main deployment target.")

    with st.sidebar:
        st.header("Input")
        mode = st.radio("Choose input", ["Camera snapshot", "Upload image"])
        st.header("Model status")
        if MODEL_PATH.exists():
            st.success(f"Found TFLite model: {MODEL_PATH.name}")
        else:
            st.warning("TFLite model not found yet. Train/export it before final demo.")

    image = read_input(mode)
    if image is None:
        st.info("Capture or upload an image to start.")
        show_pipeline_notes()
        return

    col_image, col_result = st.columns([1.2, 1])
    with col_image:
        st.subheader("Input frame")
        st.image(image, use_container_width=True)

    with col_result:
        st.subheader("Result")
        result = analyze_snapshot(image)
        st.metric("Estimated state", result["state"])
        st.write(result["message"])
        st.progress(result["confidence"])

        st.subheader("Deployment meaning")
        st.write(
            "Streamlit is useful for web demonstration and result explanation. "
            "For realtime driver warning, the Android app should run on-device with CameraX, "
            "MediaPipe and TensorFlow Lite."
        )

    show_pipeline_notes()


def read_input(mode: str) -> Image.Image | None:
    if mode == "Camera snapshot":
        captured = st.camera_input("Take a driver-face snapshot")
        if captured is None:
            return None
        return Image.open(captured).convert("RGB")

    uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "webp"])
    if uploaded is None:
        return None
    return Image.open(uploaded).convert("RGB")


def analyze_snapshot(image: Image.Image) -> dict[str, object]:
    arr = np.asarray(image)
    brightness = float(arr.mean()) / 255.0

    if not MODEL_PATH.exists():
        if brightness < 0.25:
            return {
                "state": "Need better lighting",
                "confidence": 0.45,
                "message": "Ảnh hơi tối. Với bản cuối, dùng MediaPipe landmarks và CNN/TFLite để suy luận trạng thái mắt.",
            }
        return {
            "state": "Demo ready",
            "confidence": 0.65,
            "message": "Bản Streamlit đã nhận ảnh. Hãy train model để bật inference thật.",
        }

    return {
        "state": "Model available",
        "confidence": 0.75,
        "message": "Model TFLite đã có trong assets. Bước tiếp theo là nối interpreter cho web demo nếu cần.",
    }


def show_pipeline_notes() -> None:
    st.subheader("Recommended final pipeline")
    st.code(
        "CameraX -> MediaPipe Face Landmarker -> EAR/MAR -> eye ROI -> CNN/TFLite -> smoothing -> alert",
        language="text",
    )
    st.write(
        "For IS54A, use this app as a fallback/web demonstration. "
        "Report the Android native app as the main product because it is closer to real driver deployment."
    )


if __name__ == "__main__":
    main()
