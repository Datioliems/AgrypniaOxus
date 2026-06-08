# DrowsyDriver — Streamlit demo. Ghim Python 3.12 (mediapipe có wheel) → tránh lỗi "cook".
# Deploy lên: Hugging Face Spaces (Docker), Railway, Render, Fly.io, hoặc máy chủ bất kỳ.
FROM python:3.12-slim

# System libs cho opencv/mediapipe
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY streamlit_app.py .
COPY face_landmarker.task .
# (tùy chọn) bật CNN: bỏ comment 2 dòng dưới + bỏ tflite khỏi .gitignore
# COPY app/src/main/assets/drowsiness_model.tflite app/src/main/assets/
# COPY app/src/main/assets/yawn_model.tflite app/src/main/assets/

# PORT: 8501 mặc định; Hugging Face Spaces dùng 7860 → set ENV PORT=7860 khi deploy ở đó.
ENV PORT=8501
EXPOSE 8501
HEALTHCHECK CMD curl --fail http://localhost:${PORT}/_stcore/health || exit 1

# shell-form để $PORT được mở rộng
CMD streamlit run streamlit_app.py --server.port=$PORT --server.address=0.0.0.0 --server.headless=true
