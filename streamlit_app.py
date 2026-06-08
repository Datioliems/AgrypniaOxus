# -*- coding: utf-8 -*-
"""
🚗 DrowsyDriver — Streamlit Demo (3 màn hình)
  1. Dashboard  : camera AI quét mặt + nút giám sát + thanh trạng thái tỉnh táo
  2. Alert      : màn đỏ nhấp nháy + âm thanh + nút "Tôi ổn" / "Tìm trạm dừng"
  3. Analytics  : lịch sử buồn ngủ + chỉnh độ nhạy + chọn âm thanh

Chạy:   streamlit run streamlit_app.py
Deploy: xem STREAMLIT_DEPLOY.md
"""
import base64
import math
import struct
import time

import numpy as np
import streamlit as st

# ───────────────────────── Cấu hình trang ─────────────────────────
st.set_page_config(page_title="DrowsyDriver AI", page_icon="🚗",
                   layout="wide", initial_sidebar_state="expanded")

# Import "mềm" — thiếu gói thì báo rõ thay vì crash
try:
    import cv2
    import mediapipe as mp
    import av
    from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
    DEPS_OK, DEPS_ERR = True, ""
except Exception as e:                                       # pragma: no cover
    DEPS_OK, DEPS_ERR = False, str(e)

# TFLite (tùy chọn — hiển thị thêm dự đoán CNN nếu có model)
TFLITE = None
try:
    import tensorflow as tf
    from pathlib import Path
    _m = Path("app/src/main/assets/drowsiness_model.tflite")
    if _m.exists():
        TFLITE = tf.lite.Interpreter(model_path=str(_m)); TFLITE.allocate_tensors()
except Exception:
    TFLITE = None


# MediaPipe 0.10.35+ ĐÃ BỎ mp.solutions → dùng Tasks API (FaceLandmarker)
_LANDMARK_MODEL = "face_landmarker.task"
_MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
              "face_landmarker/float16/1/face_landmarker.task")


def _make_landmarker():
    """Tạo FaceLandmarker (tự tải model ~3.7MB lần đầu)."""
    import os
    import urllib.request
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision
    if not os.path.exists(_LANDMARK_MODEL):
        urllib.request.urlretrieve(_MODEL_URL, _LANDMARK_MODEL)
    opts = vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=_LANDMARK_MODEL),
        running_mode=vision.RunningMode.IMAGE, num_faces=1)
    return vision.FaceLandmarker.create_from_options(opts)


# ───────────────────────── Landmark mắt/miệng (MediaPipe FaceMesh) ─────────────────────────
LEFT_EYE  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
MOUTH     = [13, 14, 78, 308]   # trên, dưới, trái, phải (môi trong)


def _ear(p):
    """Eye Aspect Ratio — nhỏ = mắt nhắm."""
    v = math.dist(p[1], p[5]) + math.dist(p[2], p[4])
    h = 2.0 * math.dist(p[0], p[3]) + 1e-6
    return v / h


def _mar(top, bot, left, right):
    """Mouth Aspect Ratio — lớn = đang ngáp."""
    return math.dist(top, bot) / (math.dist(left, right) + 1e-6)


# ───────────────────────── Âm thanh cảnh báo (sinh sẵn, base64) ─────────────────────────
def make_tone(freq=880, ms=600, kind="beep"):
    rate, frames = 16000, bytearray()
    n = int(rate * ms / 1000)
    for i in range(n):
        t = i / rate
        if kind == "siren":
            f = freq + 300 * math.sin(2 * math.pi * 3 * t)
        elif kind == "voice":
            f = freq if (i // (rate // 4)) % 2 == 0 else 0
        else:
            f = freq
        val = int(32767 * 0.6 * math.sin(2 * math.pi * f * t)) if f else 0
        frames += struct.pack("<h", val)
    hdr = (b"RIFF" + struct.pack("<I", 36 + len(frames)) + b"WAVE" + b"fmt "
           + struct.pack("<IHHIIHH", 16, 1, 1, rate, rate * 2, 2, 16)
           + b"data" + struct.pack("<I", len(frames)))
    return base64.b64encode(hdr + bytes(frames)).decode()


SOUNDS = {"Beep": make_tone(880, 500, "beep"),
          "Siren": make_tone(700, 900, "siren"),
          "Nhắc giọng": make_tone(600, 800, "voice")}


def play_sound(b64):
    st.markdown(f'<audio autoplay loop src="data:audio/wav;base64,{b64}"></audio>',
                unsafe_allow_html=True)


# ───────────────────────── State dùng chung ─────────────────────────
def init_state():
    ss = st.session_state
    ss.setdefault("page", "Dashboard")
    ss.setdefault("monitoring", False)
    ss.setdefault("show_cam", True)
    ss.setdefault("ear_thr", 0.21)
    ss.setdefault("mar_thr", 0.6)
    ss.setdefault("closed_sec", 1.5)
    ss.setdefault("sound", "Siren")
    ss.setdefault("history", [])


# ───────────────────────── Bộ xử lý video (chạy theo từng frame) ─────────────────────────
if DEPS_OK:
    class Processor(VideoProcessorBase):
        def __init__(self):
            self.landmarker = _make_landmarker()
            self.ear = self.mar = 0.0
            self.status = "ALERT"
            self.closed_since = None
            self.yawn_count = 0
            self._prev_yawn = False
            self.event = None
            self.ear_thr, self.mar_thr, self.closed_sec, self.show = 0.21, 0.6, 1.5, True

        def recv(self, frame):
            img = frame.to_ndarray(format="bgr24")
            h, w = img.shape[:2]
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            res = self.landmarker.detect(mp_img)
            self.event = None

            if res.face_landmarks:
                lm = res.face_landmarks[0]
                P = lambda i: (lm[i].x * w, lm[i].y * h)
                self.ear = (_ear([P(i) for i in LEFT_EYE]) + _ear([P(i) for i in RIGHT_EYE])) / 2
                self.mar = _mar(P(MOUTH[0]), P(MOUTH[1]), P(MOUTH[2]), P(MOUTH[3]))
                now = time.time()
                if self.ear < self.ear_thr:
                    self.closed_since = self.closed_since or now
                    if now - self.closed_since >= self.closed_sec:
                        self.status = "DROWSY"; self.event = ("Nhắm mắt", round(self.ear, 3))
                else:
                    self.closed_since = None
                    self.status = "WARNING" if self.mar > self.mar_thr else "ALERT"
                is_yawn = self.mar > self.mar_thr
                if is_yawn and not self._prev_yawn:
                    self.yawn_count += 1; self.event = ("Ngáp", round(self.mar, 2))
                self._prev_yawn = is_yawn

                if self.show:
                    col = {"ALERT": (0, 200, 0), "WARNING": (0, 180, 230),
                           "DROWSY": (0, 0, 255)}[self.status]
                    for i in LEFT_EYE + RIGHT_EYE:
                        cv2.circle(img, tuple(map(int, P(i))), 1, col, -1)
                    cv2.putText(img, f"EAR {self.ear:.2f}  MAR {self.mar:.2f}  {self.status}",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)
            else:
                self.status = "NO FACE"

            out = img if self.show else np.zeros_like(img)
            return av.VideoFrame.from_ndarray(out, format="bgr24")


# ───────────────────────── CSS ─────────────────────────
FLASH_CSS = """
<style>
@keyframes flash { 0%,100%{background:#7f0000} 50%{background:#ff1e1e} }
.alert-box{animation:flash .6s infinite;border-radius:20px;padding:40px;text-align:center;color:#fff}
.statusbar{padding:18px;border-radius:14px;color:#fff;font-size:24px;font-weight:800;text-align:center}
.status-alert{background:#1b8a3a}.status-warn{background:#caa300}.status-drowsy{background:#c0392b}
</style>
"""


# ════════════════════════ CÁC MÀN HÌNH ════════════════════════
def page_dashboard():
    st.title("🚗 Dashboard — Giám sát tài xế")
    ss = st.session_state
    c1, c2 = st.columns([3, 2])

    with c1:
        ss.show_cam = st.toggle("📷 Hiện camera", value=ss.show_cam)
        ctx = webrtc_streamer(
            key="cam", mode=WebRtcMode.SENDRECV, video_processor_factory=Processor,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True)
        if ctx.video_processor:
            vp = ctx.video_processor
            vp.ear_thr, vp.mar_thr = ss.ear_thr, ss.mar_thr
            vp.closed_sec, vp.show = ss.closed_sec, ss.show_cam

    with c2:
        ss.monitoring = st.toggle("🟢 KÍCH HOẠT GIÁM SÁT", value=ss.monitoring)
        vp = ctx.video_processor if ctx else None
        st.metric("Số lần ngáp (phiên này)", vp.yawn_count if vp else 0)

        status = vp.status if vp else "—"
        cls = {"ALERT": "status-alert", "WARNING": "status-warn",
               "DROWSY": "status-drowsy"}.get(status, "status-warn")
        label = {"ALERT": "✅ TỈNH TÁO", "WARNING": "🟡 CÓ DẤU HIỆU MỆT",
                 "DROWSY": "🔴 BUỒN NGỦ!", "NO FACE": "🔍 Không thấy mặt"}.get(status, status)
        st.markdown(f'<div class="statusbar {cls}">{label}</div>', unsafe_allow_html=True)
        if vp:
            st.progress(min(vp.ear / 0.4, 1.0), text=f"EAR {vp.ear:.2f}")

        if ss.monitoring and vp:
            if vp.event:
                ss.history.append((time.strftime("%H:%M:%S"), vp.event[0], vp.event[1]))
            if vp.status == "DROWSY":
                ss.page = "Alert"; st.rerun()

    st.info("Bật **Kích hoạt giám sát** rồi nhắm mắt > 1.5s hoặc ngáp để thử cảnh báo.")
    if not TFLITE:
        st.caption("ℹ️ Đang dùng EAR/MAR (MediaPipe). Có `drowsiness_model.tflite` sẽ tự nạp thêm CNN.")


def page_alert():
    ss = st.session_state
    st.markdown(FLASH_CSS, unsafe_allow_html=True)
    st.markdown('<div class="alert-box"><h1>⚠️ CẢNH BÁO BUỒN NGỦ ⚠️</h1>'
                '<h2>Hãy tỉnh táo hoặc dừng nghỉ ngay!</h2></div>', unsafe_allow_html=True)
    play_sound(SOUNDS[ss.sound])
    st.write("")
    b1, b2 = st.columns(2)
    with b1:
        if st.button("✅ TÔI ỔN", use_container_width=True, type="primary"):
            ss.page = "Dashboard"; st.rerun()
    with b2:
        st.link_button("🅿️ TÌM TRẠM DỪNG NGHỈ",
                       "https://www.google.com/maps/search/rest+area+near+me",
                       use_container_width=True)


def page_analytics():
    ss = st.session_state
    st.title("📊 Thống kê & Cài đặt")

    st.subheader("Lịch sử buồn ngủ trong phiên")
    if ss.history:
        import pandas as pd
        df = pd.DataFrame(ss.history, columns=["Thời gian", "Loại", "Giá trị"])
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.bar_chart(df["Loại"].value_counts())
        if st.button("🗑️ Xóa lịch sử"):
            ss.history = []; st.rerun()
    else:
        st.caption("Chưa có sự kiện nào. Bật giám sát ở Dashboard để ghi nhận.")

    st.divider()
    st.subheader("⚙️ Độ nhạy AI")
    ss.ear_thr = st.slider("Ngưỡng EAR (cao = nhạy nhắm mắt hơn)", 0.10, 0.35, ss.ear_thr, 0.01)
    ss.mar_thr = st.slider("Ngưỡng MAR (thấp = nhạy ngáp hơn)", 0.40, 1.00, ss.mar_thr, 0.05)
    ss.closed_sec = st.slider("Nhắm mắt liên tục bao lâu thì báo (giây)", 0.5, 4.0, ss.closed_sec, 0.5)

    st.subheader("🔊 Âm thanh cảnh báo")
    ss.sound = st.selectbox("Chọn loại âm thanh", list(SOUNDS.keys()),
                            index=list(SOUNDS.keys()).index(ss.sound))
    if st.button("▶️ Nghe thử"):
        play_sound(SOUNDS[ss.sound])


# ════════════════════════ MAIN ════════════════════════
def main():
    init_state()
    ss = st.session_state
    with st.sidebar:
        st.header("🚗 DrowsyDriver")
        st.caption("Demo AI phát hiện buồn ngủ — IS54A")
        ss.page = st.radio("Màn hình", ["Dashboard", "Alert", "Analytics"],
                           index=["Dashboard", "Alert", "Analytics"].index(ss.page))
        st.divider()
        st.caption(f"CNN TFLite: {'✅ đã nạp' if TFLITE else '— (EAR/MAR)'}")

    if not DEPS_OK:
        st.error("Thiếu thư viện. Cài: `pip install -r requirements-streamlit.txt`")
        st.code(DEPS_ERR); return

    {"Dashboard": page_dashboard, "Alert": page_alert, "Analytics": page_analytics}[ss.page]()


if __name__ == "__main__":
    main()
