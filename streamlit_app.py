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

# CORE = cv2 + mediapipe (bắt buộc cho phân tích ảnh).
# WEBRTC = av + streamlit-webrtc (TÙY CHỌN: camera trực tiếp). Thiếu vẫn chạy bằng ảnh chụp
# → hợp deploy Streamlit Cloud (nơi webrtc/av hay lỗi build hoặc cần TURN server).
try:
    import cv2
    import mediapipe as mp
    CORE_OK, CORE_ERR = True, ""
except Exception as e:                                       # pragma: no cover
    CORE_OK, CORE_ERR = False, str(e)

try:
    import av
    from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
    WEBRTC_OK = True
except Exception:                                            # pragma: no cover
    WEBRTC_OK = False
    class VideoProcessorBase:                                # placeholder để khai báo Processor
        pass

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


# ─────── Âm thanh cảnh báo: BINAURAL BEATS dải BETA (13–21 Hz) kích thích tỉnh táo ───────
# Cơ chế (CK AI §1.3): tai TRÁI nghe sóng mang f, tai PHẢI nghe f+beat → não cảm nhận nhịp
# 'beat' Hz. Dải beta (13–21 Hz) gắn với sự tỉnh táo/tập trung (Moessinger, 2021). Hai tông
# lệch nhau cũng tạo 'acoustic beating' nghe được trên loa đơn nên hiệu quả cả khi không tai nghe.
RATE = 16000


def _wav_stereo(int16_interleaved):
    data = int16_interleaved.tobytes()
    byte_rate = RATE * 2 * 2                                  # rate * channels * bytes/sample
    hdr = (b"RIFF" + struct.pack("<I", 36 + len(data)) + b"WAVE" + b"fmt "
           + struct.pack("<IHHIIHH", 16, 1, 2, RATE, byte_rate, 4, 16)   # channels = 2
           + b"data" + struct.pack("<I", len(data)))
    return base64.b64encode(hdr + bytes(data)).decode()


def make_binaural(carrier=200.0, beat=18.0, ms=4000, vol=0.55):
    """Binaural beat stereo (L=carrier, R=carrier+beat)."""
    t = np.arange(int(RATE * ms / 1000)) / RATE
    dur = ms / 1000.0
    env = np.clip(np.minimum(np.minimum(t * 6.0, (dur - t) * 6.0), 1.0), 0.0, 1.0)
    left = (32767 * vol * env * np.sin(2 * np.pi * carrier * t)).astype("<i2")
    right = (32767 * vol * env * np.sin(2 * np.pi * (carrier + beat) * t)).astype("<i2")
    inter = np.empty(left.size * 2, dtype="<i2")
    inter[0::2] = left; inter[1::2] = right
    return _wav_stereo(inter)


def make_beep(freq=880.0, ms=500, vol=0.5):
    t = np.arange(int(RATE * ms / 1000)) / RATE
    v = (32767 * vol * np.sin(2 * np.pi * freq * t)).astype("<i2")
    inter = np.empty(v.size * 2, dtype="<i2"); inter[0::2] = v; inter[1::2] = v
    return _wav_stereo(inter)


SOUNDS = {
    "Binaural beta 18 Hz (khuyến nghị)": make_binaural(200, 18, 4000),
    "Binaural beta 14 Hz": make_binaural(200, 14, 4000),
    "Binaural beta 21 Hz": make_binaural(200, 21, 4000),
    "Beep cảnh báo": make_beep(880, 600),
}


def play_sound(b64, loop=True):
    lp = "loop" if loop else ""
    st.markdown(f'<audio autoplay {lp} src="data:audio/wav;base64,{b64}"></audio>',
                unsafe_allow_html=True)


# ───────────────────────── Histogram khung giờ hay buồn ngủ ─────────────────────────
import json
HOURS_FILE = "drowsy_hours.json"


def load_hours():
    try:
        with open(HOURS_FILE, "r", encoding="utf-8") as f:
            h = json.load(f)
        if isinstance(h, list) and len(h) == 24:
            return [int(x) for x in h]
    except Exception:
        pass
    return [0] * 24


def save_hours(h):
    try:
        with open(HOURS_FILE, "w", encoding="utf-8") as f:
            json.dump(h, f)
    except Exception:
        pass


def bump_hour():
    """Ghi nhận giờ hiện tại vào histogram (gọi khi phát hiện buồn ngủ)."""
    ss = st.session_state
    ss.hours[int(time.strftime("%H"))] += 1
    save_hours(ss.hours)


# ───────────────────────── State dùng chung ─────────────────────────
def init_state():
    ss = st.session_state
    ss.setdefault("page", "Dashboard")
    ss.setdefault("monitoring", False)
    ss.setdefault("show_cam", True)
    ss.setdefault("ear_thr", 0.21)
    ss.setdefault("mar_thr", 0.6)
    ss.setdefault("closed_sec", 1.5)
    ss.setdefault("sound", "Binaural beta 18 Hz (khuyến nghị)")
    ss.setdefault("history", [])
    ss.setdefault("hours", load_hours())
    ss.setdefault("drive_start", time.time())


# ───────────────────────── Bộ xử lý video (chạy theo từng frame) ─────────────────────────
if WEBRTC_OK and CORE_OK:
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


# ───────────────────────── WebRTC config + fallback ảnh chụp ─────────────────────────
# STUN giúp webrtc kết nối qua NAT/firewall (giảm lỗi "camera không lên").
RTC_CONFIG = {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}


@st.cache_resource
def _cached_landmarker():
    return _make_landmarker()


def analyze_snapshot(pil_img, ear_thr, mar_thr):
    """Phân tích 1 ảnh chụp (fallback khi WebRTC lỗi) → (status, ear, mar, ảnh chú thích BGR)."""
    arr = np.array(pil_img.convert("RGB"))
    img = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    h, w = img.shape[:2]
    res = _cached_landmarker().detect(
        mp.Image(image_format=mp.ImageFormat.SRGB, data=arr))
    if not res.face_landmarks:
        return "NO FACE", 0.0, 0.0, img
    lm = res.face_landmarks[0]
    P = lambda i: (lm[i].x * w, lm[i].y * h)
    ear = (_ear([P(i) for i in LEFT_EYE]) + _ear([P(i) for i in RIGHT_EYE])) / 2
    mar = _mar(P(MOUTH[0]), P(MOUTH[1]), P(MOUTH[2]), P(MOUTH[3]))
    status = "DROWSY" if ear < ear_thr else ("WARNING" if mar > mar_thr else "ALERT")
    col = {"ALERT": (0, 200, 0), "WARNING": (0, 180, 230), "DROWSY": (0, 0, 255)}[status]
    for i in LEFT_EYE + RIGHT_EYE:
        cv2.circle(img, tuple(map(int, P(i))), 2, col, -1)
    cv2.putText(img, f"EAR {ear:.2f}  MAR {mar:.2f}  {status}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)
    return status, ear, mar, img


def analyze_video(video_file, ear_thr, mar_thr, sample_fps=2, max_frames=120):
    """Phân tích video upload — lấy mẫu sample_fps khung/giây, tối đa max_frames khung.

    Trả về:
      results   : list[dict{t, status, ear, mar}]   — 1 entry mỗi frame lấy mẫu
      highlights: list[dict{t, status, img}]         — tối đa 6 frame DROWSY/WARNING (ảnh nhỏ RGB)
      total_frames: tổng số frame trong video
      video_fps : FPS thực của video
    """
    import os
    import tempfile
    suffix = "." + video_file.name.rsplit(".", 1)[-1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(video_file.getvalue())
        tmp_path = tmp.name

    results, highlights = [], []
    total_frames, video_fps = 0, 25.0
    try:
        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            return results, highlights, total_frames, video_fps
        video_fps    = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        interval     = max(1, int(video_fps / sample_fps))
        lmk          = _cached_landmarker()
        frame_idx = processed = 0

        while cap.isOpened() and processed < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % interval == 0:
                t   = round(frame_idx / video_fps, 2)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res = lmk.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
                h, w = frame.shape[:2]
                if res.face_landmarks:
                    lmpts = res.face_landmarks[0]
                    P = lambda i: (lmpts[i].x * w, lmpts[i].y * h)   # noqa: E731
                    ear = (_ear([P(i) for i in LEFT_EYE]) + _ear([P(i) for i in RIGHT_EYE])) / 2
                    mar = _mar(P(MOUTH[0]), P(MOUTH[1]), P(MOUTH[2]), P(MOUTH[3]))
                    status = ("DROWSY"  if ear < ear_thr else
                              "WARNING" if mar > mar_thr else "ALERT")
                    results.append({"t": t, "status": status,
                                    "ear": round(ear, 3), "mar": round(mar, 3)})
                    if status in ("DROWSY", "WARNING") and len(highlights) < 6:
                        col = (0, 0, 255) if status == "DROWSY" else (0, 180, 230)
                        ann = frame.copy()
                        for i in LEFT_EYE + RIGHT_EYE:
                            cv2.circle(ann, tuple(map(int, P(i))), 2, col, -1)
                        cv2.putText(ann, f"{t:.1f}s | {status}",
                                    (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2)
                        highlights.append({
                            "t": t, "status": status,
                            "img": cv2.cvtColor(cv2.resize(ann, (200, 150)),
                                                cv2.COLOR_BGR2RGB)})
                else:
                    results.append({"t": t, "status": "NO FACE", "ear": 0.0, "mar": 0.0})
                processed += 1
            frame_idx += 1
        cap.release()
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass
    return results, highlights, total_frames, video_fps


# ───────────────────────── CSS (theme đỏ–đen "Stitch") ─────────────────────────
THEME_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
.stApp{background:radial-gradient(1100px 560px at 82% -12%, #2a0a12 0%, #0E0E10 55%) fixed;
       color:#F5F5F7;font-family:'Inter',system-ui,sans-serif}
#MainMenu,footer{visibility:hidden}
.block-container{padding-top:1.6rem}
/* Hero banner */
.hero{background:linear-gradient(135deg,#FC1C46 0%,#7a0d22 58%,#1B1B1E 100%);
      border-radius:24px;padding:26px 30px;margin-bottom:18px;box-shadow:0 12px 42px rgba(252,28,70,.28)}
.hero h1{color:#fff;font-size:30px;font-weight:800;margin:0;letter-spacing:.3px}
.hero p{color:#ffe;opacity:.92;margin:6px 0 0;font-size:15px}
/* Thẻ & metric */
.card{background:#1B1B1E;border:1px solid #2a2a2e;border-radius:20px;padding:20px;margin-bottom:14px}
.metric{background:#1B1B1E;border:1px solid #2a2a2e;border-radius:18px;padding:16px 18px;text-align:center}
.metric .v{font-size:34px;font-weight:800;color:#FC1C46;line-height:1.1}
.metric .l{color:#9A9AA0;font-size:13px;margin-top:4px}
/* Thanh trạng thái */
.statusbar{padding:18px;border-radius:16px;color:#fff;font-size:24px;font-weight:800;text-align:center;
           box-shadow:0 8px 28px rgba(0,0,0,.35)}
.status-alert{background:linear-gradient(135deg,#1b8a3a,#0c5022)}
.status-warn{background:linear-gradient(135deg,#caa300,#7a6200)}
.status-drowsy{background:linear-gradient(135deg,#e23,#7a1f16)}
/* Nút bấm */
.stButton>button{border-radius:14px;font-weight:700;border:1px solid #333;background:#1B1B1E;color:#fff;transition:.15s}
.stButton>button:hover{border-color:#FC1C46;color:#FC1C46;transform:translateY(-1px)}
/* Cảnh báo nhấp nháy */
@keyframes flash{0%,100%{background:#7f0000}50%{background:#ff1e1e}}
.alert-box{animation:flash .6s infinite;border-radius:24px;padding:46px;text-align:center;color:#fff;
           box-shadow:0 0 70px rgba(255,30,30,.55)}
/* Sidebar */
section[data-testid="stSidebar"]{background:#141416;border-right:1px solid #2a2a2e}
section[data-testid="stSidebar"] .stRadio label{color:#F5F5F7}
</style>
"""


def hero(title, subtitle):
    st.markdown(f'<div class="hero"><h1>{title}</h1><p>{subtitle}</p></div>', unsafe_allow_html=True)


def metric_card(label, value):
    st.markdown(f'<div class="metric"><div class="v">{value}</div><div class="l">{label}</div></div>',
                unsafe_allow_html=True)


# ════════════════════════ CÁC MÀN HÌNH ════════════════════════
def page_dashboard():
    hero("🚗 DrowsyDriver AI", "Giám sát tài xế thời gian thực — CNN + MediaPipe (EAR/MAR)")
    ss = st.session_state
    c1, c2 = st.columns([3, 2])

    vp          = None
    snap_status = None
    vid_results = None   # (vr, vh, cnt, vtot, vfps) — chỉ có giá trị khi đang ở video mode

    with c1:
        modes = ["📸 Chụp ảnh (ổn định, hợp web)",
                 "📹 Upload video (phân tích offline)"]
        if WEBRTC_OK:
            modes.append("🎥 Camera trực tiếp (WebRTC)")
        mode = st.radio("Chế độ camera", modes, horizontal=True, label_visibility="collapsed")

        # ── 🎥 WebRTC ──────────────────────────────────────────────────────────
        if mode.startswith("🎥"):
            try:
                ctx = webrtc_streamer(
                    key="cam", mode=WebRtcMode.SENDRECV, video_processor_factory=Processor,
                    rtc_configuration=RTC_CONFIG,
                    media_stream_constraints={"video": True, "audio": False},
                    async_processing=True)
                if ctx and ctx.video_processor:
                    vp = ctx.video_processor
                    vp.ear_thr, vp.mar_thr = ss.ear_thr, ss.mar_thr
                    vp.closed_sec, vp.show = ss.closed_sec, True
            except Exception as e:
                st.warning(f"⚠️ Camera trực tiếp lỗi — chuyển sang chế độ chụp ảnh. ({e})")

        # ── 📹 Upload video ────────────────────────────────────────────────────
        elif mode.startswith("📹"):
            vid_up = st.file_uploader(
                "Chọn file video để phân tích buồn ngủ (mp4 / avi / mov / mkv)",
                type=["mp4", "avi", "mov", "mkv", "webm"],
                label_visibility="collapsed")
            if vid_up is not None:
                vid_key = f"_vid_{vid_up.name}_{vid_up.size}"
                if vid_key not in ss:
                    with st.spinner(
                            f"🔍 Đang phân tích **{vid_up.name}** … "
                            f"(lấy mẫu 2 FPS, tối đa 60 giây)"):
                        ss[vid_key] = analyze_video(vid_up, ss.ear_thr, ss.mar_thr,
                                                    sample_fps=2, max_frames=120)
                vr, vh, vtot, vfps = ss[vid_key]
                if vr:
                    import pandas as pd
                    cnt = {s: sum(1 for r in vr if r["status"] == s)
                           for s in ("ALERT", "WARNING", "DROWSY", "NO FACE")}
                    snap_status = ("DROWSY"  if cnt["DROWSY"]  > 0 else
                                   "WARNING" if cnt["WARNING"] > 0 else "ALERT")
                    vid_results = (vr, vh, cnt, vtot, vfps)
                    if snap_status == "DROWSY":
                        bump_hour()

                    # ── Timeline EAR ─────────────────────────────────────────
                    df_v = pd.DataFrame(
                        [{"Thời gian (s)": r["t"], "EAR": r["ear"], "MAR": r["mar"]}
                         for r in vr if r["status"] != "NO FACE"])
                    if not df_v.empty:
                        st.caption("📈 EAR theo thời gian — dưới ngưỡng = buồn ngủ ↓")
                        df_v = df_v.set_index("Thời gian (s)")
                        st.line_chart(df_v[["EAR"]], color=["#FC1C46"], height=165)

                    # ── Gallery frame cảnh báo ────────────────────────────────
                    if vh:
                        with st.expander(f"⚠️ {len(vh)} frame cần chú ý", expanded=True):
                            gcols = st.columns(min(len(vh), 3))
                            for gi, hf in enumerate(vh):
                                gcols[gi % 3].image(
                                    hf["img"],
                                    caption=f"{hf['t']}s · {hf['status']}",
                                    use_container_width=True)
                    else:
                        st.success("✅ Không phát hiện dấu hiệu buồn ngủ trong video này.")
                else:
                    st.warning("Không đọc được video hoặc không tìm thấy khuôn mặt.")

        # ── 📸 Chụp ảnh ────────────────────────────────────────────────────────
        else:
            snap = st.camera_input("Chụp ảnh khuôn mặt để phân tích")
            if snap is not None:
                from PIL import Image
                s_status, s_ear, s_mar, s_img = analyze_snapshot(
                    Image.open(snap), ss.ear_thr, ss.mar_thr)
                snap_status = s_status
                st.image(cv2.cvtColor(s_img, cv2.COLOR_BGR2RGB), channels="RGB",
                         caption=f"EAR {s_ear:.2f} · MAR {s_mar:.2f}  →  {s_status}")
                if s_status == "DROWSY":
                    bump_hour()

    # ── Cột phải ───────────────────────────────────────────────────────────────
    with c2:
        if vid_results:
            # Metrics tổng hợp từ video
            vr, vh, cnt, vtot, vfps = vid_results
            n    = len(vr)
            dur  = round(vtot / vfps) if vfps > 0 else 0
            pct_d = round(cnt["DROWSY"]  / n * 100) if n > 0 else 0
            pct_w = round(cnt["WARNING"] / n * 100) if n > 0 else 0
            metric_card("Độ dài video", f"{dur}s ({n} mẫu)")
            metric_card("Tỉ lệ buồn ngủ",   f"{pct_d}%")
            metric_card("Tỉ lệ cảnh báo",   f"{pct_w}%")
        else:
            ss.monitoring = st.toggle("🟢 KÍCH HOẠT GIÁM SÁT", value=ss.monitoring)
            metric_card("Số lần ngáp (phiên này)", vp.yawn_count if vp else 0)

        # Thanh trạng thái chung (hiển thị cả 3 mode)
        status = (vp.status if vp else None) or snap_status or "—"
        cls    = {"ALERT": "status-alert", "WARNING": "status-warn",
                  "DROWSY": "status-drowsy"}.get(status, "status-warn")
        label  = {"ALERT": "✅ TỈNH TÁO", "WARNING": "🟡 CÓ DẤU HIỆU MỆT",
                  "DROWSY": "🔴 BUỒN NGỦ!", "NO FACE": "🔍 Không thấy mặt"}.get(status, status)
        st.markdown(f'<div class="statusbar {cls}">{label}</div>', unsafe_allow_html=True)

        if not vid_results:
            if vp:
                st.progress(min(vp.ear / 0.4, 1.0), text=f"EAR {vp.ear:.2f}")

            # ⏱️ Thời gian lái + nhắc nghỉ mốc 2 giờ / 4 giờ (CK AI §1.3)
            mins = int((time.time() - ss.drive_start) / 60)
            st.caption(f"⏱️ Thời gian lái liên tục: {mins // 60} giờ {mins % 60} phút")
            for mark in (2, 4):
                if mins >= mark * 60 and not ss.get(f"rest_{mark}"):
                    ss[f"rest_{mark}"] = True
                    st.warning(f"⏰ Bạn đã lái {mark} giờ liên tục — nên dừng nghỉ 15–30 phút!")

            if ss.monitoring:
                if vp and vp.event:
                    ss.history.append((time.strftime("%H:%M:%S"), vp.event[0], vp.event[1]))
                    if vp.event[0] == "Nhắm mắt":
                        bump_hour()
                if status == "DROWSY":
                    ss.page = "Alert"; st.rerun()

    if not vid_results:
        st.info("Bật **Kích hoạt giám sát**, rồi nhắm mắt > 1.5s hoặc ngáp để thử cảnh báo. "
                "Âm thanh dùng **binaural beats dải beta** — nghe rõ nhất khi đeo tai nghe.")
    if not WEBRTC_OK:
        st.caption("ℹ️ Bản web chạy chế độ **chụp ảnh / upload video** — ổn định khi deploy.")


def page_alert():
    ss = st.session_state
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
    hero("📊 Thống kê & Cài đặt", "Lịch sử buồn ngủ trong phiên · tinh chỉnh độ nhạy AI")

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
    st.subheader("🕐 Khung giờ hay buồn ngủ")
    hours = ss.hours
    if sum(hours) == 0:
        st.caption("Chưa đủ dữ liệu. Hệ thống sẽ tự học khung giờ bạn hay buồn ngủ để khuyến cáo tránh lái xe vào giờ đó.")
    else:
        import pandas as pd
        peak = max(range(24), key=lambda i: hours[i])
        st.markdown(f"**Bạn hay buồn ngủ nhất vào khoảng {peak:02d}:00–{(peak + 1) % 24:02d}:00** — {hours[peak]} lần.")
        st.warning("⚠️ Nên tránh lái xe vào khung giờ này — hãy nghỉ ngơi đầy đủ hoặc đổi tài xế trước khi lái.")
        dfh = pd.DataFrame({"Số lần buồn ngủ": hours}, index=[f"{h:02d}h" for h in range(24)])
        st.bar_chart(dfh, color="#FC1C46")
        if st.button("🗑️ Xóa dữ liệu khung giờ"):
            ss.hours = [0] * 24; save_hours(ss.hours); st.rerun()

    st.divider()
    st.subheader("⚙️ Độ nhạy AI")
    ss.ear_thr = st.slider("Ngưỡng EAR (cao = nhạy nhắm mắt hơn)", 0.10, 0.35, ss.ear_thr, 0.01)
    ss.mar_thr = st.slider("Ngưỡng MAR (thấp = nhạy ngáp hơn)", 0.40, 1.00, ss.mar_thr, 0.05)
    ss.closed_sec = st.slider("Nhắm mắt liên tục bao lâu thì báo (giây)", 0.5, 4.0, ss.closed_sec, 0.5)

    st.subheader("🔊 Âm thanh cảnh báo — Binaural beats (beta 13–21 Hz)")
    st.caption("Cơ chế kích thích tỉnh táo bằng nhịp sóng não dải beta (CK AI §1.3): tai trái và tai "
               "phải nghe hai tần số lệch nhau. **Nghe rõ nhất khi đeo tai nghe**; trên loa vẫn tạo "
               "nhịp âm thanh (acoustic beating) dễ nhận biết.")
    if ss.sound not in SOUNDS:
        ss.sound = list(SOUNDS.keys())[0]
    ss.sound = st.selectbox("Chọn âm thanh", list(SOUNDS.keys()),
                            index=list(SOUNDS.keys()).index(ss.sound))
    if st.button("▶️ Nghe thử"):
        play_sound(SOUNDS[ss.sound], loop=False)


# ════════════════════════ MAIN ════════════════════════
def main():
    init_state()
    st.markdown(THEME_CSS, unsafe_allow_html=True)
    ss = st.session_state
    with st.sidebar:
        st.header("🚗 DrowsyDriver")
        st.caption("Demo AI phát hiện buồn ngủ — IS54A")
        ss.page = st.radio("Màn hình", ["Dashboard", "Alert", "Analytics"],
                           index=["Dashboard", "Alert", "Analytics"].index(ss.page))
        st.divider()
        st.caption(f"Camera: {'🎥 trực tiếp + 📸 ảnh' if WEBRTC_OK else '📸 ảnh chụp (web)'}")
        st.caption(f"CNN TFLite: {'✅ đã nạp' if TFLITE else '— (EAR/MAR)'}")

    if not CORE_OK:
        st.error("Thiếu thư viện lõi (opencv-python-headless, mediapipe). "
                 "Cài: `pip install -r requirements.txt`")
        st.code(CORE_ERR); return

    {"Dashboard": page_dashboard, "Alert": page_alert, "Analytics": page_analytics}[ss.page]()


if __name__ == "__main__":
    main()
