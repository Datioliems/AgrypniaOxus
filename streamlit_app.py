# -*- coding: utf-8 -*-
"""
🚗 DrowsyDriver — Streamlit Demo hoàn thiện (4 màn hình)
  1. Dashboard  : giám sát liên tục — webcam live + auto-refresh 1.5s + EAR/MAR gauge
  2. Alert      : cảnh báo đỏ nhấp nháy + binaural beats + SOS + hướng dẫn an toàn
  3. Analytics  : thống kê đầy đủ — lịch sử, khung giờ, xuất CSV, cài đặt AI & âm thanh
  4. Hướng dẫn : thuật toán EAR/MAR, binaural beats, cách dùng hiệu quả

Deploy:  xem STREAMLIT_DEPLOY.md  |  Python 3.12  |  packages.txt: libgl1 libgles2 libegl1
"""
import base64, io, json, math, struct, time
import numpy as np
import streamlit as st

st.set_page_config(
    page_title="DrowsyDriver AI",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Thư viện lõi (bắt buộc) ──────────────────────────────────────────────────
try:
    import cv2
    import mediapipe as mp
    CORE_OK, CORE_ERR = True, ""
except Exception as e:
    CORE_OK, CORE_ERR = False, str(e)

# ── Auto-refresh (cài streamlit-autorefresh — nhẹ, không native deps) ────────
try:
    from streamlit_autorefresh import st_autorefresh
    AUTOREFRESH_OK = True
except Exception:
    AUTOREFRESH_OK = False

# ── WebRTC (tùy chọn — local / Docker) ───────────────────────────────────────
try:
    import av
    from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
    WEBRTC_OK = True
except Exception:
    WEBRTC_OK = False
    class VideoProcessorBase:   # placeholder
        pass

# ═════════════════════════════════════════════════════════════════════════════
# PHẦN 1: MEDIAPIPE — EAR / MAR
# ═════════════════════════════════════════════════════════════════════════════
_LANDMARK_MODEL = "face_landmarker.task"
_MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
              "face_landmarker/float16/1/face_landmarker.task")

LEFT_EYE  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33,  160, 158, 133, 153, 144]
MOUTH     = [13,  14,  78,  308]            # trên, dưới, trái, phải


def _ear(pts):
    """Eye Aspect Ratio — nhỏ khi mắt nhắm."""
    v = math.dist(pts[1], pts[5]) + math.dist(pts[2], pts[4])
    return v / (2.0 * math.dist(pts[0], pts[3]) + 1e-6)


def _mar(top, bot, left, right):
    """Mouth Aspect Ratio — lớn khi đang ngáp."""
    return math.dist(top, bot) / (math.dist(left, right) + 1e-6)


def _make_landmarker():
    import os, urllib.request
    from mediapipe.tasks import python as mpp
    from mediapipe.tasks.python import vision
    if not os.path.exists(_LANDMARK_MODEL):
        urllib.request.urlretrieve(_MODEL_URL, _LANDMARK_MODEL)
    opts = vision.FaceLandmarkerOptions(
        base_options=mpp.BaseOptions(model_asset_path=_LANDMARK_MODEL),
        running_mode=vision.RunningMode.IMAGE, num_faces=1)
    return vision.FaceLandmarker.create_from_options(opts)


@st.cache_resource(show_spinner="Đang nạp mô hình MediaPipe…")
def _lmk():
    return _make_landmarker()


# ═════════════════════════════════════════════════════════════════════════════
# PHẦN 2: ÂM THANH — BINAURAL BEATS BETA (13–21 Hz)
# ═════════════════════════════════════════════════════════════════════════════
RATE = 16_000


def _wav_stereo(interleaved: np.ndarray) -> str:
    data = interleaved.tobytes()
    hdr = (b"RIFF" + struct.pack("<I", 36 + len(data)) + b"WAVE" + b"fmt "
           + struct.pack("<IHHIIHH", 16, 1, 2, RATE, RATE * 4, 4, 16)
           + b"data" + struct.pack("<I", len(data)))
    return base64.b64encode(hdr + data).decode()


def make_binaural(carrier=200.0, beat=18.0, ms=4000, vol=0.55):
    """Stereo WAV: L = carrier Hz, R = carrier+beat Hz → não cảm nhận nhịp beat Hz."""
    t   = np.arange(int(RATE * ms / 1000)) / RATE
    dur = ms / 1000.0
    env = np.clip(np.minimum(t * 6.0, np.minimum((dur - t) * 6.0, 1.0)), 0.0, 1.0)
    L   = (32767 * vol * env * np.sin(2 * np.pi * carrier * t)).astype("<i2")
    R   = (32767 * vol * env * np.sin(2 * np.pi * (carrier + beat) * t)).astype("<i2")
    buf = np.empty(L.size * 2, dtype="<i2"); buf[0::2] = L; buf[1::2] = R
    return _wav_stereo(buf)


def make_beep(freq=880.0, ms=600, vol=0.5):
    t = np.arange(int(RATE * ms / 1000)) / RATE
    v = (32767 * vol * np.sin(2 * np.pi * freq * t)).astype("<i2")
    buf = np.empty(v.size * 2, dtype="<i2"); buf[0::2] = v; buf[1::2] = v
    return _wav_stereo(buf)


@st.cache_data(show_spinner=False)
def _preload_sounds():
    return {
        "Binaural beta 18 Hz (khuyến nghị)": make_binaural(200, 18, 4000),
        "Binaural beta 14 Hz":               make_binaural(200, 14, 4000),
        "Binaural beta 21 Hz":               make_binaural(200, 21, 4000),
        "Beep cảnh báo":                     make_beep(880, 600),
    }


SOUNDS = _preload_sounds()


def play_sound(b64: str, loop=True):
    lp = "loop" if loop else ""
    st.markdown(f'<audio autoplay {lp} src="data:audio/wav;base64,{b64}"></audio>',
                unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# PHẦN 3: LƯU TRỮ PERSISTENT
# ═════════════════════════════════════════════════════════════════════════════
HOURS_FILE = "drowsy_hours.json"


def load_hours():
    try:
        with open(HOURS_FILE, encoding="utf-8") as f:
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
    ss = st.session_state
    ss.hours[int(time.strftime("%H"))] += 1
    save_hours(ss.hours)


# ═════════════════════════════════════════════════════════════════════════════
# PHẦN 4: PHÂN TÍCH ẢNH / VIDEO
# ═════════════════════════════════════════════════════════════════════════════
def analyze_frame(arr_rgb: np.ndarray, ear_thr: float, mar_thr: float):
    """Phân tích frame RGB numpy → (status, ear, mar, annotated_bgr)."""
    img = cv2.cvtColor(arr_rgb, cv2.COLOR_RGB2BGR)
    h, w = img.shape[:2]
    res  = _lmk().detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=arr_rgb))
    if not res.face_landmarks:
        return "NO FACE", 0.0, 0.0, img
    lm  = res.face_landmarks[0]
    P   = lambda i: (lm[i].x * w, lm[i].y * h)
    ear = (_ear([P(i) for i in LEFT_EYE]) + _ear([P(i) for i in RIGHT_EYE])) / 2
    mar = _mar(P(MOUTH[0]), P(MOUTH[1]), P(MOUTH[2]), P(MOUTH[3]))
    status = "DROWSY" if ear < ear_thr else ("WARNING" if mar > mar_thr else "ALERT")
    col = {"ALERT": (0,200,0), "WARNING": (0,180,230), "DROWSY": (0,0,255)}[status]
    for i in LEFT_EYE + RIGHT_EYE:
        cv2.circle(img, (int(P(i)[0]), int(P(i)[1])), 2, col, -1)
    cv2.putText(img, f"EAR {ear:.3f}  MAR {mar:.3f}  {status}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.62, col, 2)
    return status, ear, mar, img


def analyze_snapshot(pil_img, ear_thr, mar_thr):
    arr = np.array(pil_img.convert("RGB"))
    return analyze_frame(arr, ear_thr, mar_thr)


def analyze_video(video_file, ear_thr, mar_thr, sample_fps=2, max_frames=120):
    import os, tempfile
    suffix = "." + video_file.name.rsplit(".", 1)[-1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(video_file.getvalue()); tmp_path = tmp.name
    results, highlights = [], []
    total_frames = video_fps = 0
    try:
        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            return results, highlights, total_frames, video_fps
        video_fps    = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        interval = max(1, int(video_fps / sample_fps))
        frame_idx = processed = 0
        while cap.isOpened() and processed < max_frames:
            ret, frame = cap.read()
            if not ret: break
            if frame_idx % interval == 0:
                t   = round(frame_idx / video_fps, 2)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                status, ear, mar, ann_bgr = analyze_frame(rgb, ear_thr, mar_thr)
                results.append({"t": t, "status": status,
                                "ear": round(ear, 3), "mar": round(mar, 3)})
                if status in ("DROWSY", "WARNING") and len(highlights) < 6:
                    highlights.append({
                        "t": t, "status": status,
                        "img": cv2.cvtColor(cv2.resize(ann_bgr, (200, 150)),
                                            cv2.COLOR_BGR2RGB)})
                processed += 1
            frame_idx += 1
        cap.release()
    finally:
        try: os.unlink(tmp_path)
        except Exception: pass
    return results, highlights, total_frames, video_fps


# ── WebRTC processor ──────────────────────────────────────────────────────────
if WEBRTC_OK and CORE_OK:
    class Processor(VideoProcessorBase):
        def __init__(self):
            self.landmarker = _make_landmarker()
            self.ear = self.mar = 0.0; self.status = "ALERT"
            self.closed_since = None; self.yawn_count = 0
            self._prev_yawn = False; self.event = None
            self.ear_thr = 0.21; self.mar_thr = 0.6; self.closed_sec = 1.5

        def recv(self, frame):
            img = frame.to_ndarray(format="bgr24")
            h, w = img.shape[:2]
            rgb  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            res  = self.landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
            self.event = None
            if res.face_landmarks:
                lm = res.face_landmarks[0]
                P  = lambda i: (lm[i].x * w, lm[i].y * h)
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
                col = {"ALERT":(0,200,0),"WARNING":(0,180,230),"DROWSY":(0,0,255)}[self.status]
                for i in LEFT_EYE + RIGHT_EYE:
                    cv2.circle(img, (int(P(i)[0]), int(P(i)[1])), 1, col, -1)
                cv2.putText(img, f"EAR {self.ear:.2f}  MAR {self.mar:.2f}  {self.status}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, col, 2)
            else:
                self.status = "NO FACE"
            return av.VideoFrame.from_ndarray(img, format="bgr24")

RTC_CONFIG = {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}


# ═════════════════════════════════════════════════════════════════════════════
# PHẦN 5: CSS THEME (Stitch — đỏ / đen)
# ═════════════════════════════════════════════════════════════════════════════
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');

/* ── Base ── */
.stApp {
  background: radial-gradient(ellipse 1200px 600px at 78% -8%,
    #2a0813 0%, #0d0d0f 62%) fixed;
  color: #F0F0F2; font-family: 'Inter', system-ui, sans-serif;
}
#MainMenu, footer, .stDeployButton { visibility: hidden }
.block-container { padding-top: 1.2rem; max-width: 1180px }

/* ── Hero banner ── */
.hero {
  background: linear-gradient(135deg, #FC1C46 0%, #820d22 52%, #1a1a1e 100%);
  border-radius: 20px; padding: 20px 26px; margin-bottom: 14px;
  box-shadow: 0 8px 38px rgba(252,28,70,.22);
}
.hero h1 { color:#fff; font-size:24px; font-weight:800; margin:0 }
.hero p  { color:rgba(255,255,210,.88); margin:4px 0 0; font-size:13.5px }
.hero-row { display:flex; justify-content:space-between; align-items:center }
.hero-badge {
  background:rgba(255,255,255,.14); backdrop-filter:blur(6px);
  border-radius:10px; padding:5px 13px; font-size:13px; color:#fff; font-weight:700;
  white-space:nowrap;
}

/* ── Metric card ── */
.metric {
  background:#1b1b1e; border:1px solid #2c2c30;
  border-radius:14px; padding:13px 15px; text-align:center; margin-bottom:8px;
}
.metric .v { font-size:26px; font-weight:800; color:#FC1C46; line-height:1.15 }
.metric .l { color:#8a8a92; font-size:11.5px; margin-top:2px }

/* ── Status bar ── */
.statusbar {
  padding:14px; border-radius:14px; color:#fff; font-size:20px; font-weight:800;
  text-align:center; box-shadow:0 5px 20px rgba(0,0,0,.3); margin-bottom:8px;
}
.s-alert  { background: linear-gradient(135deg,#1a8836,#0b4d1e) }
.s-warn   { background: linear-gradient(135deg,#c49500,#7a5c00) }
.s-drowsy { background: linear-gradient(135deg,#d92030,#781520) }
.s-noface { background: #2a2a2e }

/* ── EAR/MAR gauge ── */
.gauge { margin: 3px 0 9px }
.gauge-row { display:flex; justify-content:space-between;
  font-size:11.5px; color:#8a8a92; margin-bottom:3px }
.gauge-track { background:#252528; border-radius:5px; height:9px; overflow:hidden }
.gauge-fill  { height:9px; border-radius:5px; transition: width .4s ease }

/* ── Alert page ── */
@keyframes pulse {
  0%,100% { background:#7a0010; box-shadow:0 0 55px rgba(255,0,0,.35) }
  50%     { background:#f01828; box-shadow:0 0 80px rgba(255,20,30,.65) }
}
.alert-box {
  animation: pulse .7s ease-in-out infinite;
  border-radius:20px; padding:36px 28px; text-align:center; color:#fff;
  margin-bottom:18px;
}
.alert-box h1 { font-size:28px; margin:0 0 6px; letter-spacing:.5px }
.alert-box p  { margin:4px 0; font-size:14px; opacity:.88 }

/* ── Buttons ── */
.stButton > button {
  border-radius:12px; font-weight:700; border:1px solid #303033;
  background:#1b1b1e; color:#F0F0F2; transition:.15s;
}
.stButton > button:hover { border-color:#FC1C46; color:#FC1C46; transform:translateY(-1px) }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] { background:#1b1b1e; border-radius:10px; padding:3px; gap:2px }
.stTabs [data-baseweb="tab"]      { border-radius:8px; color:#8a8a92; font-weight:600; padding:6px 16px }
.stTabs [aria-selected="true"]    { background:#FC1C46 !important; color:#fff !important }

/* ── Sidebar ── */
section[data-testid="stSidebar"] { background:#111114; border-right:1px solid #252528 }
section[data-testid="stSidebar"] label { color:#F0F0F2 !important }

/* ── Misc ── */
hr { border-color:#252528 !important }
.stDataFrame { border-radius:10px; overflow:hidden }
.stAlert     { border-radius:12px }
.stExpanderHeader { font-weight:700 }
::-webkit-scrollbar { width:5px; height:5px }
::-webkit-scrollbar-track { background:#0d0d0f }
::-webkit-scrollbar-thumb { background:#36363a; border-radius:3px }
</style>
"""


# ═════════════════════════════════════════════════════════════════════════════
# PHẦN 6: UI HELPERS
# ═════════════════════════════════════════════════════════════════════════════
def hero(title: str, subtitle: str, badge: str = ""):
    bdg = f'<div class="hero-badge">{badge}</div>' if badge else ""
    st.markdown(f"""
    <div class="hero">
      <div class="hero-row">
        <div><h1>{title}</h1><p>{subtitle}</p></div>
        {bdg}
      </div>
    </div>""", unsafe_allow_html=True)


def metric_card(label: str, value):
    st.markdown(
        f'<div class="metric"><div class="v">{value}</div>'
        f'<div class="l">{label}</div></div>',
        unsafe_allow_html=True)


def gauge(label: str, val: float, max_val: float, color: str = "#FC1C46", unit: str = ""):
    pct = min(int(val / max_val * 100), 100) if max_val > 0 else 0
    st.markdown(f"""
    <div class="gauge">
      <div class="gauge-row"><span>{label}</span><span>{val:.3f}{unit}</span></div>
      <div class="gauge-track">
        <div class="gauge-fill" style="width:{pct}%;background:{color}"></div>
      </div>
    </div>""", unsafe_allow_html=True)


def status_bar(status: str):
    cls   = {"ALERT":"s-alert","WARNING":"s-warn","DROWSY":"s-drowsy"}.get(status, "s-noface")
    label = {"ALERT": "✅  TỈNH TÁO", "WARNING": "⚡  CÓ DẤU HIỆU MỆT",
             "DROWSY": "🔴  BUỒN NGỦ!", "NO FACE": "🔍  Chưa thấy mặt"}.get(status, "—")
    st.markdown(f'<div class="statusbar {cls}">{label}</div>', unsafe_allow_html=True)


# Component webcam live + auto-click vào nút chụp của camera_input
WEBCAM_HTML = """
<div style="position:relative;border-radius:12px;overflow:hidden;background:#0d0d0f;min-height:190px">
  <video id="wv" autoplay playsinline muted
    style="width:100%;max-height:250px;object-fit:cover;display:block"></video>
  <div id="wlive" style="position:absolute;top:7px;left:7px;
    background:rgba(252,28,70,.85);color:#fff;
    padding:2px 9px;border-radius:7px;font-size:10.5px;font-weight:800;letter-spacing:.5px">
    ● LIVE
  </div>
  <div id="wmsg" style="position:absolute;bottom:7px;left:0;right:0;text-align:center;
    color:rgba(255,255,255,.75);font-size:11px">Đang khởi động webcam…</div>
</div>
<script>
(function(){
  var vid = document.getElementById('wv');
  var msg = document.getElementById('wmsg');
  var lbl = document.getElementById('wlive');

  navigator.mediaDevices.getUserMedia({video:{width:640,height:480,facingMode:'user'},audio:false})
    .then(function(s){
      vid.srcObject = s;
      msg.textContent = 'Camera đang hoạt động — hệ thống giám sát bên dưới';
    })
    .catch(function(){
      lbl.textContent = '📵 NO CAM';
      lbl.style.background = 'rgba(80,80,80,.85)';
      msg.textContent = 'Hãy cho phép truy cập camera trong trình duyệt';
    });

  /* Auto-click nút chụp ảnh của st.camera_input mỗi 1.5s khi giám sát đang bật */
  function tryClick(){
    try{
      var p = window.parent;
      var sels = [
        'button[data-testid="stCameraInputButton"]',
        '[data-testid="stCameraInput"] button',
        '.stCameraInput button'
      ];
      for(var i=0;i<sels.length;i++){
        var b = p.document.querySelector(sels[i]);
        if(b && b.offsetParent!==null){
          b.dispatchEvent(new MouseEvent('click',{bubbles:true,cancelable:true}));
          return;
        }
      }
    }catch(e){}
  }
  setInterval(tryClick, 1500);
})();
</script>
"""


# ═════════════════════════════════════════════════════════════════════════════
# PHẦN 7: SESSION STATE
# ═════════════════════════════════════════════════════════════════════════════
def init_state():
    ss = st.session_state
    ss.setdefault("page",             "Dashboard")
    ss.setdefault("cam_mode",         "📸 Chụp ảnh")
    ss.setdefault("monitoring",       False)
    ss.setdefault("ear_thr",          0.21)
    ss.setdefault("mar_thr",          0.60)
    ss.setdefault("closed_sec",       1.5)
    ss.setdefault("sound",            "Binaural beta 18 Hz (khuyến nghị)")
    ss.setdefault("history",          [])       # [(time_str, loại, giá_trị), ...]
    ss.setdefault("hours",            load_hours())
    ss.setdefault("drive_start",      time.time())
    ss.setdefault("drowsy_count",     0)        # lần phát hiện DROWSY trong phiên
    ss.setdefault("yawn_count",       0)        # lần ngáp
    ss.setdefault("last_status",      "—")
    ss.setdefault("last_ear",         0.0)
    ss.setdefault("last_mar",         0.0)
    ss.setdefault("last_snap_b64",    None)     # bytes ảnh cuối dùng để re-analyze
    ss.setdefault("last_drowsy_ts",   0.0)      # cooldown tránh trigger liên tục
    ss.setdefault("was_yawning",      False)    # edge-detect ngáp
    ss.setdefault("rest_done",        set())    # {2,4} mốc đã nhắc
    ss.setdefault("sos_contact",      "113")


# ═════════════════════════════════════════════════════════════════════════════
# PHẦN 8: TRANG DASHBOARD
# ═════════════════════════════════════════════════════════════════════════════
def page_dashboard():
    ss  = st.session_state
    now = time.time()
    mins = int((now - ss.drive_start) / 60)

    hero("🚗 DrowsyDriver AI",
         "Phát hiện buồn ngủ thời gian thực — MediaPipe EAR/MAR · Binaural Beats Beta",
         badge=f"⏱ {mins//60:02d}:{mins%60:02d}")

    # ── Auto-refresh khi đang giám sát ──────────────────────────────────────
    if AUTOREFRESH_OK and ss.monitoring:
        st_autorefresh(interval=1500, key="ar")
    elif ss.monitoring and not AUTOREFRESH_OK:
        st.caption("⚠️ `streamlit-autorefresh` chưa cài — giám sát thủ công. "
                   "Thêm `streamlit-autorefresh>=0.0.1` vào requirements.txt.")

    c1, c2 = st.columns([3, 2], gap="medium")
    vp          = None
    vid_results = None

    # ────────────────────────────────────────────────────────────────────────
    # Cột trái — khu vực camera
    # ────────────────────────────────────────────────────────────────────────
    with c1:
        modes = ["📸 Chụp ảnh", "📹 Upload video"]
        if WEBRTC_OK:
            modes.append("🎥 WebRTC")
        if ss.cam_mode not in modes:
            ss.cam_mode = modes[0]
        ss.cam_mode = st.radio(
            "Chế độ", modes,
            index=modes.index(ss.cam_mode),
            horizontal=True,
            label_visibility="collapsed")

        # ── 📸 Chụp ảnh ─────────────────────────────────────────────────────
        if ss.cam_mode == "📸 Chụp ảnh":
            # Hiển thị live webcam (JS — chỉ visual, không phân tích)
            st.components.v1.html(WEBCAM_HTML, height=266)

            snap = st.camera_input("📸 Chụp",
                                   key="cam_snap",
                                   label_visibility="collapsed")
            if snap is not None:
                ss.last_snap_b64 = snap.getvalue()   # bytes → tồn tại qua các rerun

            # Phân tích ảnh (từ snap mới HOẶC từ cache khi auto-refresh)
            if ss.last_snap_b64:
                from PIL import Image
                pil = Image.open(io.BytesIO(ss.last_snap_b64))
                s_status, s_ear, s_mar, s_img = analyze_snapshot(
                    pil, ss.ear_thr, ss.mar_thr)
                ss.last_status = s_status
                ss.last_ear    = s_ear
                ss.last_mar    = s_mar

                # Hiển thị ảnh chỉ khi vừa chụp mới
                if snap is not None:
                    st.image(cv2.cvtColor(s_img, cv2.COLOR_BGR2RGB),
                             caption=f"EAR {s_ear:.3f} · MAR {s_mar:.3f} → {s_status}",
                             use_container_width=True)

                # ── Logic cảnh báo (có cooldown 20s tránh trigger liên tục) ──
                if ss.monitoring:
                    # Phát hiện nhắm mắt / DROWSY
                    if s_status == "DROWSY" and (now - ss.last_drowsy_ts) > 20:
                        ss.last_drowsy_ts = now
                        ss.drowsy_count  += 1
                        bump_hour()
                        ss.history.append((time.strftime("%H:%M:%S"), "Nhắm mắt", round(s_ear, 3)))
                        ss.page = "Alert"
                        st.rerun()

                    # Phát hiện ngáp (edge-detect: chỉ tính khi bắt đầu ngáp)
                    is_yawning = s_mar > ss.mar_thr
                    if is_yawning and not ss.was_yawning:
                        ss.yawn_count += 1
                        ss.history.append((time.strftime("%H:%M:%S"), "Ngáp", round(s_mar, 2)))
                    ss.was_yawning = is_yawning

        # ── 📹 Upload video ─────────────────────────────────────────────────
        elif ss.cam_mode == "📹 Upload video":
            vid_up = st.file_uploader(
                "Chọn file video (mp4 / avi / mov / mkv)",
                type=["mp4", "avi", "mov", "mkv", "webm"])
            if vid_up is not None:
                vk = f"_vid_{vid_up.name}_{vid_up.size}"
                if vk not in ss:
                    with st.spinner(f"🔍 Đang phân tích {vid_up.name} — lấy mẫu 2 FPS…"):
                        ss[vk] = analyze_video(vid_up, ss.ear_thr, ss.mar_thr)
                vr, vh, vtot, vfps = ss[vk]
                if vr:
                    import pandas as pd
                    cnt = {s: sum(1 for r in vr if r["status"] == s)
                           for s in ("ALERT","WARNING","DROWSY","NO FACE")}
                    overall = ("DROWSY" if cnt["DROWSY"]>0
                               else "WARNING" if cnt["WARNING"]>0 else "ALERT")
                    ss.last_status = overall
                    vid_results = (vr, vh, cnt, vtot, vfps)
                    if overall == "DROWSY": bump_hour()

                    dfv = pd.DataFrame(
                        [{"s": r["t"], "EAR": r["ear"], "MAR": r["mar"]}
                         for r in vr if r["status"] != "NO FACE"]).set_index("s")
                    if not dfv.empty:
                        st.caption(f"📈 EAR & MAR — {len(vr)} khung phân tích "
                                   f"({round(vtot/vfps) if vfps else 0}s)")
                        st.line_chart(dfv, height=180, color=["#FC1C46","#3b82f6"])

                    if vh:
                        with st.expander(f"⚠️ {len(vh)} frame cần chú ý", expanded=True):
                            gc = st.columns(min(len(vh), 3))
                            for gi, hf in enumerate(vh):
                                gc[gi%3].image(hf["img"],
                                               caption=f"{hf['t']}s · {hf['status']}",
                                               use_container_width=True)
                    else:
                        st.success("✅ Không phát hiện buồn ngủ trong video.")
                else:
                    st.warning("Không đọc được video hoặc không tìm thấy khuôn mặt.")

        # ── 🎥 WebRTC ────────────────────────────────────────────────────────
        else:
            try:
                ctx = webrtc_streamer(
                    key="cam", mode=WebRtcMode.SENDRECV,
                    video_processor_factory=Processor,
                    rtc_configuration=RTC_CONFIG,
                    media_stream_constraints={"video": True, "audio": False},
                    async_processing=True)
                if ctx and ctx.video_processor:
                    vp = ctx.video_processor
                    vp.ear_thr = ss.ear_thr; vp.mar_thr = ss.mar_thr
                    vp.closed_sec = ss.closed_sec
                    ss.last_status = vp.status
                    ss.last_ear    = vp.ear
                    ss.last_mar    = vp.mar
            except Exception as ex:
                st.warning(f"⚠️ WebRTC lỗi — dùng chế độ Chụp ảnh. ({ex})")

    # ────────────────────────────────────────────────────────────────────────
    # Cột phải — trạng thái & thống kê
    # ────────────────────────────────────────────────────────────────────────
    with c2:
        if vid_results:
            # Video mode: hiện tổng hợp video
            vr, vh, cnt, vtot, vfps = vid_results
            n     = len(vr)
            dur   = round(vtot / vfps) if vfps else 0
            pct_d = round(cnt["DROWSY"]  / n * 100) if n else 0
            pct_w = round(cnt["WARNING"] / n * 100) if n else 0
            r1, r2 = st.columns(2)
            with r1: metric_card("Thời lượng video", f"{dur}s")
            with r2: metric_card("Frame phân tích", n)
            r3, r4 = st.columns(2)
            with r3: metric_card("Tỉ lệ buồn ngủ", f"{pct_d}%")
            with r4: metric_card("Tỉ lệ cảnh báo", f"{pct_w}%")
        else:
            # Live / photo mode: điều khiển & số liệu
            ss.monitoring = st.toggle(
                "🟢  KÍCH HOẠT GIÁM SÁT",
                value=ss.monitoring, key="mon_toggle")
            r1, r2 = st.columns(2)
            with r1: metric_card("Buồn ngủ", ss.drowsy_count)
            with r2: metric_card("Ngáp", ss.yawn_count)

        # Thanh trạng thái (chung cho cả 3 mode)
        status_bar(ss.last_status)

        # EAR / MAR gauge (chỉ khi có dữ liệu real-time)
        if not vid_results:
            ear_col = "#FC1C46" if ss.last_ear < ss.ear_thr and ss.last_ear > 0 else "#22c55e"
            mar_col = "#FC1C46" if ss.last_mar > ss.mar_thr else "#3b82f6"
            gauge("EAR — mắt",       ss.last_ear, 0.42, color=ear_col)
            gauge("MAR — miệng/ngáp", ss.last_mar, 1.20, color=mar_col)
            st.caption(f"Ngưỡng: EAR ≤ {ss.ear_thr:.2f} · MAR ≥ {ss.mar_thr:.2f}")
            st.divider()

            # Nhắc nghỉ 2h / 4h (CK AI §1.3)
            for mark in (2, 4):
                if mins >= mark * 60 and mark not in ss.rest_done:
                    ss.rest_done.add(mark)
                    st.warning(f"⏰ Đã lái **{mark} giờ** liên tục — nghỉ 15–30 phút!")

            col_btn1, col_btn2 = st.columns(2)
            with col_btn1:
                if st.button("🔄 Phiên mới", use_container_width=True,
                             help="Reset bộ đếm và timer"):
                    for k in ("drive_start","drowsy_count","yawn_count","history",
                              "rest_done","last_status","last_ear","last_mar",
                              "last_snap_b64","last_drowsy_ts","was_yawning"):
                        if k == "drive_start":   ss.drive_start    = time.time()
                        elif k == "rest_done":   ss.rest_done      = set()
                        elif k == "drowsy_count":ss.drowsy_count   = 0
                        elif k == "yawn_count":  ss.yawn_count     = 0
                        elif k == "history":     ss.history        = []
                        elif k == "last_status": ss.last_status    = "—"
                        elif k in ("last_ear","last_mar","last_drowsy_ts"): ss[k] = 0.0
                        elif k == "last_snap_b64": ss.last_snap_b64 = None
                        elif k == "was_yawning": ss.was_yawning = False
                    st.rerun()
            with col_btn2:
                if st.button("📊 Xem thống kê", use_container_width=True):
                    ss.page = "Analytics"; st.rerun()

        # WebRTC: theo dõi sự kiện
        if ss.monitoring and vp and vp.event:
            ts = time.strftime("%H:%M:%S")
            ss.history.append((ts, vp.event[0], vp.event[1]))
            if vp.event[0] == "Nhắm mắt":
                ss.drowsy_count += 1; bump_hour()
            else:
                ss.yawn_count += 1
        if ss.monitoring and vp and vp.status == "DROWSY":
            ss.page = "Alert"; st.rerun()

    # ── Thanh thông tin dưới cùng ───────────────────────────────────────────
    if not vid_results:
        if ss.monitoring:
            mode_info = ("Tự động làm mới mỗi 1.5 giây" if AUTOREFRESH_OK
                         else "Nhấn Chụp để phân tích")
            st.info(f"🟢 **Đang giám sát** — {mode_info}. "
                    f"Giữ khuôn mặt trong khung hình. "
                    f"Nhắm mắt >{ss.closed_sec:.0f}s hoặc ngáp sẽ kích hoạt cảnh báo.")
        else:
            st.info("💡 Bật **Kích hoạt giám sát** → hệ thống tự phân tích. "
                    "Dùng **Upload video** để phân tích clip ghi sẵn.")


# ═════════════════════════════════════════════════════════════════════════════
# PHẦN 9: TRANG ALERT
# ═════════════════════════════════════════════════════════════════════════════
def page_alert():
    ss = st.session_state
    mins = int((time.time() - ss.drive_start) / 60)

    st.markdown(f"""
    <div class="alert-box">
      <h1>⚠️ CẢNH BÁO BUỒN NGỦ ⚠️</h1>
      <p>Phát hiện dấu hiệu nguy hiểm — hãy dừng xe khi đến nơi an toàn!</p>
      <p style="font-size:13px;margin-top:8px">
        Phiên lái: {mins//60}h {mins%60}m &nbsp;·&nbsp;
        Buồn ngủ: <strong>{ss.drowsy_count} lần</strong> &nbsp;·&nbsp;
        Ngáp: <strong>{ss.yawn_count} lần</strong>
      </p>
    </div>""", unsafe_allow_html=True)

    play_sound(SOUNDS[ss.sound])

    # ── Nút hành động ────────────────────────────────────────────────────────
    b1, b2, b3 = st.columns(3)
    with b1:
        if st.button("✅  TÔI ỔN — TIẾP TỤC",
                     use_container_width=True, type="primary"):
            ss.page = "Dashboard"; st.rerun()
    with b2:
        sos = ss.get("sos_contact", "113") or "113"
        st.link_button(f"📞  GỌI {sos}",
                       f"tel:{sos}", use_container_width=True)
    with b3:
        st.link_button("🅿️  TÌM TRẠM DỪNG",
                       "https://www.google.com/maps/search/rest+area+near+me",
                       use_container_width=True)

    st.divider()

    col_note, col_snd = st.columns([3, 2])
    with col_note:
        st.markdown("#### 🛡️ Hướng dẫn an toàn ngay lập tức")
        st.markdown("""
- 🚨 **Bật đèn khẩn nguy**, giảm tốc độ từ từ
- 🅿️ **Dừng xe ở vị trí an toàn** — lề đường, trạm xăng, bãi đỗ
- 💤 **Nghỉ ngơi tối thiểu 15–20 phút** (không thể thay thế bằng cà phê)
- 💧 Uống nước lạnh, rửa mặt, hít thở sâu vài lần
- 📞 **Thông báo cho người thân** nếu bạn cảm thấy không ổn
- 🔄 Sau nghỉ: bấm **"Phiên mới"** để reset bộ đếm
        """)
    with col_snd:
        st.markdown("#### 🔊 Âm thanh đang phát")
        snd_meta = {
            "Binaural beta 18 Hz (khuyến nghị)": ("18 Hz", "Beta giữa — tỉnh táo tối ưu"),
            "Binaural beta 14 Hz":               ("14 Hz", "Beta thấp — kích thích nhẹ"),
            "Binaural beta 21 Hz":               ("21 Hz", "Beta cao — cảnh giác mạnh"),
            "Beep cảnh báo":                     ("—",     "Tín hiệu chuẩn"),
        }
        nm = ss.sound
        hz, desc = snd_meta.get(nm, ("—", nm))
        st.markdown(f"**{nm}**")
        st.caption(f"Nhịp {hz} · {desc}")
        st.caption("🎧 Đeo tai nghe để hiệu quả tốt nhất")
        st.write("")
        if st.button("⚙️ Đổi âm thanh", use_container_width=True):
            ss.page = "Analytics"; st.rerun()
        if st.button("🔇 Tắt & Quay lại", use_container_width=True):
            ss.page = "Dashboard"; st.rerun()


# ═════════════════════════════════════════════════════════════════════════════
# PHẦN 10: TRANG ANALYTICS
# ═════════════════════════════════════════════════════════════════════════════
def page_analytics():
    import pandas as pd
    ss   = st.session_state
    now  = time.time()
    mins = int((now - ss.drive_start) / 60)
    rate = ss.drowsy_count / max(mins / 60, 0.01)   # lần/giờ

    hero("📊 Thống kê & Cài đặt",
         "Phân tích phiên lái · Khung giờ nguy hiểm · Điều chỉnh AI",
         badge=f"🗓 {time.strftime('%d/%m/%Y')}")

    # ── Tổng quan phiên ──────────────────────────────────────────────────────
    st.markdown("### 📋 Tổng quan phiên lái")
    m1, m2, m3, m4 = st.columns(4)
    with m1: metric_card("Thời gian lái",       f"{mins//60}h {mins%60}m")
    with m2: metric_card("Cảnh báo buồn ngủ",   ss.drowsy_count)
    with m3: metric_card("Lần ngáp",             ss.yawn_count)
    with m4: metric_card("Tần suất",             f"{rate:.1f}/h")

    # ── Lịch sử sự kiện ──────────────────────────────────────────────────────
    st.divider()
    st.markdown("### 📝 Lịch sử sự kiện trong phiên")
    if ss.history:
        df = pd.DataFrame(ss.history, columns=["Thời gian", "Loại sự kiện", "Giá trị"])
        st.dataframe(df, use_container_width=True, hide_index=True, height=200)

        ev_col, act_col = st.columns([3, 1])
        with ev_col:
            vc = df["Loại sự kiện"].value_counts().reset_index()
            vc.columns = ["Loại", "Số lần"]
            st.bar_chart(vc.set_index("Loại"), color="#FC1C46", height=140)
        with act_col:
            st.write("")
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Xuất CSV",
                data=csv,
                file_name=f"drowsy_{time.strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",
                use_container_width=True)
            if st.button("🗑️ Xóa lịch sử", use_container_width=True):
                ss.history = []; st.rerun()
    else:
        st.caption("Chưa có sự kiện. Bật **Giám sát** ở Dashboard và chụp/quay ảnh khuôn mặt.")

    # ── Khung giờ hay buồn ngủ ────────────────────────────────────────────────
    st.divider()
    st.markdown("### 🕐 Khung giờ hay buồn ngủ")
    hours = ss.hours
    total = sum(hours)
    if total == 0:
        st.info("📭 Chưa có dữ liệu. Hệ thống tự học qua các phiên lái — mỗi lần "
                "phát hiện buồn ngủ, giờ hiện tại được cộng vào biểu đồ.")
    else:
        peak    = max(range(24), key=lambda i: hours[i])
        peak_n  = (peak + 1) % 24
        top3    = sorted(range(24), key=lambda i: hours[i], reverse=True)[:3]
        top3_s  = " · ".join(f"{h:02d}h" for h in top3 if hours[h] > 0)

        wa, wb = st.columns([5, 3])
        with wa:
            st.error(f"🚨 **Nguy hiểm nhất: {peak:02d}:00 – {peak_n:02d}:00** — "
                     f"{hours[peak]} lần ({round(hours[peak]/total*100)}% tổng)")
        with wb:
            st.warning(f"⚠️ Top 3 giờ nguy hiểm: **{top3_s}**")

        dfh = pd.DataFrame({"Số lần buồn ngủ": hours},
                           index=[f"{h:02d}h" for h in range(24)])
        st.bar_chart(dfh, color="#FC1C46", height=200)

        # Khuyến nghị thông minh
        if 0 <= peak <= 5:
            rec = ("Khung giờ rạng sáng (0–5h) là lúc nhịp sinh học xuống thấp nhất. "
                   "**Tuyệt đối tránh** lái xe một mình xuyên đêm.")
        elif 13 <= peak <= 15:
            rec = ("Khung giờ sau bữa trưa (13–15h) dễ buồn ngủ sinh lý. "
                   "**Nghỉ 15–20 phút** trước khi lái sẽ giảm nguy cơ đáng kể.")
        elif 22 <= peak or peak == 0:
            rec = ("Buổi tối muộn sau 22h cơ thể đã mệt. "
                   "Hãy **đổi tài xế** hoặc nghỉ lại thay vì tiếp tục lái.")
        else:
            rec = (f"Tránh lái xe trong khung {peak:02d}:00–{peak_n:02d}:00 "
                   "nếu đã thức dậy sớm hoặc làm việc căng thẳng trước đó.")
        st.info(f"💡 **Gợi ý:** {rec}")

        if st.button("🗑️ Xóa dữ liệu khung giờ"):
            ss.hours = [0]*24; save_hours(ss.hours); st.rerun()

    # ── Cài đặt AI ───────────────────────────────────────────────────────────
    st.divider()
    st.markdown("### ⚙️ Cài đặt độ nhạy AI")
    s1, s2 = st.columns(2)
    with s1:
        ss.ear_thr = st.slider(
            "Ngưỡng EAR — phát hiện nhắm mắt",
            0.10, 0.35, ss.ear_thr, 0.01,
            help="Giá trị nhỏ hơn: cần nhắm mắt sâu hơn mới báo (ít nhạy hơn)")
        ss.closed_sec = st.slider(
            "Thời gian nhắm mắt liên tục (giây)",
            0.5, 4.0, ss.closed_sec, 0.5,
            help="Lớn hơn = cần nhắm lâu hơn mới kích hoạt cảnh báo")
    with s2:
        ss.mar_thr = st.slider(
            "Ngưỡng MAR — phát hiện ngáp",
            0.40, 1.00, ss.mar_thr, 0.05,
            help="Lớn hơn: cần mở miệng rộng hơn mới tính là ngáp")
        ss.sos_contact = st.text_input(
            "Số điện thoại SOS khẩn cấp",
            value=ss.sos_contact or "113",
            help="Hiển thị nút Gọi trên màn hình Alert")

    # ── Âm thanh ────────────────────────────────────────────────────────────
    st.divider()
    st.markdown("### 🔊 Âm thanh cảnh báo — Binaural Beats")
    st.caption(
        "Cơ chế (CK AI §1.3): tai trái nghe sóng mang f Hz, tai phải nghe f+beat Hz — "
        "não nhận ra nhịp beat Hz và đồng bộ sóng dải beta (13–21 Hz) → kích thích tỉnh táo. "
        "**Đeo tai nghe để hiệu quả cao nhất.**")

    snd_l, snd_r = st.columns([3, 2])
    with snd_l:
        if ss.sound not in SOUNDS:
            ss.sound = list(SOUNDS.keys())[0]
        ss.sound = st.radio("Chọn âm thanh cảnh báo",
                            list(SOUNDS.keys()),
                            index=list(SOUNDS.keys()).index(ss.sound))
    with snd_r:
        st.write("")
        notes = {
            "Binaural beta 18 Hz (khuyến nghị)": "18 Hz · Tối ưu tỉnh táo",
            "Binaural beta 14 Hz":               "14 Hz · Nhẹ nhàng, dễ chịu",
            "Binaural beta 21 Hz":               "21 Hz · Kích thích mạnh nhất",
            "Beep cảnh báo":                     "Chuẩn · Dễ nhận biết nhất",
        }
        st.caption(notes.get(ss.sound, ""))
        if st.button("▶️ Nghe thử", use_container_width=True):
            play_sound(SOUNDS[ss.sound], loop=False)


# ═════════════════════════════════════════════════════════════════════════════
# PHẦN 11: TRANG HƯỚNG DẪN
# ═════════════════════════════════════════════════════════════════════════════
def page_guide():
    ss = st.session_state
    hero("📖 Hướng dẫn sử dụng",
         "Thuật toán EAR/MAR · Binaural Beats Beta · Cách dùng hiệu quả nhất")

    t1, t2, t3 = st.tabs(["🚀 Cách sử dụng", "🔬 Thuật toán AI", "🔊 Binaural Beats"])

    with t1:
        st.markdown(f"""
#### Bước 1 — Dashboard
Vào màn hình **Dashboard** → chọn chế độ **📸 Chụp ảnh** (khuyến nghị cho web).

#### Bước 2 — Bật giám sát
Nhấn **🟢 KÍCH HOẠT GIÁM SÁT**. Hệ thống sẽ tự làm mới mỗi **1.5 giây**
(cần `streamlit-autorefresh` — đã có trong requirements.txt).

#### Bước 3 — Chụp ảnh
- Webcam bên trên hiển thị live feed trực tiếp (không phân tích — chỉ để xem)
- Nhấn nút chụp bên dưới → ảnh được lưu và phân tích
- Khi giám sát bật, **hệ thống tự click nút chụp** qua JavaScript mỗi 1.5s

#### Bước 4 — Nhận cảnh báo
- **EAR < {ss.ear_thr:.2f}** → mắt nhắm quá {ss.closed_sec:.0f}s → chuyển sang màn **Alert** 🔴
- **MAR > {ss.mar_thr:.2f}** → ghi nhận ngáp ⚡ (không chuyển màn)
- Âm thanh binaural beta tự phát khi vào Alert

#### Bước 5 — Phân tích video (tùy chọn)
Chọn **📹 Upload video** → tải lên clip ghi sẵn → xem biểu đồ EAR/MAR theo thời gian
và gallery các frame cảnh báo.

#### Lưu ý triển khai
| Nền tảng | Trạng thái |
|----------|-----------|
| Hugging Face Spaces (SDK Streamlit) | ✅ Hoạt động đầy đủ |
| Streamlit Community Cloud | ✅ Cần Python 3.12 (xem STREAMLIT_DEPLOY.md) |
| Docker (HF Spaces SDK Docker) | ✅ PORT=7860 đã được hỗ trợ |
| Chạy local | ✅ `py -3.12 -m streamlit run streamlit_app.py` |
        """)
        st.info("📱 **Phiên bản Android** bổ sung thêm: giám sát liên tục bằng camera native, "
                "cảnh báo rung, GPS tự động gửi vị trí SOS khi vượt ngưỡng buồn ngủ.")

    with t2:
        st.markdown("#### Eye Aspect Ratio (EAR)")
        st.latex(r"EAR = \frac{\|p_2-p_6\| + \|p_3-p_5\|}{2\,\|p_1-p_4\|}")
        st.markdown("""
| Trạng thái | Giá trị EAR điển hình |
|------------|----------------------|
| Mắt mở hoàn toàn | 0.28 – 0.38 |
| Mắt nửa nhắm (mệt) | 0.18 – 0.24 |
| Mắt nhắm (ngủ) | < 0.15 |

- Dùng **6 điểm landmark** quanh mỗi mắt từ MediaPipe FaceLandmarker (478 điểm tổng)
- Lấy trung bình hai mắt (trái + phải) để giảm nhiễu do góc nghiêng đầu
        """)
        st.markdown("#### Mouth Aspect Ratio (MAR)")
        st.latex(r"MAR = \frac{\|p_{top}-p_{bot}\|}{\|p_{left}-p_{right}\|}")
        st.markdown("""
| Trạng thái | Giá trị MAR điển hình |
|------------|----------------------|
| Miệng đóng | < 0.30 |
| Nói chuyện / há miệng | 0.35 – 0.55 |
| **Ngáp** | **> 0.60** |

- Dùng 4 điểm landmark môi trong (trên, dưới, trái, phải)
        """)
        st.markdown("#### Pipeline toàn bộ")
        st.markdown("""
```
Camera frame → RGB 640×480
  → MediaPipe FaceLandmarker (float16, ~3.7MB)
  → 478 điểm landmark khuôn mặt
  → EAR (trái) + EAR (phải) → trung bình
  → MAR (môi trong)
  → So sánh ngưỡng → ALERT / WARNING / DROWSY
  → Kích hoạt cảnh báo (âm thanh + màn Alert)
```
        """)

    with t3:
        st.markdown("""
#### Cơ chế Binaural Beats

Hai tai nghe hai tần số **khác nhau một khoảng nhỏ**:

| Tai | Tần số |
|-----|--------|
| Trái | 200 Hz (sóng mang) |
| Phải | 200 + **beat** Hz |

Não tổng hợp → cảm nhận nhịp ảo **beat Hz** → đồng bộ sóng não.

#### Dải Beta (13–21 Hz) — Kích thích tỉnh táo

| Tần số beat | Mô tả | Khuyến nghị |
|-------------|-------|-------------|
| 14 Hz | Beta thấp — thư giãn tỉnh táo | Lái xe đường dài |
| **18 Hz** | **Beta giữa — tập trung tối ưu** | **✅ Mặc định** |
| 21 Hz | Beta cao — cảnh giác mạnh | Buồn ngủ nặng |

**Nguồn khoa học:** Moessinger (2021), *Effects of binaural beats on cognitive performance*.

#### Lưu ý khi dùng
- **Đeo tai nghe** để não nhận đúng tần số riêng biệt từng tai
- Trên **loa đơn** vẫn tạo *acoustic beating* nghe được (nhịp nhịp nhẹ)
- Không khuyến nghị cho người có tiền sử **động kinh** hoặc **tim mạch**
- Âm lượng vừa phải — không cần to để có hiệu quả
        """)


# ═════════════════════════════════════════════════════════════════════════════
# PHẦN 12: MAIN
# ═════════════════════════════════════════════════════════════════════════════
def main():
    init_state()
    st.markdown(CSS, unsafe_allow_html=True)
    ss   = st.session_state
    mins = int((time.time() - ss.drive_start) / 60)

    # ── Sidebar ──────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## 🚗 DrowsyDriver")
        st.caption("AI phát hiện buồn ngủ — IS54A 2026")
        st.divider()

        ss.page = st.radio(
            "nav", ["Dashboard", "Alert", "Analytics", "Hướng dẫn"],
            index=["Dashboard","Alert","Analytics","Hướng dẫn"].index(ss.page),
            label_visibility="collapsed")

        st.divider()

        # Live status chip
        s_bg = {"ALERT":"#1a8836","WARNING":"#c49500","DROWSY":"#d92030"}.get(
                ss.last_status, "#303035")
        s_lb = {"ALERT":"✅ Tỉnh táo","WARNING":"⚡ Chú ý","DROWSY":"🔴 Buồn ngủ!"}.get(
                ss.last_status, "🔍 Chưa quét")
        st.markdown(
            f'<div style="background:{s_bg};border-radius:10px;padding:7px 12px;'
            f'text-align:center;color:#fff;font-weight:700;font-size:13px">'
            f'{s_lb}</div>',
            unsafe_allow_html=True)
        st.write("")
        st.caption(f"⏱ Lái liên tục: {mins//60}h {mins%60}m")
        st.caption(f"🔴 {ss.drowsy_count} cảnh báo  ·  😮 {ss.yawn_count} ngáp")
        st.divider()
        mon_icon = "🟢" if ss.monitoring else "⚫"
        st.caption(f"{mon_icon} Giám sát: {'Đang bật' if ss.monitoring else 'Tắt'}")
        st.caption(f"📸 {ss.cam_mode}")
        st.caption(f"🔊 {ss.sound[:22]}…" if len(ss.sound)>22 else f"🔊 {ss.sound}")
        st.divider()
        # Badges thư viện
        def badge(name, ok):
            c = "#1a8836" if ok else "#555"
            t = "✅" if ok else "—"
            return f'<span style="background:{c};color:#fff;border-radius:5px;padding:1px 6px;font-size:10px;margin-right:3px">{t} {name}</span>'
        st.markdown(
            badge("Core",        CORE_OK) +
            badge("AutoRefresh", AUTOREFRESH_OK) +
            badge("WebRTC",      WEBRTC_OK),
            unsafe_allow_html=True)

    # ── Core check ───────────────────────────────────────────────────────────
    if not CORE_OK:
        st.error(f"❌ Thiếu thư viện lõi (opencv + mediapipe): {CORE_ERR}")
        st.code("pip install -r requirements.txt")
        return

    # ── Routing ─────────────────────────────────────────────────────────────
    {
        "Dashboard":  page_dashboard,
        "Alert":      page_alert,
        "Analytics":  page_analytics,
        "Hướng dẫn": page_guide,
    }[ss.page]()


if __name__ == "__main__":
    main()
