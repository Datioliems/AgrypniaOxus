const MODEL_URL = "/app/src/main/assets/face_landmarker.task";
const CNN_EYE_URL = "./models/cnn_eye_weights.json";
const STORAGE_KEY = "drowsyGuardIphonePwaEvents";
const SOUND_SETTING_KEY = "drowsyGuardIphonePwaSoundEnabled";
const MEDIAPIPE_VERSION = "0.10.14";
const CNN_EYE_INTERVAL_MS = 500;

const els = {};
let faceLandmarker = null;
let cnnEyeModel = null;
let cnnEyeLoading = false;
let cnnEyeCanvas = null;
let latestCnnEye = null;
let lastCnnEyeAt = 0;
let mediaStream = null;
let running = false;
let animationHandle = null;
let lastFrameAt = 0;
let fps = 0;
let closedStartedAt = null;
let yawnStartedAt = null;
let lastLoggedState = "INIT";
let lastLogAt = 0;
let alertCount = 0;
let latestLocation = null;
let audioContext = null;
let soundEnabled = false;
let soundUnlocked = false;
let arousalContext = null;
let arousalNodes = [];
let arousalTimer = null;
let events = loadEvents();

const settings = {
  earThreshold: 0.21,
  marThreshold: 0.58,
  drowsyDurationMs: 1600,
};

document.addEventListener("DOMContentLoaded", () => {
  bindElements();
  restoreSoundPreference();
  bindEvents();
  updateSecureBadge();
  restoreSettings();
  renderEvents();
  renderHourChart();
  updateRisk();
  updateSoundStatus();
  loadCnnEye().catch(() => {});
  registerServiceWorker();
});

function bindElements() {
  [
    "runMode", "secureBadge", "video", "snapshot", "overlayCanvas", "stateDot",
    "stateText", "stateDetail", "earValue", "marValue", "fpsValue", "latencyValue",
    "cnnEyeValue", "cnnConfidenceValue",
    "cameraButton", "stopCameraButton", "snapshotInput", "awakeButton", "warningButton",
    "dangerButton", "loadAiButton", "cameraHelp", "soundStatus", "enableSoundButton",
    "loadCnnButton", "muteSoundButton", "beepButton", "arousalButton", "stopArousalButton", "arousalText",
    "hourChart", "riskScore", "alertCount",
    "closedTime", "yawnTime", "recommendation", "earThreshold", "marThreshold",
    "drowsyDuration", "earThresholdValue", "marThresholdValue", "drowsyDurationValue",
    "trustedContact", "locationButton", "shareButton", "locationText", "exportButton",
    "clearLogButton", "events"
  ].forEach((id) => {
    els[id] = document.getElementById(id);
  });
}

function bindEvents() {
  els.cameraButton.addEventListener("click", startCamera);
  els.stopCameraButton.addEventListener("click", stopCamera);
  els.snapshotInput.addEventListener("change", loadSnapshot);
  els.awakeButton.addEventListener("click", () => setManualState("Awake", "Manual awake check.", "safe"));
  els.warningButton.addEventListener("click", () => setManualState("Drowsy warning", "Manual warning event.", "warning"));
  els.dangerButton.addEventListener("click", () => setManualState("High risk", "Manual high-risk event.", "danger"));
  els.loadAiButton.addEventListener("click", loadMediaPipe);
  els.loadCnnButton.addEventListener("click", loadCnnEye);
  els.enableSoundButton.addEventListener("click", enableSoundMode);
  els.muteSoundButton.addEventListener("click", disableSoundMode);
  els.beepButton.addEventListener("click", async () => {
    if (await ensureSoundMode()) playTestBeep();
  });
  els.arousalButton.addEventListener("click", async () => {
    await ensureSoundMode();
    startArousal();
  });
  els.stopArousalButton.addEventListener("click", stopArousal);
  els.locationButton.addEventListener("click", requestLocation);
  els.shareButton.addEventListener("click", shareTrustedAlert);
  els.exportButton.addEventListener("click", exportCsv);
  els.clearLogButton.addEventListener("click", clearLogs);

  els.earThreshold.addEventListener("input", () => {
    settings.earThreshold = Number(els.earThreshold.value);
    saveSettings();
    updateSettingLabels();
  });
  els.marThreshold.addEventListener("input", () => {
    settings.marThreshold = Number(els.marThreshold.value);
    saveSettings();
    updateSettingLabels();
  });
  els.drowsyDuration.addEventListener("input", () => {
    settings.drowsyDurationMs = Number(els.drowsyDuration.value);
    saveSettings();
    updateSettingLabels();
  });
}

function updateSecureBadge() {
  if (window.isSecureContext) {
    els.secureBadge.textContent = "HTTPS OK";
    els.secureBadge.classList.add("ok");
    els.cameraHelp.textContent = "Trang đang ở secure context. iPhone Safari có thể mở realtime camera nếu đã cấp quyền.";
  } else {
    els.secureBadge.textContent = "Need HTTPS";
    els.secureBadge.classList.add("danger");
    els.cameraHelp.textContent = "Realtime camera trên iPhone Safari cần HTTPS. Dùng ngrok/Vercel/Netlify hoặc dùng snapshot fallback.";
  }
}

function registerServiceWorker() {
  if ("serviceWorker" in navigator) {
    navigator.serviceWorker.register("./service-worker.js").catch(() => {});
  }
}

async function startCamera() {
  if (!window.isSecureContext) {
    updateState("Need HTTPS", "Safari chỉ mở realtime camera qua HTTPS. Dùng ngrok hoặc snapshot fallback.", "danger");
    addEvent("Camera blocked", "Realtime camera needs HTTPS on iPhone Safari.", "danger", "PWA");
    return;
  }

  if (!navigator.mediaDevices?.getUserMedia) {
    updateState("Camera unsupported", "Trình duyệt không hỗ trợ getUserMedia.", "danger");
    addEvent("Camera unsupported", "Use snapshot fallback or iOS native app.", "danger", "PWA");
    return;
  }

  try {
    mediaStream = await navigator.mediaDevices.getUserMedia({
      video: {
        facingMode: "user",
        width: { ideal: 640 },
        height: { ideal: 480 },
      },
      audio: false,
    });
    els.video.srcObject = mediaStream;
    els.video.hidden = false;
    els.snapshot.hidden = true;
    await els.video.play();
    running = true;
    els.runMode.textContent = faceLandmarker ? "MediaPipe" : "Camera";
    updateState("Camera ready", faceLandmarker ? "MediaPipe đang phân tích EAR/MAR." : "Camera đã mở. Bấm Tải MediaPipe để bật phân tích.", "safe");
    loop();
  } catch (error) {
    updateState("Camera denied", "Kiểm tra quyền Camera trong Safari và đảm bảo dùng HTTPS.", "danger");
    addEvent("Camera denied", error.message || "Camera permission failed.", "danger", "PWA");
  }
}

function stopCamera() {
  running = false;
  if (animationHandle) {
    cancelAnimationFrame(animationHandle);
    animationHandle = null;
  }
  if (mediaStream) {
    mediaStream.getTracks().forEach((track) => track.stop());
    mediaStream = null;
  }
  clearCanvas();
  updateState("Camera stopped", "Camera đã tắt.", "safe");
  els.runMode.textContent = "PWA ready";
}

async function loadMediaPipe() {
  if (faceLandmarker) {
    updateState("MediaPipe ready", "Face Landmarker đã sẵn sàng.", "safe");
    return;
  }

  els.loadAiButton.disabled = true;
  els.loadAiButton.textContent = "Đang tải...";
  updateState("Loading AI", "Đang tải MediaPipe Tasks Vision từ CDN.", "warning");

  try {
    const vision = await import(`https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MEDIAPIPE_VERSION}`);
    const fileset = await vision.FilesetResolver.forVisionTasks(
      `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MEDIAPIPE_VERSION}/wasm`
    );

    try {
      faceLandmarker = await vision.FaceLandmarker.createFromOptions(fileset, {
        baseOptions: {
          modelAssetPath: MODEL_URL,
          delegate: "GPU",
        },
        runningMode: "VIDEO",
        numFaces: 1,
      });
    } catch (gpuError) {
      faceLandmarker = await vision.FaceLandmarker.createFromOptions(fileset, {
        baseOptions: {
          modelAssetPath: MODEL_URL,
        },
        runningMode: "VIDEO",
        numFaces: 1,
      });
    }

    els.runMode.textContent = "MediaPipe";
    updateState("MediaPipe ready", "EAR/MAR realtime đã sẵn sàng.", "safe");
    addEvent("MediaPipe ready", "Face Landmarker loaded in iPhone PWA.", "safe", "MediaPipe");
  } catch (error) {
    updateState("AI load failed", "Không tải được MediaPipe. Vẫn dùng được snapshot/manual demo.", "danger");
    addEvent("AI load failed", error.message || "MediaPipe CDN/model failed.", "danger", "MediaPipe");
  } finally {
    els.loadAiButton.disabled = false;
    els.loadAiButton.textContent = "Tải MediaPipe";
  }
}

async function loadCnnEye() {
  if (cnnEyeModel || cnnEyeLoading) {
    updateCnnDisplay(latestCnnEye);
    return;
  }

  cnnEyeLoading = true;
  els.loadCnnButton.disabled = true;
  els.loadCnnButton.textContent = "Đang tải CNN...";
  els.cnnEyeValue.textContent = "loading";
  els.cnnConfidenceValue.textContent = "--";

  try {
    const response = await fetch(CNN_EYE_URL, { cache: "force-cache" });
    if (!response.ok) throw new Error(`CNN weights HTTP ${response.status}`);
    const payload = await response.json();
    cnnEyeModel = prepareCnnEyeModel(payload);
    els.cnnEyeValue.textContent = "ready";
    els.cnnConfidenceValue.textContent = "web";
    addEvent("CNN Eye ready", "CNN eye classifier loaded in iPhone PWA.", "safe", "CNN");
  } catch (error) {
    els.cnnEyeValue.textContent = "unavailable";
    els.cnnConfidenceValue.textContent = "--";
    addEvent("CNN Eye failed", error.message || "Cannot load CNN weights.", "warning", "CNN");
  } finally {
    cnnEyeLoading = false;
    els.loadCnnButton.disabled = false;
    els.loadCnnButton.textContent = "Tải CNN";
  }
}

function prepareCnnEyeModel(payload) {
  return {
    inputSize: payload.inputSize || 64,
    classes: payload.classes || ["eyes_closed", "eyes_open"],
    layers: payload.layers.map((layer) => ({
      ...layer,
      kernel: layer.kernel ? new Float32Array(layer.kernel) : null,
      bias: layer.bias ? new Float32Array(layer.bias) : null,
    })),
  };
}

function loop() {
  if (!running) return;
  animationHandle = requestAnimationFrame(loop);

  const video = els.video;
  if (!faceLandmarker || video.readyState < 2) return;

  const startedAt = performance.now();
  const result = faceLandmarker.detectForVideo(video, startedAt);
  const latency = performance.now() - startedAt;
  updateFps(startedAt);

  if (!result.faceLandmarks?.length) {
    clearCanvas();
    updateState("No face", "Chưa thấy khuôn mặt. Đặt iPhone nhìn rõ mặt tài xế.", "warning");
    updateCnnDisplay(null);
    updateMetricDisplay(null, null, latency);
    return;
  }

  const landmarks = result.faceLandmarks[0];
  const metrics = computeMetrics(landmarks);
  const cnnEye = maybeRunCnnEye(video, landmarks);
  drawLandmarks(landmarks, metrics);
  handleMetrics(metrics.ear, metrics.mar, latency + (cnnEye?.latencyMs || 0), startedAt, cnnEye);
}

function computeMetrics(points) {
  const leftEar = eyeAspectRatio(points, [33, 160, 158, 133, 153, 144]);
  const rightEar = eyeAspectRatio(points, [362, 385, 387, 263, 373, 380]);
  const ear = (leftEar + rightEar) / 2;
  const mar = distance(points[13], points[14]) / Math.max(distance(points[78], points[308]), 0.001);
  return { ear, mar };
}

function eyeAspectRatio(points, ids) {
  const [outer, upper1, upper2, inner, lower2, lower1] = ids.map((id) => points[id]);
  const vertical1 = distance(upper1, lower1);
  const vertical2 = distance(upper2, lower2);
  const horizontal = distance(outer, inner);
  return (vertical1 + vertical2) / Math.max(2 * horizontal, 0.001);
}

function distance(a, b) {
  const dx = a.x - b.x;
  const dy = a.y - b.y;
  return Math.hypot(dx, dy);
}

function maybeRunCnnEye(video, landmarks) {
  if (!cnnEyeModel) return latestCnnEye;
  const now = performance.now();
  if (latestCnnEye && now - lastCnnEyeAt < CNN_EYE_INTERVAL_MS) return latestCnnEye;
  lastCnnEyeAt = now;

  const startedAt = performance.now();
  const left = cropEyeTensor(video, landmarks, [33, 160, 158, 133, 153, 144]);
  const right = cropEyeTensor(video, landmarks, [362, 385, 387, 263, 373, 380]);
  const tensors = [left, right].filter(Boolean);
  if (!tensors.length) {
    updateCnnDisplay(null);
    return null;
  }

  const avg = new Float32Array(cnnEyeModel.classes.length);
  tensors.forEach((tensor) => {
    const probs = runCnnEye(tensor);
    probs.forEach((value, index) => {
      avg[index] += value / tensors.length;
    });
  });

  let bestIndex = 0;
  for (let i = 1; i < avg.length; i += 1) {
    if (avg[i] > avg[bestIndex]) bestIndex = i;
  }

  latestCnnEye = {
    label: cnnEyeModel.classes[bestIndex],
    confidence: avg[bestIndex],
    probs: Array.from(avg),
    latencyMs: performance.now() - startedAt,
  };
  updateCnnDisplay(latestCnnEye);
  return latestCnnEye;
}

function cropEyeTensor(video, landmarks, ids) {
  const width = video.videoWidth || 640;
  const height = video.videoHeight || 480;
  const points = ids.map((id) => landmarks[id]).filter(Boolean);
  if (!points.length) return null;

  const xs = points.map((p) => p.x * width);
  const ys = points.map((p) => p.y * height);
  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const minY = Math.min(...ys);
  const maxY = Math.max(...ys);
  const centerX = (minX + maxX) / 2;
  const centerY = (minY + maxY) / 2;
  const side = Math.max(maxX - minX, (maxY - minY) * 2.6, 24) * 2.15;
  const sx = Math.max(0, Math.min(width - side, centerX - side / 2));
  const sy = Math.max(0, Math.min(height - side, centerY - side / 2));
  const sw = Math.min(side, width - sx);
  const sh = Math.min(side, height - sy);
  if (sw < 4 || sh < 4) return null;

  if (!cnnEyeCanvas) {
    cnnEyeCanvas = document.createElement("canvas");
    cnnEyeCanvas.width = cnnEyeModel.inputSize;
    cnnEyeCanvas.height = cnnEyeModel.inputSize;
  }
  const ctx = cnnEyeCanvas.getContext("2d", { willReadFrequently: true });
  ctx.drawImage(video, sx, sy, sw, sh, 0, 0, cnnEyeModel.inputSize, cnnEyeModel.inputSize);
  const pixels = ctx.getImageData(0, 0, cnnEyeModel.inputSize, cnnEyeModel.inputSize).data;
  const tensor = new Float32Array(cnnEyeModel.inputSize * cnnEyeModel.inputSize * 3);
  for (let i = 0, j = 0; i < pixels.length; i += 4) {
    tensor[j++] = pixels[i] / 255;
    tensor[j++] = pixels[i + 1] / 255;
    tensor[j++] = pixels[i + 2] / 255;
  }
  return { data: tensor, height: cnnEyeModel.inputSize, width: cnnEyeModel.inputSize, channels: 3 };
}

function runCnnEye(input) {
  let tensor = input;
  cnnEyeModel.layers.forEach((layer) => {
    if (layer.type === "conv2d") tensor = conv2d(tensor, layer);
    else if (layer.type === "maxpool2d") tensor = maxPool2d(tensor, layer);
    else if (layer.type === "gap2d") tensor = globalAveragePool2d(tensor);
    else if (layer.type === "dense") tensor = dense(tensor, layer);
  });
  return tensor.data;
}

function conv2d(input, layer) {
  const [kernelH, kernelW, inChannels, outChannels] = layer.kernelShape;
  const outH = input.height - kernelH + 1;
  const outW = input.width - kernelW + 1;
  const output = new Float32Array(outH * outW * outChannels);

  for (let y = 0; y < outH; y += 1) {
    for (let x = 0; x < outW; x += 1) {
      for (let oc = 0; oc < outChannels; oc += 1) {
        let sum = layer.bias[oc];
        for (let ky = 0; ky < kernelH; ky += 1) {
          for (let kx = 0; kx < kernelW; kx += 1) {
            const baseInput = ((y + ky) * input.width + (x + kx)) * input.channels;
            const baseKernel = ((ky * kernelW + kx) * inChannels) * outChannels + oc;
            for (let ic = 0; ic < inChannels; ic += 1) {
              sum += input.data[baseInput + ic] * layer.kernel[baseKernel + ic * outChannels];
            }
          }
        }
        output[(y * outW + x) * outChannels + oc] = layer.activation === "relu" ? Math.max(0, sum) : sum;
      }
    }
  }
  return { data: output, height: outH, width: outW, channels: outChannels };
}

function maxPool2d(input, layer) {
  const [poolH, poolW] = layer.poolSize;
  const [strideY, strideX] = layer.strides;
  const outH = Math.floor((input.height - poolH) / strideY) + 1;
  const outW = Math.floor((input.width - poolW) / strideX) + 1;
  const output = new Float32Array(outH * outW * input.channels);

  for (let y = 0; y < outH; y += 1) {
    for (let x = 0; x < outW; x += 1) {
      for (let c = 0; c < input.channels; c += 1) {
        let max = -Infinity;
        for (let py = 0; py < poolH; py += 1) {
          for (let px = 0; px < poolW; px += 1) {
            const value = input.data[((y * strideY + py) * input.width + (x * strideX + px)) * input.channels + c];
            if (value > max) max = value;
          }
        }
        output[(y * outW + x) * input.channels + c] = max;
      }
    }
  }
  return { data: output, height: outH, width: outW, channels: input.channels };
}

function globalAveragePool2d(input) {
  const output = new Float32Array(input.channels);
  const area = input.height * input.width;
  for (let y = 0; y < input.height; y += 1) {
    for (let x = 0; x < input.width; x += 1) {
      const base = (y * input.width + x) * input.channels;
      for (let c = 0; c < input.channels; c += 1) {
        output[c] += input.data[base + c] / area;
      }
    }
  }
  return { data: output, length: input.channels };
}

function dense(input, layer) {
  const [inFeatures, outFeatures] = layer.kernelShape;
  const logits = new Float32Array(outFeatures);
  for (let out = 0; out < outFeatures; out += 1) {
    let sum = layer.bias[out];
    for (let i = 0; i < inFeatures; i += 1) {
      sum += input.data[i] * layer.kernel[i * outFeatures + out];
    }
    logits[out] = sum;
  }
  return { data: layer.activation === "softmax" ? softmax(logits) : logits, length: outFeatures };
}

function softmax(logits) {
  const max = Math.max(...logits);
  const exps = Array.from(logits, (value) => Math.exp(value - max));
  const total = exps.reduce((sum, value) => sum + value, 0);
  return new Float32Array(exps.map((value) => value / Math.max(total, 1e-9)));
}

function updateCnnDisplay(cnnEye) {
  if (!els.cnnEyeValue || !els.cnnConfidenceValue) return;
  if (!cnnEye) {
    els.cnnEyeValue.textContent = cnnEyeModel ? "ready" : "pending";
    els.cnnConfidenceValue.textContent = "--";
    return;
  }
  els.cnnEyeValue.textContent = cnnEye.label.replace("eyes_", "");
  els.cnnConfidenceValue.textContent = `${Math.round(cnnEye.confidence * 100)}%`;
}

function handleMetrics(ear, mar, latency, now, cnnEye = null) {
  const cnnReady = Boolean(cnnEye && cnnEye.confidence >= 0.62);
  const closed = cnnReady ? cnnEye.label === "eyes_closed" : ear < settings.earThreshold;
  const yawning = mar > settings.marThreshold;
  const source = cnnReady ? `CNN Eye ${cnnEye.label} ${(cnnEye.confidence * 100).toFixed(0)}%` : "MediaPipe EAR/MAR";

  if (closed && closedStartedAt === null) closedStartedAt = now;
  if (!closed) closedStartedAt = null;
  if (yawning && yawnStartedAt === null) yawnStartedAt = now;
  if (!yawning) yawnStartedAt = null;

  const closedMs = closedStartedAt ? now - closedStartedAt : 0;
  const yawnMs = yawnStartedAt ? now - yawnStartedAt : 0;
  const isDrowsy = closedMs >= settings.drowsyDurationMs;
  const isWarning = closedMs > 450 || yawnMs > 600;

  updateMetricDisplay(ear, mar, latency, closedMs, yawnMs);

  if (isDrowsy) {
    updateState("Drowsy alert", `Mắt nhắm ${(closedMs / 1000).toFixed(1)}s. Hãy dừng xe nếu cảnh báo lặp lại.`, "danger");
    maybeLogState("DROWSY", `EAR=${ear.toFixed(3)}, MAR=${mar.toFixed(3)}, ${source}, closed=${(closedMs / 1000).toFixed(1)}s`, "danger", source);
    playDangerOnce();
  } else if (yawning) {
    updateState("Yawning", `MAR cao trong ${(yawnMs / 1000).toFixed(1)}s.`, "warning");
    maybeLogState("YAWNING", `EAR=${ear.toFixed(3)}, MAR=${mar.toFixed(3)}`, "warning", "MediaPipe EAR/MAR");
  } else if (isWarning) {
    updateState("Eyes closing", `EAR thấp ${(closedMs / 1000).toFixed(1)}s.`, "warning");
    maybeLogState("WARNING", `EAR=${ear.toFixed(3)}, MAR=${mar.toFixed(3)}, ${source}`, "warning", source);
  } else {
    updateState("Awake", "Khuôn mặt ổn định, chưa có cảnh báo.", "safe");
    maybeLogState("AWAKE", `EAR=${ear.toFixed(3)}, MAR=${mar.toFixed(3)}, ${source}`, "safe", source, 10000);
  }
}

let lastDangerToneAt = 0;
function playDangerOnce() {
  const now = performance.now();
  if (now - lastDangerToneAt < 2500) return;
  lastDangerToneAt = now;
  playTone(2600, 0.42, "square", 0.74);
  setTimeout(() => playTone(1200, 0.38, "square", 0.68), 420);
}

function updateMetricDisplay(ear, mar, latency, closedMs = 0, yawnMs = 0) {
  els.earValue.textContent = ear === null ? "--" : ear.toFixed(3);
  els.marValue.textContent = mar === null ? "--" : mar.toFixed(3);
  els.fpsValue.textContent = fps ? fps.toFixed(1) : "--";
  els.latencyValue.textContent = `${Math.round(latency)}ms`;
  els.closedTime.textContent = `${(closedMs / 1000).toFixed(1)}s`;
  els.yawnTime.textContent = `${(yawnMs / 1000).toFixed(1)}s`;
  updateRisk(closedMs, yawnMs);
}

function updateFps(now) {
  if (lastFrameAt) {
    const instant = 1000 / Math.max(now - lastFrameAt, 1);
    fps = fps ? fps * 0.85 + instant * 0.15 : instant;
  }
  lastFrameAt = now;
}

function updateState(title, detail, level) {
  els.stateText.textContent = title;
  els.stateDetail.textContent = detail;
  els.stateDot.style.background = level === "danger" ? "#b91c1c" : level === "warning" ? "#b7791f" : "#15803d";
}

function setManualState(title, detail, level) {
  updateState(title, detail, level);
  addEvent(title, detail, level, "Manual");
  if (level === "warning") playTone(1750);
  if (level === "danger") playDangerOnce();
  updateRisk();
}

function maybeLogState(state, detail, level, source, cooldownMs = 1800) {
  const now = performance.now();
  if (state === lastLoggedState && now - lastLogAt < cooldownMs) return;
  lastLoggedState = state;
  lastLogAt = now;
  addEvent(state, detail, level, source);
}

function addEvent(title, detail, level = "safe", source = "PWA") {
  const item = {
    timestamp: new Date().toISOString(),
    title,
    detail,
    level,
    source,
    location: latestLocation,
  };
  events.unshift(item);
  events = events.slice(0, 120);
  saveEvents();
  renderEvents();
  renderHourChart();
  updateRisk();
}

function renderEvents() {
  els.events.innerHTML = "";
  if (!events.length) {
    els.events.innerHTML = `<div class="event safe"><strong>Ready</strong><br>PWA loaded.<small>Chưa có cảnh báo.</small></div>`;
    return;
  }

  events.slice(0, 30).forEach((item) => {
    const div = document.createElement("div");
    div.className = `event ${item.level || ""}`;
    const time = new Date(item.timestamp).toLocaleTimeString("vi-VN", { hour: "2-digit", minute: "2-digit", second: "2-digit" });
    div.innerHTML = `<strong>${escapeHtml(time)} - ${escapeHtml(item.title)}</strong><br>${escapeHtml(item.detail)}<small>${escapeHtml(item.source || "PWA")}</small>`;
    els.events.appendChild(div);
  });
}

function renderHourChart() {
  const counts = Array.from({ length: 24 }, () => 0);
  events.forEach((item) => {
    const hour = new Date(item.timestamp).getHours();
    counts[hour] += item.level === "danger" ? 3 : item.level === "warning" ? 2 : 1;
  });
  const max = Math.max(1, ...counts);
  els.hourChart.innerHTML = "";
  counts.forEach((count, hour) => {
    const item = document.createElement("div");
    item.className = "hour";
    const bar = document.createElement("div");
    bar.className = "bar";
    const height = Math.max(7, Math.round((count / max) * 78));
    bar.style.height = `${height}px`;
    bar.style.background = count >= 8 ? "#dc2626" : count >= 5 ? "#f97316" : count >= 2 ? "#facc15" : "#c7d2fe";
    const label = document.createElement("small");
    label.textContent = String(hour).padStart(2, "0");
    item.appendChild(bar);
    item.appendChild(label);
    els.hourChart.appendChild(item);
  });
}

function updateRisk(closedMs = 0, yawnMs = 0) {
  alertCount = events.filter((item) => item.level === "warning" || item.level === "danger").length;
  const dangerCount = events.filter((item) => item.level === "danger").length;
  const risk = Math.min(100, Math.round(18 + alertCount * 5 + dangerCount * 9 + closedMs / 80 + yawnMs / 120));
  els.riskScore.textContent = String(risk);
  els.alertCount.textContent = String(alertCount);

  if (risk >= 80) {
    els.recommendation.textContent = "Rủi ro cao: hãy dừng xe tại vị trí an toàn, nghỉ ngắn 15-20 phút và cân nhắc gửi vị trí cho người thân.";
  } else if (risk >= 55) {
    els.recommendation.textContent = "Rủi ro trung bình: giảm tốc, tìm điểm nghỉ, không cố lái nếu cảnh báo lặp lại.";
  } else {
    els.recommendation.textContent = "Trạng thái ổn. Khi cảnh báo lặp lại, hãy dừng xe tại nơi an toàn.";
  }
}

function drawLandmarks(points, metrics) {
  const canvas = els.overlayCanvas;
  const rect = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.round(rect.width * dpr);
  canvas.height = Math.round(rect.height * dpr);
  const ctx = canvas.getContext("2d");
  ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, rect.width, rect.height);

  const eyeIds = [33, 160, 158, 133, 153, 144, 362, 385, 387, 263, 373, 380];
  const mouthIds = [13, 14, 78, 308];
  ctx.fillStyle = metrics.ear < settings.earThreshold ? "#ef4444" : "#22c55e";
  eyeIds.forEach((id) => drawPoint(ctx, points[id], rect));
  ctx.fillStyle = metrics.mar > settings.marThreshold ? "#f97316" : "#38bdf8";
  mouthIds.forEach((id) => drawPoint(ctx, points[id], rect));
}

function drawPoint(ctx, point, rect) {
  ctx.beginPath();
  ctx.arc(point.x * rect.width, point.y * rect.height, 2.6, 0, Math.PI * 2);
  ctx.fill();
}

function clearCanvas() {
  const canvas = els.overlayCanvas;
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
}

function loadSnapshot(event) {
  const file = event.target.files?.[0];
  if (!file) return;
  stopCamera();
  const url = URL.createObjectURL(file);
  els.snapshot.src = url;
  els.snapshot.hidden = false;
  els.video.hidden = true;
  updateState("Snapshot ready", "Đã nhận ảnh từ camera/file picker iPhone.", "safe");
  addEvent("Snapshot", "Fallback snapshot loaded. Realtime AI cần HTTPS.", "safe", "Snapshot");
}

function restoreSoundPreference() {
  soundEnabled = localStorage.getItem(SOUND_SETTING_KEY) === "1";
  soundUnlocked = false;
}

function getAudioContext() {
  const AudioContextClass = window.AudioContext || window.webkitAudioContext;
  if (!AudioContextClass) {
    updateState("Audio unsupported", "Trình duyệt không hỗ trợ Web Audio API.", "warning");
    return null;
  }
  if (!audioContext || audioContext.state === "closed") {
    audioContext = new AudioContextClass();
  }
  return audioContext;
}

async function enableSoundMode() {
  const context = getAudioContext();
  if (!context) return false;
  try {
    if (context.state === "suspended") {
      await context.resume();
    }
    soundEnabled = true;
    soundUnlocked = true;
    localStorage.setItem(SOUND_SETTING_KEY, "1");
    updateSoundStatus();
    playToneInternal(880, 0.12, 0.08);
    addEvent("Sound enabled", "Âm thanh cảnh báo đã được bật cho phiên PWA.", "safe", "Audio");
    return true;
  } catch (error) {
    soundEnabled = false;
    soundUnlocked = false;
    updateSoundStatus();
    addEvent("Sound blocked", error.message || "Safari chưa cho phép phát âm thanh.", "warning", "Audio");
    return false;
  }
}

function disableSoundMode() {
  soundEnabled = false;
  localStorage.setItem(SOUND_SETTING_KEY, "0");
  stopArousal();
  updateSoundStatus();
  addEvent("Sound disabled", "Âm thanh cảnh báo đã tắt.", "safe", "Audio");
}

async function ensureSoundMode() {
  if (soundEnabled && soundUnlocked) return true;
  return enableSoundMode();
}

function shouldPlaySound() {
  return soundEnabled && soundUnlocked;
}

function updateSoundStatus() {
  if (!els.soundStatus) return;
  const title = els.soundStatus.querySelector("strong");
  const detail = els.soundStatus.querySelector("span");
  const wantsSound = localStorage.getItem(SOUND_SETTING_KEY) === "1";

  if (soundEnabled && soundUnlocked) {
    els.soundStatus.classList.remove("off");
    title.textContent = "Sound On";
    detail.textContent = "Cảnh báo tự động đã bật. Nếu không nghe beep, tăng âm lượng và tắt Silent Mode trên iPhone.";
    return;
  }

  els.soundStatus.classList.add("off");
  if (wantsSound) {
    title.textContent = "Sound needs unlock";
    detail.textContent = "Bấm Bật âm thanh một lần để iPhone cho phép app phát cảnh báo trong phiên này.";
  } else {
    title.textContent = "Sound Off";
    detail.textContent = "Bấm Bật âm thanh để cho phép app phát cảnh báo trên iPhone.";
  }
}

function playTestBeep() {
  playTone(1200, 0.28, "square", 0.72);
  setTimeout(() => playTone(2200, 0.26, "square", 0.68), 340);
  if (navigator.vibrate) navigator.vibrate([80, 40, 80]);
  const detail = els.soundStatus?.querySelector("span");
  if (detail) detail.textContent = "Đã phát beep test. Nếu vẫn im lặng: tăng volume, tắt Silent Mode, rồi bấm Beep lại.";
}

function playTone(frequency, duration = 0.35, type = "square", volume = 0.55) {
  if (!shouldPlaySound()) {
    updateSoundStatus();
    return;
  }
  playToneInternal(frequency, duration, volume, type);
}

function playToneInternal(frequency, duration = 0.35, volume = 0.55, type = "square") {
  const context = getAudioContext();
  if (!context) return;
  if (context.state === "suspended") {
    context.resume().catch(() => {});
  }
  const osc = context.createOscillator();
  const gain = context.createGain();
  const now = context.currentTime;
  osc.frequency.value = frequency;
  osc.type = type;
  gain.gain.setValueAtTime(0.001, now);
  gain.gain.exponentialRampToValueAtTime(volume, now + 0.03);
  gain.gain.exponentialRampToValueAtTime(0.001, now + duration);
  osc.connect(gain).connect(context.destination);
  osc.start(now);
  osc.stop(now + duration + 0.04);
}

function startArousal() {
  if (!shouldPlaySound()) {
    updateSoundStatus();
    els.arousalText.textContent = "Hãy bấm Bật âm thanh trước khi chạy nền 1750 Hz.";
    return;
  }
  stopArousal();
  arousalContext = getAudioContext();
  if (!arousalContext) return;
  const now = arousalContext.currentTime;
  [1750, 500, 3000].forEach((frequency, index) => {
    const osc = arousalContext.createOscillator();
    const gain = arousalContext.createGain();
    const lfo = arousalContext.createOscillator();
    const lfoGain = arousalContext.createGain();
    osc.frequency.value = frequency;
    osc.type = index === 2 ? "triangle" : "sine";
    lfo.frequency.value = 1.2 + index * 0.3;
    lfoGain.gain.value = 0.025;
    gain.gain.setValueAtTime(0.0001, now);
    gain.gain.linearRampToValueAtTime(0.065, now + 0.2);
    lfo.connect(lfoGain).connect(gain.gain);
    osc.connect(gain).connect(arousalContext.destination);
    osc.start(now + index * 0.04);
    lfo.start(now + index * 0.04);
    arousalNodes.push(osc, lfo, gain);
  });
  els.arousalText.textContent = "Đang bật nền kích thích 1750 Hz có giới hạn. Tự ngắt sau 30 giây.";
  arousalTimer = window.setTimeout(() => {
    stopArousal();
    els.arousalText.textContent = "Âm nền đã tự ngắt. Nếu vẫn buồn ngủ, hãy dừng xe và nghỉ.";
  }, 30000);
}

function stopArousal() {
  if (arousalTimer) {
    clearTimeout(arousalTimer);
    arousalTimer = null;
  }
  arousalNodes.forEach((node) => {
    try {
      if (typeof node.stop === "function") node.stop();
      if (typeof node.disconnect === "function") node.disconnect();
    } catch (error) {}
  });
  arousalNodes = [];
  arousalContext = null;
  if (els.arousalText) {
    els.arousalText.textContent = "Âm nền chỉ chạy tối đa 30 giây trong demo. Nếu vẫn buồn ngủ, hệ thống phải khuyến nghị dừng xe.";
  }
}

function requestLocation() {
  if (!navigator.geolocation) {
    els.locationText.textContent = "Trình duyệt không hỗ trợ geolocation.";
    return;
  }
  els.locationText.textContent = "Đang lấy vị trí...";
  navigator.geolocation.getCurrentPosition(
    (position) => {
      latestLocation = {
        lat: position.coords.latitude,
        lon: position.coords.longitude,
      };
      els.locationText.textContent = `Vị trí gần nhất: ${latestLocation.lat.toFixed(6)}, ${latestLocation.lon.toFixed(6)}`;
      addEvent("Location ready", els.locationText.textContent, "safe", "Geolocation");
    },
    () => {
      els.locationText.textContent = "Không lấy được vị trí. Hãy cấp quyền Location trên iPhone.";
    },
    { enableHighAccuracy: true, timeout: 8000, maximumAge: 30000 }
  );
}

async function shareTrustedAlert() {
  const contact = els.trustedContact.value.trim() || "liên hệ tin cậy";
  const mapUrl = latestLocation ? `https://maps.google.com/?q=${latestLocation.lat},${latestLocation.lon}` : "Chưa có vị trí";
  const text = `DROWSY_HIGH_RISK: Người lái có dấu hiệu buồn ngủ nghiêm trọng. Liên hệ: ${contact}. Vị trí: ${mapUrl}`;

  addEvent("Trusted alert", text, "danger", "Trusted contact");

  if (navigator.share) {
    try {
      await navigator.share({ title: "Drowsy Guard Alert", text, url: latestLocation ? mapUrl : undefined });
      return;
    } catch (error) {}
  }

  const smsBody = encodeURIComponent(text);
  window.location.href = `sms:&body=${smsBody}`;
}

function exportCsv() {
  const header = ["timestamp", "title", "level", "source", "detail", "lat", "lon"];
  const rows = events.map((item) => [
    item.timestamp,
    item.title,
    item.level || "",
    item.source || "",
    item.detail || "",
    item.location?.lat ?? "",
    item.location?.lon ?? "",
  ]);
  const csv = [header, ...rows]
    .map((row) => row.map((value) => `"${String(value).replaceAll('"', '""')}"`).join(","))
    .join("\n");
  const blob = new Blob([csv], { type: "text/csv;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = `drowsy_guard_events_${new Date().toISOString().slice(0, 10)}.csv`;
  link.click();
  URL.revokeObjectURL(url);
}

function clearLogs() {
  events = [];
  saveEvents();
  renderEvents();
  renderHourChart();
  updateRisk();
}

function loadEvents() {
  try {
    return JSON.parse(localStorage.getItem(STORAGE_KEY) || "[]");
  } catch (error) {
    return [];
  }
}

function saveEvents() {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(events));
}

function restoreSettings() {
  try {
    const saved = JSON.parse(localStorage.getItem("drowsyGuardIphonePwaSettings") || "{}");
    Object.assign(settings, saved);
  } catch (error) {}
  els.earThreshold.value = settings.earThreshold;
  els.marThreshold.value = settings.marThreshold;
  els.drowsyDuration.value = settings.drowsyDurationMs;
  updateSettingLabels();
}

function saveSettings() {
  localStorage.setItem("drowsyGuardIphonePwaSettings", JSON.stringify(settings));
}

function updateSettingLabels() {
  els.earThresholdValue.textContent = settings.earThreshold.toFixed(2);
  els.marThresholdValue.textContent = settings.marThreshold.toFixed(2);
  els.drowsyDurationValue.textContent = `${settings.drowsyDurationMs}ms`;
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}
