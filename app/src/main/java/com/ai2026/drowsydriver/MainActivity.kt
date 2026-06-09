package com.ai2026.drowsydriver

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Matrix
import android.graphics.Rect
import android.os.Bundle
import android.os.SystemClock
import android.util.Log
import android.view.Gravity
import android.view.Surface
import android.widget.FrameLayout
import android.widget.TextView
import androidx.activity.ComponentActivity
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.components.containers.NormalizedLandmark
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.facelandmarker.FaceLandmarker
import com.google.mediapipe.tasks.vision.facelandmarker.FaceLandmarkerResult
import java.nio.ByteBuffer
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import kotlin.math.max
import kotlin.math.min

class MainActivity : ComponentActivity() {
    private lateinit var previewView: PreviewView
    private lateinit var overlayView: StatusOverlayView
    private lateinit var messageView: TextView
    private lateinit var cameraExecutor: ExecutorService
    private lateinit var alertController: AlertController
    private lateinit var eventLogger: EventLogger
    private val analyzer = DrowsinessAnalyzer()
    private val sessionTracker = DriverSessionTracker()
    private var faceLandmarker: FaceLandmarker? = null
    private var tfliteClassifier: TfliteDrowsinessClassifier? = null
    private var yawnClassifier: YawnClassifier? = null
    private var lastAnalyzedAt = 0L
    private var cnnClosedStartedAt: Long? = null
    @Volatile private var latestFrameStartedAt = 0L
    @Volatile private var latestBitmap: Bitmap? = null

    // Thống kê phiên lái (cho màn Analytics)
    private var sessionStart = 0L
    private var drowsyCount = 0
    private var yawnCount = 0
    private var drowsyMs = 0L
    private var prevState = DriverState.AWAKE
    private var lastStatsWrite = 0L
    private var autoLocationSent = false   // đã tự gửi định vị trong chuyến đi này chưa

    private val requestPermission = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted ->
        if (granted) startCamera() else showMessage("Camera permission is required")
    }

    // Xin quyền cho tính năng TỰ ĐỘNG gửi định vị (SMS + vị trí)
    private val requestEmergencyPerms = registerForActivityResult(
        ActivityResultContracts.RequestMultiplePermissions()
    ) { /* không chặn camera; chỉ bật được auto-SMS nếu người dùng đồng ý */ }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        cameraExecutor = Executors.newSingleThreadExecutor()
        alertController = AlertController(this)
        eventLogger = EventLogger(this)
        tfliteClassifier = TfliteDrowsinessClassifier(this)
        tfliteClassifier?.logStatus()   // log CNN status khi khởi động
        yawnClassifier = YawnClassifier(this)
        yawnClassifier?.logStatus()     // log Yawn CNN status
        setupUi()
        setupFaceLandmarker()

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
            startCamera()
        } else {
            requestPermission.launch(Manifest.permission.CAMERA)
        }

        // Xin sẵn quyền SMS + vị trí để tính năng tự động gửi định vị hoạt động khi cần
        requestEmergencyPerms.launch(arrayOf(
            Manifest.permission.SEND_SMS,
            Manifest.permission.ACCESS_FINE_LOCATION
        ))
    }

    override fun onDestroy() {
        super.onDestroy()
        cameraExecutor.shutdown()
        faceLandmarker?.close()
        tfliteClassifier?.close()
        yawnClassifier?.close()
        alertController.close()
    }

    private fun setupUi() {
        previewView = PreviewView(this).apply {
            scaleType = PreviewView.ScaleType.FILL_CENTER
        }
        overlayView = StatusOverlayView(this)
        messageView = TextView(this).apply {
            text = "Preparing camera"
            setTextColor(android.graphics.Color.WHITE)
            textSize = 16f
            gravity = Gravity.CENTER
            setBackgroundColor(android.graphics.Color.argb(120, 0, 0, 0))
            setPadding(24, 12, 24, 12)
        }

        val root = FrameLayout(this)
        root.addView(previewView, FrameLayout.LayoutParams.MATCH_PARENT, FrameLayout.LayoutParams.MATCH_PARENT)
        root.addView(overlayView, FrameLayout.LayoutParams.MATCH_PARENT, FrameLayout.LayoutParams.MATCH_PARENT)
        val messageParams = FrameLayout.LayoutParams(
            FrameLayout.LayoutParams.MATCH_PARENT,
            FrameLayout.LayoutParams.WRAP_CONTENT,
            Gravity.BOTTOM
        )
        root.addView(messageView, messageParams)

        // ── Thanh điều hướng dưới cùng: Dashboard + Khẩn cấp tự động ──────────
        val navBar = android.widget.LinearLayout(this).apply {
            orientation = android.widget.LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER
            setBackgroundColor(android.graphics.Color.argb(180, 10, 10, 14))
            setPadding(dp(24), dp(10), dp(24), dp(14))
        }

        // Nút Dashboard (xanh dương)
        val btnDashboard = makeNavButton(
            iconRes    = R.drawable.ic_nav_dashboard,
            bgRes      = R.drawable.bg_nav_dashboard,
            label      = "Dashboard",
            labelColor = android.graphics.Color.parseColor("#4D9AFF")
        ) { startActivity(android.content.Intent(this@MainActivity, AnalyticsActivity::class.java)) }

        // Nút Khẩn cấp tự động (cam)
        val btnEmergency = makeNavButton(
            iconRes    = R.drawable.ic_nav_emergency,
            bgRes      = R.drawable.bg_nav_emergency,
            label      = "Khẩn cấp tự động",
            labelColor = android.graphics.Color.parseColor("#FF8040")
        ) { startActivity(android.content.Intent(this@MainActivity, EmergencyContactActivity::class.java)) }

        navBar.addView(btnDashboard)
        navBar.addView(android.view.View(this).apply {         // spacer giữa 2 nút
            layoutParams = android.widget.LinearLayout.LayoutParams(dp(28), 1)
        })
        navBar.addView(btnEmergency)

        root.addView(navBar, FrameLayout.LayoutParams(
            FrameLayout.LayoutParams.MATCH_PARENT,
            FrameLayout.LayoutParams.WRAP_CONTENT,
            Gravity.BOTTOM))

        setContentView(root)
    }

    /** Cập nhật thống kê phiên lái → SharedPreferences (cho màn Analytics). */
    private fun updateStats(status: DriverStatus) {
        val now = System.currentTimeMillis()
        if (sessionStart == 0L) sessionStart = now
        // Cập nhật thời gian lái liên tục để AlertController nhắc nghỉ ở mốc 2h và 4h
        alertController.updateDrivingTime(((now - sessionStart) / 60000L).toInt())
        if (status.state == DriverState.DROWSY) {
            drowsyMs += 100L
            if (prevState != DriverState.DROWSY) {
                drowsyCount++
                recordDrowsyHour()
                maybeAutoSendLocation()
            }
        }
        if (status.state == DriverState.YAWNING && prevState != DriverState.YAWNING) yawnCount++
        prevState = status.state
        if (now - lastStatsWrite > 1500L) {
            lastStatsWrite = now
            val ts = java.text.SimpleDateFormat("HH:mm dd/MM", java.util.Locale.getDefault()).format(java.util.Date())
            val e = getSharedPreferences("agrypnia_prefs", MODE_PRIVATE).edit()
            e.putLong("session_sec", (now - sessionStart) / 1000)
                .putInt("drowsy_events", drowsyCount).putInt("yawn_events", yawnCount)
                .putLong("drowsy_sec", drowsyMs / 1000)
            if (status.state == DriverState.DROWSY) e.putString("last_drowsy_at", ts)
            e.apply()
        }
    }

    /**
     * Ghi nhận GIỜ trong ngày xảy ra buồn ngủ vào histogram 24 giờ (tích lũy qua mọi chuyến),
     * phục vụ phân tích khung giờ tài xế hay buồn ngủ để khuyến cáo tránh lái xe giờ đó.
     */
    private fun recordDrowsyHour() {
        val prefs = getSharedPreferences("agrypnia_prefs", MODE_PRIVATE)
        val hour = java.util.Calendar.getInstance().get(java.util.Calendar.HOUR_OF_DAY)
        val arr = (prefs.getString("drowsy_hours", null)?.split(",")
            ?.map { it.trim().toIntOrNull() ?: 0 } ?: List(24) { 0 }).toMutableList()
        while (arr.size < 24) arr.add(0)
        arr[hour] = arr[hour] + 1
        prefs.edit().putString("drowsy_hours", arr.take(24).joinToString(",")).apply()
    }

    /**
     * Khi số lần buồn ngủ vượt ngưỡng an toàn → TỰ ĐỘNG gửi định vị cho người thân.
     * Chỉ gửi MỘT lần trong mỗi chuyến đi (mỗi lần mở app) để tránh gửi lặp.
     */
    private fun maybeAutoSendLocation() {
        if (autoLocationSent) return
        val limit = EmergencyDispatcher.drowsyLimit(this)
        if (drowsyCount < limit) return
        autoLocationSent = true
        val result = EmergencyDispatcher.autoSendLocation(
            this, "buồn ngủ $drowsyCount lần (vượt ngưỡng $limit lần)"
        )
        getSharedPreferences("agrypnia_prefs", MODE_PRIVATE).edit()
            .putBoolean("auto_sent_trip", result?.startsWith("Đã") == true).apply()
        if (result != null) {
            showMessage("🆘 $result")
        } else {
            showMessage("⚠️ Vượt ngưỡng buồn ngủ — thêm SĐT người thân ở màn 🆘 để tự gửi định vị")
        }
    }

    private fun setupFaceLandmarker() {
        runCatching {
            val baseOptions = BaseOptions.builder()
                .setModelAssetPath("face_landmarker.task")
                .build()

            val options = FaceLandmarker.FaceLandmarkerOptions.builder()
                .setBaseOptions(baseOptions)
                .setRunningMode(RunningMode.LIVE_STREAM)
                .setNumFaces(1)
                .setMinFaceDetectionConfidence(0.5f)
                .setMinFacePresenceConfidence(0.5f)
                .setMinTrackingConfidence(0.5f)
                .setResultListener(::onLandmarkerResult)
                .setErrorListener { error -> Log.e("DrowsyDriver", "MediaPipe error", error) }
                .build()

            faceLandmarker = FaceLandmarker.createFromOptions(this, options)
            showMessage("MediaPipe ready")
        }.onFailure {
            showMessage("Missing face_landmarker.task in app/src/main/assets")
            Log.e("DrowsyDriver", "Cannot create FaceLandmarker", it)
        }
    }

    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        cameraProviderFuture.addListener({
            val cameraProvider = cameraProviderFuture.get()
            val targetRotation = previewView.display?.rotation ?: Surface.ROTATION_0
            val preview = Preview.Builder()
                .setTargetRotation(targetRotation)
                .build()
                .also {
                it.setSurfaceProvider(previewView.surfaceProvider)
            }

            val imageAnalysis = ImageAnalysis.Builder()
                .setTargetRotation(targetRotation)
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                .build()
                .also {
                    it.setAnalyzer(cameraExecutor, ::analyzeFrame)
                }

            cameraProvider.unbindAll()
            cameraProvider.bindToLifecycle(
                this,
                CameraSelector.DEFAULT_FRONT_CAMERA,
                preview,
                imageAnalysis
            )
            showMessage("Camera running")
        }, ContextCompat.getMainExecutor(this))
    }

    private fun analyzeFrame(imageProxy: ImageProxy) {
        val landmarker = faceLandmarker
        val now = SystemClock.uptimeMillis()
        if (landmarker == null || now - lastAnalyzedAt < 90L) {
            imageProxy.close()
            return
        }
        lastAnalyzedAt = now

        runCatching {
            latestFrameStartedAt = SystemClock.uptimeMillis()
            val bitmap = imageProxy.toRgbaBitmap().rotate(imageProxy.imageInfo.rotationDegrees)
            latestBitmap = bitmap
            val mpImage = BitmapImageBuilder(bitmap).build()
            landmarker.detectAsync(mpImage, now)
        }.onFailure {
            Log.e("DrowsyDriver", "Frame analysis failed", it)
        }
        imageProxy.close()
    }

    private fun ImageProxy.toRgbaBitmap(): Bitmap {
        val plane = planes.first()
        val buffer = plane.buffer.apply { rewind() }
        val bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
        val rowStride = plane.rowStride
        val pixelStride = plane.pixelStride
        val bytesPerPixel = 4

        if (pixelStride == bytesPerPixel && rowStride == width * bytesPerPixel) {
            bitmap.copyPixelsFromBuffer(buffer)
            return bitmap
        }

        val packed = ByteArray(width * height * bytesPerPixel)
        val row = ByteArray(rowStride)
        var outputOffset = 0
        for (y in 0 until height) {
            val bytesToRead = min(rowStride, buffer.remaining())
            buffer.get(row, 0, bytesToRead)
            if (pixelStride == bytesPerPixel) {
                System.arraycopy(row, 0, packed, outputOffset, width * bytesPerPixel)
                outputOffset += width * bytesPerPixel
            } else {
                for (x in 0 until width) {
                    val inputOffset = x * pixelStride
                    packed[outputOffset++] = row[inputOffset]
                    packed[outputOffset++] = row[inputOffset + 1]
                    packed[outputOffset++] = row[inputOffset + 2]
                    packed[outputOffset++] = row[inputOffset + 3]
                }
            }
        }
        bitmap.copyPixelsFromBuffer(ByteBuffer.wrap(packed))
        return bitmap
    }

    private fun onLandmarkerResult(result: FaceLandmarkerResult, input: com.google.mediapipe.framework.image.MPImage) {
        val resultStartedAt = latestFrameStartedAt
        val latencyMs = if (resultStartedAt > 0L) {
            SystemClock.uptimeMillis() - resultStartedAt
        } else {
            0L
        }
        val heuristicStatus = analyzer.analyze(result, SystemClock.uptimeMillis())
        val cnnPrediction   = classifyEyeRoi(result)
        val yawnPrediction  = classifyMouthRoi(result)
        val status = buildHybridStatus(heuristicStatus, cnnPrediction, yawnPrediction, latencyMs)
        runOnUiThread {
            val report = sessionTracker.update(status, SystemClock.uptimeMillis())
            overlayView.update(status, report)
            alertController.update(status)
            updateStats(status)
            eventLogger.logIfNeeded(status)
            if (status.state != DriverState.NO_FACE) {
                val cnnState = if (tfliteClassifier?.isAvailable == true) "CNN primary + EAR/MAR fallback" else "CNN model pending"
                showMessage("On-device analysis - $cnnState")
            }
        }
    }

    private fun buildHybridStatus(
        heuristicStatus: DriverStatus,
        cnnPrediction: CnnPrediction?,
        yawnPrediction: YawnClassifier.YawnPrediction?,
        latencyMs: Long
    ): DriverStatus {
        val now = SystemClock.uptimeMillis()

        // ── Eye CNN logic (temporal: mắt nhắm liên tục ≥1.2s → DROWSY) ──────
        val cnnClosed = cnnPrediction?.label == "eyes_closed" &&
                cnnPrediction.confidence >= TfliteDrowsinessClassifier.CNN_CONF_THRESHOLD
        val cnnClosedMs = if (cnnClosed) {
            if (cnnClosedStartedAt == null) cnnClosedStartedAt = now
            now - (cnnClosedStartedAt ?: now)
        } else {
            cnnClosedStartedAt = null
            0L
        }
        val cnnEyeState = when {
            cnnClosedMs >= 1_200L -> DriverState.DROWSY
            cnnClosedMs > 0L      -> DriverState.EYES_CLOSED
            else                  -> DriverState.AWAKE
        }

        // ── Ngáp: CNN Yawn CHỦ ĐẠO; MAR chỉ phụ khi KHÔNG có dự đoán CNN ───
        val cnnYawning   = yawnPrediction?.isYawning == true
        val marYawning   = heuristicStatus.state == DriverState.YAWNING
        val isYawning    = if (yawnPrediction != null) cnnYawning else marYawning

        // CNN mắt đủ tin cậy để làm PHƯƠNG PHÁP CHÍNH?
        val cnnUsable = cnnPrediction != null &&
            cnnPrediction.confidence >= TfliteDrowsinessClassifier.CNN_CONF_THRESHOLD

        // ── Fused state: CNN CHỦ ĐẠO, EAR/MAR chỉ FALLBACK ─────────────────
        val fusedState = when {
            heuristicStatus.state == DriverState.NO_FACE -> DriverState.NO_FACE
            isYawning                                    -> DriverState.YAWNING
            // CNN mắt là phương pháp CHÍNH khi đủ tin cậy
            cnnUsable -> when {
                // Lưới an toàn: nếu EAR cũng khẳng định DROWSY thì giữ DROWSY (không bỏ sót)
                cnnEyeState == DriverState.DROWSY ||
                    heuristicStatus.state == DriverState.DROWSY -> DriverState.DROWSY
                else -> cnnEyeState
            }
            // FALLBACK EAR/MAR chỉ khi CNN KHÔNG có / KHÔNG đủ tin cậy
            else -> heuristicStatus.state
        }

        val fusedConfidence = when {
            fusedState == DriverState.YAWNING && cnnYawning -> yawnPrediction?.confidence ?: heuristicStatus.confidence
            cnnUsable && fusedState != DriverState.NO_FACE  -> cnnPrediction?.confidence ?: heuristicStatus.confidence
            else -> heuristicStatus.confidence
        }

        val alertSource = when {
            cnnYawning -> "CNN Yawn (primary)"
            marYawning -> "MAR (fallback)"
            cnnUsable  -> "CNN Eye (primary)"
            else       -> "EAR/MAR (fallback)"
        }

        return heuristicStatus.copy(
            state = fusedState,
            confidence = fusedConfidence,
            closedEyeMs = max(heuristicStatus.closedEyeMs, cnnClosedMs),
            latencyMs = latencyMs,
            cnnLabel = cnnPrediction?.label,
            cnnConfidence = cnnPrediction?.confidence ?: 0f,
            alertSource = alertSource
        )
    }

    private fun Bitmap.rotate(degrees: Int): Bitmap {
        if (degrees == 0) return this
        val matrix = Matrix().apply { postRotate(degrees.toFloat()) }
        return Bitmap.createBitmap(this, 0, 0, width, height, matrix, true)
    }

    private fun classifyEyeRoi(result: FaceLandmarkerResult): CnnPrediction? {
        val classifier = tfliteClassifier ?: return null
        if (!classifier.isAvailable) return null
        val bitmap = latestBitmap ?: return null
        val face = result.faceLandmarks().firstOrNull() ?: return null
        val eyeCrops = cropEyeImages(bitmap, face)
        return classifier.predictAverage(eyeCrops)
    }

    private fun cropEyeImages(bitmap: Bitmap, face: List<NormalizedLandmark>): List<Bitmap> {
        val leftEye = cropLandmarkRegion(bitmap, face, intArrayOf(33, 133, 160, 158, 153, 144))
        val rightEye = cropLandmarkRegion(bitmap, face, intArrayOf(362, 263, 385, 387, 373, 380))
        return listOfNotNull(leftEye, rightEye)
    }

    /** Crop vùng miệng từ MediaPipe landmarks để feed CNN Yawn */
    private fun classifyMouthRoi(result: FaceLandmarkerResult): YawnClassifier.YawnPrediction? {
        val classifier = yawnClassifier ?: return null
        if (!classifier.isAvailable) return null
        val bitmap = latestBitmap ?: return null
        val face   = result.faceLandmarks().firstOrNull() ?: return null
        // Landmarks bao quanh miệng: góc trái/phải + môi trên/dưới + các điểm ngoài
        val mouthBitmap = cropLandmarkRegion(
            bitmap, face, intArrayOf(61, 291, 0, 17, 39, 269, 91, 321, 181, 405)
        ) ?: return null
        return classifier.predict(mouthBitmap)
    }

    private fun cropLandmarkRegion(
        bitmap: Bitmap,
        face: List<NormalizedLandmark>,
        eyeIndices: IntArray
    ): Bitmap? {
        val xs = eyeIndices.map { (face[it].x() * bitmap.width).toInt() }
        val ys = eyeIndices.map { (face[it].y() * bitmap.height).toInt() }
        val minX = xs.minOrNull() ?: return null
        val maxX = xs.maxOrNull() ?: return null
        val minY = ys.minOrNull() ?: return null
        val maxY = ys.maxOrNull() ?: return null

        val width = maxX - minX
        val height = maxY - minY
        if (width <= 0 || height <= 0) return null

        val marginX = (width * 0.90f).toInt()
        val marginY = (height * 1.30f).toInt()
        val rect = Rect(
            max(0, minX - marginX),
            max(0, minY - marginY),
            min(bitmap.width, maxX + marginX),
            min(bitmap.height, maxY + marginY)
        )
        if (rect.width() <= 4 || rect.height() <= 4) return null
        return Bitmap.createBitmap(bitmap, rect.left, rect.top, rect.width(), rect.height())
    }

    private fun showMessage(message: String) {
        runOnUiThread { messageView.text = message }
    }

    // ── Helpers UI ─────────────────────────────────────────────────────────

    /** Chuyển dp → px theo mật độ màn hình thực. */
    private fun dp(value: Int): Int =
        (value * resources.displayMetrics.density + 0.5f).toInt()

    /**
     * Tạo một nút điều hướng dạng icon vuông bo góc + nhãn bên dưới.
     * @param iconRes   R.drawable của vector icon (trắng trên trong suốt)
     * @param bgRes     R.drawable của background (màu bo góc)
     * @param label     Nhãn hiển thị dưới icon
     * @param labelColor Màu chữ nhãn
     * @param onClick   Hàm xử lý khi nhấn
     */
    private fun makeNavButton(
        iconRes:    Int,
        bgRes:      Int,
        label:      String,
        labelColor: Int,
        onClick:    () -> Unit
    ): android.widget.LinearLayout {
        val btnSize  = dp(72)   // kích thước vuông icon tile
        val iconSize = dp(40)   // kích thước icon bên trong

        return android.widget.LinearLayout(this).apply {
            orientation = android.widget.LinearLayout.VERTICAL
            gravity = Gravity.CENTER
            layoutParams = android.widget.LinearLayout.LayoutParams(
                android.widget.LinearLayout.LayoutParams.WRAP_CONTENT,
                android.widget.LinearLayout.LayoutParams.WRAP_CONTENT
            )
            setOnClickListener { onClick() }

            // Icon tile (vuông bo góc)
            addView(android.widget.ImageView(context).apply {
                setImageResource(iconRes)
                background = androidx.core.content.ContextCompat.getDrawable(context, bgRes)
                scaleType = android.widget.ImageView.ScaleType.CENTER_INSIDE
                setPadding(dp(14), dp(14), dp(14), dp(14))
                layoutParams = android.widget.LinearLayout.LayoutParams(btnSize, btnSize).apply {
                    bottomMargin = dp(6)
                }
                // Elevation + ripple
                elevation = dp(4).toFloat()
                isClickable = false   // click handled by parent LinearLayout
                isFocusable = false
            })

            // Nhãn dưới icon
            addView(android.widget.TextView(context).apply {
                text = label
                setTextColor(labelColor)
                textSize = 11f
                gravity = Gravity.CENTER
                maxLines = 2
                setTypeface(typeface, android.graphics.Typeface.BOLD)
                layoutParams = android.widget.LinearLayout.LayoutParams(
                    dp(90), android.widget.LinearLayout.LayoutParams.WRAP_CONTENT
                )
            })
        }
    }
}
