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
    private var faceLandmarker: FaceLandmarker? = null
    private var tfliteClassifier: TfliteDrowsinessClassifier? = null
    private var yawnClassifier: YawnClassifier? = null
    private var lastAnalyzedAt = 0L
    private var cnnClosedStartedAt: Long? = null
    @Volatile private var latestFrameStartedAt = 0L
    @Volatile private var latestBitmap: Bitmap? = null

    private val requestPermission = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted ->
        if (granted) startCamera() else showMessage("Camera permission is required")
    }

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
    }

    override fun onDestroy() {
        super.onDestroy()
        cameraExecutor.shutdown()
        faceLandmarker?.close()
        tfliteClassifier?.close()
        yawnClassifier?.close()
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
        setContentView(root)
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
            overlayView.update(status)
            alertController.update(status)
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

        // ── Yawn detection: CNN Yawn hoặc MAR heuristic ────────────────────
        val cnnYawning   = yawnPrediction?.isYawning == true
        val marYawning   = heuristicStatus.state == DriverState.YAWNING
        val isYawning    = cnnYawning || marYawning

        val shouldUseCnn = cnnPrediction != null &&
            cnnPrediction.confidence >= TfliteDrowsinessClassifier.CNN_CONF_THRESHOLD &&
            !isYawning

        // ── Fused state ────────────────────────────────────────────────────
        val fusedState = when {
            heuristicStatus.state == DriverState.NO_FACE                                       -> DriverState.NO_FACE
            isYawning                                                                          -> DriverState.YAWNING
            shouldUseCnn && cnnEyeState == DriverState.DROWSY                                  -> DriverState.DROWSY
            shouldUseCnn && cnnEyeState == DriverState.EYES_CLOSED                            -> DriverState.EYES_CLOSED
            heuristicStatus.state == DriverState.DROWSY                                        -> DriverState.DROWSY
            heuristicStatus.state == DriverState.EYES_CLOSED && cnnPrediction?.label != "eyes_open" -> DriverState.EYES_CLOSED
            shouldUseCnn && cnnEyeState == DriverState.AWAKE                                   -> DriverState.AWAKE
            else                                                                               -> heuristicStatus.state
        }

        val fusedConfidence = when {
            fusedState == DriverState.YAWNING && cnnYawning    -> yawnPrediction?.confidence ?: heuristicStatus.confidence
            fusedState == DriverState.DROWSY && cnnEyeState == DriverState.DROWSY -> cnnPrediction?.confidence ?: heuristicStatus.confidence
            fusedState == DriverState.EYES_CLOSED && cnnEyeState == DriverState.EYES_CLOSED -> cnnPrediction?.confidence ?: heuristicStatus.confidence
            else -> heuristicStatus.confidence
        }

        val alertSource = when {
            cnnYawning && marYawning -> "CNN Yawn + MAR"
            cnnYawning               -> "CNN Yawn"
            marYawning               -> "MAR"
            shouldUseCnn && (cnnEyeState == DriverState.DROWSY || cnnEyeState == DriverState.EYES_CLOSED) -> "CNN Eye + temporal"
            heuristicStatus.state == DriverState.DROWSY || heuristicStatus.state == DriverState.EYES_CLOSED -> "EAR/MAR fallback"
            shouldUseCnn             -> "CNN Eye"
            else                     -> "EAR/MAR"
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
}
