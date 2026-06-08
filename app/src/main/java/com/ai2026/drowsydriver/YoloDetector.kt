package com.ai2026.drowsydriver

import android.content.Context
import android.graphics.Bitmap
import android.graphics.RectF
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import kotlin.math.max
import kotlin.math.min

/**
 * YOLO detector cho Android — nhận diện 6 class (close_eyeL/R, open_eyeL/R, yawn, no_yawn).
 *
 * KHÁC với CNN classifier (TfliteDrowsinessClassifier):
 *  - CNN: input 64x64, output 1 nhãn cho cả ảnh.
 *  - YOLO: input 640x640, output NHIỀU box + nhãn → phải tự decode + NMS.
 *
 * Dùng model FLOAT32 export từ tools/export_yolo_tflite.py (KHÔNG --int8, vì int8 cần dequant thêm).
 * YOLO11 → output [1, 4+nc, 8400] → class này tự NMS.
 * YOLO26 (NMS-free) → output đã lọc sẵn; có thể bỏ bước NMS (xem ghi chú cuối file).
 */
class YoloDetector(
    context: Context,
    modelPath: String = "yolo_detector.tflite",
    private val labels: Array<String> = arrayOf(
        "close_eyeL", "close_eyeR", "no_yawn", "open_eyeL", "open_eyeR", "yawn"
    ),
    private val inputSize: Int = 640,
    private val confThreshold: Float = 0.30f,
    private val iouThreshold: Float = 0.45f,
) {
    data class Detection(val box: RectF, val label: String, val classId: Int, val score: Float)

    private val interpreter: Interpreter
    private val numClasses = labels.size

    init {
        val opts = Interpreter.Options().apply { numThreads = 4 }
        interpreter = Interpreter(loadModel(context, modelPath), opts)
    }

    private fun loadModel(ctx: Context, path: String): MappedByteBuffer {
        val fd = ctx.assets.openFd(path)
        FileInputStream(fd.fileDescriptor).use { fis ->
            return fis.channel.map(FileChannel.MapMode.READ_ONLY, fd.startOffset, fd.declaredLength)
        }
    }

    /** Tiền xử lý: letterbox 640x640, RGB, /255 → ByteBuffer float32 [1,640,640,3]. */
    private fun preprocess(bitmap: Bitmap): Triple<ByteBuffer, Float, Pair<Float, Float>> {
        val scale = min(inputSize / bitmap.width.toFloat(), inputSize / bitmap.height.toFloat())
        val nw = (bitmap.width * scale).toInt()
        val nh = (bitmap.height * scale).toInt()
        val padX = (inputSize - nw) / 2f
        val padY = (inputSize - nh) / 2f

        val resized = Bitmap.createScaledBitmap(bitmap, nw, nh, true)
        val canvasBmp = Bitmap.createBitmap(inputSize, inputSize, Bitmap.Config.ARGB_8888)
        android.graphics.Canvas(canvasBmp).apply {
            drawColor(android.graphics.Color.rgb(114, 114, 114)) // màu nền letterbox
            drawBitmap(resized, padX, padY, null)
        }

        val buf = ByteBuffer.allocateDirect(1 * inputSize * inputSize * 3 * 4)
        buf.order(ByteOrder.nativeOrder())
        val px = IntArray(inputSize * inputSize)
        canvasBmp.getPixels(px, 0, inputSize, 0, 0, inputSize, inputSize)
        for (p in px) {
            buf.putFloat(((p shr 16) and 0xFF) / 255f) // R
            buf.putFloat(((p shr 8) and 0xFF) / 255f)  // G
            buf.putFloat((p and 0xFF) / 255f)          // B
        }
        buf.rewind()
        return Triple(buf, scale, Pair(padX, padY))
    }

    fun detect(bitmap: Bitmap): List<Detection> {
        val (input, scale, pad) = preprocess(bitmap)

        // Output YOLO11: [1, 4+nc, numAnchors]
        val outShape = interpreter.getOutputTensor(0).shape() // [1, 4+nc, N]
        val ch = outShape[1]
        val n = outShape[2]
        val output = Array(1) { Array(ch) { FloatArray(n) } }
        interpreter.run(input, output)

        val (padX, padY) = pad
        val dets = ArrayList<Detection>()
        for (a in 0 until n) {
            // tìm class có score cao nhất
            var bestC = 0; var bestS = 0f
            for (c in 0 until numClasses) {
                val s = output[0][4 + c][a]
                if (s > bestS) { bestS = s; bestC = c }
            }
            if (bestS < confThreshold) continue

            // box (cx,cy,w,h) normalized [0,1] theo input letterbox → pixel input
            val cx = output[0][0][a] * inputSize
            val cy = output[0][1][a] * inputSize
            val w = output[0][2][a] * inputSize
            val h = output[0][3][a] * inputSize

            // gỡ letterbox → toạ độ ảnh gốc
            val x1 = (cx - w / 2 - padX) / scale
            val y1 = (cy - h / 2 - padY) / scale
            val x2 = (cx + w / 2 - padX) / scale
            val y2 = (cy + h / 2 - padY) / scale
            dets.add(Detection(RectF(x1, y1, x2, y2), labels[bestC], bestC, bestS))
        }
        return nms(dets)
    }

    /** Non-Max Suppression: bỏ box trùng IoU cao, giữ box điểm cao. */
    private fun nms(dets: List<Detection>): List<Detection> {
        val sorted = dets.sortedByDescending { it.score }.toMutableList()
        val keep = ArrayList<Detection>()
        while (sorted.isNotEmpty()) {
            val best = sorted.removeAt(0)
            keep.add(best)
            sorted.removeAll { iou(best.box, it.box) > iouThreshold && it.classId == best.classId }
        }
        return keep
    }

    private fun iou(a: RectF, b: RectF): Float {
        val x1 = max(a.left, b.left); val y1 = max(a.top, b.top)
        val x2 = min(a.right, b.right); val y2 = min(a.bottom, b.bottom)
        val inter = max(0f, x2 - x1) * max(0f, y2 - y1)
        val areaA = (a.right - a.left) * (a.bottom - a.top)
        val areaB = (b.right - b.left) * (b.bottom - b.top)
        return inter / (areaA + areaB - inter + 1e-6f)
    }

    fun close() = interpreter.close()
}

/*
 * ─── GHI CHÚ TÍCH HỢP ───
 * 1. CNN hiện tại (64x64) NHẸ hơn YOLO (640x640) ~10x → YOLO nặng hơn nhiều trên điện thoại.
 *    Cân nhắc: chỉ chạy YOLO mỗi vài frame, hoặc dùng yolo11n (nano) thay yolo11s.
 * 2. YOLO26 (NMS-free): output đã lọc sẵn dạng [1, N, 6] = [x1,y1,x2,y2,conf,cls] → có thể
 *    BỎ hàm nms() và đọc trực tiếp 6 giá trị/ box. Đơn giản hơn YOLO11.
 * 3. Suy ra trạng thái buồn ngủ từ 6 class:
 *      val d = detect(faceBitmap)
 *      val closed = d.any { it.label.startsWith("close_eye") }
 *      val yawning = d.any { it.label == "yawn" }
 *    → kết hợp PERCLOS/temporal như DrowsinessAnalyzer.
 * 4. Nếu dùng model --int8: input/output là int8, phải dequant bằng scale/zero_point
 *    từ interpreter.getInputTensor(0).quantizationParams() — thêm code; nên dùng float32 trước.
 */
