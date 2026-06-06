package com.ai2026.drowsydriver

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import org.tensorflow.lite.Interpreter
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.min

data class CnnPrediction(
    val label: String,
    val confidence: Float,
    val probabilities: FloatArray
)

/**
 * TFLite classifier cho bài toán phân loại trạng thái mắt.
 *
 * QUAN TRỌNG — Class order:
 *   Model được train với image_dataset_from_directory → sắp xếp alphabetically:
 *     index 0 = "eyes_closed"
 *     index 1 = "eyes_open"
 *   Mảng [labels] bên dưới phải khớp đúng thứ tự này.
 *
 * Input cho TFLite:
 *   - Bitmap resize về 64×64
 *   - Normalize pixel về [0, 1] bằng cách chia /255f
 *   - Shape: [1, 64, 64, 3] float32
 *
 * Output:
 *   - Shape: [1, 2] float32 → [P(eyes_closed), P(eyes_open)]
 */
class TfliteDrowsinessClassifier(context: Context) {

    companion object {
        private const val TAG          = "DrowsyCNN"
        private const val INPUT_SIZE   = 64
        private const val MODEL_FILE   = "drowsiness_model.tflite"

        /**
         * Ngưỡng confidence để tin vào dự đoán của CNN.
         * Nếu max(probs) < CNN_CONF_THRESHOLD → trả về null → dùng EAR/MAR fallback.
         *
         * ĐẶT THẤP HƠN 0.70 ở đây vì:
         *   - Model mới chưa cực kỳ confident → 0.55 là hợp lý ban đầu
         *   - Sau khi test thực tế có thể tăng lên 0.65–0.70
         */
        const val CNN_CONF_THRESHOLD = 0.55f

        /** Class order PHẢI khớp với thứ tự alphabetical của folder trong dataset */
        val LABELS = arrayOf("eyes_closed", "eyes_open")
    }

    // ── Load TFLite model ─────────────────────────────────────────────
    private val interpreter: Interpreter? = loadInterpreter(context)

    val isAvailable: Boolean get() = interpreter != null

    /** Gọi để log trạng thái CNN khi khởi động app */
    fun logStatus() {
        if (isAvailable) {
            val inp = interpreter!!.getInputTensor(0)
            val out = interpreter.getOutputTensor(0)
            Log.i(TAG, "✅ CNN loaded OK")
            Log.i(TAG, "   Input shape : ${inp.shape().toList()}")
            Log.i(TAG, "   Output shape: ${out.shape().toList()}")
            Log.i(TAG, "   Labels      : ${LABELS.toList()}")
            Log.i(TAG, "   Conf thresh : $CNN_CONF_THRESHOLD")
        } else {
            Log.e(TAG, "❌ CNN FAILED TO LOAD — app sẽ dùng EAR/MAR only")
            Log.e(TAG, "   Kiểm tra: app/src/main/assets/$MODEL_FILE có tồn tại không?")
        }
    }

    /**
     * Dự đoán trung bình trên danh sách ảnh mắt (thường là mắt trái + mắt phải).
     *
     * @return CnnPrediction nếu model OK và confidence đủ, null nếu không.
     */
    fun predictAverage(bitmaps: List<Bitmap>): CnnPrediction? {
        val model = interpreter ?: return null
        if (bitmaps.isEmpty()) return null

        val summed = FloatArray(LABELS.size)
        var count  = 0

        for (bitmap in bitmaps) {
            val input  = bitmap.toModelInput()
            val output = Array(1) { FloatArray(LABELS.size) }
            runCatching {
                model.run(input, output)
            }.onFailure { e ->
                Log.e(TAG, "CNN inference failed: ${e.message}")
                return null
            }
            for (i in output[0].indices) summed[i] += output[0][i]
            count++
        }

        if (count == 0) return null
        val avg = FloatArray(LABELS.size) { summed[it] / count }

        // Log raw probabilities để debug (verbose → chỉ hiện khi cần)
        Log.v(TAG, "CNN raw avg: eyes_closed=${"%.3f".format(avg[0])} eyes_open=${"%.3f".format(avg[1])}")

        val pred = avg.toPrediction()

        // Chỉ trả về kết quả nếu đủ tự tin
        return if (pred.confidence >= CNN_CONF_THRESHOLD) pred else null
    }

    /** Predict đơn lẻ một bitmap (dùng để test/debug) */
    fun predict(bitmap: Bitmap): CnnPrediction? {
        val model = interpreter ?: return null
        val input  = bitmap.toModelInput()
        val output = Array(1) { FloatArray(LABELS.size) }
        runCatching { model.run(input, output) }.onFailure { return null }
        return output[0].toPrediction()
    }

    fun close() {
        interpreter?.close()
    }

    // ── Private helpers ───────────────────────────────────────────────

    private fun loadInterpreter(context: Context): Interpreter? {
        return runCatching {
            val bytes = context.assets.open(MODEL_FILE).use { it.readBytes() }
            val buf   = ByteBuffer
                .allocateDirect(bytes.size)
                .order(ByteOrder.nativeOrder())
                .put(bytes)
                .apply { rewind() }
            val options = Interpreter.Options().apply {
                numThreads = 2          // 2 thread đủ cho mobile
            }
            Interpreter(buf, options)
        }.onFailure { e ->
            Log.e(TAG, "Cannot load $MODEL_FILE: ${e.javaClass.simpleName}: ${e.message}")
        }.getOrNull()
    }

    /**
     * Chuyển Bitmap thành ByteBuffer float32 cho TFLite.
     *
     * NORMALIZE: pixel [0,255] → [0.0f, 1.0f] bằng /255f
     * Shape output: [1, INPUT_SIZE, INPUT_SIZE, 3] (batch dim handled by run())
     */
    private fun Bitmap.toModelInput(): ByteBuffer {
        val resized = Bitmap.createScaledBitmap(this, INPUT_SIZE, INPUT_SIZE, true)
        val buffer  = ByteBuffer
            .allocateDirect(1 * INPUT_SIZE * INPUT_SIZE * 3 * 4)  // float32 = 4 bytes
            .order(ByteOrder.nativeOrder())

        val pixels = IntArray(INPUT_SIZE * INPUT_SIZE)
        resized.getPixels(pixels, 0, INPUT_SIZE, 0, 0, INPUT_SIZE, INPUT_SIZE)

        for (pixel in pixels) {
            // ARGB → RGB, normalize /255f → [0, 1]
            buffer.putFloat(((pixel shr 16) and 0xFF) / 255f)   // R
            buffer.putFloat(((pixel shr 8)  and 0xFF) / 255f)   // G
            buffer.putFloat((pixel          and 0xFF) / 255f)   // B
        }
        buffer.rewind()
        return buffer
    }

    private fun FloatArray.toPrediction(): CnnPrediction {
        var bestIndex = 0
        for (i in 1 until min(LABELS.size, size)) {
            if (this[i] > this[bestIndex]) bestIndex = i
        }
        return CnnPrediction(
            label        = LABELS[bestIndex],
            confidence   = this[bestIndex],
            probabilities = this.copyOf()
        )
    }
}

// Extension để format Float với 3 chữ số thập phân trong log
private val Float.fmt: String get() = "%.3f".format(this)
