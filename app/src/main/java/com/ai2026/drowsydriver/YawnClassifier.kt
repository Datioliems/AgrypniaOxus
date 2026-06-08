package com.ai2026.drowsydriver

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import org.tensorflow.lite.Interpreter
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * CNN Yawn Classifier — phân loại vùng miệng: yawn / no_yawn
 *
 * Class order (alphabetical từ image_dataset_from_directory):
 *   index 0 = "no_yawn"
 *   index 1 = "yawn"
 *
 * Input: vùng miệng crop 64×64 từ MediaPipe landmarks
 * Output: [P(no_yawn), P(yawn)]
 *
 * Cách dùng:
 *   val yawnClassifier = YawnClassifier(context)
 *   val pred = yawnClassifier.predict(mouthBitmap)
 *   if (pred?.isYawning == true) { ... }
 */
class YawnClassifier(context: Context) {

    companion object {
        private const val TAG          = "YawnCNN"
        private const val INPUT_SIZE   = 64
        private const val MODEL_FILE   = "yawn_cbam.tflite"
        const val YAWN_CONF_THRESHOLD  = 0.60f  // cần 60% confidence để kết luận yawning

        val LABELS = arrayOf("no_yawn", "yawn")  // alphabetical order!
    }

    data class YawnPrediction(
        val label: String,
        val confidence: Float,
        val isYawning: Boolean,
    )

    private val interpreter: Interpreter? = loadInterpreter(context)
    val isAvailable: Boolean get() = interpreter != null

    fun logStatus() {
        if (isAvailable) {
            Log.i(TAG, "✅ Yawn CNN loaded OK — labels: ${LABELS.toList()}")
        } else {
            Log.w(TAG, "⚠️  yawn_model.tflite not found — dùng MAR fallback")
        }
    }

    /**
     * Phân loại 1 vùng miệng.
     * @param bitmap Vùng miệng crop từ MediaPipe landmarks (bất kỳ kích thước)
     * @return YawnPrediction hoặc null nếu model không khả dụng
     */
    fun predict(bitmap: Bitmap): YawnPrediction? {
        val model = interpreter ?: return null
        val input  = bitmap.toModelInput()
        val output = Array(1) { FloatArray(LABELS.size) }

        runCatching {
            model.run(input, output)
        }.onFailure { e ->
            Log.e(TAG, "Yawn inference failed: ${e.message}")
            return null
        }

        val probs     = output[0]
        val yawnIdx   = LABELS.indexOf("yawn").coerceAtLeast(1)
        val yawnProb  = probs[yawnIdx]
        val bestIdx   = probs.indices.maxByOrNull { probs[it] } ?: return null

        Log.v(TAG, "Yawn probs: no_yawn=${"%.3f".format(probs[0])} yawn=${"%.3f".format(probs[1])}")

        return YawnPrediction(
            label      = LABELS[bestIdx],
            confidence = probs[bestIdx],
            isYawning  = yawnProb >= YAWN_CONF_THRESHOLD,
        )
    }

    fun close() { interpreter?.close() }

    private fun loadInterpreter(context: Context): Interpreter? {
        return runCatching {
            val bytes   = context.assets.open(MODEL_FILE).use { it.readBytes() }
            val buf     = ByteBuffer
                .allocateDirect(bytes.size)
                .order(ByteOrder.nativeOrder())
                .put(bytes).apply { rewind() }
            Interpreter(buf, Interpreter.Options().apply { numThreads = 2 })
        }.onFailure { e ->
            Log.w(TAG, "Cannot load $MODEL_FILE: ${e.message} (MAR will be used instead)")
        }.getOrNull()
    }

    private fun Bitmap.toModelInput(): ByteBuffer {
        val resized = Bitmap.createScaledBitmap(this, INPUT_SIZE, INPUT_SIZE, true)
        val buffer  = ByteBuffer
            .allocateDirect(INPUT_SIZE * INPUT_SIZE * 3 * 4)
            .order(ByteOrder.nativeOrder())
        val pixels  = IntArray(INPUT_SIZE * INPUT_SIZE)
        resized.getPixels(pixels, 0, INPUT_SIZE, 0, 0, INPUT_SIZE, INPUT_SIZE)
        for (pixel in pixels) {
            buffer.putFloat(((pixel shr 16) and 0xFF) / 255f)
            buffer.putFloat(((pixel shr 8)  and 0xFF) / 255f)
            buffer.putFloat((pixel          and 0xFF) / 255f)
        }
        buffer.rewind()
        return buffer
    }
}
