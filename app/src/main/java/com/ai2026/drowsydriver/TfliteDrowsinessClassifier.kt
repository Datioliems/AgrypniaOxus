package com.ai2026.drowsydriver

import android.content.Context
import android.graphics.Bitmap
import org.tensorflow.lite.Interpreter
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.min

data class CnnPrediction(
    val label: String,
    val confidence: Float,
    val probabilities: FloatArray
)

class TfliteDrowsinessClassifier(context: Context) {
    private val inputSize = 64
    private val labels = arrayOf("eyes_closed", "eyes_open")

    private val interpreter: Interpreter? = runCatching {
        val bytes = context.assets.open("drowsiness_model.tflite").readBytes()
        Interpreter(ByteBuffer.allocateDirect(bytes.size).order(ByteOrder.nativeOrder()).put(bytes).apply { rewind() })
    }.getOrNull()

    val isAvailable: Boolean get() = interpreter != null

    fun predict(bitmap: Bitmap): CnnPrediction? {
        val model = interpreter ?: return null
        val input = bitmap.toModelInput()
        val output = Array(1) { FloatArray(labels.size) }
        model.run(input, output)

        return output[0].toPrediction()
    }

    fun predictAverage(bitmaps: List<Bitmap>): CnnPrediction? {
        val model = interpreter ?: return null
        if (bitmaps.isEmpty()) return null

        val summed = FloatArray(labels.size)
        var count = 0
        for (bitmap in bitmaps) {
            val input = bitmap.toModelInput()
            val output = Array(1) { FloatArray(labels.size) }
            model.run(input, output)
            for (i in output[0].indices) {
                summed[i] += output[0][i]
            }
            count++
        }
        if (count == 0) return null
        val average = FloatArray(labels.size) { index -> summed[index] / count }
        return average.toPrediction()
    }

    fun close() {
        interpreter?.close()
    }

    private fun Bitmap.toModelInput(): ByteBuffer {
        val resized = Bitmap.createScaledBitmap(this, inputSize, inputSize, true)
        val buffer = ByteBuffer
            .allocateDirect(1 * inputSize * inputSize * 3 * 4)
            .order(ByteOrder.nativeOrder())

        val pixels = IntArray(inputSize * inputSize)
        resized.getPixels(pixels, 0, inputSize, 0, 0, inputSize, inputSize)
        for (pixel in pixels) {
            buffer.putFloat(((pixel shr 16) and 0xFF) / 255f)
            buffer.putFloat(((pixel shr 8) and 0xFF) / 255f)
            buffer.putFloat((pixel and 0xFF) / 255f)
        }
        buffer.rewind()
        return buffer
    }

    private fun FloatArray.toPrediction(): CnnPrediction {
        var bestIndex = 0
        for (i in 1 until min(labels.size, size)) {
            if (this[i] > this[bestIndex]) bestIndex = i
        }

        return CnnPrediction(
            label = labels[bestIndex],
            confidence = this[bestIndex],
            probabilities = this
        )
    }
}
