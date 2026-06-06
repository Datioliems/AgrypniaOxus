package com.ai2026.drowsydriver

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import android.util.AttributeSet
import android.view.View

class StatusOverlayView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null
) : View(context, attrs) {
    private val panelPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply { color = Color.argb(210, 8, 12, 18) }
    private val titlePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.WHITE
        textSize = 42f
        typeface = android.graphics.Typeface.DEFAULT_BOLD
    }
    private val textPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.rgb(219, 226, 236)
        textSize = 27f
    }
    private val warningPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply { color = Color.rgb(239, 68, 68) }
    private var status = DriverStatus(DriverState.NO_FACE, 0f, 0f, 0f, 0L, 0L, 0f)

    fun update(status: DriverStatus) {
        this.status = status
        invalidate()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        val stateColor = when (status.state) {
            DriverState.AWAKE -> Color.rgb(34, 197, 94)
            DriverState.EYES_CLOSED -> Color.rgb(245, 158, 11)
            DriverState.YAWNING -> Color.rgb(249, 115, 22)
            DriverState.DROWSY -> Color.rgb(239, 68, 68)
            DriverState.NO_FACE -> Color.rgb(148, 163, 184)
        }

        canvas.drawRoundRect(RectF(24f, 36f, width - 24f, 292f), 18f, 18f, panelPaint)
        warningPaint.color = stateColor
        canvas.drawCircle(58f, 76f, 14f, warningPaint)
        canvas.drawText(labelFor(status.state), 84f, 90f, titlePaint)
        canvas.drawText(
            "EAR %.3f   MAR %.3f   FPS %.1f".format(status.ear, status.mar, status.fps),
            42f,
            145f,
            textPaint
        )
        canvas.drawText(
            "Closed %.1fs   Yawn %.1fs   Confidence %.0f%%".format(
                status.closedEyeMs / 1000f,
                status.yawningMs / 1000f,
                status.confidence * 100f
            ),
            42f,
            194f,
            textPaint
        )
        val cnnText = if (status.cnnLabel != null) {
            "CNN ${status.cnnLabel}   %.0f%%".format(status.cnnConfidence * 100f)
        } else {
            "CNN model pending"
        }
        canvas.drawText(cnnText, 42f, 242f, textPaint)
    }

    private fun labelFor(state: DriverState): String = when (state) {
        DriverState.NO_FACE -> "No face"
        DriverState.AWAKE -> "Awake"
        DriverState.EYES_CLOSED -> "Eyes closed"
        DriverState.YAWNING -> "Yawning"
        DriverState.DROWSY -> "Drowsy alert"
    }
}
