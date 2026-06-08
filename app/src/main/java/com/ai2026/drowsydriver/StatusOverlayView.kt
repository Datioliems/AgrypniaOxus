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
    private val panelPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply { color = Color.argb(220, 19, 19, 19) }
    private val titlePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.WHITE
        textSize = 42f
        typeface = android.graphics.Typeface.DEFAULT_BOLD
    }
    private val textPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.rgb(219, 226, 236)
        textSize = 27f
    }
    private val smallPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.rgb(196, 205, 218)
        textSize = 23f
    }
    private val accentPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.rgb(252, 28, 70)
        textSize = 24f
        typeface = android.graphics.Typeface.DEFAULT_BOLD
    }
    private val warningPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply { color = Color.rgb(239, 68, 68) }
    private var status = DriverStatus(DriverState.NO_FACE, 0f, 0f, 0f, 0L, 0L, 0f)
    private var report = DriverSessionReport()

    fun update(status: DriverStatus, report: DriverSessionReport = DriverSessionReport()) {
        this.status = status
        this.report = report
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

        canvas.drawRoundRect(RectF(24f, 36f, width - 24f, 548f), 18f, 18f, panelPaint)
        warningPaint.color = stateColor
        canvas.drawCircle(58f, 76f, 14f, warningPaint)
        canvas.drawText(labelFor(status.state), 84f, 90f, titlePaint)
        canvas.drawText(
            "EAR %.3f   MAR %.3f   FPS %.1f   LAT %dms".format(
                status.ear,
                status.mar,
                status.fps,
                status.latencyMs
            ),
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
        canvas.drawText("Alert source: ${status.alertSource}", 42f, 290f, textPaint)

        drawSessionReport(canvas, stateColor)
    }

    private fun drawSessionReport(canvas: Canvas, stateColor: Int) {
        warningPaint.color = Color.rgb(48, 56, 70)
        canvas.drawRoundRect(RectF(42f, 320f, width - 42f, 528f), 14f, 14f, warningPaint)

        accentPaint.color = stateColor
        canvas.drawText("Sleep-risk report", 60f, 360f, accentPaint)
        canvas.drawText(
            "Session ${formatDuration(report.sessionMs)}   Risk ${report.riskScore}/100",
            60f,
            397f,
            smallPaint
        )
        canvas.drawText(
            "Drowsy ${formatDuration(report.drowsyMs)} (${report.drowsyEvents})   Yawn ${formatDuration(report.yawningMs)} (${report.yawnEvents})",
            60f,
            432f,
            smallPaint
        )
        canvas.drawText(
            "Eyes closed ${formatDuration(report.eyesClosedMs)}   Awake ${formatDuration(report.awakeMs)}",
            60f,
            467f,
            smallPaint
        )

        warningPaint.color = riskColor(report.riskScore)
        canvas.drawRoundRect(RectF(60f, 486f, width - 60f, 512f), 12f, 12f, warningPaint)
        smallPaint.color = Color.WHITE
        smallPaint.textSize = 21f
        canvas.drawText(report.recommendation, 72f, 506f, smallPaint)
        smallPaint.textSize = 23f
        smallPaint.color = Color.rgb(196, 205, 218)
    }

    private fun labelFor(state: DriverState): String = when (state) {
        DriverState.NO_FACE -> "No face"
        DriverState.AWAKE -> "Awake"
        DriverState.EYES_CLOSED -> "Eyes closed"
        DriverState.YAWNING -> "Yawning"
        DriverState.DROWSY -> "Drowsy alert"
    }

    private fun riskColor(score: Int): Int = when {
        score >= 75 -> Color.rgb(185, 28, 28)
        score >= 45 -> Color.rgb(217, 119, 6)
        score >= 20 -> Color.rgb(202, 138, 4)
        else -> Color.rgb(22, 101, 52)
    }

    private fun formatDuration(ms: Long): String {
        val totalSeconds = (ms / 1000L).coerceAtLeast(0L)
        val minutes = totalSeconds / 60L
        val seconds = totalSeconds % 60L
        return if (minutes > 0L) "%dm%02ds".format(minutes, seconds) else "%ds".format(seconds)
    }
}
