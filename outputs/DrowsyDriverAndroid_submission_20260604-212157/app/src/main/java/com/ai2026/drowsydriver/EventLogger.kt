package com.ai2026.drowsydriver

import android.content.Context
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

class EventLogger(context: Context) {
    private val logFile = File(context.filesDir, "drowsiness_events.csv")
    private val dateFormat = SimpleDateFormat("yyyy-MM-dd HH:mm:ss.SSS", Locale.US)
    private var lastLoggedState: DriverState? = null
    private var lastLoggedAt = 0L

    init {
        if (!logFile.exists()) {
            logFile.writeText(
                "timestamp,state,confidence,ear,mar,closed_eye_ms,yawning_ms,fps,cnn_label,cnn_confidence\n"
            )
        }
    }

    fun logIfNeeded(status: DriverStatus) {
        if (status.state != DriverState.DROWSY && status.state != DriverState.YAWNING) return

        val now = System.currentTimeMillis()
        val sameStateTooSoon = status.state == lastLoggedState && now - lastLoggedAt < 2_000L
        if (sameStateTooSoon) return

        lastLoggedState = status.state
        lastLoggedAt = now

        val row = listOf(
            dateFormat.format(Date(now)),
            status.state.name,
            "%.4f".format(Locale.US, status.confidence),
            "%.4f".format(Locale.US, status.ear),
            "%.4f".format(Locale.US, status.mar),
            status.closedEyeMs.toString(),
            status.yawningMs.toString(),
            "%.2f".format(Locale.US, status.fps),
            status.cnnLabel ?: "",
            "%.4f".format(Locale.US, status.cnnConfidence)
        ).joinToString(",")

        logFile.appendText(row + "\n")
    }

    fun path(): String = logFile.absolutePath
}
