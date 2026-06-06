package com.ai2026.drowsydriver

import com.google.mediapipe.tasks.components.containers.NormalizedLandmark
import com.google.mediapipe.tasks.vision.facelandmarker.FaceLandmarkerResult
import kotlin.math.hypot

class DrowsinessAnalyzer {
    private var eyeClosedStartedAt: Long? = null
    private var yawnStartedAt: Long? = null
    private var lastFrameAt = 0L
    private var fps = 0f

    private val eyeClosedThreshold = 0.19f
    private val yawnThreshold = 0.58f
    private val drowsyDurationMs = 1_700L

    fun analyze(result: FaceLandmarkerResult?, timestampMs: Long): DriverStatus {
        updateFps(timestampMs)

        val face = result?.faceLandmarks()?.firstOrNull()
        if (face == null) {
            eyeClosedStartedAt = null
            yawnStartedAt = null
            return DriverStatus(DriverState.NO_FACE, 0f, 0f, 0f, 0L, 0L, fps)
        }

        val leftEar = eyeAspectRatio(face, 33, 160, 158, 133, 153, 144)
        val rightEar = eyeAspectRatio(face, 362, 385, 387, 263, 373, 380)
        val ear = (leftEar + rightEar) / 2f
        val mar = mouthAspectRatio(face)

        val closedEyeMs = if (ear < eyeClosedThreshold) {
            if (eyeClosedStartedAt == null) eyeClosedStartedAt = timestampMs
            timestampMs - (eyeClosedStartedAt ?: timestampMs)
        } else {
            eyeClosedStartedAt = null
            0L
        }

        val yawningMs = if (mar > yawnThreshold) {
            if (yawnStartedAt == null) yawnStartedAt = timestampMs
            timestampMs - (yawnStartedAt ?: timestampMs)
        } else {
            yawnStartedAt = null
            0L
        }

        val state = when {
            closedEyeMs >= drowsyDurationMs -> DriverState.DROWSY
            yawningMs >= 1_000L -> DriverState.YAWNING
            closedEyeMs > 0L -> DriverState.EYES_CLOSED
            else -> DriverState.AWAKE
        }

        val confidence = when (state) {
            DriverState.DROWSY -> (closedEyeMs / drowsyDurationMs.toFloat()).coerceIn(0f, 1f)
            DriverState.YAWNING -> ((mar - yawnThreshold) / 0.25f).coerceIn(0.35f, 1f)
            DriverState.EYES_CLOSED -> ((eyeClosedThreshold - ear) / eyeClosedThreshold).coerceIn(0.35f, 1f)
            DriverState.AWAKE -> 1f - ((eyeClosedThreshold - ear).coerceAtLeast(0f) / eyeClosedThreshold)
            DriverState.NO_FACE -> 0f
        }

        return DriverStatus(state, confidence, ear, mar, closedEyeMs, yawningMs, fps)
    }

    private fun eyeAspectRatio(
        face: List<NormalizedLandmark>,
        p1: Int,
        p2: Int,
        p3: Int,
        p4: Int,
        p5: Int,
        p6: Int
    ): Float {
        val verticalA = distance(face[p2], face[p6])
        val verticalB = distance(face[p3], face[p5])
        val horizontal = distance(face[p1], face[p4]).coerceAtLeast(0.0001f)
        return (verticalA + verticalB) / (2f * horizontal)
    }

    private fun mouthAspectRatio(face: List<NormalizedLandmark>): Float {
        val verticalA = distance(face[13], face[14])
        val verticalB = distance(face[82], face[87])
        val verticalC = distance(face[312], face[317])
        val horizontal = distance(face[78], face[308]).coerceAtLeast(0.0001f)
        return (verticalA + verticalB + verticalC) / (3f * horizontal)
    }

    private fun distance(a: NormalizedLandmark, b: NormalizedLandmark): Float {
        return hypot((a.x() - b.x()).toDouble(), (a.y() - b.y()).toDouble()).toFloat()
    }

    private fun updateFps(timestampMs: Long) {
        val delta = timestampMs - lastFrameAt
        if (lastFrameAt > 0L && delta > 0L) {
            val instant = 1000f / delta
            fps = if (fps == 0f) instant else fps * 0.85f + instant * 0.15f
        }
        lastFrameAt = timestampMs
    }
}
