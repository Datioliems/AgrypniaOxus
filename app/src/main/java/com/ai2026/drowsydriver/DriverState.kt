package com.ai2026.drowsydriver

enum class DriverState {
    NO_FACE,
    AWAKE,
    EYES_CLOSED,
    YAWNING,
    DROWSY
}

data class DriverStatus(
    val state: DriverState,
    val confidence: Float,
    val ear: Float,
    val mar: Float,
    val closedEyeMs: Long,
    val yawningMs: Long,
    val fps: Float,
    val latencyMs: Long = 0L,
    val cnnLabel: String? = null,
    val cnnConfidence: Float = 0f,
    val alertSource: String = "EAR/MAR"
)
