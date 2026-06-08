package com.ai2026.drowsydriver

import kotlin.math.min

data class DriverSessionReport(
    val sessionMs: Long = 0L,
    val awakeMs: Long = 0L,
    val eyesClosedMs: Long = 0L,
    val yawningMs: Long = 0L,
    val drowsyMs: Long = 0L,
    val yawnEvents: Int = 0,
    val drowsyEvents: Int = 0,
    val riskScore: Int = 0,
    val recommendation: String = "Start camera to begin monitoring",
)

class DriverSessionTracker {
    private var sessionStartedAt = 0L
    private var lastUpdateAt = 0L
    private var lastState = DriverState.NO_FACE
    private var awakeMs = 0L
    private var eyesClosedMs = 0L
    private var yawningMs = 0L
    private var drowsyMs = 0L
    private var yawnEvents = 0
    private var drowsyEvents = 0

    fun update(status: DriverStatus, nowMs: Long): DriverSessionReport {
        if (sessionStartedAt == 0L) {
            sessionStartedAt = nowMs
            lastUpdateAt = nowMs
            lastState = status.state
        }

        val deltaMs = (nowMs - lastUpdateAt).coerceIn(0L, 2_000L)
        when (lastState) {
            DriverState.AWAKE -> awakeMs += deltaMs
            DriverState.EYES_CLOSED -> eyesClosedMs += deltaMs
            DriverState.YAWNING -> yawningMs += deltaMs
            DriverState.DROWSY -> drowsyMs += deltaMs
            DriverState.NO_FACE -> Unit
        }

        if (lastState != DriverState.YAWNING && status.state == DriverState.YAWNING) yawnEvents++
        if (lastState != DriverState.DROWSY && status.state == DriverState.DROWSY) drowsyEvents++

        lastUpdateAt = nowMs
        lastState = status.state
        return currentReport(nowMs)
    }

    private fun currentReport(nowMs: Long): DriverSessionReport {
        val sessionMs = (nowMs - sessionStartedAt).coerceAtLeast(0L)
        val riskScore = calculateRiskScore(sessionMs)
        return DriverSessionReport(
            sessionMs = sessionMs,
            awakeMs = awakeMs,
            eyesClosedMs = eyesClosedMs,
            yawningMs = yawningMs,
            drowsyMs = drowsyMs,
            yawnEvents = yawnEvents,
            drowsyEvents = drowsyEvents,
            riskScore = riskScore,
            recommendation = recommendationFor(riskScore)
        )
    }

    private fun calculateRiskScore(sessionMs: Long): Int {
        if (sessionMs <= 0L) return 0
        val drowsyRatio = drowsyMs.toFloat() / sessionMs
        val eyesClosedRatio = eyesClosedMs.toFloat() / sessionMs
        val yawnRatio = yawningMs.toFloat() / sessionMs
        val score = drowsyRatio * 70f +
            eyesClosedRatio * 35f +
            yawnRatio * 25f +
            drowsyEvents * 12f +
            yawnEvents * 6f
        return min(100, score.toInt())
    }

    private fun recommendationFor(score: Int): String = when {
        score >= 75 -> "High risk: stop driving and rest 15-20 minutes"
        score >= 45 -> "Warning: take a break, hydrate, and ventilate the cabin"
        score >= 20 -> "Caution: keep posture upright and watch for repeated yawns"
        else -> "Status stable: continue monitoring"
    }
}
