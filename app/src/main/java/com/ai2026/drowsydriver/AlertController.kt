package com.ai2026.drowsydriver

import android.content.Context
import android.media.AudioManager
import android.media.ToneGenerator
import android.os.Build
import android.os.VibrationEffect
import android.os.Vibrator
import android.os.VibratorManager

class AlertController(private val context: Context) {
    private val tone = ToneGenerator(AudioManager.STREAM_ALARM, 85)
    private var lastAlertAt = 0L

    fun update(status: DriverStatus) {
        if (status.state != DriverState.DROWSY) return
        val now = System.currentTimeMillis()
        if (now - lastAlertAt < 1200L) return
        lastAlertAt = now
        tone.startTone(ToneGenerator.TONE_CDMA_ALERT_CALL_GUARD, 350)
        vibrate()
    }

    private fun vibrate() {
        val vibrator = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
            val manager = context.getSystemService(Context.VIBRATOR_MANAGER_SERVICE) as VibratorManager
            manager.defaultVibrator
        } else {
            @Suppress("DEPRECATION")
            context.getSystemService(Context.VIBRATOR_SERVICE) as Vibrator
        }

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            vibrator.vibrate(VibrationEffect.createOneShot(450, VibrationEffect.DEFAULT_AMPLITUDE))
        } else {
            @Suppress("DEPRECATION")
            vibrator.vibrate(450)
        }
    }
}
