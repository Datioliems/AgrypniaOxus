package com.ai2026.drowsydriver

import android.content.Context
import android.media.AudioAttributes
import android.media.AudioFormat
import android.media.AudioManager
import android.media.AudioTrack
import android.media.ToneGenerator
import android.os.Build
import android.os.VibrationEffect
import android.os.Vibrator
import android.os.VibratorManager
import kotlin.math.PI
import kotlin.math.sin

/**
 * Cảnh báo LEO THANG ĐA GIÁC QUAN + ÂM THANH BINAURAL BEATS
 * (theo CK AI.docx — 1.3.1.6 và 1.3.1.7).
 *
 * Cơ sở khoa học:
 *  • 1.3.1.6 (Moessinger 2021, PLOS ONE): binaural beats dải BETA THẤP 13–21 Hz kích thích
 *    hoạt động vỏ não dải beta → tăng tỉnh táo. Triển khai: tai TRÁI phát tần số carrier f,
 *    tai PHẢI phát f+Δ (Δ=18 Hz ∈ beta) → não cảm nhận "nhịp" 18 Hz tỉnh táo.
 *  • 1.3.1.7 (Beles 2024, Sensors): cảnh báo LEO THANG thị giác → âm thanh → rung; ngưỡng
 *    giờ lái (Wang 2014): hiệu suất suy giảm sau 2h → nhắc nghỉ 15–30 phút.
 */
class AlertController(private val context: Context) {

    private val sampleRate = 44100
    // ToneGenerator ở mức TỐI ĐA (100) — STREAM_ALARM tự theo âm lượng báo thức hệ thống
    private val tone = ToneGenerator(AudioManager.STREAM_ALARM, ToneGenerator.MAX_VOLUME)
    private var lastAlertAt = 0L
    private var drowsyStreakStart = 0L
    private var drivingMinutes = 0
    private var lastRestNudgeAt = 0L
    private var restMark = 0   // mốc nghỉ đã nhắc (0 → 2h → 4h)

    // Binaural beta-beat: tăng vol lên mức to hơn rõ rệt so với trước.
    //   betaSoft: cảnh báo lần đầu / ngáp  (vol 0.78 → nghe rõ mà không chói)
    //   betaLoud: DROWSY kéo dài / mốc nghỉ (vol 1.00 = full scale)
    private val betaSoft by lazy { binauralPcm(carrier = 200.0, beat = 18.0, ms = 1600, vol = 0.78f) }
    private val betaLoud by lazy { binauralPcm(carrier = 220.0, beat = 18.0, ms = 2400, vol = 1.00f) }

    /** Cập nhật thời gian lái (phút) để nhắc nghỉ ở mốc 2h và 4h (Wang 2014). Gọi từ MainActivity. */
    fun updateDrivingTime(minutes: Int) { drivingMinutes = minutes }

    fun update(status: DriverStatus) {
        val now = System.currentTimeMillis()

        // Nhắc nghỉ ở MỐC 2 GIỜ và 4 GIỜ lái liên tục (CK AI §1.3); 4h mạnh hơn 2h.
        val mark = when {
            drivingMinutes >= 240 -> 4
            drivingMinutes >= 120 -> 2
            else -> 0
        }
        if (mark > restMark) {                       // lần đầu chạm mốc → cảnh báo nghỉ mạnh
            restMark = mark
            lastRestNudgeAt = now
            playBinaural(if (mark >= 4) betaLoud else betaSoft)
            vibrate(if (mark >= 4) longArrayOf(0, 400, 200, 400, 200, 400) else longArrayOf(0, 300, 150, 300))
        } else if (mark > 0 && now - lastRestNudgeAt > 600_000L) {  // sau đó nhắc lại mỗi 10 phút
            lastRestNudgeAt = now
            tone.startTone(ToneGenerator.TONE_PROP_PROMPT, 300)
        }

        when (status.state) {
            DriverState.DROWSY -> {
                if (drowsyStreakStart == 0L) drowsyStreakStart = now
                val streak = now - drowsyStreakStart
                val interval = if (streak > 4_000L) 1_200L else 900L
                if (now - lastAlertAt < interval) return
                lastAlertAt = now
                if (streak > 4_000L) {
                    // Tier 3 — buồn ngủ KÉO DÀI: binaural beat TO + rung mạnh
                    playBinaural(betaLoud)
                    vibrate(longArrayOf(0, 400, 150, 400))
                } else {
                    // Tier 2 — buồn ngủ: binaural beta beat + rung vừa
                    playBinaural(betaSoft)
                    vibrate(longArrayOf(0, 250))
                }
            }
            DriverState.YAWNING -> {
                drowsyStreakStart = 0L
                if (now - lastAlertAt < 2_500L) return
                lastAlertAt = now
                tone.startTone(ToneGenerator.TONE_PROP_BEEP2, 220)
                vibrate(longArrayOf(0, 150))
            }
            DriverState.EYES_CLOSED -> {
                // Tier 1 — nhắc nhẹ (thị giác là overlay + beep nhỏ)
                drowsyStreakStart = 0L
                if (now - lastAlertAt < 1_500L) return
                lastAlertAt = now
                tone.startTone(ToneGenerator.TONE_PROP_BEEP, 120)
            }
            else -> drowsyStreakStart = 0L
        }
    }

    // ── Sinh PCM stereo binaural: L=carrier, R=carrier+beat (có fade chống tiếng "tách") ──
    private fun binauralPcm(carrier: Double, beat: Double, ms: Int, vol: Float): ShortArray {
        val n = (ms / 1000.0 * sampleRate).toInt()
        val buf = ShortArray(n * 2)
        val fadeN = (0.02 * sampleRate).toInt().coerceAtLeast(1)
        for (i in 0 until n) {
            val t = i.toDouble() / sampleRate
            val env = vol * when {
                i < fadeN -> i.toFloat() / fadeN
                i > n - fadeN -> (n - i).toFloat() / fadeN
                else -> 1f
            }
            buf[i * 2] = (sin(2 * PI * carrier * t) * Short.MAX_VALUE * env).toInt().toShort()
            buf[i * 2 + 1] = (sin(2 * PI * (carrier + beat) * t) * Short.MAX_VALUE * env).toInt().toShort()
        }
        return buf
    }

    private fun playBinaural(buf: ShortArray) {
        Thread {
            try {
                val track = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
                    AudioTrack.Builder()
                        .setAudioAttributes(
                            AudioAttributes.Builder()
                                .setUsage(AudioAttributes.USAGE_ALARM)
                                .setContentType(AudioAttributes.CONTENT_TYPE_SONIFICATION).build()
                        )
                        .setAudioFormat(
                            AudioFormat.Builder()
                                .setEncoding(AudioFormat.ENCODING_PCM_16BIT)
                                .setSampleRate(sampleRate)
                                .setChannelMask(AudioFormat.CHANNEL_OUT_STEREO).build()
                        )
                        .setBufferSizeInBytes(buf.size * 2)
                        .setTransferMode(AudioTrack.MODE_STATIC).build()
                } else {
                    @Suppress("DEPRECATION")
                    AudioTrack(
                        AudioManager.STREAM_ALARM, sampleRate, AudioFormat.CHANNEL_OUT_STEREO,
                        AudioFormat.ENCODING_PCM_16BIT, buf.size * 2, AudioTrack.MODE_STATIC
                    )
                }
                track.write(buf, 0, buf.size)
                track.play()
                Thread.sleep((buf.size / 2 * 1000L / sampleRate) + 120)
                track.stop(); track.release()
            } catch (_: Exception) {
            }
        }.start()
    }

    private fun vibrate(pattern: LongArray) {
        val vibrator = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
            (context.getSystemService(Context.VIBRATOR_MANAGER_SERVICE) as VibratorManager).defaultVibrator
        } else {
            @Suppress("DEPRECATION") context.getSystemService(Context.VIBRATOR_SERVICE) as Vibrator
        }
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            vibrator.vibrate(VibrationEffect.createWaveform(pattern, -1))
        } else {
            @Suppress("DEPRECATION") vibrator.vibrate(pattern, -1)
        }
    }

    fun close() {
        tone.release()
    }
}
