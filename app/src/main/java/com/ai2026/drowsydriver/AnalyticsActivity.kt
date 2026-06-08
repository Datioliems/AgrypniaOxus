package com.ai2026.drowsydriver

import android.app.Activity
import android.os.Bundle
import android.widget.LinearLayout
import android.widget.TextView

/**
 * Màn hình THỐNG KÊ & PHÂN TÍCH — tổng quan thời gian buồn ngủ của phiên lái.
 * Đọc số liệu MainActivity ghi vào SharedPreferences.
 */
class AnalyticsActivity : Activity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val p = getSharedPreferences("agrypnia_prefs", MODE_PRIVATE)

        val sessionSec = p.getLong("session_sec", 0L)
        val drowsyEvents = p.getInt("drowsy_events", 0)
        val yawnEvents = p.getInt("yawn_events", 0)
        val drowsySec = p.getLong("drowsy_sec", 0L)
        val lastDrowsyAt = p.getString("last_drowsy_at", "—") ?: "—"
        val perclos = if (sessionSec > 0) drowsySec * 100.0 / sessionSec else 0.0

        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(48, 64, 48, 48)
            setBackgroundColor(0xFF131313.toInt())
        }
        fun stat(title: String, value: String, hot: Boolean = false) {
            root.addView(TextView(this).apply {
                text = title; setTextColor(0xFFAAAAAA.toInt()); textSize = 14f; setPadding(0, 22, 0, 2)
            })
            root.addView(TextView(this).apply {
                text = value; setTextColor(if (hot) 0xFFFC1C46.toInt() else 0xFFE5E2E1.toInt())
                textSize = 26f
            })
        }
        root.addView(TextView(this).apply {
            text = "📊 Thống kê phiên lái"; setTextColor(0xFFFC1C46.toInt()); textSize = 26f; setPadding(0, 0, 0, 8)
        })
        stat("Thời gian lái phiên này", fmt(sessionSec))
        stat("Số lần buồn ngủ (DROWSY)", "$drowsyEvents lần", drowsyEvents > 0)
        stat("Số lần ngáp", "$yawnEvents lần")
        stat("Tổng thời gian buồn ngủ", fmt(drowsySec), drowsySec > 0)
        stat("PERCLOS (tỉ lệ nhắm mắt)", String.format("%.1f%%", perclos), perclos > 15)
        stat("Lần buồn ngủ gần nhất", lastDrowsyAt)
        root.addView(TextView(this).apply {
            text = "\n💡 PERCLOS > 15% hoặc nhiều lần DROWSY → nên dừng nghỉ 15-30 phút (khuyến cáo an toàn)."
            setTextColor(0xFFAAAAAA.toInt()); textSize = 13f; setPadding(0, 28, 0, 0)
        })
        setContentView(root)
    }

    private fun fmt(sec: Long): String {
        val m = sec / 60; val s = sec % 60
        return if (m > 0) "$m phút $s giây" else "$s giây"
    }
}
