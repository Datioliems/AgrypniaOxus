package com.ai2026.drowsydriver

import android.app.Activity
import android.os.Bundle
import android.view.Gravity
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.TextView

/**
 * Màn hình BÁO CÁO & PHÂN TÍCH — tổng quan mức độ buồn ngủ của phiên lái.
 * Thiết kế lại theo theme đỏ–đen "Stitch": hero card đánh giá rủi ro, các thẻ số liệu
 * bo góc, thanh PERCLOS, và trạng thái tự động gửi định vị.
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
        val limit = EmergencyDispatcher.drowsyLimit(this)
        val autoSent = p.getBoolean("auto_sent_trip", false)
        val lastSentAt = p.getString("last_auto_sent_at", null)

        // Đánh giá mức rủi ro tổng thể
        val (riskTxt, riskColor, riskIcon) = when {
            perclos > 25 || drowsyEvents >= limit -> Triple("RỦI RO CAO", UiKit.RED, "🔴")
            perclos > 12 || drowsyEvents > 0 || yawnEvents >= 3 -> Triple("CẦN CHÚ Ý", UiKit.AMBER, "🟡")
            else -> Triple("AN TOÀN", UiKit.GREEN, "🟢")
        }

        val content = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(UiKit.dp(this@AnalyticsActivity, 16), UiKit.dp(this@AnalyticsActivity, 28),
                       UiKit.dp(this@AnalyticsActivity, 16), UiKit.dp(this@AnalyticsActivity, 28))
        }

        // Tiêu đề
        content.addView(UiKit.title(this, "📊  Báo cáo phiên lái", UiKit.TXT, 24f).apply {
            setPadding(0, 0, 0, UiKit.dp(this@AnalyticsActivity, 16))
        })

        // HERO — đánh giá rủi ro (gradient)
        val hero = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            background = UiKit.gradient(riskColor, 0xFF101012.toInt(), UiKit.dp(this@AnalyticsActivity, 24).toFloat())
            setPadding(UiKit.dp(this@AnalyticsActivity, 22), UiKit.dp(this@AnalyticsActivity, 22),
                       UiKit.dp(this@AnalyticsActivity, 22), UiKit.dp(this@AnalyticsActivity, 22))
            layoutParams = LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT)
                .apply { bottomMargin = UiKit.dp(this@AnalyticsActivity, 16) }
        }
        hero.addView(TextView(this).apply { text = "MỨC ĐỘ AN TOÀN"; setTextColor(0xCCFFFFFF.toInt()); textSize = 13f })
        hero.addView(TextView(this).apply {
            text = "$riskIcon  $riskTxt"; setTextColor(0xFFFFFFFF.toInt()); textSize = 30f
            setTypeface(typeface, android.graphics.Typeface.BOLD); setPadding(0, UiKit.dp(this@AnalyticsActivity, 4), 0, 0)
        })
        hero.addView(TextView(this).apply {
            text = "Thời gian lái: ${fmt(sessionSec)}"; setTextColor(0xDDFFFFFF.toInt()); textSize = 14f
            setPadding(0, UiKit.dp(this@AnalyticsActivity, 8), 0, 0)
        })
        content.addView(hero)

        // Hàng 2 thẻ số liệu (buồn ngủ | ngáp)
        val row = LinearLayout(this).apply { orientation = LinearLayout.HORIZONTAL }
        val s1 = UiKit.statCard(this, "😴", "Số lần buồn ngủ", "$drowsyEvents lần",
            if (drowsyEvents >= limit) UiKit.RED else if (drowsyEvents > 0) UiKit.AMBER else UiKit.GREEN)
        val s2 = UiKit.statCard(this, "🥱", "Số lần ngáp", "$yawnEvents lần", UiKit.TXT)
        (s1.layoutParams as LinearLayout.LayoutParams).apply { width = 0; weight = 1f; rightMargin = UiKit.dp(this@AnalyticsActivity, 6) }
        (s2.layoutParams as LinearLayout.LayoutParams).apply { width = 0; weight = 1f; leftMargin = UiKit.dp(this@AnalyticsActivity, 6) }
        row.addView(s1); row.addView(s2)
        content.addView(row)

        // PERCLOS card có thanh tiến trình
        val perclosCard = UiKit.card(this)
        perclosCard.addView(UiKit.label(this, "PERCLOS — tỉ lệ thời gian nhắm mắt").apply { textSize = 14f })
        perclosCard.addView(UiKit.value(this, String.format("%.1f%%", perclos),
            if (perclos > 15) UiKit.RED else UiKit.GREEN))
        perclosCard.addView(UiKit.progressBar(this, perclos, if (perclos > 15) UiKit.RED else UiKit.GREEN))
        perclosCard.addView(TextView(this).apply {
            text = "Ngưỡng cảnh báo khoa học: 15%"; setTextColor(UiKit.SUB); textSize = 12f
            setPadding(0, UiKit.dp(this@AnalyticsActivity, 6), 0, 0)
        })
        content.addView(perclosCard)

        // Thẻ thời gian buồn ngủ + lần gần nhất
        val timeCard = UiKit.card(this)
        timeCard.addView(UiKit.label(this, "Tổng thời gian buồn ngủ").apply { textSize = 14f })
        timeCard.addView(UiKit.value(this, fmt(drowsySec), if (drowsySec > 0) UiKit.AMBER else UiKit.TXT, 22f))
        timeCard.addView(UiKit.label(this, "Lần buồn ngủ gần nhất: $lastDrowsyAt").apply {
            setPadding(0, UiKit.dp(this@AnalyticsActivity, 8), 0, 0) })
        content.addView(timeCard)

        // Thẻ TỰ ĐỘNG GỬI ĐỊNH VỊ
        val autoCard = UiKit.card(this, if (autoSent) 0xFF2A1417.toInt() else UiKit.CARD)
        autoCard.addView(UiKit.title(this, "📍 Tự động gửi định vị", UiKit.TXT, 17f))
        autoCard.addView(TextView(this).apply {
            text = "Ngưỡng kích hoạt: $limit lần buồn ngủ / chuyến"; setTextColor(UiKit.SUB); textSize = 13f
            setPadding(0, UiKit.dp(this@AnalyticsActivity, 6), 0, UiKit.dp(this@AnalyticsActivity, 10))
        })
        autoCard.addView(UiKit.chip(this,
            if (autoSent) "✓ Đã gửi định vị lúc ${lastSentAt ?: "—"}" else "Chưa kích hoạt trong chuyến này",
            if (autoSent) UiKit.RED else UiKit.GREEN))
        content.addView(autoCard)

        // Khuyến cáo an toàn
        val tip = UiKit.card(this, UiKit.CARD2)
        tip.addView(TextView(this).apply {
            text = "💡 Khuyến cáo an toàn"; setTextColor(UiKit.AMBER); textSize = 15f
            setTypeface(typeface, android.graphics.Typeface.BOLD)
        })
        tip.addView(TextView(this).apply {
            text = "PERCLOS > 15% hoặc nhiều lần buồn ngủ là dấu hiệu mệt mỏi nguy hiểm. " +
                   "Hãy dừng nghỉ 15–30 phút, uống nước hoặc đổi người lái trước khi tiếp tục hành trình."
            setTextColor(UiKit.SUB); textSize = 13f; setPadding(0, UiKit.dp(this@AnalyticsActivity, 6), 0, 0)
        })
        content.addView(tip)

        val scroll = ScrollView(this).apply {
            setBackgroundColor(UiKit.BG)
            addView(content)
        }
        setContentView(scroll)
    }

    private fun fmt(sec: Long): String {
        val h = sec / 3600; val m = (sec % 3600) / 60; val s = sec % 60
        return when {
            h > 0 -> "$h giờ $m phút"
            m > 0 -> "$m phút $s giây"
            else -> "$s giây"
        }
    }
}
